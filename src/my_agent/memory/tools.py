"""记忆工具：``MemoryRead`` / ``MemoryWrite`` / ``MemoryForget``（§8.7）。

| 工具 | 参数 | 权限类别 | 行为 |
| --- | --- | --- | --- |
| ``MemoryRead`` | ``scope?``, ``section?`` | read | 返回指定范围的条目（含 id） |
| ``MemoryWrite`` | ``text``, ``scope?``, ``section?``, ``confidence?`` | write | 走完整管线（§8.4 步骤 3–5） |
| ``MemoryForget`` | ``id`` | write | 删除指定条目（备份可恢复） |

三个工具都由 :class:`~my_agent.tools.registry.ToolRegistry` 统一执行，因此**都走权限流水线**：
``MemoryWrite`` / ``MemoryForget`` 属写类，在 ``plan`` 模式下被 deny（§8.7）。

规格：``my_agent-开发文档.md`` §8.7、§14.3；``CONTRACTS.md`` §17.1。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from my_agent.errors import MyAgentError
from my_agent.memory.extractor import coerce_confidence
from my_agent.memory.sections import label_for, sections_for
from my_agent.memory.store import MemoryEntry, MemoryStore
from my_agent.models.tools import ToolContext, ToolResult, ToolSpec
from my_agent.tools.base import BaseTool

if TYPE_CHECKING:
    from my_agent.memory.writer import MemoryWriter

__all__ = ["MemoryForgetTool", "MemoryReadTool", "MemoryWriteTool"]

_MEMORY_READ_DESCRIPTION = """读取长期记忆条目（全局 + 项目），返回带条目 id 的列表。

何时用：需要确认「用户偏好 / 本项目约定」等已记住的事实，或写入前先查重时。
何时不用：① 不要用它查代码或文件内容——用 Read / Grep / Glob；
② 不要用它查历史对话——用 ArchiveSearch；③ 不要用它写记忆——用 MemoryWrite。
副作用：无（只读，不修改磁盘文件）。可能触发旧格式条目补 id 的一次性回写。
参数：scope 可选（global / project / all，默认 all）；section 可选（§8.2 的章节键）。"""

_MEMORY_WRITE_DESCRIPTION = """写入一条长期记忆（走完整写入管线：范围选择 → 冲突合并 → 落盘 + 审计）。

何时用：用户明确要求「记住」某事，或对话中出现稳定的偏好/约定/红线。
何时不用：① 不要用它记本次任务的临时细节或可从代码推断的常识；
② 不要用它保存大段文件内容——用 Write；③ 重复内容会被去噪丢弃（返回 SKIP，属正常结果）。
副作用：**会修改磁盘文件**（`.myagent/memory.md` 或 `~/.myagent/memory.md`，改动前自动备份 `.bak`）。
参数：text 必填（单行 ≤300 字符）；scope / section 可选，不指定时由记忆归档器选择。"""

_MEMORY_FORGET_DESCRIPTION = """删除一条长期记忆（按条目 id）。

何时用：用户明确要求「别再记着 X」或某条记忆已被确认过时。
何时不用：① 不确定 id 时先用 MemoryRead 查；② 不要用它删除代码或文件——用 Bash/Write。
副作用：**会修改磁盘文件**（删除前自动备份 `.bak`，可人工恢复）。"""

_SCOPES: tuple[str, ...] = ("global", "project", "all")


class MemoryReadTool(BaseTool):
    """``MemoryRead``：返回指定范围的记忆条目（含 id，§8.7）。

    Args:
        global_store: 全局记忆存储。
        project_store: 项目记忆存储。
    """

    spec = ToolSpec(
        name="MemoryRead",
        description=_MEMORY_READ_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": list(_SCOPES),
                    "description": "读取范围：global / project / all（默认 all）",
                },
                "section": {
                    "type": "string",
                    "description": "只读某个章节键（§8.2），如 user_preference、stack_commands",
                },
            },
        },
        category="read",
        is_read_only=True,
    )

    def __init__(self, global_store: MemoryStore, project_store: MemoryStore) -> None:
        self.global_store = global_store
        self.project_store = project_store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """读取记忆条目。

        Args:
            args: ``scope?``（global/project/all）、``section?``。
            ctx: 执行上下文（提供 ``call_id``）。

        Returns:
            成功时 ``content`` 为带 id 的条目列表；参数非法时 ``ok=False``。
        """
        scope = str(args.get("scope") or "all").strip().lower()
        if scope not in _SCOPES:
            return self.fail(ctx, f"参数 scope 非法：{scope!r}（应为 global / project / all）")
        section = args.get("section")
        section_key = str(section).strip() if section else ""
        if section_key and not any(
            section_key in sections_for(item) for item in ("global", "project")
        ):
            return self.fail(ctx, f"参数 section 非法：{section_key!r}（见 §8.2 章节表）")

        try:
            lines: list[str] = []
            for name, store in (("global", self.global_store), ("project", self.project_store)):
                if scope != "all" and scope != name:
                    continue
                entries = await asyncio.to_thread(store.read)
                if section_key:
                    entries = [entry for entry in entries if entry.section == section_key]
                lines.append(f"[{name}] {store.path}")
                lines.extend(_render(entries))
                lines.append("")
        except MyAgentError as exc:
            return self.fail(ctx, exc.user_message())

        body = "\n".join(lines).strip()
        return self.ok(ctx, body or "（暂无记忆）")


class MemoryWriteTool(BaseTool):
    """``MemoryWrite``：走完整写入管线写一条记忆（§8.7、§8.4 步骤 3–5）。

    Args:
        writer: 记忆写入器（``/memory add`` 用的同一个实例，验收 D-9）。
    """

    spec = ToolSpec(
        name="MemoryWrite",
        description=_MEMORY_WRITE_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要记住的内容（单行，≤300 字符）"},
                "scope": {
                    "type": "string",
                    "enum": ["global", "project"],
                    "description": "写入范围；不指定时由记忆归档器选择",
                },
                "section": {
                    "type": "string",
                    "description": "章节键（§8.2）；不指定时由记忆归档器选择",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "置信度：用户明确陈述=high（默认），模型推断=medium",
                },
            },
            "required": ["text"],
        },
        category="write",
        is_read_only=False,
    )

    def __init__(self, writer: MemoryWriter) -> None:
        self.writer = writer

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """写入记忆（不指定 scope/section 时由模型选择，§8.7）。

        Args:
            args: ``text`` 必填；``scope?`` / ``section?`` / ``confidence?``。
            ctx: 执行上下文（提供 ``turn_index`` / ``call_id``）。

        Returns:
            成功时 ``content`` 为「写入位置 + 动作 + 条目 id」；参数非法或落盘失败时 ``ok=False``。
        """
        raw_text = args.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            return self.fail(ctx, "缺少必填参数 text（要记住的内容，单行字符串）")
        scope = args.get("scope")
        scope_value = str(scope).strip().lower() if scope else None
        if scope_value is not None and scope_value not in ("global", "project"):
            return self.fail(ctx, f"参数 scope 非法：{scope_value!r}（应为 global / project）")
        section = args.get("section")
        section_value = str(section).strip() if section else None
        confidence = coerce_confidence(args.get("confidence", "high"), default="high")

        try:
            records = await self.writer.write_candidates(
                [(raw_text, confidence)],
                turn=ctx.turn_index,
                scope=scope_value,
                section=section_value,
            )
        except MyAgentError as exc:
            return self.fail(ctx, exc.user_message())
        if not records:
            return self.ok(ctx, "未写入：该内容与已有记忆重复，或未通过去噪门槛（§8.4 步骤 2）。")
        record = records[0]
        where = f"{record.scope}/{record.section}"
        target = record.target_id or "新条目"
        return self.ok(ctx, f"{record.action} → {where}（id={target}）\n理由：{record.why}")


class MemoryForgetTool(BaseTool):
    """``MemoryForget``：按 id 删除一条记忆（§8.7）。

    Args:
        global_store: 全局记忆存储。
        project_store: 项目记忆存储。
    """

    spec = ToolSpec(
        name="MemoryForget",
        description=_MEMORY_FORGET_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "条目 id（形如 m-3f2a1b，用 MemoryRead 查）"},
            },
            "required": ["id"],
        },
        category="write",
        is_read_only=False,
        is_destructive=True,
    )

    def __init__(self, global_store: MemoryStore, project_store: MemoryStore) -> None:
        self.global_store = global_store
        self.project_store = project_store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """删除指定记忆条目（删除前自动 ``.bak``）。

        Args:
            args: ``id`` 必填。
            ctx: 执行上下文（提供 ``call_id``）。

        Returns:
            成功时 ``content`` 为被删条目全文；id 不存在时 ``ok=False``。
        """
        entry_id = args.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            return self.fail(ctx, "缺少必填参数 id（条目 id，形如 m-3f2a1b）")
        key = entry_id.strip()
        try:
            for store in (self.global_store, self.project_store):
                removed = await asyncio.to_thread(store.delete, key)
                if removed is not None:
                    return self.ok(ctx, f"已删除 {key}（{removed.section}）：{removed.text}")
        except MyAgentError as exc:
            return self.fail(ctx, exc.user_message())
        return self.fail(ctx, f"未找到条目 {key}（可先用 MemoryRead 查看有效 id）")


def _render(entries: Sequence[MemoryEntry]) -> list[str]:
    """把条目渲染成给模型看的行（保留 id/date/conf，便于后续 UPDATE/DELETE）。

    Args:
        entries: 记忆条目列表。

    Returns:
        每个章节一行标题 + 若干条目行；无条目时返回占位行。
    """
    if not entries:
        return ["（暂无记忆）"]
    out: list[str] = []
    current = ""
    for entry in entries:
        if entry.section != current:
            current = entry.section
            out.append(f"### {label_for(current)}（{current}）")
        out.append(entry.to_line())
    return out
