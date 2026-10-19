"""``Edit`` / ``MultiEdit`` 工具：字面替换，写前备份，失败即保持原样（§14.2、§14.5）。

关键不变量：
- **J-2**：``old_string`` 在文件中不唯一且未开 ``replace_all`` → 报错，且**文件一个字节都不改**。
- **J-3**：``MultiEdit`` 顺序应用多个编辑，任一失败 → 用备份把原文件恢复原状。
- 两个工具都在写前调用 :meth:`BackupManager.snapshot`，并把「写入后 sha1」登记到 manifest，
  供 ``/undo`` 做 §14.5 冲突检测。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from my_agent.models.tools import ToolContext, ToolResult, ToolSpec
from my_agent.tools.backup import BackupEntry, BackupManager
from my_agent.tools.base import (
    BaseTool,
    atomic_write_text,
    count_lines,
    display_path,
    read_text_normalized,
    resolve_path,
)

__all__ = ["EditTool", "MultiEditTool"]

_EDIT_DESCRIPTION = """在文件中做**字面量**（非正则）字符串替换，要求 old_string 在文件中唯一。

何时用：已明确要改哪几行、改动局部且该片段在文件中唯一时（先用 Read 确认上下文）。
何时不用：① 同一片段出现多次而只想改一处时不要硬改——先补足上下文让 old_string 唯一，
   或者确认要全改时设置 replace_all=true；② 整文件重写请用 Write；
   ③ 本工具是字面匹配，不要指望用正则或模糊匹配。
副作用：**会修改磁盘文件**；old_string 不唯一时直接报错且文件保持原样；写入前原文件自动备份，可用 /undo 撤销。"""

_MULTIEDIT_DESCRIPTION = """对**同一个文件**按顺序应用多个 Edit（字面替换），任一失败则整体回滚，文件保持原样。

何时用：同一文件内有多处互相独立的修改，且每处 old_string 都能唯一匹配时。
何时不用：① 跨多个文件时不要用它——分别调用 Edit；② 只有一处修改时用 Edit 即可；
   ③ 编辑之间互相依赖（后一处依赖前一处的文本）时不要用——用 Edit 逐次确认再改。
副作用：**会修改磁盘文件**；全部编辑成功才落盘，任一失败即从备份恢复原内容；写入前自动备份，可用 /undo 撤销。"""


def _first_hit_line(text: str, index: int) -> int:
    """计算某个字符下标所在的行号（1 起）。

    Args:
        text: 全文（已归一化为 ``\\n``）。
        index: 字符下标。

    Returns:
        行号。
    """
    return text.count("\n", 0, index) + 1


class EditTool(BaseTool):
    """字面替换单个片段（唯一性不满足则报错且不改文件）。

    Args:
        backups: 备份管理器（§14.5）。
    """

    spec = ToolSpec(
        name="Edit",
        description=_EDIT_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文件路径（绝对或相对 cwd）"},
                "old_string": {"type": "string", "description": "要被替换的原文（字面匹配）"},
                "new_string": {"type": "string", "description": "替换后的新文本"},
                "replace_all": {
                    "type": "boolean",
                    "description": "true = 替换所有出现处（默认 false，要求唯一匹配）",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
        category="write",
        is_read_only=False,
        is_destructive=True,
        requires_user_approval_hint=True,
    )

    def __init__(self, backups: BackupManager) -> None:
        self.backups = backups

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """执行字面替换。

        Args:
            args: ``file_path``、``old_string``、``new_string``、可选 ``replace_all``。
            ctx: 执行上下文。

        Returns:
            成功时为简短确认；参数非法、文件不存在、未匹配、不唯一或写入失败时 ``ok=False``。
        """
        raw_path = args.get("file_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return self.fail(ctx, "缺少必填参数 file_path（应为文件路径字符串）")
        old_string = args.get("old_string")
        new_string = args.get("new_string")
        if not isinstance(old_string, str) or old_string == "":
            return self.fail(ctx, "缺少必填参数 old_string（应为非空字符串，字面匹配）")
        if not isinstance(new_string, str):
            return self.fail(ctx, "缺少必填参数 new_string（应为字符串，可为空串表示删除）")
        if old_string == new_string:
            return self.fail(ctx, "old_string 与 new_string 相同，没有任何修改")
        replace_all = bool(args.get("replace_all", False))

        path = resolve_path(raw_path, ctx)
        if path.is_dir():
            return self.fail(ctx, f"{display_path(path, ctx)} 是目录，无法编辑")
        return await asyncio.to_thread(self._edit, path, old_string, new_string, replace_all, ctx)

    def _edit(
        self,
        path: Path,
        old_string: str,
        new_string: str,
        replace_all: bool,
        ctx: ToolContext,
    ) -> ToolResult:
        """同步替换实现（由 :meth:`run` 放进线程执行）。

        Args:
            path: 已解析的绝对路径。
            old_string: 待替换原文。
            new_string: 替换后文本。
            replace_all: 是否替换全部出现处。
            ctx: 执行上下文。

        Returns:
            编辑结果；唯一性不满足时**不触碰文件**。
        """
        shown = display_path(path, ctx)
        if not path.is_file():
            return self.fail(ctx, f"文件不存在：{shown}（新建文件请用 Write）")
        try:
            text, newline = read_text_normalized(path)
        except (UnicodeDecodeError, OSError) as exc:
            return self.fail(ctx, f"无法按文本读取 {shown}（疑似二进制）：{exc}")

        count = text.count(old_string)
        if count == 0:
            return self.fail(
                ctx,
                f"在 {shown} 中未找到 old_string（文件未修改）。"
                "请先用 Read 确认原文（注意缩进与换行必须完全一致）",
            )
        if count > 1 and not replace_all:
            return self.fail(
                ctx,
                f"old_string 在 {shown} 中出现 {count} 次，无法确定替换哪一处（文件未修改，J-2）。"
                "请补充前后文使其唯一，或设置 replace_all=true 全部替换",
            )

        line = _first_hit_line(text, text.index(old_string))
        new_text = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )
        result = self._persist(path, new_text, newline, ctx, shown)
        if not result.ok:
            return result
        replaced = count if replace_all else 1
        delta = count_lines(new_text) - count_lines(text)
        return self.ok(
            ctx,
            f"已编辑 {shown}：替换 {replaced} 处（自第 {line} 行起），"
            f"行数 {delta:+d}；原文件已备份，可用 /undo 撤销",
            duration_ms=0,
        )

    def _persist(
        self,
        path: Path,
        new_text: str,
        newline: str,
        ctx: ToolContext,
        shown: str,
    ) -> ToolResult:
        """备份 + 原子写入 + 登记写后指纹。

        Args:
            path: 目标文件。
            new_text: 新内容（``\\n`` 归一化）。
            newline: 原文件换行符。
            ctx: 执行上下文。
            shown: 展示用路径。

        Returns:
            写入失败时返回 ``ok=False``；成功返回 ``ok=True``（content 为空，由调用方组装确认）。
        """
        try:
            self.backups.snapshot(path, tool=self.spec.name, call_id=ctx.call_id)
            atomic_write_text(path, new_text, newline=newline)
        except OSError as exc:
            return self.fail(ctx, f"写入失败：{shown}（{exc}）")
        self.backups.mark_written(path, call_id=ctx.call_id)
        return self.ok(ctx, "")


class MultiEditTool(BaseTool):
    """同一文件上顺序应用多个编辑，失败整体回滚。

    Args:
        backups: 备份管理器（§14.5）。
    """

    spec = ToolSpec(
        name="MultiEdit",
        description=_MULTIEDIT_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文件路径（绝对或相对 cwd）"},
                "edits": {
                    "type": "array",
                    "description": "按顺序应用的编辑列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string", "description": "要被替换的原文"},
                            "new_string": {"type": "string", "description": "替换后的新文本"},
                            "replace_all": {"type": "boolean", "description": "是否全部替换"},
                        },
                        "required": ["old_string", "new_string"],
                    },
                },
            },
            "required": ["file_path", "edits"],
        },
        category="write",
        is_read_only=False,
        is_destructive=True,
        requires_user_approval_hint=True,
    )

    def __init__(self, backups: BackupManager) -> None:
        self.backups = backups

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """顺序应用多个编辑。

        Args:
            args: ``file_path`` 与 ``edits``（每项含 ``old_string``/``new_string``/``replace_all``）。
            ctx: 执行上下文。

        Returns:
            全部成功时为简短确认；任一失败时为 ``ok=False`` 且文件保持原样（J-3）。
        """
        raw_path = args.get("file_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return self.fail(ctx, "缺少必填参数 file_path（应为文件路径字符串）")
        raw_edits = args.get("edits")
        if not isinstance(raw_edits, list) or not raw_edits:
            return self.fail(ctx, "缺少必填参数 edits（应为非空的编辑数组）")

        edits: list[tuple[str, str, bool]] = []
        for index, item in enumerate(raw_edits):
            if not isinstance(item, dict):
                return self.fail(ctx, f"edits[{index}] 非法：应为对象 {{old_string, new_string}}")
            old_string = item.get("old_string")
            new_string = item.get("new_string")
            if not isinstance(old_string, str) or old_string == "":
                return self.fail(ctx, f"edits[{index}].old_string 非法：应为非空字符串")
            if not isinstance(new_string, str):
                return self.fail(ctx, f"edits[{index}].new_string 非法：应为字符串")
            if old_string == new_string:
                return self.fail(ctx, f"edits[{index}] 的 old_string 与 new_string 相同")
            edits.append((old_string, new_string, bool(item.get("replace_all", False))))

        path = resolve_path(raw_path, ctx)
        if path.is_dir():
            return self.fail(ctx, f"{display_path(path, ctx)} 是目录，无法编辑")
        return await asyncio.to_thread(self._multi_edit, path, edits, ctx)

    def _multi_edit(
        self,
        path: Path,
        edits: list[tuple[str, str, bool]],
        ctx: ToolContext,
    ) -> ToolResult:
        """同步多编辑实现：先备份，再在内存里顺序应用，最后一次性落盘。

        Args:
            path: 已解析的绝对路径。
            edits: ``(old_string, new_string, replace_all)`` 列表。
            ctx: 执行上下文。

        Returns:
            结果；任一编辑失败时从备份恢复原文件并返回 ``ok=False``。
        """
        shown = display_path(path, ctx)
        if not path.is_file():
            return self.fail(ctx, f"文件不存在：{shown}（新建文件请用 Write）")
        try:
            original, newline = read_text_normalized(path)
        except (UnicodeDecodeError, OSError) as exc:
            return self.fail(ctx, f"无法按文本读取 {shown}（疑似二进制）：{exc}")

        entry: BackupEntry | None = None
        try:
            backup_path = self.backups.snapshot(path, tool=self.spec.name, call_id=ctx.call_id)
        except OSError as exc:
            return self.fail(ctx, f"备份失败，已放弃编辑：{shown}（{exc}）")
        if backup_path is not None:
            entry = self.backups.latest()

        working = original
        hits: list[int] = []
        for index, (old_string, new_string, replace_all) in enumerate(edits):
            count = working.count(old_string)
            if count == 0:
                return self._rollback(
                    ctx, entry, shown, f"edits[{index}] 的 old_string 未匹配（已回滚，文件未改变）"
                )
            if count > 1 and not replace_all:
                return self._rollback(
                    ctx,
                    entry,
                    shown,
                    f"edits[{index}] 的 old_string 出现 {count} 次且未开 replace_all"
                    "（已回滚，文件未改变，J-3）",
                )
            hits.append(_first_hit_line(working, working.index(old_string)))
            working = (
                working.replace(old_string, new_string)
                if replace_all
                else working.replace(old_string, new_string, 1)
            )

        if working == original:
            return self.fail(ctx, f"edits 未产生任何变化：{shown}")
        try:
            atomic_write_text(path, working, newline=newline)
        except OSError as exc:
            return self._rollback(ctx, entry, shown, f"写入失败：{exc}（已回滚）")
        self.backups.mark_written(path, call_id=ctx.call_id)
        delta = count_lines(working) - count_lines(original)
        return self.ok(
            ctx,
            f"已在 {shown} 应用 {len(edits)} 处编辑（起始行 {hits}），行数 {delta:+d}；"
            "原文件已备份，可用 /undo 撤销",
        )

    def _rollback(
        self,
        ctx: ToolContext,
        entry: BackupEntry | None,
        shown: str,
        reason: str,
    ) -> ToolResult:
        """用备份把文件恢复原状后返回失败结果（J-3）。

        Args:
            ctx: 执行上下文。
            entry: 备份记录（``None`` 表示未产生备份，此时文件本就未被写入）。
            shown: 展示用路径。
            reason: 失败原因。

        Returns:
            ``ok=False`` 的结果。
        """
        if entry is None:
            return self.fail(ctx, f"{reason}；{shown} 从未被写入")
        try:
            self.backups.restore(entry)
        except OSError as exc:
            return self.fail(ctx, f"{reason}；且回滚失败（{exc}），请检查 {shown} 与备份 {entry.backup}")
        return self.fail(ctx, f"{reason}；已从备份恢复 {shown}")
