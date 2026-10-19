"""上下文组装（§7.6）与 ``/context`` 报告（§7.7.5）。

最终消息顺序（**不可乱**，§7.6）：

1. ``system``：``MAIN_SYSTEM_PROMPT`` + 工作目录摘要 + 记忆块 + skills/mcp 索引块
   （用 ``\\n\\n---\\n\\n`` 分隔）；
2. ``history``：按 §7.5 裁剪 + §7.7 压缩后的视图；
3. 当前用户消息（若尚未加入 ``history``）。

超预算丢弃优先级（§7.6，低 → 高）：MCP/skills 明细 → 低优先级记忆 →
工作目录摘要包明细 → 工具描述（压到 ``name + 一行描述``，**永不整体删除**）→
**系统提示词永不裁剪**。具体降级算法在 :mod:`my_agent.context.budget`。

:meth:`ContextAssembler.build` 是**纯函数语义**（C3/C-10）：
不改 ``SessionState.history``、不改传参块文本、同一输入产生同一输出。

``format_context_report`` 按 §7.7.5 的样例输出渲染 ``/context`` 文本。
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from my_agent.config.defaults import (
    KEEP_RECENT_LARGE_RESULTS,
    MODEL_WINDOW_DEFAULT,
    PROMPT_BUDGET_TOKENS,
)
from my_agent.context.budget import (
    BLOCK_MEMORY,
    BLOCK_REPOMAP,
    BLOCK_SKILLS_MCP,
    BLOCK_SYSTEM,
    PromptBudget,
    context_value,
    effective_context,
    fit_to_budget,
    needs_compaction,
    watermark,
)
from my_agent.context.pruner import prune_large_results
from my_agent.llm.prompts import MAIN_SYSTEM_PROMPT
from my_agent.models.messages import Message

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期导入
    from my_agent.config.schema import Config
    from my_agent.llm.tokenizer import TokenCounter
    from my_agent.models.session import SessionState

__all__ = [
    "BAR_CELLS",
    "ContextAssembler",
    "BuildResult",
    "SECTION_SEPARATOR",
    "format_context_report",
]

#: 系统消息内各块的连接符（§7.6 逐字）。
SECTION_SEPARATOR: str = "\n\n---\n\n"

#: 水位条的格子数（§7.7.5 示例为 10 格）。
BAR_CELLS: int = 10
#: 水位条填充/空白字符（§7.7.5 逐字）。
BAR_FILLED: str = "▓"
BAR_EMPTY: str = "░"
#: 报告里「标签 + 数字」的对齐宽度（显示宽度，非行为阈值）。
_LABEL_WIDTH: int = 14
_NUMBER_WIDTH: int = 8


class BuildResult(BaseModel):
    """``ContextAssembler.build()`` 的产物（CONTRACTS §7 + 报告所需补充字段）。"""

    messages: list[Message]
    total_tokens: int = 0
    system_tokens: int = 0
    history_tokens: int = 0
    watermark: float = 0.0
    pruned_count: int = 0
    compacted: bool = False
    dropped_blocks: list[str] = Field(default_factory=list)
    system_text: str = ""
    prompt_budget: PromptBudget = Field(default_factory=PromptBudget)

    # ---- 【补齐】调用方（engine/ui）渲染与装配所需的补充字段 ----
    tools_schema: list[dict[str, Any]] = Field(
        default_factory=list, description="可能被降级的工具 schema，供 Provider tools 参数使用"
    )
    effective_tokens: int = Field(default=0, description="有效上下文（水位分母）")
    model_window: int = Field(default=0, description="模型上下文窗口")
    pruned_total: int = Field(default=0, description="视图中累计被清除的大结果条数")
    keep_large_results: int = Field(
        default=KEEP_RECENT_LARGE_RESULTS, description="本次裁剪保留的大结果条数"
    )
    needs_compaction: bool = Field(default=False, description="水位是否已达压缩触发点（§7.7）")


class ContextAssembler:
    """把五件套 + 历史视图组装成一次请求（§7.6）。

    Args:
        config: 全局配置（读 ``context.*`` 覆盖 defaults）。
        counter: token 计数器；**必须**与其他预算判断共用同一实例（§7.2）。
    """

    def __init__(self, *, config: Config, counter: TokenCounter) -> None:
        self._config = config
        self._counter = counter

    def build(
        self,
        *,
        session: SessionState,
        tools_schema: list[dict[str, Any]],
        memory_block: str,
        skills_block: str,
        repomap: str,
        pending_user: Message | None = None,
    ) -> BuildResult:
        """组装一次请求的消息与预算报告（纯函数语义）。

        Args:
            session: 会话状态；``history`` 为唯一真相源，本方法**不修改**它。
            tools_schema: 工具注册表给出的 function schema 列表（第 3 件套）。
            memory_block: 记忆注入块（第 4 件套）。
            skills_block: skills + MCP 索引块（第 5 件套）。
            repomap: 工作目录摘要块（第 2 件套）。
            pending_user: 当前用户消息；若尚未出现在 ``history`` 中则追加到最后。

        Returns:
            :class:`BuildResult`（messages 顺序：system → history → 当前用户消息）。
        """
        keep = int(
            context_value(self._config, "keep_recent_large_results", KEEP_RECENT_LARGE_RESULTS)
        )
        history_view = prune_large_results(list(session.history), keep)
        pruned_count = sum(
            1 for old, new in zip(session.history, history_view, strict=False) if old is not new
        )
        pruned_total = sum(1 for message in history_view if message.kind == "cleared")
        history_messages = list(history_view)
        if pending_user is not None and all(
            message.id != pending_user.id for message in history_messages
        ):
            history_messages.append(pending_user)

        fitted = fit_to_budget(
            {
                BLOCK_SYSTEM: MAIN_SYSTEM_PROMPT,
                BLOCK_REPOMAP: repomap,
                BLOCK_MEMORY: memory_block,
                BLOCK_SKILLS_MCP: skills_block,
            },
            counter=self._counter,
            tools_schema=tools_schema,
            limit=int(context_value(self._config, "prompt_budget_tokens", PROMPT_BUDGET_TOKENS)),
        )
        system_text = _compose_system(fitted.blocks)
        system_message = Message(role="system", content=system_text)

        model_window = int(context_value(self._config, "model_window", MODEL_WINDOW_DEFAULT))
        effective = effective_context(model_window, self._config)
        history_tokens = self._counter.count_messages(history_messages)
        ratio = watermark(history_tokens, effective)
        return BuildResult(
            messages=[system_message, *history_messages],
            total_tokens=fitted.budget.total + history_tokens,
            system_tokens=self._counter.count_text(system_text),
            history_tokens=history_tokens,
            watermark=ratio,
            pruned_count=pruned_count,
            compacted=session.compact_count > 0,
            dropped_blocks=list(fitted.budget.dropped_blocks),
            system_text=system_text,
            prompt_budget=fitted.budget,
            tools_schema=fitted.tools_schema,
            effective_tokens=effective,
            model_window=model_window,
            pruned_total=pruned_total,
            keep_large_results=keep,
            needs_compaction=needs_compaction(ratio),
        )


def format_context_report(build_result: BuildResult, session: SessionState) -> str:
    """渲染 ``/context`` 报告（§7.7.5 样例格式，C-12）。

    Args:
        build_result: 最近一次 :meth:`ContextAssembler.build` 的结果。
        session: 会话状态（提供压缩次数、最近压缩轮次、归档文件）。

    Returns:
        多行报告文本：水位（含进度条与 ``历史 / 有效上下文``）、各块 token、
        裁剪统计、压缩统计、归档路径（有则显示）。
    """
    effective = build_result.effective_tokens
    lines = [
        f"上下文水位  {_bar(build_result.watermark)}  "
        f"{build_result.watermark * 100:.1f}%  "
        f"({build_result.history_tokens:,} / {effective:,} tokens)"
    ]
    lines.append(_row("系统提示词", build_result.prompt_budget.system))
    lines.append(_row("工作目录摘要", build_result.prompt_budget.repomap))
    lines.append(_row("tools 提示词", build_result.prompt_budget.tools))
    lines.append(_row("记忆", build_result.prompt_budget.memory))
    lines.append(_row("skills/mcp", build_result.prompt_budget.skills_mcp))
    history_line = _row("对话历史", build_result.history_tokens)
    if session.compact_count:
        turns_ago = max(0, session.turn_index - session.last_compact_turn)
        history_line += f"  (压缩 <{session.compact_count}> 次，最近一次 {turns_ago} 轮前)"
    lines.append(history_line)
    lines.append(
        f"裁剪：已清理 {build_result.pruned_total} 条大结果"
        f"（保留最近 {build_result.keep_large_results} 条）"
    )
    archive = _latest_archive(session)
    if archive:
        lines.append(f"归档：{archive}")
    if build_result.dropped_blocks:
        labels = "、".join(_block_label(name) for name in build_result.dropped_blocks)
        lines.append(f"超预算降级：{labels}")
    if build_result.prompt_budget.warning:
        lines.append(build_result.prompt_budget.warning)
    if build_result.compacted and build_result.needs_compaction:
        lines.append("⚠ 水位仍 ≥ 80%，下一轮 ASSEMBLE 前会再次自动压缩")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _compose_system(blocks: dict[str, str]) -> str:
    """按 §7.6 顺序拼接系统消息（system → repomap → memory → skills/mcp）。"""
    ordered = (
        blocks.get(BLOCK_SYSTEM, ""),
        blocks.get(BLOCK_REPOMAP, ""),
        blocks.get(BLOCK_MEMORY, ""),
        blocks.get(BLOCK_SKILLS_MCP, ""),
    )
    return SECTION_SEPARATOR.join(part for part in ordered if part.strip())


def _bar(ratio: float) -> str:
    """水位进度条（§7.7.5：10 格 ``▓``/``░``）。"""
    filled = min(BAR_CELLS, max(0, round(ratio * BAR_CELLS)))
    return BAR_FILLED * filled + BAR_EMPTY * (BAR_CELLS - filled)


def _row(label: str, value: int) -> str:
    """渲染一行「标签 + 右对齐千分位数字」。"""
    padding = " " * max(1, _LABEL_WIDTH - _display_width(label))
    return f"  {label}{padding}{value:>{_NUMBER_WIDTH},}"


def _display_width(text: str) -> int:
    """终端显示宽度（CJK 全角按 2 计），用于报告对齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _block_label(name: str) -> str:
    """块名 → 中文标签（§7.6 表格用词）。"""
    mapping = {
        BLOCK_SYSTEM: "系统提示词",
        BLOCK_REPOMAP: "工作目录摘要的包明细",
        BLOCK_MEMORY: "记忆块的低优先级条目",
        BLOCK_SKILLS_MCP: "MCP/skills 索引的详细描述",
        "tools": "工具描述",
    }
    return mapping.get(name, name)


def _latest_archive(session: SessionState) -> str | None:
    """从 ``SessionState.archive_files`` 里找出最近的归档分片（相对项目根显示）。"""
    for recorded in reversed(session.archive_files):
        name = str(recorded)
        normalized = name.replace("\\", "/")
        is_shard = normalized.rsplit("/", 1)[-1].startswith(_SHARD_PREFIX)
        if _ARCHIVE_MARKER not in normalized and not is_shard:
            continue
        return _relative(name, session.project_root)
    return None


#: 归档目录标记（用于在 archive_files 里识别归档分片，§7.7.2）。
_ARCHIVE_MARKER: str = "/archive/"
#: 归档分片文件名前缀（§7.7.2）。
_SHARD_PREFIX: str = "turns-"


def _relative(path: str, project_root: str) -> str:
    """尽量把绝对路径显示为项目内相对路径（§7.7.5 样例为相对路径）。"""
    if not project_root:
        return path
    try:
        return Path(path).relative_to(Path(project_root)).as_posix()
    except ValueError:
        return path
