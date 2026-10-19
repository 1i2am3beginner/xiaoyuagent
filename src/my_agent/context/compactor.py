"""全文压缩（§7.7）。

触发：``watermark >= COMPRESS_TRIGGER_RATIO``（每次 ``ASSEMBLE`` 前检查）或用户 ``/compact``。
核心要求（上游）：**不是简单丢弃** —— 先把完整对话历史冻结进归档，再用摘要替换，
并用 ``ArchiveSearch`` 工具回读原文。

流程（§7.7.1）：

1. **冻结快照**：把当前 ``history`` 全量写入归档文件（§7.7.2）；
2. 选取被压缩区间：除「最近 K 轮」外的所有历史（``K = COMPACT_KEEP_RECENT_TURNS``）；
3. 对被压缩区间分段摘要（``SEGMENT_TURNS`` 轮一段，附录 B.4 模板）；
4. 用「分层摘要消息」替换被压缩区间：
   ``history = [summary_msg_1, ..., summary_msg_n] + 最近 K 轮原文``；
5. 压缩后立即重算水位；仍 ``>= 0.80`` → 减小 K 重试一次，仍超则强制 K=1 并告警；
6. 记录 :class:`~my_agent.models.audit.CompactEvent`（由调用方交给 ``AuditWriter``）。

关键约束：

- 摘要中每个事实必须带轮次号或文件路径（提示词硬性要求 + 本模块兜底补头）；
- 摘要本身不超过 ``SUMMARY_MAX_TOKENS``，超出要求模型重写**一次**，再超则截断；
- **幂等可重入**：压缩失败（模型不可用）→ 走 §7.7.4 **规则化摘要**降级，
  归档照写、显式告警，**绝不产生半压缩状态**（新 history 只在全部摘要就绪后一次性替换）。

所有文件 IO 经 :func:`asyncio.to_thread`，不阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from my_agent.config.defaults import (
    COMPACT_KEEP_RECENT_TURNS,
    COMPRESS_TRIGGER_RATIO,
    MODEL_WINDOW_DEFAULT,
    SEGMENT_TURNS,
    SUMMARY_MAX_TOKENS,
)
from my_agent.context.budget import (
    context_value,
    effective_context,
    needs_compaction,
    truncate_to_tokens,
    watermark,
)
from my_agent.errors import MyAgentError
from my_agent.llm.prompts import COMPACTOR_PROMPT
from my_agent.models.audit import CompactEvent
from my_agent.models.messages import Message, new_message_id

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期导入
    from my_agent.config.schema import Config
    from my_agent.context.archive import ArchiveStore
    from my_agent.llm.client import LLMClient
    from my_agent.llm.tokenizer import TokenCounter
    from my_agent.models.session import SessionState

__all__ = [
    "Compactor",
    "RULE_SUMMARY_WARNING",
    "SegmentSummary",
]

#: §7.7.4 规则化摘要里用户消息原文的 token 上限。
RULE_SUMMARY_USER_TOKENS: int = 2_000
#: 单条用户消息在规则化摘要里的最大字符数（展示细节，非行为阈值）。
RULE_SUMMARY_USER_MAX_CHARS: int = 600
#: 规则化摘要里列出的文件路径条数上限（展示细节）。
RULE_SUMMARY_MAX_FILES: int = 20
#: 送入模型的 transcript 单条消息最大字符数（成本保护，非行为阈值）。
TRANSCRIPT_MSG_MAX_CHARS: int = 4_000
#: transcript 二次压缩时单条消息的最小字符数。
TRANSCRIPT_MIN_MSG_CHARS: int = 200
#: 摘要缺头时补的标题模板（§7.7.1 字段，逐字）。
SUMMARY_HEADING: str = "## 会话摘要（第 {turn_start}–{turn_end} 轮）"
#: 摘要缺归档行时补的行模板（§7.7.1 字段，逐字）。
SUMMARY_ARCHIVE_LINE: str = "- 归档文件：{archive_path}（轮次区间 {turn_start}-{turn_end}）"
#: §7.7.4 规定的告警行（逐字）。
RULE_SUMMARY_WARNING: str = "⚠ 本次压缩未使用模型，采用规则化摘要；可用 ArchiveSearch 回读原文"
#: 摘要超限时的重写指令（§7.7.1「超出则要求模型压缩重写一次」）。
REWRITE_INSTRUCTION: str = (
    "\n\n【重要】你上一次输出约 {actual} tokens，超出 {max_tokens} tokens 上限。"
    "请重新输出一份更短的摘要，只保留带轮次号/文件路径的关键事实，"
    "并保持既有 Markdown 结构。"
)
#: 提取「像路径的 token」。
_PATH_RE = re.compile(r"[\w.\-]+(?:[/\\][\w.\-]+)+")
#: transcript / 规则化摘要里的换行折叠符。
_FLATTEN_SEP: str = " / "
#: 摘要消息固定为 ``system``（分层摘要 = 背景信息，不是模型发言）。
SUMMARY_ROLE: Literal["system"] = "system"


class SegmentSummary(BaseModel):
    """一段被压缩历史的摘要（§7.7.1 步骤 3–4）。"""

    turn_start: int
    turn_end: int
    text: str
    tokens: int = 0
    mode: Literal["llm", "rule_based"] = "llm"


class Compactor:
    """把长历史压缩为「分层摘要消息 + 最近 K 轮原文」（§7.7）。

    Args:
        config: 全局配置（读 ``context.compact_keep_recent_turns`` 等覆盖项）。
        counter: 默认 token 计数器；每次调用也可单独传入，二者取先者。
    """

    def __init__(self, *, config: Config, counter: TokenCounter | None = None) -> None:
        self._config = config
        self._counter = counter

    async def maybe_compact(
        self,
        *,
        session: SessionState,
        llm: LLMClient,
        archive: ArchiveStore,
        counter: TokenCounter | None = None,
        focus: str | None = None,
        force: bool = False,
    ) -> CompactEvent | None:
        """水位达到触发点时自动压缩（§7.7「触发」）。

        Args:
            session: 会话状态。
            llm: 模型客户端（角色 ``compactor``）。
            archive: 归档库。
            counter: token 计数器（缺省用构造时的实例）。
            focus: 本次压缩的重点（可选，透传给摘要提示词）。
            force: ``True`` 时忽略水位直接压缩（``/compact``）。

        Returns:
            :class:`CompactEvent`；水位未达触发点且 ``force=False`` 时返回 ``None``。
        """
        resolved = self._resolve_counter(counter)
        current = watermark(
            resolved.count_messages(list(session.history)),
            effective_context(self._model_window(), self._config),
        )
        if not force and not needs_compaction(current, ratio=self._ratio()):
            return None
        return await self.compact(
            session=session, llm=llm, archive=archive, counter=resolved, focus=focus
        )

    async def compact(
        self,
        *,
        session: SessionState,
        llm: LLMClient,
        archive: ArchiveStore,
        counter: TokenCounter | None = None,
        focus: str | None = None,
        force_keep_recent: int | None = None,
    ) -> CompactEvent:
        """执行一次全文压缩（§7.7.1 六步，失败降级 §7.7.4）。

        Args:
            session: 会话状态；``history`` 会被替换为「摘要 + 最近 K 轮原文」。
            llm: 模型客户端（角色 ``compactor``）。
            archive: 归档库（冻结快照的落盘目标）。
            counter: token 计数器（缺省用构造时的实例）。
            focus: 本次压缩的重点（可选）。
            force_keep_recent: 强制保留最近轮数（缺省读配置/默认值）。

        Returns:
            :class:`CompactEvent`（含模式、归档路径、压缩前后水位、摘要 token 数）。

        Raises:
            OSError: 归档落盘失败（此时 history 保持不变，绝不半压缩）。
        """
        resolved = self._resolve_counter(counter)
        effective = effective_context(self._model_window(), self._config)
        ratio = self._ratio()
        history = list(session.history)
        turns = sorted({message.turn_index for message in history})
        turn_start = turns[0] if turns else 0
        turn_end = turns[-1] if turns else session.turn_index
        before = watermark(resolved.count_messages(history), effective)
        if not history:
            return CompactEvent(
                session_id=session.session_id,
                turn_start=turn_start,
                turn_end=turn_end,
                mode="llm",
                watermark_before=before,
                watermark_after=before,
                note="无历史可压缩",
            )

        # 步骤 1：冻结快照（先归档；失败则整体放弃，history 不变）
        archive_path = await asyncio.to_thread(
            archive.archive_messages,
            session.session_id,
            history,
            turn_start=turn_start,
            turn_end=turn_end,
        )
        self._remember_archive(session, archive_path)

        base_keep = (
            force_keep_recent
            if force_keep_recent is not None
            else int(
                context_value(
                    self._config, "compact_keep_recent_turns", COMPACT_KEEP_RECENT_TURNS
                )
            )
        )
        cache: dict[tuple[int, int], SegmentSummary] = {}
        notes: list[str] = []
        allow_llm = True
        chosen: tuple[list[Message], list[SegmentSummary], int, float] | None = None
        attempts = _keep_attempts(base_keep)
        for index, keep in enumerate(attempts):
            compactable, kept = _split_history(history, keep)
            if not compactable:
                notes.append("无可压缩区间（历史轮次不超过保留轮数）")
                break
            if all(message.kind == "summary" for message in compactable):
                notes.append("可压缩区间仅含摘要消息，已跳过（幂等：无新增原文）")
                break
            summaries: list[SegmentSummary] = []
            for seg_start, seg_end, segment in _segment(compactable, SEGMENT_TURNS):
                summary = cache.get((seg_start, seg_end))
                if summary is None:
                    summary = await self._summarize_segment(
                        segment,
                        turn_start=seg_start,
                        turn_end=seg_end,
                        archive_path=archive_path,
                        llm=llm,
                        counter=resolved,
                        focus=focus,
                        allow_llm=allow_llm,
                    )
                    cache[(seg_start, seg_end)] = summary
                if summary.mode == "rule_based":
                    allow_llm = False
                summaries.append(summary)
            new_history = [
                _summary_message(summary, archive_path=archive_path) for summary in summaries
            ] + kept
            after = watermark(resolved.count_messages(new_history), effective)
            chosen = (new_history, summaries, keep, after)
            if after < ratio:
                break
            if index < len(attempts) - 1:
                notes.append(
                    f"压缩后水位仍为 {after * 100:.1f}%（≥ {ratio * 100:.0f}%），"
                    f"把保留轮次从 {keep} 降到 {attempts[index + 1]} 重试"
                )
            else:
                notes.append(
                    f"⚠ 已压缩到只剩最近 {keep} 轮，水位仍为 {after * 100:.1f}%"
                    f"（≥ {ratio * 100:.0f}%），请 /clear 或缩小单轮输入"
                )

        if chosen is None:
            return CompactEvent(
                session_id=session.session_id,
                turn_start=turn_start,
                turn_end=turn_end,
                archive_path=str(archive_path),
                messages_archived=len(history),
                mode="llm",
                watermark_before=before,
                watermark_after=before,
                keep_recent_turns=base_keep,
                note="；".join(notes) or "未发生压缩",
            )

        new_history, summaries, keep_used, after = chosen
        mode: Literal["llm", "rule_based"] = (
            "rule_based" if any(item.mode == "rule_based" for item in summaries) else "llm"
        )
        if mode == "rule_based":
            notes.insert(0, RULE_SUMMARY_WARNING)
        session.history = new_history
        session.compact_count += 1
        session.last_compact_turn = session.turn_index
        session.context_watermark = after
        session.updated_at = time.time()
        return CompactEvent(
            session_id=session.session_id,
            turn_start=turn_start,
            turn_end=turn_end,
            archive_path=str(archive_path),
            messages_archived=len(history),
            mode=mode,
            summary_tokens=sum(item.tokens for item in summaries),
            summary_ids=[message.id for message in new_history if message.kind == "summary"],
            watermark_before=before,
            watermark_after=after,
            keep_recent_turns=keep_used,
            note="；".join(notes),
        )

    # ------------------------------------------------------------------ #
    # 内部：摘要生成
    # ------------------------------------------------------------------ #
    async def _summarize_segment(
        self,
        messages: Sequence[Message],
        *,
        turn_start: int,
        turn_end: int,
        archive_path: Path,
        llm: LLMClient,
        counter: TokenCounter,
        focus: str | None,
        allow_llm: bool,
    ) -> SegmentSummary:
        """生成一段摘要；模型不可用时降级为规则化摘要（§7.7.4）。"""
        reason = ""
        if allow_llm:
            try:
                text = await self._llm_summary(
                    messages,
                    turn_start=turn_start,
                    turn_end=turn_end,
                    archive_path=archive_path,
                    llm=llm,
                    counter=counter,
                    focus=focus,
                )
            except MyAgentError as exc:
                reason = f"{exc.code}：{exc.detail or exc}"
            else:
                return SegmentSummary(
                    turn_start=turn_start,
                    turn_end=turn_end,
                    text=text,
                    tokens=counter.count_text(text),
                    mode="llm",
                )
        text = _rule_summary(
            messages,
            turn_start=turn_start,
            turn_end=turn_end,
            archive_path=archive_path,
            counter=counter,
            reason=reason,
        )
        return SegmentSummary(
            turn_start=turn_start,
            turn_end=turn_end,
            text=text,
            tokens=counter.count_text(text),
            mode="rule_based",
        )

    async def _llm_summary(
        self,
        messages: Sequence[Message],
        *,
        turn_start: int,
        turn_end: int,
        archive_path: Path,
        llm: LLMClient,
        counter: TokenCounter,
        focus: str | None,
    ) -> str:
        """调用 ``compactor`` 角色生成摘要，并把结果压进 ``SUMMARY_MAX_TOKENS``。"""
        prompt = _render_template(
            COMPACTOR_PROMPT,
            max_tokens=SUMMARY_MAX_TOKENS,
            turn_start=turn_start,
            turn_end=turn_end,
            archive_path=str(archive_path),
            history=_transcript(
                messages,
                counter=counter,
                max_tokens=max(1, self._model_window() - SUMMARY_MAX_TOKENS),
            ),
        )
        if focus:
            prompt += f"\n\n【本次压缩重点】{focus}"
        response = await llm.complete_role(
            "compactor", user=prompt, max_tokens=SUMMARY_MAX_TOKENS
        )
        text = _ensure_header(
            str(getattr(response, "text", "") or ""),
            turn_start=turn_start,
            turn_end=turn_end,
            archive_path=archive_path,
        )
        actual = counter.count_text(text)
        if actual > SUMMARY_MAX_TOKENS:
            retry_prompt = prompt + REWRITE_INSTRUCTION.format(
                actual=actual, max_tokens=SUMMARY_MAX_TOKENS
            )
            rewritten = await llm.complete_role(
                "compactor", user=retry_prompt, max_tokens=SUMMARY_MAX_TOKENS
            )
            candidate = _ensure_header(
                str(getattr(rewritten, "text", "") or ""),
                turn_start=turn_start,
                turn_end=turn_end,
                archive_path=archive_path,
            )
            if candidate.strip() and counter.count_text(candidate) < actual:
                text = candidate
        if counter.count_text(text) > SUMMARY_MAX_TOKENS:
            text = truncate_to_tokens(text, max_tokens=SUMMARY_MAX_TOKENS, counter=counter)
        return text

    # ------------------------------------------------------------------ #
    # 内部：配置与登记
    # ------------------------------------------------------------------ #
    def _resolve_counter(self, counter: TokenCounter | None) -> TokenCounter:
        """取本次调用使用的计数器（显式传入 > 构造实例 > 默认实例）。"""
        if counter is not None:
            return counter
        if self._counter is not None:
            return self._counter
        from my_agent.llm.tokenizer import get_counter

        self._counter = get_counter(None)
        return self._counter

    def _model_window(self) -> int:
        """模型上下文窗口（§15.3 ``llm.context_window``，缺省 200k）。"""
        return int(context_value(self._config, "model_window", MODEL_WINDOW_DEFAULT))

    def _ratio(self) -> float:
        """压缩触发比例（§7.2 ``COMPRESS_TRIGGER_RATIO``）。"""
        return float(context_value(self._config, "compress_trigger_ratio", COMPRESS_TRIGGER_RATIO))

    def _remember_archive(self, session: SessionState, path: Path) -> None:
        """把归档分片登记进 ``SessionState.archive_files``（供导出/清理）。"""
        recorded = str(path)
        if recorded not in session.archive_files:
            session.archive_files.append(recorded)


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _keep_attempts(base_keep: int) -> list[int]:
    """§7.7.1 步骤 5 的 K 递减序列：``K → K/2 → 1``（去重、保序）。"""
    attempts = [max(1, int(base_keep))]
    for candidate in (max(1, attempts[0] // 2), 1):
        if candidate not in attempts:
            attempts.append(candidate)
    return attempts


def _split_history(
    history: Sequence[Message], keep_recent: int
) -> tuple[list[Message], list[Message]]:
    """切出 ``(被压缩区间, 最近 K 轮原文)``。"""
    turns = sorted({message.turn_index for message in history})
    if keep_recent <= 0 or len(turns) <= keep_recent:
        return [], list(history)
    keep_turns = set(turns[-keep_recent:])
    split = next(
        index for index, message in enumerate(history) if message.turn_index in keep_turns
    )
    return list(history[:split]), list(history[split:])


def _segment(
    messages: Sequence[Message], segment_turns: int
) -> list[tuple[int, int, list[Message]]]:
    """按 ``SEGMENT_TURNS`` 轮分窗（锚定首轮，保证多次尝试的窗口可复用）。"""
    if not messages:
        return []
    anchor = min(message.turn_index for message in messages)
    width = max(1, segment_turns)
    groups: dict[int, list[Message]] = {}
    for message in messages:
        groups.setdefault((message.turn_index - anchor) // width, []).append(message)
    return [
        (
            min(item.turn_index for item in groups[key]),
            max(item.turn_index for item in groups[key]),
            groups[key],
        )
        for key in sorted(groups)
    ]


def _summary_message(summary: SegmentSummary, *, archive_path: Path) -> Message:
    """把分段摘要包成「分层摘要消息」（§7.7.1 步骤 4）。"""
    return Message(
        id=new_message_id(),
        role=SUMMARY_ROLE,
        content=summary.text,
        kind="summary",
        turn_index=summary.turn_end,
        archive_ref=str(archive_path),
        tokens=summary.tokens,
    )


def _ensure_header(
    text: str, *, turn_start: int, turn_end: int, archive_path: Path
) -> str:
    """保证摘要带标题与归档行（否则回读工具无法定位，§7.7.1 关键约束）。"""
    heading = SUMMARY_HEADING.format(turn_start=turn_start, turn_end=turn_end)
    archive_line = SUMMARY_ARCHIVE_LINE.format(
        archive_path=archive_path, turn_start=turn_start, turn_end=turn_end
    )
    lines = [line for line in text.strip().splitlines()]
    if not lines:
        return f"{heading}\n{archive_line}"
    if not lines[0].lstrip().startswith("#"):
        lines.insert(0, heading)
    if not any("归档文件" in line for line in lines):
        lines.insert(1, archive_line)
    return "\n".join(lines)


def _transcript(
    messages: Sequence[Message], *, counter: TokenCounter, max_tokens: int
) -> str:
    """把消息渲染成带轮次号的 transcript；超预算时按条压缩后再截断。"""
    text = "\n".join(_transcript_line(message, TRANSCRIPT_MSG_MAX_CHARS) for message in messages)
    if counter.count_text(text) <= max_tokens:
        return text
    per_message = max(
        TRANSCRIPT_MIN_MSG_CHARS,
        int(TRANSCRIPT_MSG_MAX_CHARS * max_tokens / max(1, counter.count_text(text))),
    )
    text = "\n".join(_transcript_line(message, per_message) for message in messages)
    return truncate_to_tokens(text, max_tokens=max_tokens, counter=counter)


def _transcript_line(message: Message, limit: int) -> str:
    """单条消息的 transcript 行：``[turn N | role] 内容 (tool_call: …)``。"""
    parts = [f"[turn {message.turn_index} | {message.role}]"]
    if message.content:
        parts.append(_flatten(message.content, limit))
    for call in message.tool_calls:
        arguments = json.dumps(call.arguments, ensure_ascii=False, default=str)
        parts.append(f"(tool_call: {call.name}({_clip(arguments, limit)}))")
    return " ".join(parts)


def _rule_summary(
    messages: Sequence[Message],
    *,
    turn_start: int,
    turn_end: int,
    archive_path: Path,
    counter: TokenCounter,
    reason: str,
) -> str:
    """§7.7.4 确定性摘要：用户消息原文 + 工具调用摘要 + 文件路径集合 + 告警行。"""
    lines = [
        SUMMARY_HEADING.format(turn_start=turn_start, turn_end=turn_end),
        SUMMARY_ARCHIVE_LINE.format(
            archive_path=archive_path, turn_start=turn_start, turn_end=turn_end
        ),
        RULE_SUMMARY_WARNING + (f"（原因：{reason}）" if reason else ""),
        "- 用户目标：",
    ]
    users = [message for message in messages if message.role == "user" and message.content.strip()]
    used = 0
    for index, message in enumerate(users):
        text = _flatten(message.content, RULE_SUMMARY_USER_MAX_CHARS)
        cost = counter.count_text(text)
        if used + cost > RULE_SUMMARY_USER_TOKENS:
            lines.append(f"  - …（其余 {len(users) - index} 条用户消息见归档文件）")
            break
        lines.append(f"  - [turn {message.turn_index}] {text}")
        used += cost
    if not users:
        lines.append("  - （该区间无用户消息）")
    calls = Counter(
        call.name for message in messages for call in message.tool_calls if call.name
    )
    if calls:
        summary = "，".join(f"{name}×{count}" for name, count in sorted(calls.items()))
        lines.append(f"- 工具调用：{summary}")
    failures = [
        message
        for message in messages
        if message.role == "tool" and ("执行失败" in message.content or "权限被拒绝" in message.content)
    ]
    if failures:
        lines.append(f"- 错误与修复：{len(failures)} 条工具失败记录（详见归档文件）")
    files = sorted({token for message in messages for token in _PATH_RE.findall(message.content)})
    if files:
        listed = "，".join(files[:RULE_SUMMARY_MAX_FILES])
        lines.append(f"- 关键文件与改动：{listed}")
    lines.append("- 未决问题 / 待办：见归档文件与最近保留轮次")
    lines.append("- 重要决策与原因：未使用模型，规则化摘要不臆测决策")
    return "\n".join(lines)


def _render_template(template: str, /, **values: Any) -> str:
    """渲染附录 B 模板（占位符为 ``{name}``，逐字采用附录 B.4）。"""
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        return template


def _flatten(text: str, limit: int) -> str:
    """把多行文本压成单行并截断。"""
    flat = _FLATTEN_SEP.join(part.strip() for part in text.splitlines() if part.strip())
    return _clip(flat, limit)


def _clip(text: str, limit: int) -> str:
    """截断到 ``limit`` 字符（超出加省略号）。"""
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"
