"""记忆写入管线步骤 4：冲突合并（§8.4、§8.5、附录 B.7）。

对每条已定位的候选：

1. 读该 ``(scope, section)`` 下**全部**已有记忆（上游「拿到那个选项下所有的记忆」）；
2. 调模型（角色 ``memory_merge``，提示词附录 B.7）判定 ``ADD / UPDATE / DELETE / SKIP``；
3. 用**代码**强制 §8.5 硬约束（模型输出不可信）：

   - 章节键以 ``redlines`` 结尾 → 只允许 ``ADD`` / ``DELETE``；模型请求的 ``UPDATE``
     **降级为 SKIP 并记录**（验收 D-5）；
   - ``UPDATE`` / ``DELETE`` 必须引用真实存在的 ``target_id``，否则降级为 SKIP；
   - ``DELETE`` 必须在审计里记录被删条目全文（``before``）；
   - 无法判断 → ``SKIP``，并标记 ``unresolved``，由 writer 记入 ``memory-unresolved.jsonl``。

冲突解决优先级（§8.5）：用户显式语句 > 最近时间 > 项目记忆 > 全局记忆。

规格：``my_agent-开发文档.md`` §8.4 步骤 4、§8.5、附录 B.7；``CONTRACTS.md`` §8。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from my_agent.config.defaults import MEMORY_DUP_SIMILARITY
from my_agent.llm.prompts import MEMORY_MERGE_PROMPT, render
from my_agent.memory.extractor import extract_json, similarity
from my_agent.memory.sections import Scope, is_redline, label_for
from my_agent.models.audit import MemoryWriteRecord
from my_agent.models.permission import Confidence

if TYPE_CHECKING:
    from my_agent.memory import MemoryLLM
    from my_agent.memory.scope_router import ScopedCandidate
    from my_agent.memory.store import MemoryEntry, MemoryStore

__all__ = [
    "MergeAction",
    "MergeResult",
    "AuditSink",
    "apply_result",
    "build_prompt",
    "decide",
    "format_existing",
    "merge",
    "parse_decision",
]

_LOG = logging.getLogger(__name__)

#: 合并动作（§8.5）。
MergeAction = Literal["ADD", "UPDATE", "DELETE", "SKIP"]

#: 审计回调：writer 传入，用于收集/落盘 :class:`MemoryWriteRecord`。
AuditSink = Callable[[MemoryWriteRecord], None]

_ACTIONS: frozenset[str] = frozenset({"ADD", "UPDATE", "DELETE", "SKIP"})

#: 章节下无已有记忆时占位（避免提示词出现空块）。
_NO_EXISTING = "（该章节暂无已有记忆）"


class MergeResult(BaseModel):
    """一条候选的合并判定与落地结果（§8.4 步骤 4 + 步骤 6 审计素材）。

    Attributes:
        candidate: 候选文本。
        scope: 作用域。
        section: 章节键。
        action: 最终动作（可能已被 §8.5 硬约束降级）。
        why: **范围选择**的理由（上游「解决 2」的「因为选择了哪个选项」，§8.4 步骤 3）。
        reason: 合并模型给出的理由，或硬约束降级说明（§8.4 步骤 4）。
        target_id: ``UPDATE`` / ``DELETE`` 的目标 id；``ADD`` 落地后为新建 id。
        before: 被替换/删除的原文（``DELETE`` 时必须记录全文）。
        after: 写入后的文本。
        confidence: 置信度。
        applied: 是否真的落盘。
        unresolved: 是否属于「无法判断」（§8.5 → 记入 ``memory-unresolved.jsonl``）。
    """

    candidate: str
    scope: Scope
    section: str
    action: MergeAction = "SKIP"
    why: str = ""
    reason: str = ""
    target_id: str | None = None
    before: str | None = None
    after: str | None = None
    confidence: Confidence = "medium"
    applied: bool = False
    unresolved: bool = False

    model_config = {"extra": "ignore"}

    def to_record(self, *, session_id: str, turn: int) -> MemoryWriteRecord:
        """转换成审计记录（§8.4 步骤 6）。

        ``why`` 以「范围选择理由」为主（上游「解决 2」要求），追加上合并理由。

        Args:
            session_id: 会话 id。
            turn: 轮次。

        Returns:
            审计记录。
        """
        why = self.why
        if self.reason:
            why = f"{why}｜合并：{self.reason}" if why else self.reason
        return MemoryWriteRecord(
            session_id=session_id,
            turn=turn,
            candidate=self.candidate,
            scope=self.scope,
            section=self.section,
            why=why,
            action=self.action,
            target_id=self.target_id,
            before=self.before,
            after=self.after,
            confidence=self.confidence,
        )


def format_existing(entries: Sequence[MemoryEntry]) -> str:
    """把该章节的全部已有记忆渲染进提示词（B.7 的 ``{existing}``）。

    Args:
        entries: 该 ``(scope, section)`` 下的全部条目。

    Returns:
        每行 ``- [id] (date, conf=x) text`` 的文本；无条目时为占位说明。
    """
    if not entries:
        return _NO_EXISTING
    return "\n".join(entry.to_line() for entry in entries)


def build_prompt(candidate: ScopedCandidate, existing: Sequence[MemoryEntry]) -> str:
    """组装冲突合并提示词（附录 B.7）。

    Args:
        candidate: 已定位候选。
        existing: 该章节全部已有记忆。

    Returns:
        渲染后的提示词文本。
    """
    return render(
        MEMORY_MERGE_PROMPT,
        candidate=candidate.text,
        scope=candidate.scope,
        section=candidate.section,
        existing=format_existing(existing),
    )


def parse_decision(raw: str) -> dict[str, Any]:
    """解析 ``memory_merge`` 角色的 JSON 输出（B.7）。

    Args:
        raw: 模型输出原文。

    Returns:
        ``{"action", "target_id", "new_text", "reason"}``；无法解析时返回全 ``None`` 占位。
    """
    payload = extract_json(raw, expect="{")
    if not isinstance(payload, dict):
        _LOG.warning("合并输出无法解析，按 SKIP 处理")
        return {"action": None, "target_id": None, "new_text": None, "reason": ""}
    return {
        "action": str(payload.get("action", "")).strip().upper() or None,
        "target_id": (str(payload["target_id"]).strip() if payload.get("target_id") else None),
        "new_text": (str(payload["new_text"]).strip() if payload.get("new_text") else None),
        "reason": str(payload.get("reason", "")).strip(),
    }


def decide(
    candidate: ScopedCandidate,
    existing: Sequence[MemoryEntry],
    decision: dict[str, Any],
) -> MergeResult:
    """把模型判定套上 §8.5 硬约束，得到最终动作（纯函数，便于单测）。

    Args:
        candidate: 已定位候选。
        existing: 该章节全部已有记忆。
        decision: :func:`parse_decision` 的输出。

    Returns:
        最终合并结果（尚未落盘，``applied=False``）。
    """
    action = decision.get("action")
    reason = decision.get("reason") or ""
    target_id = decision.get("target_id")
    new_text = decision.get("new_text")
    result = MergeResult(
        candidate=candidate.text,
        scope=candidate.scope,
        section=candidate.section,
        confidence=candidate.confidence,
        why=candidate.why,
        reason=reason,
    )
    if action not in _ACTIONS:
        result.action = "SKIP"
        result.reason = f"合并输出非法或无法判断，按 SKIP 处理：{reason}" if reason else "合并输出非法"
        result.unresolved = True
        return result

    if action == "SKIP":
        result.action = "SKIP"
        result.unresolved = not any(
            similarity(candidate.text, entry.text) >= MEMORY_DUP_SIMILARITY for entry in existing
        )
        if result.unresolved:
            result.reason = f"无法判断（宁少勿多）：{reason}" if reason else "无法判断（宁少勿多）"
        return result

    if is_redline(candidate.section) and action == "UPDATE":
        result.action = "SKIP"
        result.reason = (
            f"红线章节「{label_for(candidate.section)}」只允许 ADD/DELETE（§8.5），"
            f"模型请求的 UPDATE 已降级为 SKIP；模型理由：{reason}"
        )
        return result

    by_id = {entry.id: entry for entry in existing}
    target = by_id.get(target_id) if target_id else None
    if action in ("UPDATE", "DELETE") and target is None:
        result.action = "SKIP"
        result.reason = f"引用了不存在的 target_id={target_id!r}，已降级为 SKIP"
        result.unresolved = True
        return result

    if action == "UPDATE":
        text = new_text or candidate.text
        result.action = "UPDATE"
        result.target_id = target.id if target else None
        result.before = target.text if target else None
        result.after = text
        return result

    if action == "DELETE":
        result.action = "DELETE"
        result.target_id = target.id if target else None
        result.before = target.text if target else None
        return result

    result.action = "ADD"
    result.after = new_text or candidate.text
    return result


async def apply_result(store: MemoryStore, result: MergeResult) -> MergeResult:
    """把判定落到磁盘（原子写 + 改动前 ``.bak``，§8.4 步骤 5）。

    Args:
        store: 目标作用域的存储。
        result: 合并判定。

    Returns:
        就地更新后的结果（``applied`` / ``target_id`` / ``after`` 已回填）。

    Raises:
        MyAgentError: 章节非法或条目不存在。
        MemoryLockError: 锁文件被占用超时。
    """
    if result.action == "ADD":
        entry = await asyncio.to_thread(
            store.add, result.candidate, result.section, result.confidence
        )
        result.target_id = entry.id
        result.after = entry.text
        result.applied = True
    elif result.action == "UPDATE" and result.target_id:
        entry = await asyncio.to_thread(
            store.update, result.target_id, result.after or result.candidate, result.section,
            result.confidence,
        )
        result.after = entry.text
        result.applied = True
    elif result.action == "DELETE" and result.target_id:
        removed = await asyncio.to_thread(store.delete, result.target_id)
        result.applied = removed is not None
    return result


async def merge(
    scoped: Sequence[ScopedCandidate],
    store_for: Callable[[Scope], MemoryStore],
    llm: MemoryLLM,
    *,
    audit: AuditSink | None = None,
    session_id: str = "",
    turn: int = 0,
) -> list[MergeResult]:
    """步骤 4：逐条做冲突判定与落盘（§8.4、§8.5）。

    Args:
        scoped: 步骤 3 的输出。
        store_for: 作用域 → :class:`~my_agent.memory.store.MemoryStore`。
        llm: 角色化模型客户端（``complete_role``）。
        audit: 审计回调（每条候选调用一次，含 SKIP）。
        session_id: 会话 id（写审计用）。
        turn: 轮次（写审计用）。

    Returns:
        与 ``scoped`` 等长的合并结果列表。

    Raises:
        MyAgentError: 落盘失败（章节非法等）。
        MemoryLockError: 锁文件被占用超时。
    """
    results: list[MergeResult] = []
    for candidate in scoped:
        store = store_for(candidate.scope)
        # 进程内锁保护「读全部已有记忆 → 判定 → 落盘」整段，避免并发回合丢更新
        async with store.locked():
            existing = await asyncio.to_thread(_entries_of, store, candidate.section)
            prompt = build_prompt(candidate, existing)
            response = await llm.complete_role(role="memory_merge", user=prompt)
            decision = parse_decision(getattr(response, "text", "") or "")
            result = decide(candidate, existing, decision)
            result = await apply_result(store, result)
        results.append(result)
        if audit is not None:
            audit(result.to_record(session_id=session_id, turn=turn))
    return results


def _entries_of(store: MemoryStore, section: str) -> list[MemoryEntry]:
    """读取某章节的全部已有记忆。

    Args:
        store: 目标作用域存储。
        section: 章节键。

    Returns:
        条目列表（文件顺序）。
    """
    return store.sections().get(section, [])
