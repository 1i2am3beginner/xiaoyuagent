"""记忆写入管线步骤 3：范围选择（§8.4、附录 B.6）。

对应上游「解决 1」：写入时选全局还是项目，**只给范围名 + 一句说明**，由小模型选择。

**硬约束（验收 D-1）**：本模块组装的提示词里**只允许**出现 §8.2 的范围键与一句说明，
以及**待归档的候选文本**；绝不允许把任何已有记忆内容拼进去。

规格：``my_agent-开发文档.md`` §8.4 步骤 3、附录 B.6。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from my_agent.llm.prompts import MEMORY_SCOPE_PROMPT, render
from my_agent.memory.extractor import coerce_confidence, extract_json
from my_agent.memory.sections import Scope, render_section_catalog, scope_of_section
from my_agent.models.permission import Confidence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from my_agent.memory import MemoryLLM
    from my_agent.memory.extractor import Candidate

__all__ = ["ScopedCandidate", "choose", "build_prompt", "normalize_scoped"]

_LOG = logging.getLogger(__name__)


class ScopedCandidate(BaseModel):
    """一条已确定写入位置（作用域 + 章节）的候选（§8.4 步骤 3 输出）。

    Attributes:
        text: 候选文本。
        scope: 写入作用域。
        section: §8.2 的章节键。
        why: 「因为选择了哪个选项」——上游「解决 2」要求的理由，写进审计。
        confidence: 继承自候选的置信度。
        evidence_turn: 出自哪一轮。
    """

    text: str
    scope: Scope
    section: str
    why: str = ""
    confidence: Confidence = "medium"
    evidence_turn: int = 0

    model_config = {"extra": "ignore"}


def build_prompt(candidates: Sequence[Candidate]) -> str:
    """组装范围选择提示词（附录 B.6）。

    提示词内容 = 范围键 + 一句说明 + 候选文本；**不含任何已有记忆**（D-1）。

    Args:
        candidates: 待归档候选。

    Returns:
        渲染后的提示词文本。
    """
    listing = "\n".join(f"- {candidate.text}" for candidate in candidates)
    return render(
        MEMORY_SCOPE_PROMPT,
        global_sections=render_section_catalog("global"),
        project_sections=render_section_catalog("project"),
        candidates=listing,
    )


def normalize_scoped(
    *,
    text: str,
    scope: object,
    section: object,
    why: object = "",
    confidence: object = "medium",
    evidence_turn: int = 0,
) -> ScopedCandidate | None:
    """校验并规范化模型给出的「作用域 + 章节」。

    容错：作用域与章节键不一致时以**章节键所属作用域**为准（两张表的键不重叠）；
    章节键在两张表里都不存在时返回 ``None``（宁少勿多，§8.5「无法判断 → SKIP」）。

    Args:
        text: 候选文本。
        scope: 模型给出的作用域。
        section: 模型给出的章节键。
        why: 模型给出的理由。
        confidence: 置信度。
        evidence_turn: 出自哪一轮。

    Returns:
        规范化后的 :class:`ScopedCandidate`；无法判定写入位置时返回 ``None``。
    """
    key = str(section).strip()
    owner = scope_of_section(key)
    if owner is None:
        _LOG.warning("范围选择返回了未知章节键 %r，已跳过该候选", section)
        return None
    raw_scope = str(scope).strip()
    if raw_scope in ("global", "project") and raw_scope != owner:
        _LOG.debug("范围 %s 与章节键 %s 不一致，按章节键归属修正为 %s", raw_scope, key, owner)
    return ScopedCandidate(
        text=text,
        scope=owner,
        section=key,
        why=str(why or "").strip(),
        confidence=coerce_confidence(confidence),
        evidence_turn=evidence_turn,
    )


async def choose(candidates: Sequence[Candidate], llm: MemoryLLM) -> list[ScopedCandidate]:
    """步骤 3：为每条候选选择写入范围与章节（§8.4、附录 B.6）。

    Args:
        candidates: 步骤 2 保留下来的候选。
        llm: 角色化模型客户端（``complete_role``）。

    Returns:
        已定位的候选列表；模型输出不可解析或章节键非法时对应条目被丢弃。
    """
    if not candidates:
        return []
    prompt = build_prompt(candidates)
    result = await llm.complete_role(role="memory_scope", user=prompt)
    payload = extract_json(getattr(result, "text", "") or "", expect="[")
    if not isinstance(payload, list):
        _LOG.warning("范围选择输出无法解析，本轮不写入记忆")
        return []

    by_text = {candidate.text: candidate for candidate in candidates}
    out: list[ScopedCandidate] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        source = by_text.get(text)
        if source is None:
            # 模型改写/漏写文本时按顺序兜底，避免整条候选丢失
            source = candidates[index] if index < len(candidates) else None
        if source is None:
            continue
        scoped = normalize_scoped(
            text=source.text,
            scope=item.get("scope"),
            section=item.get("section"),
            why=item.get("why", ""),
            confidence=source.confidence,
            evidence_turn=source.evidence_turn,
        )
        if scoped is not None:
            out.append(scoped)
    return out
