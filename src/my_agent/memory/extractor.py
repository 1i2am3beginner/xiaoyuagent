"""记忆写入管线步骤 1–2：候选抽取与去噪门槛（§8.4）。

- 步骤 1 :func:`extract`：调小模型（角色 ``memory_extract``，提示词附录 B.5），
  输入本轮 user 输入 + assistant 终答 + 工具调用摘要，输出 0..n 条候选。
- 步骤 2 :func:`denoise`：丢弃临时性内容与与已有记忆高度重复的候选
  （相似度 ≥ :data:`~my_agent.config.defaults.MEMORY_DUP_SIMILARITY`），
  并按 confidence 截断到 :data:`~my_agent.config.defaults.MAX_MEMORY_WRITES_PER_TURN` 条。

「可从代码推断的常识」这类语义过滤交由 B.5 提示词完成（规则无法可靠判定），
本模块只做**确定性**过滤，保证「宁可少写，不可写错」（§8.5）。

规格：``my_agent-开发文档.md`` §8.4 步骤 1–2、附录 B.5。
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from my_agent.config.defaults import MAX_MEMORY_WRITES_PER_TURN, MEMORY_DUP_SIMILARITY
from my_agent.llm.prompts import MEMORY_EXTRACT_PROMPT, render
from my_agent.memory.store import sanitize_text
from my_agent.models.permission import Confidence

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from my_agent.memory import MemoryLLM

__all__ = [
    "TEMPORARY_MARKERS",
    "Candidate",
    "confidence_rank",
    "coerce_confidence",
    "denoise",
    "extract",
    "extract_json",
    "is_duplicate",
    "is_temporary",
    "parse_candidates",
    "similarity",
]

_LOG = logging.getLogger(__name__)

#: 临时性内容的确定性标记（§8.4 步骤 2「过滤：临时性内容」的示例：『今天先这样』）。
TEMPORARY_MARKERS: tuple[str, ...] = (
    "今天先这样",
    "今天就到这",
    "暂时先",
    "先这样吧",
    "下次再说",
    "回头再说",
    "本次任务",
    "这个任务的临时",
    "只是这次",
    "仅限这次",
    "临时约定",
)

_CONFIDENCE_RANK: dict[Confidence, int] = {"high": 0, "medium": 1, "low": 2}


class Candidate(BaseModel):
    """一条待写入的记忆候选（§8.4 步骤 1 输出）。

    Attributes:
        text: 候选文本（单行 ≤300 字符）。
        confidence: 模型给出的置信度。
        evidence: 模型给出的依据（一句话）。
        evidence_turn: 出自哪一轮（由调用方回填）。
    """

    text: str
    confidence: Confidence = "medium"
    evidence: str = ""
    evidence_turn: int = 0

    model_config = {"extra": "ignore"}


def confidence_rank(confidence: str) -> int:
    """返回置信度的排序权重（越小越可信）。

    Args:
        confidence: ``high`` / ``medium`` / ``low``（其它值按 ``medium`` 处理）。

    Returns:
        0（high）、1（medium）、2（low）。
    """
    return _CONFIDENCE_RANK.get(coerce_confidence(confidence), 1)


def coerce_confidence(value: object, *, default: Confidence = "medium") -> Confidence:
    """把任意值规范成合法置信度。

    Args:
        value: 原始值（模型输出/工具参数）。
        default: 无法识别时的兜底值。

    Returns:
        ``"high"`` / ``"medium"`` / ``"low"``。
    """
    text = str(value).strip().lower() if value is not None else ""
    if text == "high":
        return "high"
    if text == "medium":
        return "medium"
    if text == "low":
        return "low"
    return default


def similarity(left: str, right: str) -> float:
    """计算两段文本的相似度（rapidfuzz 归一化到 0..1）。

    Args:
        left: 文本 A。
        right: 文本 B。

    Returns:
        0..1 的相似度；两边都为空时返回 1.0。
    """
    from rapidfuzz import fuzz  # 延迟导入：仅记忆管线使用

    a = sanitize_text(left).lower()
    b = sanitize_text(right).lower()
    if not a and not b:
        return 1.0
    return fuzz.ratio(a, b) / 100.0


def is_temporary(text: str) -> bool:
    """判断文本是否是临时性内容（§8.4 步骤 2 的第一条过滤）。

    Args:
        text: 候选文本。

    Returns:
        命中 :data:`TEMPORARY_MARKERS` 任一标记时返回 ``True``。
    """
    return any(marker in text for marker in TEMPORARY_MARKERS)


def is_duplicate(text: str, existing: Iterable[str]) -> bool:
    """判断文本是否与已有记忆重复（§8.4 步骤 2：相似度 ≥ 阈值直接丢）。

    Args:
        text: 候选文本。
        existing: 已有记忆文本集合。

    Returns:
        存在相似度 ≥ :data:`~my_agent.config.defaults.MEMORY_DUP_SIMILARITY` 的条目时返回 ``True``。
    """
    return any(similarity(text, other) >= MEMORY_DUP_SIMILARITY for other in existing)


def denoise(
    candidates: Sequence[Candidate],
    existing: Iterable[str] = (),
    *,
    max_writes: int = MAX_MEMORY_WRITES_PER_TURN,
) -> list[Candidate]:
    """步骤 2：去噪与门槛（§8.4）。

    顺序：丢弃空文本 → 丢弃临时性内容 → 丢弃与已有记忆重复的 → 丢弃彼此重复的 →
    按 confidence 排序后取前 ``max_writes`` 条。

    Args:
        candidates: 步骤 1 的输出。
        existing: 已有记忆文本（全局 + 项目）。
        max_writes: 单轮最多写入条数。

    Returns:
        保留下来的候选（按 confidence 从高到低）。
    """
    kept: list[Candidate] = []
    pool: list[str] = list(existing)
    for candidate in sorted(candidates, key=lambda item: confidence_rank(item.confidence)):
        text = sanitize_text(candidate.text)
        if not text or is_temporary(text) or is_duplicate(text, pool):
            continue
        kept.append(candidate.model_copy(update={"text": text}))
        pool.append(text)
        if len(kept) >= max_writes:
            break
    return kept


def parse_candidates(raw: str) -> list[Candidate]:
    """解析 ``memory_extract`` 角色的 JSON 输出（B.5）。

    容忍 ```json 代码块围栏与前后噪声文本；解析失败时返回空列表（宁少勿多）。

    Args:
        raw: 模型输出的原始文本。

    Returns:
        候选列表；无法解析或结构非法时返回空列表。
    """
    payload = extract_json(raw, expect="[")
    if payload is None:
        _LOG.warning("记忆抽取输出无法解析，按无候选处理")
        return []
    if isinstance(payload, dict):
        payload = payload.get("candidates", [])
    if not isinstance(payload, list):
        return []
    out: list[Candidate] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        text = sanitize_text(str(item.get("text", "")))
        if not text:
            continue
        out.append(
            Candidate(
                text=text,
                confidence=coerce_confidence(item.get("confidence")),
                evidence=str(item.get("evidence", "")),
            )
        )
    return out


async def extract(
    *,
    user_input: str,
    assistant_final: str,
    tool_summary: str,
    llm: MemoryLLM,
) -> list[Candidate]:
    """步骤 1：从本轮对话抽取记忆候选（§8.4、附录 B.5）。

    Args:
        user_input: 本轮用户输入。
        assistant_final: 本轮助手终答。
        tool_summary: 本轮工具调用摘要（工具名 + 参数 + 成败）。
        llm: 角色化模型客户端（``complete_role``）。

    Returns:
        0..``max_items`` 条候选；模型输出不可解析时返回空列表。
    """
    prompt = render(
        MEMORY_EXTRACT_PROMPT,
        max_items=MAX_MEMORY_WRITES_PER_TURN,
        user_input=user_input,
        assistant_final=assistant_final,
        tool_summary=tool_summary,
    )
    result = await llm.complete_role(role="memory_extract", user=prompt)
    return parse_candidates(getattr(result, "text", "") or "")


def extract_json(raw: str, *, expect: str) -> Any:
    """从模型输出中抠出 JSON（容忍代码块围栏与前后噪声）。

    Args:
        raw: 原始文本。
        expect: ``"["`` 或 ``"{"``，决定优先尝试的容器类型。

    Returns:
        解析后的 JSON 对象；失败时返回 ``None``。
    """
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    opener, closer = ("[", "]") if expect == "[" else ("{", "}")
    start = text.find(opener)
    end = text.rfind(closer)
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
