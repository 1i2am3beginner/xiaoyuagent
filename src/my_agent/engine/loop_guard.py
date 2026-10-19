"""工具调用死循环检测（§5.3.3 B11 的补充；【待确认假设 U33】）。

用户口径（U33）：**同一工具的同一参数被模型反复调用，且每次结果内容高度相似**
才判为死循环。因此判定有**两个条件，必须同时满足**：

1. 同一个 ``(tool, 参数)`` 指纹重复出现 ≥ :data:`~my_agent.config.defaults.LOOP_REPEAT_THRESHOLD`
   次（默认 3）；
2. 这几次调用的**结果内容**高度相似：最高相似度 ≥
   :data:`~my_agent.config.defaults.LOOP_RESULT_SIMILARITY`（默认 0.95）。

相似度**复用** §9.2 的 :class:`~my_agent.skills.similarity.SimilarityBackend` 抽象
（``similarity(a, b) -> float``）：默认 ``trigram``（字符 trigram 余弦 ∪ token_set_ratio，
离线零依赖），可用 ``config.skills.similarity_backend = "embedding"`` 切到真向量余弦。
这样「用向量做内容相似度」的要求与 skills 去重是**同一套口径**，不引入第二套比较逻辑。

只满足条件 1 → :attr:`LoopVerdict.REPEAT_SUSPECTED`（可疑但**不**终止：结果在变，说明工具
还在推进，例如带计数器的轮询）；两条都满足 → :attr:`LoopVerdict.LOOP_DETECTED`。

设计约束：

* **有界内存**：最多保留 ``window`` 个指纹（LRU 淘汰最旧），每个指纹最多保留最近
  ``repeat_threshold`` 条结果文本——相似度只需与本指纹的历史比，更早的文本没有新信息。
  指纹被淘汰后再次出现会从第 1 次重新计数（有界内存的必然取舍，仅影响极端长尾）。
* **截断**：结果文本先截到 :data:`RESULT_TEXT_CAP` 字符再算相似度。理由：工具结果可能极大
  （``WebFetch`` 上限 2MB、``Read`` 默认 2000 行），而字符 trigram 向量规模与文本长度成正比；
  结果头部的判别力最强（文件名/首行/结构），截断让「向量这一步」保持廉价且开销确定。
* **绝不抛异常**：空结果、``None`` 形态、两条完全相同的空结果都是合法输入（工具连续 3 次以
  相同参数返回空，本身就是真死循环），不得因输入形态中断 ReAct 循环。
* **后端不可用不算死循环**：相似度后端抛错（如 embedding 端点不可用）时按「不相似」处理，
  宁可漏报也不误杀一个正常回合（fail-open）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from my_agent.config.defaults import (
    LOOP_GUARD_WINDOW,
    LOOP_REPEAT_THRESHOLD,
    LOOP_RESULT_SIMILARITY,
)
from my_agent.errors import MyAgentError
from my_agent.skills.similarity import SimilarityBackend, TrigramCosineBackend

__all__ = [
    "RESULT_TEXT_CAP",
    "LoopCheck",
    "LoopGuard",
    "LoopVerdict",
    "fingerprint",
]

_LOG = logging.getLogger(__name__)

RESULT_TEXT_CAP = 4_000
"""参与相似度计算的结果文本字符上限（超出只取头部；见模块 docstring「截断」）。"""


class LoopVerdict(StrEnum):
    """一次工具调用记录后的判定结果。"""

    CONTINUE = "continue"
    """正常：该指纹首次出现（未发现重复）。"""

    REPEAT_SUSPECTED = "repeat_suspected"
    """指纹重复了，但结果还不相似（工具仍在推进）——只观察，不终止。"""

    LOOP_DETECTED = "loop_detected"
    """判定死循环：指纹重复 ≥ 阈值 **且** 结果高度相似。"""


class LoopCheck(BaseModel):
    """一次死循环检查的结论（供引擎决策与测试断言）。

    Attributes:
        verdict: 判定结果。
        tool: 工具名。
        repeat_count: 该 ``(tool, 参数)`` 指纹累计出现的次数（含本次）。
        max_similarity: 本次结果与历史结果的**最高**相似度（首次调用为 ``0.0``）。
        message: 中文说明（检测到死循环时直接回填给模型与用户）。
    """

    verdict: LoopVerdict
    tool: str
    repeat_count: int
    max_similarity: float
    message: str

    model_config = {"extra": "ignore"}


def fingerprint(tool: str, arguments: Mapping[str, Any]) -> str:
    """计算 ``(tool, 参数)`` 指纹（与 :mod:`my_agent.security.cache` 同款规范化）。

    规范化 JSON：``sort_keys=True``（**键序无关**）、``ensure_ascii=False``、
    紧凑分隔符，不可序列化的值退化为 ``str``；因此 ``{"a": 1, "b": 2}`` 与
    ``{"b": 2, "a": 1}`` 得到同一指纹，而参数值不同必然得到不同指纹。

    Args:
        tool: 工具名。
        arguments: 工具参数（任意 Mapping；非法输入退化为 ``repr`` 而非抛异常）。

    Returns:
        ``sha1(tool + "\\x00" + canonical_json(arguments))`` 的 40 位十六进制摘要。
    """
    payload = _canonical_json(arguments)
    return hashlib.sha1(f"{tool}\x00{payload}".encode()).hexdigest()


@dataclass
class _FingerprintEntry:
    """一个指纹的历史（``count`` 计数不设上限，文本只留最近几条）。

    Attributes:
        tool: 工具名（写进报告文案）。
        count: 该指纹累计出现次数。
        results: 最近 ``repeat_threshold`` 条结果文本（已截断）。
    """

    tool: str
    count: int = 0
    results: deque[str] = field(default_factory=deque)


class LoopGuard:
    """工具调用死循环守卫（每个用户回合一个实例，回合开始 :meth:`reset`）。

    Args:
        backend: 相似度后端（§9.2）。``None`` = :class:`TrigramCosineBackend`（离线默认）。
            注入点用于测试与 embedding 切换，**不在本类内触网**。
        repeat_threshold: 指纹重复到第几次开始做结果相似度判定（默认 3）。
        similarity_threshold: 结果相似度达到该值即判死循环（默认 0.95）。
        window: 最多跟踪的指纹数（LRU 淘汰最旧的，默认 32）。

    Attributes:
        backend: 生效的相似度后端。
    """

    def __init__(
        self,
        *,
        backend: SimilarityBackend | None = None,
        repeat_threshold: int = LOOP_REPEAT_THRESHOLD,
        similarity_threshold: float = LOOP_RESULT_SIMILARITY,
        window: int = LOOP_GUARD_WINDOW,
    ) -> None:
        self.backend: SimilarityBackend = backend if backend is not None else TrigramCosineBackend()
        self._repeat_threshold = max(2, int(repeat_threshold))
        self._similarity_threshold = float(similarity_threshold)
        self._window = max(1, int(window))
        self._entries: OrderedDict[str, _FingerprintEntry] = OrderedDict()
        self._total_calls = 0
        self._loops_detected = 0

    # ------------------------------------------------------------------ 只读视图
    @property
    def repeat_threshold(self) -> int:
        """生效的重复次数阈值（≥ 2）。"""
        return self._repeat_threshold

    @property
    def similarity_threshold(self) -> float:
        """生效的结果相似度阈值。"""
        return self._similarity_threshold

    @property
    def window(self) -> int:
        """生效的指纹窗口大小（≥ 1）。"""
        return self._window

    # ------------------------------------------------------------------ 主流程
    def record(self, *, tool: str, arguments: Mapping[str, Any], result_text: str) -> LoopCheck:
        """记录一次**已完成**的工具调用并给出判定。

        流程：算指纹 → 与同指纹的历史结果逐条比相似度取**最大值** → 存下本次文本 →
        按「重复次数 ≥ 阈值 **且** 最高相似度 ≥ 阈值」判定。

        Args:
            tool: 工具名。
            arguments: 本次调用的参数（键序无关）。
            result_text: 回填给模型的结果文本（超长自动截断；``None``/非字符串按空串处理）。

        Returns:
            :class:`LoopCheck`：``CONTINUE`` / ``REPEAT_SUSPECTED`` / ``LOOP_DETECTED``。

        Raises:
            MyAgentError: 不抛。相似度后端故障会被吞掉并按「不相似」处理（fail-open）。
        """
        self._total_calls += 1
        tool_name = tool if isinstance(tool, str) and tool else "(未知工具)"
        text = _clip_result_text(result_text)
        key = fingerprint(tool_name, arguments if isinstance(arguments, Mapping) else {})

        entry = self._entries.get(key)
        if entry is None:
            entry = _FingerprintEntry(tool=tool_name)
            self._entries[key] = entry
            self._evict()
        else:
            self._entries.move_to_end(key)

        previous = list(entry.results)
        entry.count += 1
        max_similarity = max((self._similarity(text, other) for other in previous), default=0.0)

        entry.results.append(text)
        while len(entry.results) > self._repeat_threshold:
            entry.results.popleft()

        return self._verdict(entry, max_similarity)

    def reset(self) -> None:
        """清空全部状态（每个用户回合开始时调用：指纹表与计数器一并归零）。"""
        self._entries.clear()
        self._total_calls = 0
        self._loops_detected = 0

    def stats(self) -> dict[str, int]:
        """返回小型自省字典（测试与 ``/context`` 式报告用）。

        Returns:
            ``tracked_fingerprints``（当前跟踪的指纹数，≤ ``window``）、
            ``total_calls``（本回合已记录的调用数）、``loops_detected``（判定次数）。
        """
        return {
            "tracked_fingerprints": len(self._entries),
            "total_calls": self._total_calls,
            "loops_detected": self._loops_detected,
        }

    # ------------------------------------------------------------------ 内部
    def _verdict(self, entry: _FingerprintEntry, max_similarity: float) -> LoopCheck:
        """把「次数 + 相似度」折成判定与中文说明。

        Args:
            entry: 该指纹的历史（``count`` 已含本次）。
            max_similarity: 本次结果与历史结果的最高相似度。

        Returns:
            :class:`LoopCheck`。
        """
        repeat_count = entry.count
        enough_repeats = repeat_count >= self._repeat_threshold
        similar_enough = max_similarity >= self._similarity_threshold

        if enough_repeats and similar_enough:
            self._loops_detected += 1
            message = (
                f"检测到工具调用死循环：{entry.tool} 以相同参数被调用 {repeat_count} 次"
                f"且结果高度相似（相似度 {max_similarity:.2f}）。已终止本回合。"
            )
            return LoopCheck(
                verdict=LoopVerdict.LOOP_DETECTED,
                tool=entry.tool,
                repeat_count=repeat_count,
                max_similarity=max_similarity,
                message=message,
            )

        if repeat_count > 1:
            if enough_repeats:
                message = (
                    f"{entry.tool} 以相同参数被调用 {repeat_count} 次，"
                    f"但结果差异明显（最高相似度 {max_similarity:.2f}），暂不判定死循环。"
                )
            else:
                message = (
                    f"{entry.tool} 以相同参数重复调用（第 {repeat_count} 次，"
                    f"最高相似度 {max_similarity:.2f}），继续观察。"
                )
            return LoopCheck(
                verdict=LoopVerdict.REPEAT_SUSPECTED,
                tool=entry.tool,
                repeat_count=repeat_count,
                max_similarity=max_similarity,
                message=message,
            )

        return LoopCheck(
            verdict=LoopVerdict.CONTINUE,
            tool=entry.tool,
            repeat_count=repeat_count,
            max_similarity=max_similarity,
            message=f"{entry.tool} 首次以该参数调用，未发现重复。",
        )

    def _similarity(self, left: str, right: str) -> float:
        """算两段结果文本的相似度（**任何后端故障都不外泄**）。

        Args:
            left: 本次结果文本（已截断）。
            right: 历史结果文本（已截断）。

        Returns:
            ``[0, 1]`` 的相似度；两条都为空时返回 ``1.0``（空 == 空，语义上完全一致，
            而 trigram/embedding 后端对空串都只能返回 ``0.0``，必须在此显式定义）；
            后端抛错或返回值非法时返回 ``0.0``（fail-open：不据此终止回合）。
        """
        if not left.strip() and not right.strip():
            return 1.0
        try:
            raw = self.backend.similarity(left, right)
        except MyAgentError as exc:
            _LOG.warning("相似度后端不可用，按「不相似」处理：%s", exc.user_message())
            return 0.0
        except (AttributeError, TypeError, ValueError) as exc:
            _LOG.warning("相似度后端返回非法结果，按「不相似」处理：%s", exc)
            return 0.0
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    def _evict(self) -> None:
        """按 LRU 淘汰最旧指纹，保证跟踪表不超过 ``window`` 条。"""
        while len(self._entries) > self._window:
            self._entries.popitem(last=False)


def _clip_result_text(result_text: str) -> str:
    """把结果文本规范化为「字符串 + 截断到 :data:`RESULT_TEXT_CAP`」。

    Args:
        result_text: 任意结果文本（``None`` 或非字符串按空串/``str()`` 处理，绝不抛异常）。

    Returns:
        长度不超过 :data:`RESULT_TEXT_CAP` 的文本。
    """
    if result_text is None:
        return ""
    text = result_text if isinstance(result_text, str) else str(result_text)
    return text if len(text) <= RESULT_TEXT_CAP else text[:RESULT_TEXT_CAP]


def _canonical_json(arguments: Mapping[str, Any]) -> str:
    """生成指纹用的规范化 JSON（键排序 + 非 ASCII 原样 + 紧凑分隔符）。

    Args:
        arguments: 参数映射；不可 ``dict()`` 化时退化为 ``repr``。

    Returns:
        规范化 JSON 串；值不可 JSON 序列化时用 ``default=str``，
        键类型混杂导致排序失败时退化为 ``repr(排序后的 (str 键, repr 值) 列表)``
        （同一逻辑参数仍得到同一指纹，绝不抛异常）。
    """
    try:
        payload = dict(arguments)
    except (TypeError, ValueError):
        return repr(arguments)
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        return repr(sorted((str(key), repr(value)) for key, value in payload.items()))
