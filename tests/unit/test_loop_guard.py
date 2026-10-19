"""``engine/loop_guard.py`` 单元测试：指纹、双条件判定、有界内存、输入鲁棒性。

覆盖【待确认假设 U33】的用户口径：**同一 (tool, 参数) 指纹重复 ≥3 次**（条件 1）
**且**「这几次的结果内容相似度 ≥0.95」（条件 2）才判死循环。只满足条件 1 时结果在变
（计数器/时间戳在推进），必须**不**终止回合——这是用户对原始需求的关键修正。

相似度走的是 §9.2 的 :class:`~my_agent.skills.similarity.SimilarityBackend` 抽象：
默认 trigram（离线、确定性），本文件另用注入的假后端证明「比较这一步真的发生了」。
"""

from __future__ import annotations

from typing import Any

from my_agent.config.defaults import (
    LOOP_GUARD_WINDOW,
    LOOP_REPEAT_THRESHOLD,
    LOOP_RESULT_SIMILARITY,
)
from my_agent.engine.loop_guard import (
    RESULT_TEXT_CAP,
    LoopCheck,
    LoopGuard,
    LoopVerdict,
    fingerprint,
)
from my_agent.skills.similarity import TrigramCosineBackend

#: 一小组「同一参数反复读取同一文件」的结果样本（第 3 次多了个空行 → 高度相似）。
READ_ARGS: dict[str, Any] = {"file_path": "hello.txt"}
READ_TEXT = "1\thello world\n2\tsecond line\n"
READ_TEXT_NEAR = "1\thello world\n2\tsecond line\n3\t\n"

#: 同一参数但结果**明显在变**（轮询任务状态：时间戳与级别都不同）。
#: 实测最高相似度 0.72，远低于 0.95 阈值。
CHANGING_RESULTS = (
    "2024-05-01 10:00:00 INFO job started\n",
    "2024-05-01 10:03:12 WARN job still running\n",
    "2024-05-01 10:07:41 ERROR job failed: disk full\n",
)


class _RecordingBackend:
    """记录调用并固定返回值的假相似度后端（证明守卫**真的**调了相似度）。

    Args:
        value: 每次 ``similarity()`` 的固定返回值。
    """

    backend_id = "recording-fake"

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple[str, str]] = []

    def similarity(self, a: str, b: str) -> float:
        """记录一对文本并返回预设相似度。

        Args:
            a: 左文本。
            b: 右文本。

        Returns:
            构造时给定的固定值。
        """
        self.calls.append((a, b))
        return self.value


def _record_n(guard: LoopGuard, times: int, text: str) -> list[LoopCheck]:
    """以同一 (tool, 参数) 连记 ``times`` 次。

    Args:
        guard: 守卫。
        times: 记录次数。
        text: 每次的结果文本。

    Returns:
        :class:`~my_agent.engine.loop_guard.LoopCheck` 列表。
    """
    return [
        guard.record(tool="Read", arguments=READ_ARGS, result_text=text) for _ in range(times)
    ]


# ---------------------------------------------------------------------------- 指纹
def test_fingerprint_ignores_key_order() -> None:
    """键序不影响指纹（同一逻辑参数 → 同一指纹）。"""
    left = fingerprint("Read", {"file_path": "a.py", "limit": 10})
    right = fingerprint("Read", {"limit": 10, "file_path": "a.py"})

    assert left == right
    assert len(left) == 40, "sha1 十六进制摘要"


def test_fingerprint_changes_with_arguments_or_tool() -> None:
    """参数值不同、或工具名不同 → 指纹必须不同。"""
    base = fingerprint("Read", {"file_path": "a.py"})

    assert fingerprint("Read", {"file_path": "b.py"}) != base
    assert fingerprint("Read", {"file_path": "a.py", "limit": 10}) != base
    assert fingerprint("Glob", {"file_path": "a.py"}) != base


def test_fingerprint_survives_weird_arguments() -> None:
    """不可 JSON 序列化的参数不得让指纹计算崩溃（``default=str`` 或排序回退）。"""
    assert len(fingerprint("Bash", {"blob": {1, 2, 3}})) == 40
    assert fingerprint("Bash", {"blob": {1, 2, 3}}) == fingerprint("Bash", {"blob": {3, 2, 1}})
    assert fingerprint("Bash", {(1, 2): "x"}) == fingerprint("Bash", {(1, 2): "x"})


# ---------------------------------------------------------------------- 双条件判定
def test_first_call_continues_and_second_is_only_suspected() -> None:
    """第 1 次 CONTINUE、第 2 次 REPEAT_SUSPECTED——阈值未到，绝不能判死循环。"""
    guard = LoopGuard()
    first, second = _record_n(guard, 2, READ_TEXT)

    assert first.verdict is LoopVerdict.CONTINUE
    assert first.repeat_count == 1
    assert second.verdict is LoopVerdict.REPEAT_SUSPECTED
    assert second.verdict is not LoopVerdict.LOOP_DETECTED
    assert second.repeat_count == 2
    assert second.max_similarity >= LOOP_RESULT_SIMILARITY
    assert guard.stats()["loops_detected"] == 0


def test_third_call_with_similar_result_is_a_loop() -> None:
    """条件 1 + 条件 2 同时满足 → LOOP_DETECTED，且计数/相似度/文案都对。"""
    guard = LoopGuard()
    _record_n(guard, 2, READ_TEXT)
    third = guard.record(tool="Read", arguments=READ_ARGS, result_text=READ_TEXT_NEAR)

    assert third.verdict is LoopVerdict.LOOP_DETECTED
    assert third.repeat_count == LOOP_REPEAT_THRESHOLD == 3
    assert third.max_similarity >= LOOP_RESULT_SIMILARITY
    assert third.tool == "Read"
    assert "死循环" in third.message
    assert "Read" in third.message
    assert "3 次" in third.message
    assert "已终止本回合" in third.message
    assert guard.stats()["loops_detected"] == 1


def test_third_call_with_changing_results_is_not_a_loop() -> None:
    """**条件 2 的守门测试**：参数相同但结果在变 → 绝不判死循环（用户的关键修正）。

    若把结果相似度这一步去掉（只看「同 tool 同参数重复 3 次」），本测试立刻失败：
    第 3 次会返回 ``LOOP_DETECTED``。
    """
    guard = LoopGuard()
    checks = [
        guard.record(tool="Read", arguments=READ_ARGS, result_text=text)
        for text in CHANGING_RESULTS
    ]

    assert checks[-1].repeat_count == 3, "指纹确实重复到了阈值"
    assert checks[-1].max_similarity < LOOP_RESULT_SIMILARITY
    assert checks[-1].verdict is LoopVerdict.REPEAT_SUSPECTED
    assert checks[-1].verdict is not LoopVerdict.LOOP_DETECTED
    assert guard.stats()["loops_detected"] == 0

    # 对照：同一守卫、同一工具与参数，换成高度相似的结果就**会**判死循环。
    identical = [
        guard.record(tool="Read", arguments=READ_ARGS, result_text=READ_TEXT) for _ in range(3)
    ]
    assert identical[-1].verdict is LoopVerdict.LOOP_DETECTED


def test_same_arguments_with_different_tool_is_not_a_loop() -> None:
    """工具名不同 → 指纹不同 → 各算各的，一次都不重复。"""
    guard = LoopGuard()
    checks = [
        guard.record(tool=tool, arguments={"path": "."}, result_text="same\n")
        for tool in ("Read", "Glob", "Grep")
    ]

    assert [check.verdict for check in checks] == [LoopVerdict.CONTINUE] * 3
    assert guard.stats() == {
        "tracked_fingerprints": 3,
        "total_calls": 3,
        "loops_detected": 0,
    }


def test_verdict_follows_injected_backend_not_text_equality() -> None:
    """同一批文本，后端说相似 → 死循环；后端说不相似 → 不算（条件 2 是硬门槛）。"""
    hot_backend = _RecordingBackend(1.0)
    cold_backend = _RecordingBackend(0.5)
    same_text = "identical result text\n"

    hot = LoopGuard(backend=hot_backend)
    cold = LoopGuard(backend=cold_backend)
    for _ in range(3):
        hot_check = hot.record(tool="Glob", arguments={"pattern": "*.py"}, result_text=same_text)
        cold_check = cold.record(tool="Glob", arguments={"pattern": "*.py"}, result_text=same_text)

    assert hot_check.verdict is LoopVerdict.LOOP_DETECTED
    assert cold_check.verdict is LoopVerdict.REPEAT_SUSPECTED
    assert cold_check.max_similarity == 0.5


def test_guard_calls_the_injected_similarity_backend() -> None:
    """注入的假后端被真正调用：第 2 次比 1 条、第 3 次比 2 条 = 共 3 次「向量」比较。"""
    backend = _RecordingBackend(0.99)
    guard = LoopGuard(backend=backend)
    checks = [
        guard.record(tool="Glob", arguments={"pattern": "*.py"}, result_text="a.py\nb.py\n")
        for _ in range(3)
    ]

    assert checks[-1].verdict is LoopVerdict.LOOP_DETECTED
    assert len(backend.calls) == 3
    assert backend.calls[-1] == ("a.py\nb.py\n", "a.py\nb.py\n")


def test_backend_failure_does_not_raise_and_does_not_fake_a_loop() -> None:
    """后端抛 MyAgentError（如 embedding 端点不可用）→ 按「不相似」处理，不终止回合。"""
    guard = LoopGuard(backend=_ExplodingBackend())
    checks = [guard.record(tool="Read", arguments=READ_ARGS, result_text=READ_TEXT) for _ in range(3)]

    assert all(check.verdict is LoopVerdict.REPEAT_SUSPECTED for check in checks[1:])
    assert checks[-1].max_similarity == 0.0
    assert guard.stats()["loops_detected"] == 0


class _ExplodingBackend:
    """每次比较都抛 ``MyAgentError`` 的后端（模拟向量服务不可用）。"""

    backend_id = "exploding-fake"

    def similarity(self, a: str, b: str) -> float:
        """抛出 :class:`~my_agent.errors.LLMUnavailableError`。

        Args:
            a: 左文本（忽略）。
            b: 右文本（忽略）。

        Raises:
            LLMUnavailableError: 恒抛。
        """
        from my_agent.errors import LLMUnavailableError

        raise LLMUnavailableError("fake embedding endpoint down", detail="unit-test")


# ---------------------------------------------------------------------- 有界内存
def test_window_eviction_bounds_memory() -> None:
    """窗口外的指纹被 LRU 淘汰：跟踪数不随调用数增长。"""
    guard = LoopGuard(window=4)
    for index in range(50):
        guard.record(tool="Read", arguments={"file_path": f"f{index}.py"}, result_text="same\n")

    assert guard.stats()["tracked_fingerprints"] == 4
    assert guard.stats()["total_calls"] == 50


def test_evicted_fingerprint_restarts_counting() -> None:
    """被淘汰的指纹再次出现时从第 1 次重新计数（有界内存的取舍，不误判）。"""
    guard = LoopGuard(window=2)
    for index in range(3):
        guard.record(tool="Read", arguments={"file_path": f"f{index}.py"}, result_text="same\n")

    again = guard.record(tool="Read", arguments={"file_path": "f0.py"}, result_text="same\n")

    assert again.verdict is LoopVerdict.CONTINUE
    assert again.repeat_count == 1


def test_default_window_is_bounded_by_config() -> None:
    """默认窗口 = ``LOOP_GUARD_WINDOW``，超量指纹不会无界增长。"""
    guard = LoopGuard()
    for index in range(LOOP_GUARD_WINDOW + 5):
        guard.record(tool="Read", arguments={"file_path": f"f{index}.py"}, result_text="same\n")

    assert guard.stats()["tracked_fingerprints"] <= LOOP_GUARD_WINDOW
    assert guard.window == LOOP_GUARD_WINDOW


# ------------------------------------------------------------------ 输入鲁棒性
def test_empty_and_none_results_do_not_crash() -> None:
    """空串 / ``None`` / 纯空白都不得崩溃；「空 == 空」也算相似（真死循环）。"""
    guard = LoopGuard()
    first = guard.record(tool="Glob", arguments={"pattern": "*.nope"}, result_text="")
    second = guard.record(
        tool="Glob",
        arguments={"pattern": "*.nope"},
        result_text=None,  # type: ignore[arg-type]
    )
    third = guard.record(tool="Glob", arguments={"pattern": "*.nope"}, result_text="   ")

    assert first.verdict is LoopVerdict.CONTINUE
    assert second.verdict is LoopVerdict.REPEAT_SUSPECTED
    assert third.verdict is LoopVerdict.LOOP_DETECTED, "同参数返回空 3 次确实是死循环"
    assert third.max_similarity == 1.0


def test_weird_arguments_do_not_crash() -> None:
    """非 Mapping 参数、空工具名都不抛异常（防御性，不影响主流程）。"""
    guard = LoopGuard()
    first = guard.record(tool="", arguments=None, result_text="x")  # type: ignore[arg-type]
    second = guard.record(tool="", arguments=42, result_text="x")  # type: ignore[arg-type]

    assert first.tool == "(未知工具)"
    assert first.repeat_count == 1
    assert second.verdict is LoopVerdict.REPEAT_SUSPECTED


def test_long_results_are_truncated_before_similarity() -> None:
    """超过 :data:`RESULT_TEXT_CAP` 的文本先截断（向量规模与耗时可控）。"""
    guard = LoopGuard()
    head = "x" * RESULT_TEXT_CAP
    third = None
    for index in range(3):
        third = guard.record(
            tool="Read",
            arguments={"file_path": "big.log"},
            result_text=f"{head}tail-{index}",
        )

    assert third is not None
    assert third.verdict is LoopVerdict.LOOP_DETECTED
    assert third.max_similarity == 1.0


# ------------------------------------------------------------------------ reset
def test_reset_clears_all_state() -> None:
    """``reset()`` 把指纹表与三个计数器全部归零（每回合起点干净）。"""
    guard = LoopGuard()
    _record_n(guard, 3, READ_TEXT)
    assert guard.stats() == {
        "tracked_fingerprints": 1,
        "total_calls": 3,
        "loops_detected": 1,
    }

    guard.reset()

    assert guard.stats() == {
        "tracked_fingerprints": 0,
        "total_calls": 0,
        "loops_detected": 0,
    }
    again = guard.record(tool="Read", arguments=READ_ARGS, result_text=READ_TEXT)
    assert again.verdict is LoopVerdict.CONTINUE
    assert again.repeat_count == 1


def test_thresholds_default_to_shared_config_constants() -> None:
    """阈值只能来自 ``config/defaults.py``（不许硬编码）。"""
    guard = LoopGuard()

    assert guard.repeat_threshold == LOOP_REPEAT_THRESHOLD
    assert guard.similarity_threshold == LOOP_RESULT_SIMILARITY
    assert guard.window == LOOP_GUARD_WINDOW
    assert isinstance(guard.backend, TrigramCosineBackend), "默认后端 = 离线 trigram"


def test_verdict_wire_values_are_frozen() -> None:
    """``LoopVerdict`` 的字符串值冻结（审计/日志/测试按值断言）。"""
    assert LoopVerdict.CONTINUE == "continue"
    assert LoopVerdict.REPEAT_SUSPECTED == "repeat_suspected"
    assert LoopVerdict.LOOP_DETECTED == "loop_detected"
