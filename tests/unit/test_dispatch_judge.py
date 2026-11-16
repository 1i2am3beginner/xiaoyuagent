"""单元测试：任务分发审判器（U35）。

覆盖用户明确要求的四件事：

1. **最小信息集**：只给「原始需求 / 子任务 / 约束 / 文件路径」，不打包会话历史；
   文件由审判器**自己读**，内容不回流主上下文（``read_file_excerpts``）；
2. **过粗** → 审判器**亲自补发**（第二条派生路径），结论带专属标记；
3. **过细** → **不派生**，返回结论/补丁/影响分析/理由，交主 Agent 重新分发；
4. **审判模型不可用** → 指数退避重试（1s→2s→4s），仍失败则**放行 + 显式告警用户**。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from my_agent.agents.dispatch_judge import (
    DISPATCH_JUDGE_ROLE,
    JUDGE_DOWN_WARNING,
    SUPPLEMENT_MARKER,
    DispatchJudge,
    DispatchVerdict,
    SubtaskSpec,
    build_context,
    extract_paths,
    read_file_excerpts,
    supplement_text,
)
from my_agent.errors import ConfigError, MyAgentError


class _StubLLM:
    """按脚本返回 ``text`` 或抛异常的假 LLM。"""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    async def complete_role(self, role: str, **kwargs: Any) -> Any:
        """记录调用并按脚本返回。

        Args:
            role: 角色名。
            **kwargs: 透传（其中 ``user`` 是提示词）。

        Returns:
            带 ``text`` 属性的对象。

        Raises:
            BaseException: 脚本里是异常时原样抛出。
        """
        self.calls.append({"role": role, **kwargs})
        item = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        if isinstance(item, BaseException):
            raise item
        return SimpleNamespace(text=item)


class _StubRunner:
    """记录补发请求的假 SubagentRunner。"""

    def __init__(self, *, ok: bool = True) -> None:
        self.ok = ok
        self.spawned: list[dict[str, Any]] = []

    async def spawn(self, **kwargs: Any) -> Any:
        """记录并返回结论。

        Args:
            **kwargs: spawn 实参。

        Returns:
            带 ``ok``/``text``/``error`` 的对象。
        """
        self.spawned.append(kwargs)
        return SimpleNamespace(
            ok=self.ok,
            text=f"补发结论：{kwargs.get('description')}",
            error=None if self.ok else "补发失败",
        )


class _RecordingDisplay:
    """记录输出消息的假 DisplayPort。"""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def stream_text(self, delta: str) -> None:
        """忽略。"""

    def info(self, msg: str) -> None:
        """记录 info。"""
        self.infos.append(msg)

    def warn(self, msg: str) -> None:
        """记录 warning。"""
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        """忽略。"""

    def tool_start(self, name: str, summary: str) -> None:
        """忽略。"""

    def tool_result(self, name: str, ok: bool, summary: str) -> None:
        """忽略。"""


def _judgment(**overrides: Any) -> str:
    """生成审判 JSON 文本。

    Args:
        **overrides: 覆盖字段。

    Returns:
        JSON 字符串。
    """
    payload: dict[str, Any] = {
        "verdict": "reasonable",
        "confidence": "high",
        "reason": "覆盖完整、粒度合适",
        "missing_tasks": [],
        "merged_into": None,
        "patch": [],
        "impact": [],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def _judge(llm: _StubLLM, **kwargs: Any) -> DispatchJudge:
    """构造审判器（sleep 注入为无等待）。

    Args:
        llm: 假 LLM。
        **kwargs: 透传给 :class:`DispatchJudge`。

    Returns:
        审判器实例。
    """
    sleeps: list[float] = []

    async def _no_sleep(delay: float) -> None:
        sleeps.append(delay)

    judge = DispatchJudge(
        llm=llm,  # type: ignore[arg-type]
        config=None,  # type: ignore[arg-type]
        display=kwargs.pop("display", _RecordingDisplay()),
        sleep=_no_sleep,
        **kwargs,
    )
    judge._test_sleeps = sleeps  # type: ignore[attr-defined]
    return judge


def _context(**overrides: Any) -> Any:
    """构造最小信息集。

    Args:
        **overrides: 覆盖字段。

    Returns:
        :class:`~my_agent.agents.dispatch_judge.JudgmentContext`。
    """
    from my_agent.agents.dispatch_judge import JudgmentContext

    payload: dict[str, Any] = {
        "user_requirement": "把 auth.py 的 JWT 校验改成 RS256 并补测试",
        "subtasks": [
            SubtaskSpec(description="改校验", prompt="修改 auth.py 的 verify 函数"),
            SubtaskSpec(description="补测试", prompt="为 auth.py 增加测试"),
        ],
        "constraints": "禁止改动公开 API",
        "file_paths": ["auth.py"],
    }
    payload.update(overrides)
    return JudgmentContext(**payload)


# ---------------------------------------------------------------------------- 最小信息集


def test_extract_paths_picks_path_like_tokens() -> None:
    """从子任务描述里提取路径（主 Agent 只传路径，不传内容）。"""
    text = "修改 src/my_agent/auth.py 的 verify，并更新 tests/test_auth.py"
    paths = extract_paths(text)
    assert "src/my_agent/auth.py" in paths
    assert "tests/test_auth.py" in paths


def test_read_file_excerpts_reads_caps_and_skips_missing(workdir: Path) -> None:
    """审判器自己读文件：只读存在的、截断超长、跳过缺失。"""
    long_file = workdir / "long.py"
    long_file.write_text("x" * 50, encoding="utf-8")
    excerpts = read_file_excerpts(
        ["long.py", "missing.py"], root=workdir, max_files=5, max_chars=20
    )
    assert "missing.py" not in excerpts
    assert "已截断" in excerpts["long.py"]


def test_read_file_excerpts_respects_max_files(workdir: Path) -> None:
    """文件数量受 ``max_files`` 限制（防止审判上下文爆掉）。"""
    for index in range(5):
        (workdir / f"f{index}.txt").write_text("a", encoding="utf-8")
    excerpts = read_file_excerpts(
        [f"f{i}.txt" for i in range(5)], root=workdir, max_files=2, max_chars=100
    )
    assert len(excerpts) == 2


def test_build_context_reads_files_itself(workdir: Path) -> None:
    """``build_context`` 自行读文件；路径清单来自子任务描述。"""
    (workdir / "auth.py").write_text("def verify(): ...\n", encoding="utf-8")
    context = build_context(
        user_requirement="改 auth.py",
        subtasks=[SubtaskSpec(description="改", prompt="修改 auth.py 的 verify")],
        root=workdir,
    )
    assert "auth.py" in context.file_paths
    assert "def verify" in context.file_excerpts["auth.py"]


def test_build_context_can_skip_file_reading(workdir: Path) -> None:
    """``read_files=False`` 时只给路径不给内容。"""
    (workdir / "auth.py").write_text("secret\n", encoding="utf-8")
    context = build_context(
        user_requirement="改",
        subtasks=[SubtaskSpec(description="改", prompt="修改 auth.py")],
        root=workdir,
        read_files=False,
    )
    assert "auth.py" in context.file_paths
    assert context.file_excerpts == {}


def test_prompt_contains_minimal_set_and_not_full_history() -> None:
    """提示词只含最小信息集四要素，且不含无关会话内容。"""
    judge = _judge(_StubLLM([_judgment()]))
    prompt = judge._build_prompt(_context())
    assert "把 auth.py 的 JWT 校验改成 RS256" in prompt
    assert "改校验" in prompt and "补测试" in prompt
    assert "禁止改动公开 API" in prompt
    assert "auth.py" in prompt


# ---------------------------------------------------------------------------- 三种裁决


async def test_reasonable_passes_through_without_supplement() -> None:
    """合理 → 不补发、不打回。"""
    judge = _judge(_StubLLM([_judgment()]), runner=_StubRunner())
    judgment, supplements = await judge.review(_context())
    assert judgment.verdict is DispatchVerdict.REASONABLE
    assert supplements == []
    assert judge.judge_degraded is False


async def test_too_coarse_triggers_judge_side_dispatch() -> None:
    """过粗 → 审判器亲自补发（第二条派生路径）。"""
    runner = _StubRunner()
    judge = _judge(
        _StubLLM(
            [
                _judgment(
                    verdict="too_coarse",
                    reason="漏了集成测试与文档",
                    missing_tasks=[
                        {"description": "补集成测试", "prompt": "为 auth 增加集成测试"},
                        {"description": "更新文档", "prompt": "更新 README 的鉴权说明"},
                    ],
                )
            ]
        ),
        runner=runner,  # type: ignore[arg-type]
    )
    judgment, supplements = await judge.review(_context())
    assert judgment.verdict is DispatchVerdict.TOO_COARSE
    assert len(supplements) == 2
    assert [item.description for item in supplements] == ["补集成测试", "更新文档"]
    assert len(runner.spawned) == 2
    assert all(call["run_in_background"] is False for call in runner.spawned)


async def test_too_coarse_without_runner_does_not_crash() -> None:
    """过粗但没注入 runner → 记录警告、不补发、不抛异常。"""
    judge = _judge(_StubLLM([_judgment(verdict="too_coarse", missing_tasks=[])]) , runner=None)
    judgment, supplements = await judge.review(_context())
    assert judgment.verdict is DispatchVerdict.TOO_COARSE
    assert supplements == []


async def test_too_fine_returns_patch_and_does_not_spawn() -> None:
    """过细 → **不派生**，返回结论/补丁/影响分析/理由。"""
    runner = _StubRunner()
    judge = _judge(
        _StubLLM(
            [
                _judgment(
                    verdict="too_fine",
                    reason="两个子任务本质是同一件事",
                    merged_into=1,
                    impact=["src/my_agent/auth.py"],
                    patch=[
                        {
                            "description": "统一改校验",
                            "prompt": "在 auth.py 里一次性改完校验并补测试",
                            "agent_type": "general",
                        }
                    ],
                )
            ]
        ),
        runner=runner,  # type: ignore[arg-type]
    )
    judgment, supplements = await judge.review(_context())
    assert judgment.verdict is DispatchVerdict.TOO_FINE
    assert supplements == []
    assert runner.spawned == [], "过细必须由主 Agent 重新分发，审判器不得派生"
    feedback = judgment.feedback_text()
    assert "过细" in feedback
    assert "合并为：1" in feedback
    assert "src/my_agent/auth.py" in feedback
    assert "统一改校验" in feedback


def test_supplement_text_carries_marker_and_conclusions() -> None:
    """补发结论文本带专属标记（测试与主 Agent 都据此识别）。"""
    from my_agent.agents.dispatch_judge import MissingDispatch

    text = supplement_text(
        [
            MissingDispatch(description="补集成测试", ok=True, text="已补 3 个用例"),
            MissingDispatch(description="更新文档", ok=False, error="超时"),
        ]
    )
    assert text.startswith(SUPPLEMENT_MARKER)
    assert "已补 3 个用例" in text
    assert "失败：超时" in text


# ---------------------------------------------------------------------------- 降级路径


async def test_fallback_retries_with_exponential_backoff_then_warns() -> None:
    """失败 → 指数退避重试 → 仍失败则放行并**显式告警用户**。"""
    display = _RecordingDisplay()
    llm = _StubLLM([ConfigError("boom")] * 8)
    judge = _judge(llm, display=display, max_retries=3)
    judgment, supplements = await judge.review(_context())

    # 4 次尝试（首试 + 3 次重试）与 1s → 2s → 4s 的指数退避
    assert len(llm.calls) == 4
    assert judge._test_sleeps == [1.0, 2.0, 4.0]  # type: ignore[attr-defined]
    # 放行（fail-open），但必须告警
    assert judgment.verdict is DispatchVerdict.REASONABLE
    assert judgment.is_fallback is True
    assert judge.judge_degraded is True
    assert supplements == []
    assert display.warnings and JUDGE_DOWN_WARNING.format(retries=3)[:20] in display.warnings[0]


async def test_transient_failure_recovers_without_degrading() -> None:
    """一次失败后重试成功 → 不降级、不告警。"""
    display = _RecordingDisplay()
    judge = _judge(_StubLLM([MyAgentError("临时故障"), _judgment()]), display=display)
    judgment, _ = await judge.review(_context())
    assert judgment.verdict is DispatchVerdict.REASONABLE
    assert judgment.is_fallback is False
    assert judge.judge_degraded is False
    assert display.warnings == []
    assert judge._test_sleeps == [1.0]  # type: ignore[attr-defined]


async def test_invalid_json_is_retried_then_degrades() -> None:
    """非法 JSON 同样走重试；重试耗尽后放行并告警。"""
    display = _RecordingDisplay()
    judge = _judge(_StubLLM(["不是 JSON"] * 6), display=display, max_retries=1)
    judgment, _ = await judge.review(_context())
    assert judgment.is_fallback is True
    assert display.warnings


async def test_invalid_verdict_value_degrades() -> None:
    """verdict 非法枚举值视为解析失败（不放行成误判）。"""
    judge = _judge(
        _StubLLM([_judgment(verdict="maybe")] * 4), max_retries=1
    )
    judgment, _ = await judge.review(_context())
    assert judgment.is_fallback is True


# ---------------------------------------------------------------------------- 解析


def test_parse_tolerates_prose_around_json() -> None:
    """模型输出带前后噪声时仍能取出 JSON 对象。"""
    judge = _judge(_StubLLM([""]))
    parsed = judge._parse(f"好的，判决如下：\n{_judgment()}\n以上。")
    assert parsed.verdict is DispatchVerdict.REASONABLE


def test_parse_skips_incomplete_patch_items() -> None:
    """补丁项缺 description/prompt 时被丢弃，不产生半截任务。"""
    judge = _judge(_StubLLM([""]))
    parsed = judge._parse(
        _judgment(verdict="too_fine", patch=[{"description": "只有标题"}, {"prompt": "只有说明"}])
    )
    assert parsed.patch == []


def test_parse_normalizes_unknown_agent_type() -> None:
    """未知 agent_type 归一为 general。"""
    judge = _judge(_StubLLM([""]))
    parsed = judge._parse(
        _judgment(
            verdict="too_coarse",
            missing_tasks=[{"description": "d", "prompt": "p", "agent_type": "weird"}],
        )
    )
    assert parsed.missing_tasks[0].agent_type == "general"


def test_judge_uses_dedicated_role() -> None:
    """审判走独立角色 ``dispatch_judge``（而非 main/judge）。"""
    llm = _StubLLM([_judgment()])
    judge = _judge(llm)

    import asyncio

    asyncio.get_event_loop_policy()
    asyncio.run(judge.review(_context()))
    assert llm.calls[0]["role"] == DISPATCH_JUDGE_ROLE


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("reasonable", DispatchVerdict.REASONABLE),
        ("too_coarse", DispatchVerdict.TOO_COARSE),
        ("too_fine", DispatchVerdict.TOO_FINE),
    ],
)
def test_verdict_values_roundtrip(verdict: str, expected: DispatchVerdict) -> None:
    """三种裁决值都能被正确解析（JSON 契约稳定）。"""
    judge = _judge(_StubLLM([""]))
    assert judge._parse(_judgment(verdict=verdict)).verdict is expected
