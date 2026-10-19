"""``llm/client.py`` 与 ``llm/tokenizer.py`` 回归测试（§13.3 验收 I-2 / I-6 / I-7 / I-10）。

覆盖点与验收项对照：

* ``I-2`` 流式 ``delta.tool_calls[]`` 按 ``index`` **拼接** ``arguments``（不是覆盖）；
* ``I-6`` ``finish_reason == "length"`` → 丢弃部分工具调用并提示；纯文本保留 + ``truncated``；
* ``I-7`` 末帧 usage 采集；缺失 → tokenizer 估算 + ``usage_estimated`` 标记；累加正确；
* ``I-10`` ``aclose()`` 必须 await ``provider.close()``；
* 重试策略（§13.3）：网络/5xx/429 → 退避 ``LLM_BACKOFF_S``，最多 ``LLM_MAX_RETRIES``；
  4xx 非 429 **不重试**；重试次数可审计（``UsageTracker.retries``）；
* 超时按请求传：``main`` 120s / ``judge`` 4s / 其它小任务 30s；
* 消息净化：内部元数据字段绝不进入请求；
* 流式显示：``display.stream_text(delta)`` 逐帧回调。
"""

from __future__ import annotations

import json

import pytest

from my_agent.config import defaults as D
from my_agent.config.schema import Config
from my_agent.errors import (
    ConfigError,
    LLMRequestError,
    LLMUnavailableError,
    MyAgentError,
    ToolCallParseError,
)
from my_agent.llm.client import (
    TRUNCATED_NOTICE,
    LLMClient,
    UsageTracker,
    role_timeout,
    sanitize,
)
from my_agent.llm.provider import StreamChunk, ToolCallDelta
from my_agent.models.messages import Message, ToolCallRequest
from tests.fakes.provider import FakeProvider

#: 三个分片的工具调用脚本（B7：``arguments`` 必须拼接，覆盖会得到非法 JSON）。
_SPLIT_ARGUMENTS = ['{"file_p', 'ath": "a.py"', "}"]

#: 拼接后的期望原始参数串。
_JOINED_ARGUMENTS = '{"file_path": "a.py"}'


class _RecordingDisplay:
    """记录 ``stream_text`` 调用的最小 DisplayPort 实现。"""

    def __init__(self) -> None:
        """初始化空记录。"""
        self.deltas: list[str] = []

    def stream_text(self, delta: str) -> None:
        """记录一帧文本增量。

        Args:
            delta: 文本增量。
        """
        self.deltas.append(delta)

    def info(self, msg: str) -> None:
        """忽略提示。"""

    def warn(self, msg: str) -> None:
        """忽略警告。"""

    def error(self, msg: str) -> None:
        """忽略错误。"""

    def tool_start(self, name: str, summary: str) -> None:
        """忽略工具开始事件。"""

    def tool_result(self, name: str, ok: bool, summary: str) -> None:
        """忽略工具结束事件。"""


def _client(
    provider: FakeProvider,
    *,
    config: Config | None = None,
    usage: UsageTracker | None = None,
    sleeps: list[float] | None = None,
    real_sleep: bool = False,
) -> LLMClient:
    """装配一个 ``LLMClient``（默认不真正睡眠）。

    Args:
        provider: 脚本化 Provider。
        config: 配置；``None`` = 默认配置（端点为空，走默认窗口）。
        usage: 用量累计器；``None`` = 新建。
        sleeps: 传入时记录退避秒数而不真正睡眠。
        real_sleep: ``True`` = 使用生产路径的 ``asyncio.sleep``。

    Returns:
        可直接 await 的客户端。
    """
    if real_sleep:
        return LLMClient(provider, config or Config(), usage or UsageTracker())

    async def _sleep(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    return LLMClient(provider, config or Config(), usage or UsageTracker(), sleep=_sleep)


def _user(text: str = "hi") -> Message:
    """构造一条用户消息。

    Args:
        text: 消息正文。

    Returns:
        用户 :class:`~my_agent.models.messages.Message`。
    """
    return Message(role="user", content=text)


# --------------------------------------------------------------------------------------
# I-2：流式工具调用分片聚合（按 index 拼接 arguments）
# --------------------------------------------------------------------------------------


async def test_tool_call_arguments_are_concatenated_across_chunks() -> None:
    """I-2 / B7：index 相同的三个分片必须**拼接**成一份合法 JSON。"""
    provider = FakeProvider(
        [
            FakeProvider.chunks(
                [
                    StreamChunk(text_delta="我来读文件："),
                    StreamChunk(tool_call_delta=ToolCallDelta(index=0, id="call_a", name="Read")),
                    *[
                        StreamChunk(tool_call_delta=ToolCallDelta(index=0, arguments_delta=piece))
                        for piece in _SPLIT_ARGUMENTS
                    ],
                    StreamChunk(finish_reason="tool_calls"),
                    StreamChunk(
                        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
                    ),
                ]
            )
        ]
    )
    turn = await _client(provider).assistant_turn("main", messages=[_user()])

    assert turn.text == "我来读文件："
    assert turn.finish_reason == "tool_calls"
    assert turn.truncated is False
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.id == "call_a"
    assert call.name == "Read"
    assert call.raw_arguments == _JOINED_ARGUMENTS
    assert call.arguments == {"file_path": "a.py"}
    assert turn.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


async def test_interleaved_indices_are_aggregated_separately() -> None:
    """不同 ``index`` 的工具调用必须各自聚合，且按 index 顺序输出。"""
    provider = FakeProvider(
        [
            FakeProvider.chunks(
                [
                    StreamChunk(tool_call_delta=ToolCallDelta(index=1, id="b", name="Grep")),
                    StreamChunk(tool_call_delta=ToolCallDelta(index=0, id="a", name="Read")),
                    StreamChunk(tool_call_delta=ToolCallDelta(index=1, arguments_delta='{"p')),
                    StreamChunk(tool_call_delta=ToolCallDelta(index=0, arguments_delta='{"f')),
                    StreamChunk(tool_call_delta=ToolCallDelta(index=1, arguments_delta='attern":"x"}')),
                    StreamChunk(tool_call_delta=ToolCallDelta(index=0, arguments_delta='ile":"y"}')),
                    StreamChunk(finish_reason="tool_calls"),
                ]
            )
        ]
    )
    turn = await _client(provider).assistant_turn("main", messages=[_user()])

    assert [call.id for call in turn.tool_calls] == ["a", "b"]
    assert turn.tool_calls[0].arguments == {"file": "y"}
    assert turn.tool_calls[1].arguments == {"pattern": "x"}


async def test_partial_id_or_name_does_not_overwrite_earlier_fragment() -> None:
    """后续分片里的 ``None`` 字段不得覆盖首帧已给出的 ``id`` / ``name``。"""
    provider = FakeProvider(
        [
            FakeProvider.chunks(
                [
                    StreamChunk(tool_call_delta=ToolCallDelta(index=0, id="keep", name="Read")),
                    StreamChunk(tool_call_delta=ToolCallDelta(index=0, arguments_delta="{}")),
                    StreamChunk(finish_reason="tool_calls"),
                ]
            )
        ]
    )
    turn = await _client(provider).assistant_turn("main", messages=[_user()])

    assert turn.tool_calls[0].id == "keep"
    assert turn.tool_calls[0].name == "Read"


async def test_invalid_json_arguments_are_kept_raw_and_do_not_raise() -> None:
    """B7：``json.loads`` 失败 → 保留 ``raw_arguments``、记 ``ToolCallParseError``、**不抛**。"""
    provider = FakeProvider([FakeProvider.tool_call("Read", "{not json")])
    turn = await _client(provider).assistant_turn("main", messages=[_user()])

    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.raw_arguments == "{not json"
    assert call.arguments == {}
    error = ToolCallParseError("x", detail="x", name=call.name, call_id=call.id)
    assert error.user_message()  # 附录 A 文案可用（模型下一轮据此自纠）


async def test_display_receives_every_text_delta() -> None:
    """流式显示：每个 ``text_delta`` 都要回调 ``display.stream_text``（CLI 实时渲染）。"""
    display = _RecordingDisplay()
    provider = FakeProvider(
        [
            FakeProvider.chunks(
                [
                    StreamChunk(text_delta="a"),
                    StreamChunk(text_delta="b"),
                    StreamChunk(finish_reason="stop"),
                    StreamChunk(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
                ]
            )
        ]
    )
    turn = await _client(provider).assistant_turn(
        "main", messages=[_user()], display=display
    )

    assert turn.text == "ab"
    assert display.deltas == ["a", "b"]


# --------------------------------------------------------------------------------------
# I-6：finish_reason == "length"
# --------------------------------------------------------------------------------------


async def test_length_with_partial_tool_call_drops_call_and_adds_notice() -> None:
    """I-6：被截断的工具调用整条丢弃，并追加「输出过长，请分批」。"""
    provider = FakeProvider(
        [
            FakeProvider.tool_call(
                "Read", {"file_path": "a.py"}, text="先看文件", finish_reason="length"
            )
        ]
    )
    turn = await _client(provider).assistant_turn("main", messages=[_user()])

    assert turn.tool_calls == []
    assert turn.truncated is True
    assert turn.finish_reason == "length"
    assert turn.text.startswith("先看文件")
    assert TRUNCATED_NOTICE in turn.text


async def test_length_with_pure_text_keeps_text_and_marks_truncated() -> None:
    """I-6：纯文本截断 → 文本保留并标记 ``truncated=True``。"""
    provider = FakeProvider([FakeProvider.text("被截断的回答", finish_reason="length")])
    turn = await _client(provider).assistant_turn("main", messages=[_user()])

    assert "被截断的回答" in turn.text
    assert turn.truncated is True
    assert turn.tool_calls == []


async def test_stop_finish_reason_is_not_marked_truncated() -> None:
    """对照组：``finish_reason="stop"`` 不得被误判为截断。"""
    provider = FakeProvider([FakeProvider.text("完整回答")])
    turn = await _client(provider).assistant_turn("main", messages=[_user()])

    assert turn.truncated is False
    assert turn.text == "完整回答"


# --------------------------------------------------------------------------------------
# I-7：usage 采集与估算兜底
# --------------------------------------------------------------------------------------


async def test_usage_from_final_chunk_is_accumulated_exactly() -> None:
    """I-7：末帧 usage 原样累加，不做估算。"""
    usage = UsageTracker()
    provider = FakeProvider(
        [
            FakeProvider.text("a", usage={"prompt_tokens": 10, "completion_tokens": 4}),
            FakeProvider.text("b", usage={"prompt_tokens": 1, "completion_tokens": 2}),
        ]
    )
    client = _client(provider, usage=usage)
    await client.assistant_turn("main", messages=[_user("q1")])
    await client.assistant_turn("main", messages=[_user("q2")])

    assert usage.totals["prompt"] == 11
    assert usage.totals["completion"] == 6
    assert usage.totals["total"] == 17
    assert usage.estimated_usage_calls == 0
    assert "estimated_total" not in usage.snapshot()
    assert usage.snapshot()["prompt_tokens"] == 11


async def test_missing_usage_falls_back_to_tokenizer_estimate() -> None:
    """I-7：端点未返回 usage → tokenizer 估算 + ``usage_estimated`` 标记。"""
    usage = UsageTracker()
    provider = FakeProvider([FakeProvider.text("估一估")])
    turn = await _client(provider, usage=usage).assistant_turn(
        "main", messages=[_user("请估算")]
    )

    assert turn.usage["prompt_tokens"] > 0
    assert turn.usage["completion_tokens"] > 0
    assert turn.usage["total_tokens"] == turn.usage["prompt_tokens"] + turn.usage["completion_tokens"]
    assert usage.estimated_usage_calls == 1
    assert usage.totals["estimated_total"] == usage.totals["total"] > 0
    assert "估算" in usage.cost_report()


async def test_usage_tracker_report_and_retry_counters() -> None:
    """I-7 / 重试可审计：``cost_report`` 与重试计数一致。"""
    usage = UsageTracker()
    usage.add({"prompt_tokens": 3, "completion_tokens": 4}, estimated=True)
    usage.note_retry()
    usage.note_retry(2)
    report = usage.cost_report()

    assert usage.totals["total"] == 7
    assert usage.retries == 3
    assert "prompt" in report and "completion" in report
    assert "重试 3 次" in report


# --------------------------------------------------------------------------------------
# 重试策略（§13.3）
# --------------------------------------------------------------------------------------


async def test_retries_network_errors_with_backoff_then_succeeds() -> None:
    """网络/5xx/429 → 退避 ``LLM_BACKOFF_S``，第 4 次成功；重试次数可审计。"""
    sleeps: list[float] = []
    usage = UsageTracker()
    provider = FakeProvider(
        [
            FakeProvider.raise_(LLMUnavailableError("429")),
            FakeProvider.raise_(LLMUnavailableError("500")),
            FakeProvider.raise_(LLMUnavailableError("connection")),
            FakeProvider.text("ok", usage={"prompt_tokens": 1, "completion_tokens": 1}),
        ]
    )
    result = await _client(provider, usage=usage, sleeps=sleeps).complete_role("judge", user="hi")

    assert result.text == "ok"
    assert provider.call_count == D.LLM_MAX_RETRIES + 1
    assert sleeps == list(D.LLM_BACKOFF_S)
    assert usage.retries == D.LLM_MAX_RETRIES
    assert usage.errors == 0


async def test_retries_are_capped_at_llm_max_retries() -> None:
    """重试上限：**恰好** ``LLM_MAX_RETRIES`` 次重试后抛出，不再尝试。"""
    sleeps: list[float] = []
    usage = UsageTracker()
    provider = FakeProvider(
        [FakeProvider.raise_(LLMUnavailableError("boom")) for _ in range(D.LLM_MAX_RETRIES + 1)]
    )
    with pytest.raises(LLMUnavailableError):
        await _client(provider, usage=usage, sleeps=sleeps).complete_role("judge", user="hi")

    assert provider.call_count == D.LLM_MAX_RETRIES + 1
    assert usage.retries == D.LLM_MAX_RETRIES
    assert usage.errors == 1


async def test_client_error_is_never_retried() -> None:
    """4xx（非 429）→ ``LLMRequestError``，**不重试**（调用次数必须是 1）。"""
    sleeps: list[float] = []
    usage = UsageTracker()
    provider = FakeProvider([FakeProvider.raise_(LLMRequestError("400 bad schema"))])
    with pytest.raises(LLMRequestError):
        await _client(provider, usage=usage, sleeps=sleeps).complete_role("judge", user="hi")

    assert provider.call_count == 1
    assert sleeps == []
    assert usage.retries == 0


async def test_config_error_is_never_retried() -> None:
    """鉴权/配置类错误（``ConfigError``）同样不重试。"""
    provider = FakeProvider([FakeProvider.raise_(ConfigError("缺少 API Key", detail="缺少 API Key"))])
    with pytest.raises(ConfigError):
        await _client(provider).complete_role("judge", user="hi")

    assert provider.call_count == 1


async def test_unknown_role_is_rejected_before_any_call() -> None:
    """未知角色必须在**发起请求前**被拒绝（避免业务代码静默用错提示词）。"""
    provider = FakeProvider([FakeProvider.text("never")])
    with pytest.raises(ValueError, match="未知角色"):
        await _client(provider).complete_role("nope", user="hi")

    assert provider.call_count == 0


# --------------------------------------------------------------------------------------
# 超时与角色默认参数（§13.3）
# --------------------------------------------------------------------------------------


def test_role_timeout_mapping_uses_defaults() -> None:
    """角色 → 超时映射：main 120s、judge 4s、其它小任务 30s（全部取自 defaults）。"""
    assert role_timeout("main") == D.TIMEOUT_MAIN_S == 120.0
    assert role_timeout("judge") == D.JUDGE_TIMEOUT_S == 4.0
    for role in ("compactor", "memory_extract", "memory_scope", "memory_merge",
                 "skill_synonym", "repomap", "router_gen"):
        assert role_timeout(role) == D.TIMEOUT_SMALL_S == 30.0


async def test_timeout_is_passed_per_request_not_at_client_level() -> None:
    """超时按角色**按请求**传（provider 每次都能看到 ``timeout_s``）。"""
    provider = FakeProvider(
        [
            FakeProvider.text("m", usage={"prompt_tokens": 1, "completion_tokens": 1}),
            FakeProvider.text("j", usage={"prompt_tokens": 1, "completion_tokens": 1}),
            FakeProvider.text("s", usage={"prompt_tokens": 1, "completion_tokens": 1}),
        ]
    )
    client = _client(provider)
    await client.complete_role("main", user="a")
    await client.complete_role("judge", user="b")
    await client.complete_role("compactor", user="c")

    assert [call.timeout_s for call in provider.calls] == [120.0, 4.0, 30.0]
    assert [call.temperature for call in provider.calls] == [0.2, 0.0, 0.0]


async def test_temperature_and_max_tokens_come_from_config() -> None:
    """温度取自 config（主对话 vs 判定），max_tokens 取自 ``config.max_output_tokens()``。"""
    config = Config()
    config.llm.temperature = 0.9
    config.llm.judge_temperature = 0.1
    provider = FakeProvider(
        [
            FakeProvider.text("a", usage={"prompt_tokens": 1, "completion_tokens": 1}),
            FakeProvider.text("b", usage={"prompt_tokens": 1, "completion_tokens": 1}),
        ]
    )
    client = _client(provider, config=config)
    await client.complete_role("main", user="a")
    await client.complete_role("judge", user="b")

    assert [call.temperature for call in provider.calls] == [0.9, 0.1]
    assert all(call.max_tokens == config.max_output_tokens() for call in provider.calls)
    assert config.max_output_tokens() > 0


async def test_explicit_kwargs_override_role_defaults() -> None:
    """显式 ``max_tokens`` / ``temperature`` / ``timeout_s`` 覆盖角色默认值。"""
    provider = FakeProvider([FakeProvider.text("x", usage={"prompt_tokens": 1, "completion_tokens": 1})])
    await _client(provider).complete_role(
        "judge", user="hi", max_tokens=7, temperature=0.5, timeout_s=9.5
    )
    call = provider.calls[-1]

    assert (call.max_tokens, call.temperature, call.timeout_s) == (7, 0.5, 9.5)


async def test_system_prompt_comes_from_role_table() -> None:
    """角色 → system prompt 映射必须来自 ``prompts.ROLE_PROMPTS``（业务代码无内联提示词）。"""
    from my_agent.llm.prompts import JUDGE_SYSTEM_PROMPT, ROLE_PROMPTS

    provider = FakeProvider([FakeProvider.text("x", usage={"prompt_tokens": 1, "completion_tokens": 1})])
    await _client(provider).complete_role("judge", user="hi")

    assert provider.calls[-1].system == JUDGE_SYSTEM_PROMPT == ROLE_PROMPTS["judge"]


# --------------------------------------------------------------------------------------
# 消息净化（§13.3）
# --------------------------------------------------------------------------------------


def test_sanitize_strips_internal_metadata_fields() -> None:
    """``sanitize`` 必须剥掉全部内部元数据字段（§4.1 不变量）。"""
    msg = Message(
        role="user",
        content="hi",
        turn_index=7,
        subagent_id="sub-1",
        tokens=42,
        kind="tool_result",
        archive_ref="archive/x.jsonl",
        is_large_result=True,
    )
    [clean] = sanitize([msg])

    assert set(clean) == {"role", "content"}
    assert clean == {"role": "user", "content": "hi"}
    for field in ("id", "tokens", "kind", "archive_ref", "turn_index", "subagent_id"):
        assert field not in clean


def test_sanitize_keeps_protocol_fields_and_tool_calls() -> None:
    """``tool_calls`` / ``tool_call_id`` 必须保留（回填工具结果的前提）。"""
    assistant = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCallRequest(id="c1", name="Read", arguments={"file_path": "a.py"})
        ],
        turn_index=3,
    )
    tool = Message(role="tool", content="file body", tool_call_id="c1", tokens=9)
    clean = sanitize([assistant, tool])

    assert clean[0]["tool_calls"][0]["function"]["name"] == "Read"
    assert json.loads(clean[0]["tool_calls"][0]["function"]["arguments"]) == {"file_path": "a.py"}
    assert clean[1]["tool_call_id"] == "c1"
    assert "tokens" not in clean[1]


def test_sanitize_accepts_already_converted_dicts() -> None:
    """``sanitize`` 也要能吃字典（同一份消息可能被上层转换过一次）。"""
    clean = sanitize([{"role": "user", "content": "x", "turn_index": 4, "tokens": 1}])

    assert clean == [{"role": "user", "content": "x"}]


async def test_internal_fields_never_reach_the_provider() -> None:
    """端到端：带内部元数据的消息发出去时已被剥离。"""
    provider = FakeProvider([FakeProvider.text("ok", usage={"prompt_tokens": 1, "completion_tokens": 1})])
    await _client(provider).assistant_turn(
        "main", messages=[Message(role="user", content="hi", turn_index=5, tokens=12)]
    )

    [sent] = provider.calls[-1].messages
    assert isinstance(sent, dict)
    assert set(sent) == {"role", "content"}


# --------------------------------------------------------------------------------------
# I-10：aclose
# --------------------------------------------------------------------------------------


async def test_aclose_awaits_provider_close() -> None:
    """I-10：``LLMClient.aclose()`` 必须调用 ``provider.close()``。"""
    provider = FakeProvider([FakeProvider.text("x")])
    client = _client(provider)
    assert provider.closed is False

    await client.aclose()

    assert provider.closed is True


# --------------------------------------------------------------------------------------
# stream_role 透传
# --------------------------------------------------------------------------------------


async def test_stream_role_yields_provider_chunks_verbatim() -> None:
    """``stream_role`` 逐帧透传（不聚合），并把消息净化后交给 Provider。"""
    provider = FakeProvider([FakeProvider.text("ab")])
    frames = [
        chunk
        async for chunk in _client(provider).stream_role(
            "main", messages=[Message(role="user", content="q", turn_index=2)], tools=None
        )
    ]

    assert [chunk.text_delta for chunk in frames] == ["ab", ""]
    assert frames[-1].finish_reason == "stop"
    assert provider.calls[-1].timeout_s == D.TIMEOUT_MAIN_S
    assert provider.calls[-1].messages == [{"role": "user", "content": "q"}]


async def test_unknown_myagent_error_from_provider_is_propagated() -> None:
    """Provider 抛出的其它内部异常原样上抛（不吞异常）。"""
    provider = FakeProvider([FakeProvider.raise_(MyAgentError("boom", detail="boom"))])
    with pytest.raises(MyAgentError):
        await _client(provider).complete_role("judge", user="hi")


async def test_retry_uses_asyncio_sleep_from_defaults_when_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未注入 ``sleep`` 时走生产路径：``asyncio.sleep(LLM_BACKOFF_S[0])``。"""
    slept: list[float] = []

    async def _spy(seconds: float) -> None:
        slept.append(seconds)

    import my_agent.llm.client as client_module

    monkeypatch.setattr(client_module.asyncio, "sleep", _spy)
    provider = FakeProvider(
        [
            FakeProvider.raise_(LLMUnavailableError("429")),
            FakeProvider.text("ok", usage={"prompt_tokens": 1, "completion_tokens": 1}),
        ]
    )
    usage = UsageTracker()
    result = await _client(provider, usage=usage, real_sleep=True).complete_role("judge", user="hi")

    assert result.text == "ok"
    assert usage.retries == 1
    assert slept == [D.LLM_BACKOFF_S[0]]
