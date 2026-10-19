"""``FakeProvider``：脚本化的 ``ModelProvider`` 替身（§18.1，附录 C 硬性要求）。

**本模块绝不 import ``openai``**（附录 C：`FakeProvider` 已实现且不依赖 ``openai`` 包，
保证 §5–§12 测试不触网）。它只实现 **我们自己的** :class:`~my_agent.llm.provider.ModelProvider`
协议，因此 SDK 升级不会打碎这套测试（这正是 C7 依赖隔离的直接收益）。

===============================================================================
快速上手（模块 docstring 即 API 文档）
===============================================================================

``FakeProvider`` 由**脚本**驱动：构造时给一串「这一轮要返回什么 / 抛什么」，
之后每次 ``complete()`` / ``stream()`` 按顺序消费一条。**:class:`ScriptedResult`
同时是「非流式结果」与「流式分片脚本」的来源** —— 传入 ``chunks`` 的条目就按你的
分片逐帧下发，没传 ``chunks`` 的条目由 ``FakeProvider`` 自动切成帧（先文本、再按
``index`` 分片下发每个工具调用的 ``arguments``、然后 ``finish_reason``、最后 ``usage``）。

构造
----

.. code-block:: python

    FakeProvider(text="你好")            # 位置参数：任意 ScriptedResult 或异常实例
    FakeProvider([...], name="fake")     # 或list；name 覆盖 provider.name
    FakeProvider(sleep=0.01)             # 每帧前 await asyncio.sleep(...)

便捷构造器（推荐；其它 worker 可只用这些）
------------------------------------------

============================== ======================================================
``FakeProvider.text("hi")``    脚本化为一条纯文本回复（``finish_reason="stop"``）
                               —— **构造器**，返回 :class:`ScriptedResult`，
                               用 ``FakeProvider(FakeProvider.text("hi"))`` 装配。
``FakeProvider.tool_call(name, args)``
                               脚本化为一次工具调用（``args`` 可以是 dict 或原始
                               JSON 字符串；字符串原样下发，用于构造非法 JSON 场景）。
``FakeProvider.raise_(exc)``   脚本化为一抛：下一次调用抛 ``exc``（**同一个对象**，
                               便于用 ``pytest.raises`` 捕获并断言 identity）。
``FakeProvider.scripted([...])``
                               把 ``list[ScriptedResult | BaseException]`` 一次装成
                               一段脚本（列表里也可以直接写 ``"字符串"``）。
``FakeProvider.chunks([...])`` 脚本化为一段显式流式分片（``list[StreamChunk]``）。
============================== ======================================================

典型用法
--------

.. code-block:: python

    from my_agent.errors import LLMUnavailableError
    from tests.fakes.provider import FakeProvider

    # 1) 纯文本
    fake = FakeProvider([FakeProvider.text("第一轮"), FakeProvider.text("第二轮")])

    # 2) 工具调用（非流式或流式都会得到 arguments={"file_path": "a.py"}）
    fake = FakeProvider([FakeProvider.tool_call("Read", {"file_path": "a.py"})])

    # 3) 重试：先两次 429，再成功（异常按顺序消费）
    fake = FakeProvider([
        FakeProvider.raise_(LLMUnavailableError("429")),
        FakeProvider.raise_(LLMUnavailableError("429")),
        FakeProvider.text("ok"),
    ])

    # 4) 手工分片：验证「按 index 拼接 arguments」（B7）
    from my_agent.llm.provider import StreamChunk, ToolCallDelta
    fake = FakeProvider([
        FakeProvider.chunks([
            StreamChunk(tool_call_delta=ToolCallDelta(index=0, id="c1", name="Read",
                                                      arguments_delta='{"file_p')),
            StreamChunk(tool_call_delta=ToolCallDelta(index=0, arguments_delta='ath": "a.py"}')),
            StreamChunk(finish_reason="tool_calls"),
            StreamChunk(usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
        ]),
    ])

    # 5) 非法 JSON 工具参数（模型自纠场景）
    fake = FakeProvider([FakeProvider.tool_call("Read", "{not json")])

    # 6) 流中途超时（已聚合内容保留 + truncated=True）
    fake = FakeProvider([FakeProvider.raise_(asyncio.TimeoutError())])

请求记录（其它 worker 断言 ``call_count == 0`` 用）
---------------------------------------------------

每次 ``complete()`` / ``stream()`` 都会把请求原样记进 :attr:`FakeProvider.calls`
（:class:`RecordedCall`，可直接属性访问，也可像元组一样解包成
``(system, messages)``）：

.. code-block:: python

    assert fake.call_count == 0              # 未调用（例如权限流水线直接拒绝）
    assert fake.calls[-1].tools is None      # 本次没带 tools
    assert fake.calls[-1].timeout_s == 4.0   # 判定角色超时
    assert fake.texts() == ["hello"]         # 每次调用的 user 文本
    assert fake.last_call is not None and fake.last_call.stream is True
    assert fake.closed is True               # await provider.close() 之后

约定
----

* 脚本耗尽后再被调用 → 抛 :class:`AssertionError`（说明测试给的脚本不够），
  并把已消费的调用次数写进消息，避免静默拿到空回复。
* ``raise_(exc)`` 抛的是**同一个异常对象**，不是副本。
* ``complete()`` 对带 ``chunks`` 的条目会由分片**反推**结果（文本拼接、arguments
  按 index 拼接），因此同一份脚本既能喂流式也能喂非流式调用。
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from my_agent.llm.provider import CompletionResult, StreamChunk, ToolCallDelta
from my_agent.models.messages import Message, ToolCallRequest

__all__ = [
    "FakeProvider",
    "RecordedCall",
    "ScriptedResult",
]

#: 自动切帧时，工具参数被拆成的分片数（模拟真实端点把 ``arguments`` 拆着下发）。
_ARGUMENT_CHUNKS = 2


class RecordedCall(BaseModel):
    """一次被记录的 Provider 请求（供断言调用次数与参数）。

    Attributes:
        stream: 本次是流式（``stream()``）还是非流式（``complete()``）。
        system: system prompt。
        messages: 收到的消息（``Message`` 或已净化字典，原样保存）。
        tools: function schema 列表；未带工具时为 ``None``。
        max_tokens: 输出上限。
        temperature: 采样温度。
        timeout_s: 本次请求超时。
        extra: 其余关键字实参（透传项）。
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}

    stream: bool = False
    system: str = ""
    messages: list[Any] = Field(default_factory=list)
    tools: list[dict] | None = None
    max_tokens: int = 0
    temperature: float = 0.0
    timeout_s: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)

    def __iter__(self) -> Any:  # type: ignore[override]
        """支持 ``system, messages = fake.calls[0]`` 式解包。

        Yields:
            ``system`` 与 ``messages``。
        """
        return iter((self.system, self.messages))

    def message_texts(self) -> list[str]:
        """提取本次请求里所有消息的文本内容。

        Returns:
            文本列表（按消息顺序；``content`` 为空的条目产出空串）。
        """
        out: list[str] = []
        for msg in self.messages:
            if isinstance(msg, Message):
                out.append(msg.content)
            elif isinstance(msg, dict):
                out.append(str(msg.get("content") or ""))
            else:  # pragma: no cover - 防御性
                out.append(str(msg))
        return out


class ScriptedResult(BaseModel):
    """脚本里的一条结果：本轮「返回什么 / 抛什么」。

    它同时描述非流式结果与流式分片：``chunks`` 非空时流式按 ``chunks`` 逐帧下发，
    否则由 :meth:`iter_chunks` 自动切帧。
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}

    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = Field(default_factory=dict)
    model: str = "fake-model"
    latency_ms: int = 0
    error: BaseException | None = Field(default=None, description="非 None 则抛出该异常")
    chunks: list[StreamChunk] | None = Field(default=None, description="显式流式分片脚本")
    delay_s: float = Field(default=0.0, description="下发前额外延迟（模拟慢端点）")

    def completion_result(self) -> CompletionResult:
        """生成非流式结果。

        Returns:
            与本条目等价的 :class:`~my_agent.llm.provider.CompletionResult`；
            仅有 ``chunks`` 时由分片反推（文本拼接 + arguments 按 index 拼接）。
        """
        text, calls = self._content()
        return CompletionResult(
            text=text,
            tool_calls=calls,
            finish_reason=self.finish_reason,
            usage=dict(self.usage),
            model=self.model,
            latency_ms=self.latency_ms,
        )

    async def iter_chunks(self) -> AsyncIterator[StreamChunk]:
        """产出本条的流式分片。

        Yields:
            :class:`~my_agent.llm.provider.StreamChunk`：显式 ``chunks`` 原样透传；
            否则按「文本 → 工具调用分片 → finish_reason → usage」自动切帧。

        Raises:
            BaseException: ``error`` 非空时原样抛出（模拟建流/流中断失败）。
        """
        if self.error is not None:
            raise self.error
        if self.chunks is not None:
            for chunk in self.chunks:
                yield chunk
            return
        text, calls = self._content()
        if text:
            yield StreamChunk(text_delta=text)
        for index, call in enumerate(calls):
            yield StreamChunk(
                tool_call_delta=ToolCallDelta(index=index, id=call.id, name=call.name)
            )
            for piece in _split_arguments(call.raw_arguments or _dumps(call.arguments)):
                yield StreamChunk(
                    tool_call_delta=ToolCallDelta(index=index, arguments_delta=piece)
                )
        yield StreamChunk(finish_reason=self.finish_reason)
        if self.usage:
            yield StreamChunk(usage=dict(self.usage))

    def _content(self) -> tuple[str, list[ToolCallRequest]]:
        """取本条的（文本, 工具调用）。

        Returns:
            显式字段优先；仅有 ``chunks`` 时由分片反推。
        """
        if self.text or self.tool_calls or not self.chunks:
            return self.text, list(self.tool_calls)
        text_parts: list[str] = []
        aggregator: dict[int, dict[str, str]] = {}
        for chunk in self.chunks:
            if chunk.text_delta:
                text_parts.append(chunk.text_delta)
            delta = chunk.tool_call_delta
            if delta is None:
                continue
            slot = aggregator.setdefault(delta.index, {"id": "", "name": "", "arguments": ""})
            if delta.id:
                slot["id"] = delta.id
            if delta.name:
                slot["name"] = delta.name
            slot["arguments"] += delta.arguments_delta
        calls = [
            ToolCallRequest(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments=_safe_arguments(slot["arguments"]),
                raw_arguments=slot["arguments"],
            )
            for index, slot in sorted(aggregator.items())
        ]
        return "".join(text_parts), calls


class FakeProvider:
    """脚本化 ``ModelProvider`` 替身（§18.1；**不依赖 openai**）。

    见本模块 docstring 的「快速上手」——构造、便捷构造器与请求记录约定都在那里。
    """

    def __init__(
        self,
        script: Sequence[ScriptedResult | BaseException | str] | ScriptedResult | BaseException | str = (),
        *,
        name: str = "fake",
        sleep: float = 0.0,
    ) -> None:
        """装配脚本。

        Args:
            script: 脚本条目，按顺序消费。元素可以是 :class:`ScriptedResult`、
                异常实例（等价于 ``raise_``）、或字符串（等价于 ``text``）。
                也接受单个条目（自动包成一段脚本）。
            name: ``provider.name`` 的取值（默认 ``"fake"``）。
            sleep: 每帧/每次返回前 ``await asyncio.sleep(sleep)`` 的秒数
                （默认 0，即不await；用于模拟慢端点）。
        """
        self.name = name
        self.sleep = sleep
        self._script: deque[ScriptedResult] = deque(_normalize(script))
        self.calls: list[RecordedCall] = []
        self.closed: bool = False

    # ------------------------------------------------------------------ 便捷构造器
    @staticmethod
    def text(
        text: str = "",
        *,
        finish_reason: str = "stop",
        usage: dict[str, int] | None = None,
        model: str = "fake-model",
        delay_s: float = 0.0,
    ) -> ScriptedResult:
        """构造「一轮纯文本回复」。

        Args:
            text: 回复正文。
            finish_reason: 结束原因（``"length"`` 用于截断场景）。
            usage: usage 字典；``None``/空 = 端点未返回 usage（上层走估算兜底）。
            model: 结果里的模型名。
            delay_s: 下发前的额外延迟秒数。

        Returns:
            可直接放进脚本的 :class:`ScriptedResult`。
        """
        return ScriptedResult(
            text=text,
            finish_reason=finish_reason,
            usage=dict(usage or {}),
            model=model,
            delay_s=delay_s,
        )

    @staticmethod
    def tool_call(
        name: str,
        arguments: dict[str, Any] | str | None = None,
        *,
        call_id: str = "call_1",
        text: str = "",
        finish_reason: str = "tool_calls",
        usage: dict[str, int] | None = None,
    ) -> ScriptedResult:
        """构造「一轮工具调用」。

        Args:
            name: 工具名。
            arguments: 参数字典，或**原始 JSON 字符串**（传字符串可构造非法 JSON /
                分片拼接场景；非流式与流式都会按原串下发）。
            call_id: 调用 id（回填 ``tool_call_id`` 用）。
            text: 同轮附带的文本（可为空）。
            finish_reason: 结束原因。
            usage: usage 字典。

        Returns:
            可直接放进脚本的 :class:`ScriptedResult`。
        """
        raw = arguments if isinstance(arguments, str) else _dumps(arguments or {})
        parsed = arguments if isinstance(arguments, dict) else _safe_arguments(raw)
        return ScriptedResult(
            text=text,
            tool_calls=[
                ToolCallRequest(id=call_id, name=name, arguments=parsed, raw_arguments=raw)
            ],
            finish_reason=finish_reason,
            usage=dict(usage or {}),
        )

    @staticmethod
    def raise_(exc: BaseException) -> ScriptedResult:
        """构造「一抛」：下一次调用抛出 ``exc``（同一个对象）。

        Args:
            exc: 要抛出的异常实例。

        Returns:
            可直接放进脚本的 :class:`ScriptedResult`。

        Raises:
            TypeError: ``exc`` 不是异常实例（避免把普通值当异常塞进脚本）。
        """
        if not isinstance(exc, BaseException):
            raise TypeError(f"raise_ 需要异常实例，收到 {type(exc).__name__}")
        return ScriptedResult(error=exc)

    @staticmethod
    def chunks(chunks: Iterable[StreamChunk], *, finish_reason: str = "stop") -> ScriptedResult:
        """构造「一段显式流式分片」。

        Args:
            chunks: :class:`~my_agent.llm.provider.StreamChunk` 序列（原样下发）。
            finish_reason: 非流式反推时使用的结束原因。

        Returns:
            可直接放进脚本的 :class:`ScriptedResult`。
        """
        return ScriptedResult(chunks=list(chunks), finish_reason=finish_reason)

    @staticmethod
    def scripted(items: Iterable[ScriptedResult | BaseException | str]) -> list[ScriptedResult]:
        """把一串条目规范化为 ``list[ScriptedResult]``。

        Args:
            items: :class:`ScriptedResult` / 异常实例 / 字符串混合序列。

        Returns:
            规范化后的脚本列表（可直接传给 ``FakeProvider(...)``）。
        """
        return _normalize(items)

    # ------------------------------------------------------------------ 断言辅助
    @property
    def call_count(self) -> int:
        """已发生的调用次数（其它 worker 用它断言 ``call_count == 0``）。"""
        return len(self.calls)

    @property
    def last_call(self) -> RecordedCall | None:
        """最近一次请求记录；未调用过时为 ``None``。"""
        return self.calls[-1] if self.calls else None

    @property
    def remaining(self) -> int:
        """脚本里尚未消费的条目数。"""
        return len(self._script)

    def texts(self) -> list[str]:
        """每次调用的消息文本（每项为该次请求所有消息 ``content`` 的拼接）。

        Returns:
            与 :attr:`calls` 等长的字符串列表。
        """
        return ["".join(call.message_texts()) for call in self.calls]

    # ------------------------------------------------------------------ ModelProvider
    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int,
        temperature: float,
        timeout_s: float = 60.0,
        **extra: Any,
    ) -> CompletionResult:
        """消费一条脚本并返回非流式结果。

        Args:
            system: system prompt。
            messages: 消息列表。
            tools: function schema 列表。
            max_tokens: 输出上限。
            temperature: 采样温度。
            timeout_s: 本次请求超时。
            **extra: 其余关键字实参（原样记录，便于断言透传项）。

        Returns:
            :class:`~my_agent.llm.provider.CompletionResult`。

        Raises:
            BaseException: 该条脚本是 ``raise_(...)`` 时原样抛出（同一对象）。
            AssertionError: 脚本已耗尽。
        """
        item = self._consume("complete", system, messages, tools, max_tokens, temperature, timeout_s, extra)
        await self._tick(item.delay_s)
        if item.error is not None:
            raise item.error
        return item.completion_result()

    async def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int,
        temperature: float,
        timeout_s: float = 60.0,
        **extra: Any,
    ) -> AsyncIterator[StreamChunk]:
        """消费一条脚本并按帧下发。

        Args:
            system: system prompt。
            messages: 消息列表。
            tools: function schema 列表。
            max_tokens: 输出上限。
            temperature: 采样温度。
            timeout_s: 本次请求超时。
            **extra: 其余关键字实参（原样记录）。

        Yields:
            :class:`~my_agent.llm.provider.StreamChunk` 帧。

        Raises:
            BaseException: 该条脚本是 ``raise_(...)`` 时**在首帧前**原样抛出。
            AssertionError: 脚本已耗尽。
        """
        item = self._consume("stream", system, messages, tools, max_tokens, temperature, timeout_s, extra)
        await self._tick(item.delay_s)
        async for chunk in item.iter_chunks():
            await self._tick(self.sleep)
            yield chunk

    async def close(self) -> None:
        """标记自身已关闭（``LLMClient.aclose()`` → ``provider.close()``，I-10）。"""
        self.closed = True

    # ------------------------------------------------------------------ 内部
    def _consume(
        self,
        mode: str,
        system: str,
        messages: list[Message],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
        extra: dict[str, Any],
    ) -> ScriptedResult:
        """记录一次请求并取出下一条脚本。

        Args:
            mode: ``"complete"`` 或 ``"stream"``。
            system: system prompt。
            messages: 消息列表。
            tools: function schema 列表。
            max_tokens: 输出上限。
            temperature: 采样温度。
            timeout_s: 本次请求超时。
            extra: 其余关键字实参。

        Returns:
            下一条 :class:`ScriptedResult`。

        Raises:
            AssertionError: 脚本已耗尽（测试给的脚本不够）。
        """
        self.calls.append(
            RecordedCall(
                stream=mode == "stream",
                system=system,
                messages=list(messages),
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
                extra=dict(extra),
            )
        )
        if not self._script:
            raise AssertionError(
                f"FakeProvider 脚本已耗尽：已消费 {len(self.calls)} 次调用（{mode}），"
                "请给 FakeProvider 追加更多条目"
            )
        return self._script.popleft()

    async def _tick(self, delay_s: float) -> None:
        """按需 await 一小段延迟（``delay_s`` 为 0 时完全不 await）。

        Args:
            delay_s: 延迟秒数。
        """
        if (delay_s or self.sleep) > 0:
            await asyncio.sleep(delay_s or self.sleep)


def _normalize(
    script: Sequence[ScriptedResult | BaseException | str] | ScriptedResult | BaseException | str,
) -> list[ScriptedResult]:
    """把用户给的脚本规范化为 ``list[ScriptedResult]``。

    Args:
        script: 单个条目或条目序列；元素可为 :class:`ScriptedResult`、异常实例或字符串。

    Returns:
        规范化后的列表。

    Raises:
        TypeError: 出现不支持的条目类型（早失败，避免测试里出现难查的静默错误）。
    """
    items: Iterable[Any]
    if isinstance(script, (ScriptedResult, BaseException, str)):
        items = (script,)
    elif isinstance(script, Sequence):
        items = script
    else:
        raise TypeError(f"FakeProvider 脚本必须是序列或单个条目，收到 {type(script).__name__}")
    out: list[ScriptedResult] = []
    for item in items:
        if isinstance(item, ScriptedResult):
            out.append(item)
        elif isinstance(item, BaseException):
            out.append(FakeProvider.raise_(item))
        elif isinstance(item, str):
            out.append(FakeProvider.text(item))
        else:
            raise TypeError(f"不支持的脚本条目类型：{type(item).__name__}")
    return out


def _split_arguments(raw: str, pieces: int = _ARGUMENT_CHUNKS) -> list[str]:
    """把参数 JSON 串拆成若干分片（模拟真实端点的分片下发）。

    Args:
        raw: 完整参数 JSON 串。
        pieces: 目标分片数；串太短时自动减少。

    Returns:
        分片列表；``raw`` 为空时返回空列表。
    """
    if not raw:
        return []
    if len(raw) <= pieces:
        return [raw[index : index + 1] for index in range(len(raw))]
    size = len(raw) // pieces or 1
    chunks = [raw[index : index + size] for index in range(0, len(raw), size)]
    return chunks


def _dumps(value: Any) -> str:
    """把参数序列化为紧凑 JSON 文本。

    Args:
        value: 待序列化对象。

    Returns:
        JSON 字符串；不可序列化时退化为 ``str(value)``。
    """
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _safe_arguments(raw: str) -> dict[str, Any]:
    """尽力解析参数 JSON。

    Args:
        raw: 原始 JSON 串。

    Returns:
        解析后的字典；非法 JSON 或非对象时返回 ``{}``（``raw_arguments`` 已保留原文）。
    """
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}
