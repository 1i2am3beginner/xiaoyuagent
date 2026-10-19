"""内部 Provider 协议与 OpenAI SDK 实现（§13.1.3–§13.1.5）。

分层（§13.1.1）::

    业务代码 → LLMClient（角色提示词/重试/超时/降级/用量/流式聚合）
             → OpenAISDKClient（本模块：仅做参数映射 + 分片 → 内部模型 + 异常翻译）
             → openai.AsyncOpenAI（官方 SDK）

**C7 依赖隔离**：本文件是 ``src/`` 下唯一允许 ``import openai`` 的地方（I-9 用 ``ast``
扫描断言）。除本文件外，任何模块都不得接触 SDK 类型，因此换实现（如将来的原生
Anthropic 端点）只需在本文件新增一个 :class:`ModelProvider` 实现。

**C9**：不手写 HTTP / SSE / 退避重试——那三件事分别交给 SDK 与 ``LLMClient``。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# C7：唯一允许出现 openai 的位置。
import openai
from pydantic import BaseModel, Field

from my_agent.errors import (
    ConfigError,
    LLMRequestError,
    LLMUnavailableError,
    ModelUnavailableError,
    MyAgentError,
)
from my_agent.models.messages import ToolCallRequest

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查器
    from my_agent.config.schema import EndpointConfig
    from my_agent.models.messages import Message

__all__ = [
    "API_KEY_MASK",
    "CompletionResult",
    "ModelProvider",
    "OpenAISDKClient",
    "SDK_ERROR_MAP",
    "StreamChunk",
    "ToolCallDelta",
    "map_sdk_error",
    "read_api_key",
]

_LOG = logging.getLogger(__name__)

#: 日志中替代密钥材料的固定占位串（**永不**记录密钥本体，I-5）。
API_KEY_MASK = "***"


# --------------------------------------------------------------------------------------
# 内部模型（§13.1.3【补齐】/ §13.3）
# --------------------------------------------------------------------------------------


class ToolCallDelta(BaseModel):
    """流式响应中一个 ``delta.tool_calls[]`` 分片。

    SDK 把同一次工具调用拆成多帧下发：通常第一帧带 ``id`` 与 ``function.name``，
    后续帧只带 ``function.arguments`` 的**字符串片段**（必须按下标拼接，见 B7）。
    """

    index: int = Field(description="同一响应内工具调用的下标（聚合键）")
    id: str | None = Field(default=None, description="调用 id，通常只在首帧出现")
    name: str | None = Field(default=None, description="函数名，通常只在首帧出现")
    arguments_delta: str = Field(default="", description="本次分片携带的参数 JSON 片段")

    model_config = {"extra": "ignore"}


class StreamChunk(BaseModel):
    """流式响应的一个内部帧（已脱离 SDK 类型）。"""

    text_delta: str = Field(default="", description="文本增量")
    tool_call_delta: ToolCallDelta | None = Field(default=None, description="工具调用分片")
    finish_reason: str | None = Field(default=None, description="仅末帧非空")
    usage: dict[str, int] | None = Field(
        default=None,
        description="末帧 usage（``stream_options.include_usage=true``）；缺失则上层估算",
    )

    model_config = {"extra": "ignore"}


class CompletionResult(BaseModel):
    """一次**非流式**完成的内部结果（§13.3）。"""

    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str = Field(default="", description="stop | tool_calls | length | content_filter")
    usage: dict[str, int] = Field(default_factory=dict)
    model: str = ""
    latency_ms: int = 0

    model_config = {"extra": "ignore"}


@runtime_checkable
class ModelProvider(Protocol):
    """内部模型提供方协议（§13.1.3）。

    **只被 ``LLMClient`` 调用**；实现类不得把 SDK 类型泄漏到上层。流式为**异步生成器**
    （``async def`` + ``yield``），调用方直接 ``async for`` 消费。
    """

    name: str

    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int,
        temperature: float,
        timeout_s: float = 60.0,
    ) -> CompletionResult:
        """非流式完成一次对话。

        Args:
            system: system prompt（由角色决定，§13.2）。
            messages: 已净化的对话消息。
            tools: OpenAI 兼容的 function schema 列表；``None`` = 不用工具。
            max_tokens: 输出上限。
            temperature: 采样温度。
            timeout_s: 本次请求的超时（按请求传，不设 client 级默认）。

        Returns:
            内部 :class:`CompletionResult`。

        Raises:
            MyAgentError: SDK 异常已翻译为内部异常（见 :func:`map_sdk_error`）。
        """
        ...

    def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int,
        temperature: float,
        timeout_s: float = 60.0,
    ) -> AsyncIterator[StreamChunk]:
        """流式完成一次对话（异步生成器）。

        Args:
            system: system prompt。
            messages: 已净化的对话消息。
            tools: function schema 列表。
            max_tokens: 输出上限。
            temperature: 采样温度。
            timeout_s: 本次请求的超时。

        Yields:
            :class:`StreamChunk` 帧序列。

        Raises:
            MyAgentError: SDK 异常已翻译为内部异常。
        """
        ...

    async def close(self) -> None:
        """释放底层 HTTP 连接池。

        Raises:
            MyAgentError: 关闭失败。
        """
        ...


# --------------------------------------------------------------------------------------
# SDK 异常 → 内部异常映射表（§13.1.5，必须逐条实现）
# --------------------------------------------------------------------------------------

#: §13.1.5 映射表，**顺序敏感**：``map_sdk_error`` 自上而下用 ``isinstance`` 匹配，
#: 因此「子类必须排在父类之前」——这正是规格强调的易错点：
#:
#:   * ``APITimeoutError`` 是 ``APIConnectionError`` 的子类 → 必须先判超时；
#:   * ``RateLimitError`` / ``InternalServerError`` / ``BadRequestError`` 等
#:     都是 ``APIStatusError`` 的子类 → 必须先判具体状态码类；
#:   * ``APIStatusError`` 本身排在最后做「其它状态码」兜底。
SDK_ERROR_MAP: tuple[tuple[type[BaseException], type[MyAgentError]], ...] = (
    (openai.APITimeoutError, ModelUnavailableError),  # 判定角色 → §6.5.4 降级
    (openai.APIConnectionError, LLMUnavailableError),
    (openai.RateLimitError, LLMUnavailableError),  # 429
    (openai.InternalServerError, LLMUnavailableError),  # 5xx
    (openai.AuthenticationError, ConfigError),  # 401/403 → 提示密钥无效
    (openai.PermissionDeniedError, ConfigError),
    (openai.BadRequestError, LLMRequestError),  # 400：参数/schema 非法，**不重试**
    (openai.NotFoundError, ConfigError),  # 模型名或 base_url 错
    (openai.APIStatusError, LLMUnavailableError),  # 其它状态码兜底
)


def map_sdk_error(exc: BaseException) -> MyAgentError:
    """把 SDK 异常翻译为内部异常（§13.1.5）。

    已映射顺序遍历 :data:`SDK_ERROR_MAP`：**子类在前**，故
    ``APITimeoutError`` 先于 ``APIConnectionError`` 命中，
    ``RateLimitError`` / ``InternalServerError`` 先于 ``APIStatusError`` 命中。

    Args:
        exc: 任意异常对象。

    Returns:
        对应的内部异常实例；``exc`` 本身已是 :class:`~my_agent.errors.MyAgentError`
        时原样返回。非 SDK 的未知异常收敛为 :class:`LLMUnavailableError`
        （内部异常体系内没有更贴切的兜底类，且该错误可重试）。

    Raises:
        TypeError: ``exc`` 不是异常实例（编程错误，直接暴露）。
    """
    if isinstance(exc, MyAgentError):
        return exc
    if not isinstance(exc, BaseException):  # pragma: no cover - 防御性
        raise TypeError(f"map_sdk_error 需要异常实例，收到 {type(exc).__name__}")
    detail = _describe(exc)
    for sdk_cls, internal_cls in SDK_ERROR_MAP:
        if isinstance(exc, sdk_cls):
            return internal_cls(detail, detail=detail, sdk_error=type(exc).__name__)
    return LLMUnavailableError(
        f"未预期的 SDK 错误 {type(exc).__name__}: {detail}",
        detail=f"未预期的 SDK 错误 {type(exc).__name__}: {detail}",
        sdk_error=type(exc).__name__,
    )


def _describe(exc: BaseException) -> str:
    """生成**不含密钥材料**的异常摘要。

    SDK 的 ``APIStatusError.__str__`` 只含状态码 / 请求 id / 响应体片段，不含
    ``Authorization`` 头；此处再做一次防御性处理后截断。

    Args:
        exc: 异常对象。

    Returns:
        单行摘要（最长 500 字符）。
    """
    text = " ".join(str(exc).split()) or type(exc).__name__
    status = getattr(exc, "status_code", None)
    if status is not None and str(status) not in text:
        text = f"HTTP {status}: {text}"
    return text[:500]


# --------------------------------------------------------------------------------------
# API Key 读取（I-4：绝不记录密钥本体）
# --------------------------------------------------------------------------------------


def read_api_key(api_key_env: str, env: Mapping[str, str] | None = None) -> str:
    """从环境变量读取 API Key。

    Args:
        api_key_env: 环境变量名（``EndpointConfig.api_key_env``；**只存名字不存密钥**）。
        env: 环境变量映射；``None`` = ``os.environ``。

    Returns:
        密钥值。

    Raises:
        ConfigError: 环境变量未设置或为空（消息含环境变量名与配置指引，I-4）。
    """
    source = os.environ if env is None else env
    name = api_key_env or "<未配置 api_key_env>"
    value = (source.get(api_key_env, "") if api_key_env else "").strip()
    if not value:
        hint = (
            f"环境变量 {name} 未设置，无法初始化模型端点。请在 ~/.myagent/config.toml 的 "
            f"[llm.endpoints.<name>] 中配置 api_key_env，并导出该环境变量（本地端点可填占位值）。"
        )
        raise ConfigError(hint, detail=hint, api_key_env=name)
    return value


# --------------------------------------------------------------------------------------
# OpenAI SDK 实现
# --------------------------------------------------------------------------------------


class OpenAISDKClient:
    """``ModelProvider`` 的 OpenAI SDK 实现（§13.1.3）。

    **thin wrapper**：只做「参数映射 + 分片 → 内部模型 + 异常翻译」，不做重试
    （``max_retries=0``，重试统一在 ``LLMClient``，便于审计与计数）也不写字面 HTTP。

    SDK client 与 API Key 均**惰性初始化**：``QueryEngine.start()`` 会先构造
    Provider，再按需注入 ``provider.api_key``；密钥缺失在首次调用时以
    :class:`ConfigError` 报出（I-4），消息含环境变量名。
    """

    def __init__(
        self,
        cfg: EndpointConfig,
        *,
        http_client: Any | None = None,
        api_key: str = "",
    ) -> None:
        """初始化 Provider（不建连接）。

        Args:
            cfg: 端点配置（§13.1.3）。
            http_client: 仅供测试注入（如 ``httpx.MockTransport`` 包装的 AsyncClient）；
                生产环境传 ``None``，由 SDK 自建。
            api_key: 直接注入的密钥（测试或上层已从环境变量读出时使用）；
                留空则回退到 ``cfg.api_key_env`` 指向的环境变量。
        """
        self._cfg = cfg
        self._http_client = http_client
        self.api_key = api_key
        self._client: Any | None = None
        self._client_key = ""
        self._client_headers: dict[str, str] = {}
        self._closed = False

    # ---------------------------------------------------------------- 元信息
    @property
    def name(self) -> str:
        """Provider 名（``openai-sdk:<model>``，供日志与审计展示）。"""
        return f"openai-sdk:{self._cfg.model}"

    @property
    def model(self) -> str:
        """当前使用的模型名。"""
        return self._cfg.model

    @property
    def client(self) -> Any:
        """底层 ``AsyncOpenAI`` 实例（仅供同文件内使用与测试断言）。

        Returns:
            SDK 客户端；尚未初始化时先按需创建。

        Raises:
            ConfigError: API Key 缺失。
        """
        return self._ensure_client()

    # ---------------------------------------------------------------- SDK 客户端
    def _ensure_client(self) -> Any:
        """按需构造 ``AsyncOpenAI``（密钥或 headers 变化时重建）。

        Returns:
            可用的 SDK 客户端。

        Raises:
            ConfigError: API Key 缺失（消息含环境变量名与配置指引）。
        """
        key = self.api_key.strip() or read_api_key(self._cfg.api_key_env)
        headers = dict(self._cfg.extra_headers)
        if self._client is not None and (key != self._client_key or headers != self._client_headers):
            self._client = None  # 密钥/请求头变更 → 重建（P2：不落盘、不入日志）
        if self._client is None:
            kwargs: dict[str, Any] = {
                "api_key": key,
                "base_url": self._cfg.base_url,
                "max_retries": 0,  # 重试统一在 LLMClient（可审计、可计数）
                "timeout": None,  # 超时按请求传（角色差异大，见 §13.3）
            }
            if self._cfg.extra_headers:
                kwargs["default_headers"] = dict(self._cfg.extra_headers)
            if self._http_client is not None:
                kwargs["http_client"] = self._http_client
            self._client = openai.AsyncOpenAI(**kwargs)
            self._client_key = key
            self._client_headers = headers
        return self._client

    # ---------------------------------------------------------------- 参数映射
    def _call_kwargs(
        self,
        *,
        system: str,
        messages: Sequence[Message | Mapping[str, Any]],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
    ) -> dict[str, Any]:
        """组装 ``chat.completions.create`` 的实参（§13.1.4 逐条映射）。

        Args:
            system: system prompt。
            messages: 已净化的消息（``Message`` 或已是 SDK 字典）。
            tools: function schema 列表。
            max_tokens: 输出上限。
            temperature: 采样温度。
            timeout_s: 本次请求超时。

        Returns:
            传给 SDK 的关键字实参。
        """
        return {
            "model": self._cfg.model,
            "messages": [{"role": "system", "content": system}, *to_sdk_messages(messages)],
            # C9：``tools=None`` 时用 SDK 哨兵而非 ``None``（``None`` 语义为「显式禁用」）。
            "tools": tools if tools else openai.NOT_GIVEN,
            # 规格用 max_tokens=；新 SDK 另有 max_completion_tokens（推理模型要求它），
            # 为保证任意 OpenAI 兼容端点都能用，此处统一传 max_tokens。
            "max_tokens": max_tokens,
            "temperature": temperature,
            # 超时按请求传（不设 client 级默认，避免主对话/判定互相干扰）。
            "timeout": timeout_s,
        }

    # ---------------------------------------------------------------- 非流式
    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int,
        temperature: float,
        timeout_s: float = 60.0,
    ) -> CompletionResult:
        """非流式完成一次对话（§13.1.4「非流式」）。

        Args:
            system: system prompt。
            messages: 已净化的对话消息。
            tools: function schema 列表；``None`` = 不带工具。
            max_tokens: 输出上限。
            temperature: 采样温度。
            timeout_s: 本次请求超时秒数。

        Returns:
            内部 :class:`CompletionResult`（含 latency_ms 与 usage）。

        Raises:
            MyAgentError: SDK 异常经 :func:`map_sdk_error` 翻译后的内部异常。
        """
        kwargs = self._call_kwargs(
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        started = time.monotonic()
        try:
            # 双保险：SDK 的 timeout= 在部分版本/自定义 transport 下**不被强制执行**
            # （实测：timeout=0.05 的请求仍会等待 1s+ 并成功返回）。判定角色的 4s
            # 超时是 §6.5.3 的降级触发条件，绝不能依赖 SDK 行为，因此这里自己守挂钟。
            # 用 asyncio.timeout() 而非 wait_for：后者会等待被取消的内层任务真正结束，
            # 而 httpx 在部分路径上会吞掉取消，导致「超时」实际不发生。
            async with asyncio.timeout(timeout_s):
                resp = await self._ensure_client().chat.completions.create(**kwargs)
        except TimeoutError as exc:
            raise ModelUnavailableError(
                f"请求超时（{timeout_s}s）", detail=f"timeout={timeout_s}s"
            ) from exc
        except Exception as exc:  # SDK 异常 → 内部异常（§13.1.5）；不记录 exc_info
            raise map_sdk_error(exc) from exc
        return _result_from_response(resp, started)

    # ---------------------------------------------------------------- 流式
    async def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int,
        temperature: float,
        timeout_s: float = 60.0,
    ) -> AsyncIterator[StreamChunk]:
        """流式完成一次对话（§13.1.4「流式」）。

        先 ``await`` 建流（HTTP 错误在此抛出），再 ``async for`` 消费 SDK 已解析好的 SSE；
        本方法只把 SDK chunk 映射成 :class:`StreamChunk`，聚合与容错在上层 ``LLMClient``。

        Args:
            system: system prompt。
            messages: 已净化的对话消息。
            tools: function schema 列表。
            max_tokens: 输出上限。
            temperature: 采样温度。
            timeout_s: 本次请求超时秒数。

        Yields:
            :class:`StreamChunk`：文本增量 / 工具调用分片 / 末帧 finish_reason 与 usage。

        Raises:
            MyAgentError: 建流或消费过程中的 SDK 异常（已翻译为内部异常）。
        """
        kwargs = self._call_kwargs(
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        kwargs["stream"] = True
        # 末帧带 usage（部分兼容端点会忽略 → 上层走 tokenizer 估算兜底，§13.3）。
        kwargs["stream_options"] = {"include_usage": True}
        try:
            stream = await self._ensure_client().chat.completions.create(**kwargs)
        except Exception as exc:
            raise map_sdk_error(exc) from exc
        try:
            async for raw in stream:
                chunk = _chunk_from_sdk(raw)
                if chunk is not None:
                    yield chunk
        except Exception as exc:
            # 连接在流中途断开：已 yield 的分片由上层保留并标记 truncated=True（§13.3）。
            raise map_sdk_error(exc) from exc

    # ---------------------------------------------------------------- 生命周期
    async def close(self) -> None:
        """关闭底层 SDK client（幂等）。

        Raises:
            MyAgentError: 关闭过程抛出 SDK 异常。
        """
        client, self._client = self._client, None
        self._closed = True
        if client is None:
            return
        try:
            await client.close()
        except Exception as exc:  # SDK 关闭失败同样要翻译为内部异常
            raise map_sdk_error(exc) from exc


# --------------------------------------------------------------------------------------
# SDK 结构 → 内部模型
# --------------------------------------------------------------------------------------


def to_sdk_messages(messages: Sequence[Message | Mapping[str, Any]]) -> list[dict[str, Any]]:
    """把内部消息转成 SDK 的 ``{"role","content","tool_calls","tool_call_id"}`` 结构。

    已是 ``dict`` 的条目（``LLMClient.sanitize`` 的产物）原样透传；``Message`` 对象
    走 ``Message.to_provider_dict()``（内部元数据字段在那里被剥离）。

    Args:
        messages: 内部 ``Message`` 或已净化的字典。

    Returns:
        SDK 消息列表。
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, Mapping):
            out.append(dict(msg))
        else:  # Message
            out.append(msg.to_provider_dict())
    return out


def to_sdk_tools(tools: Iterable[Mapping[str, Any]] | None) -> Any:
    """把内部工具 schema 列表转成 SDK 的 ``tools=`` 实参。

    Args:
        tools: ``{"type": "function", "function": {...}}`` 列表。

    Returns:
        列表本身；``None`` / 空列表 → SDK 哨兵 ``NOT_GIVEN``。
    """
    items = [dict(item) for item in tools or ()]
    return items if items else openai.NOT_GIVEN


def usage_to_dict(usage: Any | None) -> dict[str, int]:
    """把 SDK ``CompletionUsage`` 转成内部 usage 字典。

    Args:
        usage: SDK usage 对象或 ``None``。

    Returns:
        ``{"prompt_tokens","completion_tokens","total","total_tokens"}``；缺失字段省略。
    """
    if usage is None:
        return {}
    out: dict[str, int] = {}
    for src, dst in (
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = getattr(usage, src, None)
        if isinstance(value, int):
            out[dst] = value
    if "total_tokens" in out:
        out["total"] = out["total_tokens"]
    elif "prompt_tokens" in out or "completion_tokens" in out:
        out["total"] = out.get("prompt_tokens", 0) + out.get("completion_tokens", 0)
    return out


def _result_from_response(resp: Any, started: float) -> CompletionResult:
    """把 SDK 非流式响应映射为 :class:`CompletionResult`。

    Args:
        resp: ``ChatCompletion``。
        started: 请求发起时刻（``time.monotonic()``）。

    Returns:
        内部结果对象。
    """
    latency_ms = int((time.monotonic() - started) * 1000)
    choices = getattr(resp, "choices", None) or []
    if not choices:  # pragma: no cover - 兼容端点的异常形状
        return CompletionResult(
            model=str(getattr(resp, "model", "") or ""),
            usage=usage_to_dict(getattr(resp, "usage", None)),
            latency_ms=latency_ms,
        )
    choice = choices[0]
    message = getattr(choice, "message", None)
    finish = getattr(choice, "finish_reason", None)
    return CompletionResult(
        text=_text_of(getattr(message, "content", None)),
        tool_calls=_tool_calls_of(message),
        finish_reason=str(finish or ""),
        usage=usage_to_dict(getattr(resp, "usage", None)),
        model=str(getattr(resp, "model", "") or ""),
        latency_ms=latency_ms,
    )


def _tool_calls_of(message: Any) -> list[ToolCallRequest]:
    """把 SDK 消息里的 ``tool_calls`` 映射为内部模型。

    Args:
        message: SDK ``ChatCompletionMessage``（可能为 ``None``）。

    Returns:
        内部 :class:`~my_agent.models.messages.ToolCallRequest` 列表。
    """
    out: list[ToolCallRequest] = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        raw = str(getattr(function, "arguments", "") or "")
        out.append(
            ToolCallRequest(
                id=str(getattr(call, "id", "") or ""),
                name=str(getattr(function, "name", "") or ""),
                arguments=_arguments_of(raw),
                raw_arguments=raw,
            )
        )
    return out


def _arguments_of(raw: str) -> dict[str, Any]:
    """解析工具参数 JSON（失败不抛，留给上层记 ``ToolCallParseError``）。

    Args:
        raw: 模型给出的原始参数串。

    Returns:
        解析后的字典；空串 → ``{}``；非法 JSON → ``{}``（``raw_arguments`` 已保留原文）。
    """
    import json

    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _chunk_from_sdk(raw: Any) -> StreamChunk | None:
    """把 SDK 流式帧映射为 :class:`StreamChunk`。

    Args:
        raw: ``ChatCompletionChunk``。

    Returns:
        内部帧；该帧不含任何可用增量时返回 ``None``（上层跳过）。
    """
    usage = usage_to_dict(getattr(raw, "usage", None)) or None
    choices = getattr(raw, "choices", None) or []
    if not choices:
        # 末帧：choices 为空、只带 usage（include_usage=true 的契约）。
        return StreamChunk(usage=usage) if usage else None
    choice = choices[0]
    delta = getattr(choice, "delta", None)
    text = _text_of(getattr(delta, "content", None))
    finish = getattr(choice, "finish_reason", None)
    deltas = _tool_call_deltas(delta)
    if not text and not deltas and finish is None and not usage:
        return None
    return StreamChunk(
        text_delta=text,
        tool_call_delta=deltas[0] if deltas else None,
        finish_reason=str(finish) if finish else None,
        usage=usage,
    )


def _tool_call_deltas(delta: Any) -> list[ToolCallDelta]:
    """提取一帧里的全部 ``delta.tool_calls[]`` 分片。

    参数增量必须**按 index 拼接**（B7），故每个分片独立成帧，不做合并。

    Args:
        delta: SDK 的 ``ChoiceDelta``（可能为 ``None``）。

    Returns:
        :class:`ToolCallDelta` 列表（可能为空）。
    """
    out: list[ToolCallDelta] = []
    for call in getattr(delta, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        index = getattr(call, "index", 0)
        out.append(
            ToolCallDelta(
                index=int(index or 0),
                id=getattr(call, "id", None),
                name=getattr(function, "name", None),
                arguments_delta=str(getattr(function, "arguments", "") or ""),
            )
        )
    return out


def _text_of(content: Any) -> str:
    """把 SDK 的 ``content`` 归一为字符串。

    Args:
        content: ``str`` / ``None`` / 多模态 part 列表。

    Returns:
        文本内容。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content)
