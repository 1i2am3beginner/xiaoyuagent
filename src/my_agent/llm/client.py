"""``LLMClient``：业务代码唯一的模型入口（§13.3）。

职责（C8：对外签名冻结，§5–§12 全部模块只依赖本类）：

* **角色提示词**：``role`` → system prompt（§13.2）与默认温度 / max_tokens；
* **重试**：网络错误 / 5xx / 429 → 退避 ``LLM_BACKOFF_S``，最多 ``LLM_MAX_RETRIES`` 次；
  4xx（非 429）**不重试**（SDK 侧 ``max_retries=0``，重试计数集中在本类，便于审计）；
* **超时**：按角色选 ``TIMEOUT_MAIN_S`` / ``JUDGE_TIMEOUT_S`` / ``TIMEOUT_SMALL_S``，
  一律**按请求传** ``timeout=``；
* **流式聚合**：``delta.tool_calls[]`` 按 ``index`` **拼接** ``arguments``（B7），
  再走 ``llm/tool_args.py`` 的容错解析链（剥 markdown → 提平衡 ``{}`` → ``json.loads``
  → 修复 → schema 校验）；修复成功即下发，彻底失败才保留 ``raw_arguments`` 并记
  ``ToolCallParseError``（不抛，留给模型自纠）；
* **截断处理**：``finish_reason == "length"`` → 丢弃部分工具调用并提示「输出过长，请分批」，
  纯文本保留并标记 ``truncated=True``（I-6）；
* **用量统计**：末帧 usage → :class:`UsageTracker`；缺失则用 ``llm/tokenizer.py`` 估算
  并记 ``usage_estimated=true``（I-7）；
* **日志**：摘要写 ``<myagent_home>/logs/llm-YYYY-MM-DD.log``，全文仅 DEBUG（I-5：不含密钥）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from my_agent.config import defaults as D
from my_agent.config.loader import myagent_home
from my_agent.errors import LLMUnavailableError, MyAgentError, ToolCallParseError
from my_agent.llm.prompts import ROLE_DEFAULTS, ROLE_PROMPTS
from my_agent.llm.provider import CompletionResult, StreamChunk
from my_agent.llm.tool_args import INVALID_ARGS_MESSAGE, ArgParseStatus, parse_tool_arguments
from my_agent.models.messages import INTERNAL_FIELDS, Message, ToolCallRequest
from my_agent.models.protocols import DisplayPort, NullDisplay

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查器
    from my_agent.config.schema import Config
    from my_agent.llm.provider import ModelProvider

__all__ = [
    "TRUNCATED_NOTICE",
    "AssistantTurn",
    "LLMClient",
    "UsageTracker",
    "role_timeout",
    "sanitize",
]

_LOG = logging.getLogger("my_agent.llm")

#: 截断（``finish_reason="length"``）时回填给模型的提示（§13.3 逐字）。
TRUNCATED_NOTICE = "输出过长，请分批"

#: 日志文件名前缀与后缀（``llm-YYYY-MM-DD.log``）。
_LOG_PREFIX = "llm-"
_LOG_SUFFIX = ".log"

#: 成本单价缺失时的说明（§13.4：代码内不得出现任何厂商名或模型名常量，
#: 也不存价格表，故 ``/cost`` 只给 token 口径）。
_PRICE_NOTE = "未配置单价（代码内不含厂商/价格常量）→ 本报告只给 token 口径"

#: 角色 → 超时秒数（§13.3：主对话 120s、判定角色 4s、其余小任务 30s）。
_ROLE_TIMEOUTS: dict[str, float] = {
    "main": D.TIMEOUT_MAIN_S,
    "subagent": D.TIMEOUT_MAIN_S,
    "judge": D.JUDGE_TIMEOUT_S,
}

#: §13.1.5 中「可重试」的异常族（网络 / 5xx / 429）。
_RETRYABLE_ERRORS: tuple[type[MyAgentError], ...] = (LLMUnavailableError,)


def role_timeout(role: str) -> float:
    """返回角色的默认请求超时（秒）。

    Args:
        role: 角色名（§13.2 的 10 个之一；未知角色按小任务处理）。

    Returns:
        ``main``/``subagent`` → :data:`~my_agent.config.defaults.TIMEOUT_MAIN_S`；
        ``judge`` → :data:`~my_agent.config.defaults.JUDGE_TIMEOUT_S`；
        其它小任务角色 → :data:`~my_agent.config.defaults.TIMEOUT_SMALL_S`。
    """
    return _ROLE_TIMEOUTS.get(role, D.TIMEOUT_SMALL_S)


# --------------------------------------------------------------------------------------
# 消息净化（§13.3「消息净化」）
# --------------------------------------------------------------------------------------


def sanitize(messages: Sequence[Message | Mapping[str, Any]]) -> list[dict[str, Any]]:
    """剥离内部元数据，生成发往 Provider 的消息结构。

    ``models/messages.py::INTERNAL_FIELDS`` 列出的字段（``id``/``tokens``/``kind``/
    ``archive_ref``/``turn_index``/``subagent_id`` …）绝不进入请求体。

    Args:
        messages: 内部 ``Message`` 或已是字典的消息。

    Returns:
        仅含 ``role`` / ``content`` / ``tool_calls`` / ``tool_call_id`` / ``name`` 的列表。
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, Message):
            out.append(msg.to_provider_dict())
        else:
            out.append({k: v for k, v in msg.items() if k not in INTERNAL_FIELDS})
    return out


# --------------------------------------------------------------------------------------
# 用量统计（§13.3「用量统计」）
# --------------------------------------------------------------------------------------


class UsageTracker:
    """进程内 token 用量累计器（``/cost`` 数据源，§13.3）。

    ``totals`` 的键为**冻结契约**的三元组 ``prompt`` / ``completion`` / ``total``
    （CONTRACTS §4.2）；:meth:`snapshot` 同时给出 ``*_tokens`` 别名，便于直接拼报告。
    """

    def __init__(self) -> None:
        """初始化空计数器。"""
        self.totals: dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}
        self.calls: int = 0
        self.retries: int = 0
        self.errors: int = 0
        self.estimated_usage_calls: int = 0

    def add(self, usage: Mapping[str, int], *, estimated: bool = False) -> None:
        """累加一次调用的用量。

        Args:
            usage: usage 字典；接受 ``prompt_tokens`` / ``completion_tokens`` /
                ``total`` / ``total_tokens`` 任一写法，缺失的字段按 0 计。
            estimated: 该次用量是否为估算值（端点未返回 usage）；估算值会同时计入
                ``estimated_*`` 口径并在 :meth:`cost_report` 中标注（I-7）。
        """
        prompt = _int_of(usage, "prompt_tokens", "prompt")
        completion = _int_of(usage, "completion_tokens", "completion")
        total = _int_of(usage, "total", "total_tokens") or (prompt + completion)
        self.totals["prompt"] += prompt
        self.totals["completion"] += completion
        self.totals["total"] += total
        self.calls += 1
        if estimated:
            self.estimated_usage_calls += 1
            self.totals.setdefault("estimated_prompt", 0)
            self.totals.setdefault("estimated_completion", 0)
            self.totals.setdefault("estimated_total", 0)
            self.totals["estimated_prompt"] += prompt
            self.totals["estimated_completion"] += completion
            self.totals["estimated_total"] += total

    def note_retry(self, count: int = 1) -> None:
        """记录重试次数（供审计，§13.3「重试计数走 LLMClient」）。

        Args:
            count: 本次新增的重试次数。
        """
        self.retries += max(0, count)

    def note_error(self) -> None:
        """记录一次最终失败的模型调用。"""
        self.errors += 1

    def snapshot(self) -> dict[str, int]:
        """返回累计用量的快照（``/cost`` 与 ``SessionState`` 共用）。

        Returns:
            ``prompt`` / ``completion`` / ``total`` 三项，外加
            ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` 别名；
            端点未返回 usage 时还有 ``estimated_*`` 三项。
        """
        snap = dict(self.totals)
        snap["prompt_tokens"] = snap.get("prompt", 0)
        snap["completion_tokens"] = snap.get("completion", 0)
        snap["total_tokens"] = snap.get("total", 0)
        return snap

    def cost_report(self) -> str:
        """生成 ``/cost`` 报告文本。

        Returns:
            多行报告；因代码内不存厂商价格常量，报告只给 token 口径与调用统计。
        """
        snap = self.snapshot()
        lines = [
            "模型用量（本次进程）",
            f"  prompt      {snap.get('prompt', 0):>10,}",
            f"  completion  {snap.get('completion', 0):>10,}",
            f"  total       {snap.get('total', 0):>10,}",
            f"  调用 {self.calls} 次 · 重试 {self.retries} 次 · 失败 {self.errors} 次",
        ]
        if self.estimated_usage_calls:
            lines.append(
                f"  其中估算 {self.estimated_usage_calls} 次"
                f"（约 {snap.get('estimated_total', 0):,} tokens，端点未返回 usage）"
            )
        lines.append(f"  {_PRICE_NOTE}")
        return "\n".join(lines)


def _int_of(usage: Mapping[str, int], *keys: str) -> int:
    """从 usage 字典里取第一个可用键的整数值。

    Args:
        usage: usage 字典。
        *keys: 候选键名（按优先级）。

    Returns:
        整数值；全部缺失或非法时为 0。
    """
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


# --------------------------------------------------------------------------------------
# 聚合结果（§5.3.2 引用、CONTRACTS §4.2 补齐）
# --------------------------------------------------------------------------------------


class AssistantTurn(BaseModel):
    """一次 assistant 回合的聚合结果（供 §5 引擎循环消费）。"""

    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    truncated: bool = Field(default=False, description="finish_reason=length 或流中断")

    model_config = {"extra": "ignore"}


# --------------------------------------------------------------------------------------
# LLMClient
# --------------------------------------------------------------------------------------


class LLMClient:
    """角色化模型调用入口（§13.3）。

    所有业务模块（Query / PermissionPipeline / MemoryWriter / Compactor …）只经本类
    调用模型；``role`` 决定 system prompt 与默认生成参数，**禁止**在业务代码内联提示词。
    """

    def __init__(
        self,
        provider: ModelProvider,
        config: Config,
        usage: UsageTracker,
        *,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """初始化客户端。

        Args:
            provider: 内部 Provider 实现（生产为 ``OpenAISDKClient``，测试为 ``FakeProvider``）。
            config: 完整配置（提供温度、max_tokens、端点信息）。
            usage: 进程内用量累计器。
            sleep: 退避睡眠函数（测试注入以跳过真实等待）；``None`` = ``asyncio.sleep``。
        """
        self._provider = provider
        self._config = config
        self._usage = usage
        self._sleep: Callable[[float], Awaitable[None]] = sleep or asyncio.sleep

    # ---------------------------------------------------------------- 只读属性
    @property
    def provider(self) -> ModelProvider:
        """当前 Provider（测试与调试用，业务代码不应直接调用）。"""
        return self._provider

    @property
    def usage(self) -> UsageTracker:
        """进程内用量累计器。"""
        return self._usage

    @property
    def config(self) -> Config:
        """当前配置。"""
        return self._config

    # ---------------------------------------------------------------- 参数解析
    def _role_params(self, role: str, kw: dict[str, Any]) -> dict[str, Any]:
        """解析角色的生成参数（温度、max_tokens、超时）。

        Args:
            role: 角色名。
            kw: 调用方覆盖项，支持 ``temperature`` / ``max_tokens`` / ``timeout_s``
                / ``system``；其余键透传给 Provider。

        Returns:
            含 ``system`` / ``temperature`` / ``max_tokens`` / ``timeout_s`` / ``extra``
            的字典。

        Raises:
            ValueError: ``role`` 不在 §13.2 冻结的 10 个角色内。
        """
        if role not in ROLE_PROMPTS:
            raise ValueError(f"未知角色 {role!r}；合法角色：{', '.join(ROLE_PROMPTS)}")
        defaults = ROLE_DEFAULTS[role]
        temperature = kw.pop("temperature", None)
        if temperature is None:
            temperature = self._config.llm.temperature if role == "main" else (
                self._config.llm.judge_temperature
            )
        max_tokens = kw.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = self._config.max_output_tokens() or int(defaults["max_tokens"])
        timeout_s = kw.pop("timeout_s", None)
        if timeout_s is None:
            timeout_s = role_timeout(role)
        system = kw.pop("system", None)
        if system is None:
            system = ROLE_PROMPTS[role]
        return {
            "system": system,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "timeout_s": float(timeout_s),
            "extra": kw,
        }

    # ---------------------------------------------------------------- 非流式
    async def complete_role(
        self,
        role: str,
        *,
        user: str | list[Message],
        **kw: Any,
    ) -> CompletionResult:
        """以指定角色做一次**非流式**调用（小任务 / 判定 / 压缩等首选）。

        Args:
            role: 角色名（§13.2）。
            user: 用户消息文本，或完整的内部消息列表。
            **kw: ``system`` / ``temperature`` / ``max_tokens`` / ``timeout_s`` 覆盖项；
                其余键透传给 Provider。

        Returns:
            Provider 返回的 :class:`~my_agent.llm.provider.CompletionResult`。

        Raises:
            ValueError: 角色名非法。
            LLMUnavailableError: 重试耗尽仍不可用（网络 / 5xx / 429）。
            LLMRequestError: 4xx（非 429）参数错误，**不重试**。
            ConfigError: 密钥缺失或鉴权失败。
        """
        params = self._role_params(role, kw)
        messages = (
            [Message(role="user", content=user)]
            if isinstance(user, str)
            else [item if isinstance(item, Message) else Message.model_validate(item) for item in user]
        )
        request = {
            "system": params["system"],
            "messages": sanitize(messages),
            "tools": None,
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "timeout_s": params["timeout_s"],
            **params["extra"],
        }
        started = time.monotonic()
        try:
            result = await self._with_retry(role, "complete", request)
        except MyAgentError as exc:
            # 失败也要落摘要（§13.3「日志」/ I-5）：便于排查「判定为何降级」。
            # 只记角色 + 异常类型与内部文案，**绝不含密钥**。
            _log_call(
                self._config,
                role=role,
                mode="complete",
                request=request,
                text="",
                tool_calls=0,
                finish_reason="error",
                usage={},
                latency_ms=int((time.monotonic() - started) * 1000),
                extra={"error": f"{type(exc).__name__}:{exc.user_message()}"},
            )
            raise
        latency_ms = result.latency_ms or int((time.monotonic() - started) * 1000)
        self._record_usage(role, result.usage, estimated=False)
        _log_call(
            self._config,
            role=role,
            mode="complete",
            request=request,
            text=result.text,
            tool_calls=len(result.tool_calls),
            finish_reason=result.finish_reason,
            usage=result.usage,
            latency_ms=latency_ms,
        )
        return result

    # ---------------------------------------------------------------- 流式（透传）
    def stream_role(
        self,
        role: str,
        *,
        messages: list[Message],
        tools: list[dict] | None = None,
        system: str | None = None,
        **kw: Any,
    ) -> AsyncIterator[StreamChunk]:
        """以指定角色流式调用，逐帧透传（不做工具调用聚合）。

        需要完整回合（含按 index 聚合的工具调用）时请用 :meth:`assistant_turn`。

        Args:
            role: 角色名（§13.2）。
            messages: 内部消息列表。
            tools: OpenAI 兼容的 function schema 列表。
            system: 覆盖 system prompt；``None`` = 角色默认。
            **kw: ``temperature`` / ``max_tokens`` / ``timeout_s`` 覆盖项。

        Yields:
            Provider 原样透传的 :class:`~my_agent.llm.provider.StreamChunk`。

        Raises:
            ValueError: 角色名非法。
            MyAgentError: Provider 抛出的内部异常（建流阶段）。
        """
        params = self._role_params(role, kw)
        if system is not None:
            params["system"] = system
        request = {
            "system": params["system"],
            "messages": sanitize(messages),
            "tools": tools or None,
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "timeout_s": params["timeout_s"],
            **params["extra"],
        }
        return self._provider.stream(**request)

    # ---------------------------------------------------------------- 流式聚合
    async def assistant_turn(
        self,
        role: str,
        *,
        messages: list[Message],
        tools: list[dict] | None = None,
        system: str | None = None,
        display: DisplayPort | None = None,
        **kw: Any,
    ) -> AssistantTurn:
        """流式完成一个 assistant 回合并聚合（§13.3，B7 / I-6 / I-7）。

        行为：

        1. 逐帧调用 ``display.stream_text(delta)``（CLI 的实时渲染入口）；
        2. ``delta.tool_calls[]`` 按 ``index`` 聚合，``arguments`` **拼接**（B7）；
        3. 聚合完成后对每个 ``arguments`` 做 ``json.loads``；失败则保留
           ``raw_arguments``、记 :class:`~my_agent.errors.ToolCallParseError` 并**不抛**
           （模型下一轮可自纠）；
        4. ``finish_reason == "length"`` → 丢弃部分工具调用并追加
           :data:`TRUNCATED_NOTICE`；纯文本保留并置 ``truncated=True``（I-6）；
        5. 末帧无 usage → 用 tokenizer 估算，置 ``usage_estimated=true``（I-7）。

        Args:
            role: 角色名（§13.2）。
            messages: 内部消息列表。
            tools: function schema 列表。
            system: 覆盖 system prompt。
            display: 流式渲染端口；``None`` = :class:`~my_agent.models.protocols.NullDisplay`。
            **kw: ``temperature`` / ``max_tokens`` / ``timeout_s`` 覆盖项。

        Returns:
            聚合后的 :class:`AssistantTurn`（含 usage 与 truncated 标记）。

        Raises:
            ValueError: 角色名非法。
            LLMUnavailableError: 流建不起来且重试耗尽（此时**尚未**产生任何分片）。
            MyAgentError: 其它 Provider 异常。
        """
        view: DisplayPort = display or NullDisplay()
        params = self._role_params(role, kw)
        if system is not None:
            params["system"] = system
        request = {
            "system": params["system"],
            "messages": sanitize(messages),
            "tools": tools or None,
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "timeout_s": params["timeout_s"],
            **params["extra"],
        }
        started = time.monotonic()
        text, calls, finish_reason, usage, truncated = await self._stream_turn(
            role, request, view
        )
        parsed, parse_errors = _finalize_tool_calls(
            calls, role=role, schemas=_tool_schemas(request.get("tools"))
        )
        if finish_reason == "length":
            # I-6：被截断的 tool_call 一定不完整 → 丢弃该调用，提示模型分批输出。
            if parsed:
                _LOG.warning(
                    "role=%s finish_reason=length → 丢弃 %d 个部分工具调用", role, len(parsed)
                )
                parsed = []
            if TRUNCATED_NOTICE not in text:
                text = f"{text}\n\n{TRUNCATED_NOTICE}" if text else TRUNCATED_NOTICE
            truncated = True
        self._record_usage(role, usage, estimated=usage.pop("estimated", 0) == 1)
        _log_call(
            self._config,
            role=role,
            mode="stream",
            request=request,
            text=text,
            tool_calls=len(parsed),
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=int((time.monotonic() - started) * 1000),
            extra={"parse_errors": parse_errors, "truncated": truncated},
        )
        return AssistantTurn(
            text=text,
            tool_calls=parsed,
            finish_reason=finish_reason,
            usage=usage,
            truncated=truncated,
        )

    # ---------------------------------------------------------------- 生命周期
    async def aclose(self) -> None:
        """关闭 Provider（退出时由 ``QueryEngine.shutdown()`` 调用，I-10）。

        Raises:
            MyAgentError: Provider 关闭失败。
        """
        await self._provider.close()

    # ---------------------------------------------------------------- 内部：重试
    async def _with_retry(self, role: str, mode: str, request: dict[str, Any]) -> Any:
        """执行一次非流式调用，按 §13.3 策略重试。

        Args:
            role: 角色名（仅用于日志）。
            mode: 调用模式（仅用于日志）。
            request: 已净化的 Provider 实参。

        Returns:
            Provider 的 ``CompletionResult``。

        Raises:
            MyAgentError: 不可重试的错误，或重试耗尽后的最后一个错误。
        """
        attempt = 0
        while True:
            try:
                return await self._provider.complete(**request)
            except MyAgentError as exc:
                if not self._should_retry(exc, attempt):
                    self._usage.note_error()
                    raise
                self._usage.note_retry()
                delay = _backoff_delay(attempt)
                _LOG.warning(
                    "role=%s mode=%s 第 %d 次重试（%.1fs 后）：%s",
                    role,
                    mode,
                    attempt + 1,
                    delay,
                    exc.code,
                )
                await self._sleep(delay)
                attempt += 1

    def _should_retry(self, exc: MyAgentError, attempt: int) -> bool:
        """判定该错误是否应重试（§13.3「重试」）。

        可重试 = 网络错误 / 5xx / 429（§13.1.5 中映射为 :class:`LLMUnavailableError`
        的那些）；4xx（非 429）映射为 :class:`LLMRequestError` / :class:`ConfigError`，
        **绝不重试**。

        Args:
            exc: Provider 抛出的内部异常。
            attempt: 已发生的重试次数（0 起）。

        Returns:
            ``True`` = 属于可重试族且未超过 ``LLM_MAX_RETRIES``。
        """
        if attempt >= D.LLM_MAX_RETRIES:
            return False
        return type(exc) in _RETRYABLE_ERRORS

    # ---------------------------------------------------------------- 内部：流聚合
    async def _stream_turn(
        self,
        role: str,
        request: dict[str, Any],
        display: DisplayPort,
    ) -> tuple[str, dict[int, dict[str, str]], str, dict[str, int], bool]:
        """消费 Provider 的流并聚合（含建流阶段重试）。

        建流阶段（尚未产出任何分片）失败可按 §13.3 重试；一旦已经产出分片，
        重试会产生重复输出，因此**保留已聚合内容并标记 truncated=True**（§13.3
        「流被中断 → 已聚合的内容保留并标记 truncated=True，未闭合的 tool_call 丢弃」）。

        Args:
            role: 角色名。
            request: Provider 实参。
            display: 流式渲染端口。

        Returns:
            ``(text, aggregated_tool_calls, finish_reason, usage, truncated)``。

        Raises:
            MyAgentError: 流始终无法建立（重试耗尽），或中断且已无可保留内容时。
        """
        attempt = 0
        while True:
            text_parts: list[str] = []
            calls: dict[int, dict[str, str]] = {}
            finish_reason = ""
            usage: dict[str, int] = {}
            interrupted = False
            try:
                async for chunk in self._provider.stream(**request):
                    if chunk.text_delta:
                        text_parts.append(chunk.text_delta)
                        display.stream_text(chunk.text_delta)
                    if chunk.tool_call_delta is not None:
                        _aggregate_tool_call(calls, chunk.tool_call_delta)
                    if chunk.finish_reason:
                        finish_reason = chunk.finish_reason
                    if chunk.usage:
                        usage = dict(chunk.usage)
            except MyAgentError as exc:
                if not text_parts and not calls and self._should_retry(exc, attempt):
                    self._usage.note_retry()
                    delay = _backoff_delay(attempt)
                    _LOG.warning(
                        "role=%s stream 建流失败，第 %d 次重试（%.1fs 后）：%s",
                        role,
                        attempt + 1,
                        delay,
                        exc.code,
                    )
                    await self._sleep(delay)
                    attempt += 1
                    continue
                if not text_parts and not calls:
                    self._usage.note_error()
                    raise
                _LOG.warning("role=%s 流中断，保留已聚合内容：%s", role, exc.code)
                interrupted = True
            text = "".join(text_parts)
            if not usage:
                usage = self._estimate_usage(request, text, calls)
            if interrupted:
                # 未闭合的 tool_call 丢弃（§13.3）。
                finish_reason = finish_reason or "interrupted"
            return text, calls, finish_reason, usage, interrupted

    def _estimate_usage(
        self,
        request: Mapping[str, Any],
        text: str,
        calls: Mapping[int, Mapping[str, str]],
    ) -> dict[str, int]:
        """端点未返回 usage 时的估算兜底（I-7）。

        Args:
            request: Provider 实参（含 messages）。
            text: 已聚合的文本。
            calls: 已聚合的工具调用分片。

        Returns:
            估算 usage；键 ``estimated`` 置 1，供上层标记 ``usage_estimated=true``。
        """
        from my_agent.llm.tokenizer import get_counter

        counter = get_counter(_endpoint_model(self._config))
        prompt = counter.count_messages(_as_messages(request.get("messages")))
        completion = counter.count_text(text) + sum(
            counter.count_text(json.dumps(item, ensure_ascii=False)) for item in calls.values()
        )
        _LOG.info("usage_estimated=true（端点未返回 usage，已用 tokenizer 估算）")
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "total": prompt + completion,
            "estimated": 1,
        }

    # ---------------------------------------------------------------- 内部：用量
    def _record_usage(self, role: str, usage: Mapping[str, int], *, estimated: bool) -> None:
        """把一次调用的用量累计进 :class:`UsageTracker`。

        Args:
            role: 角色名（仅用于日志）。
            usage: usage 字典；含内部标记 ``estimated`` 时会被剔除。
            estimated: 是否估算值。
        """
        clean = {k: v for k, v in usage.items() if k != "estimated"}
        if not clean:
            return
        self._usage.add(clean, estimated=estimated or usage.get("estimated") == 1)
        if estimated or usage.get("estimated") == 1:
            _LOG.info(
                "role=%s usage_estimated=true prompt=%s completion=%s",
                role,
                clean.get("prompt_tokens"),
                clean.get("completion_tokens"),
            )


# --------------------------------------------------------------------------------------
# 聚合工具函数
# --------------------------------------------------------------------------------------


def _aggregate_tool_call(
    acc: dict[int, dict[str, str]],
    delta: Any,
) -> None:
    """把一个工具调用分片并入聚合表（B7：``arguments`` 必须**拼接**）。

    Args:
        acc: ``index`` → ``{"id", "name", "arguments"}`` 的累加表（原地修改）。
        delta: Provider 给出的 :class:`~my_agent.llm.provider.ToolCallDelta`。
    """
    slot = acc.setdefault(delta.index, {"id": "", "name": "", "arguments": ""})
    if delta.id:
        slot["id"] = delta.id
    if delta.name:
        slot["name"] = delta.name
    if delta.arguments_delta:
        slot["arguments"] = slot["arguments"] + delta.arguments_delta


def _finalize_tool_calls(
    acc: Mapping[int, Mapping[str, str]],
    *,
    role: str,
    schemas: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[list[ToolCallRequest], int]:
    """把聚合表转成 :class:`ToolCallRequest` 列表（B7 后半段）。

    参数解析走 ``llm/tool_args.py`` 的容错解析链（预处理 → ``json.loads`` → 修复 →
    schema 校验），本函数**永不抛**：

    * ``OK`` / ``REPAIRED`` → 用解析（可能被修复）出的参数下发；``REPAIRED`` **不计**
      解析错误（可选传 ``schemas`` 以对该工具参数做 Pydantic 校验）；
    * ``INVALID`` → 保留 ``raw_arguments``、置 ``arguments={}``、错误计数 +1，并记
      :class:`~my_agent.errors.ToolCallParseError`（让模型下一轮自纠，B7）。

    Args:
        acc: ``index`` → 分片聚合结果。
        role: 角色名（仅用于日志）。
        schemas: 工具名 → 参数 JSON Schema（可选）。给了才做 schema 校验。

    Returns:
        ``(tool_calls, parse_error_count)``。
    """
    calls: list[ToolCallRequest] = []
    errors = 0
    repaired = 0
    for index in sorted(acc):
        slot = acc[index]
        raw = slot.get("arguments", "")
        name = slot.get("name", "")
        call_id = slot.get("id", "") or f"call_{index}"
        if not raw.strip():
            calls.append(ToolCallRequest(id=call_id, name=name, arguments={}, raw_arguments=raw))
            continue
        result = parse_tool_arguments(raw, schema=(schemas or {}).get(name))
        if result.status == ArgParseStatus.INVALID:
            errors += 1
            error = ToolCallParseError(
                result.error or INVALID_ARGS_MESSAGE,
                detail=result.error or INVALID_ARGS_MESSAGE,
                name=name,
                call_id=call_id,
            )
            _LOG.warning(
                "role=%s ToolCallParseError name=%s call_id=%s detail=%s（保留 raw 串供自纠）",
                role,
                name,
                call_id,
                "; ".join(result.notes) or error.detail,
            )
            calls.append(ToolCallRequest(id=call_id, name=name, arguments={}, raw_arguments=raw))
            continue
        arguments = result.arguments
        if not arguments:
            # 顶层不是对象（数组 / 标量）：与旧行为一致，包成 value 字段并计一次错误。
            parsed = _try_json_loads(raw)
            if parsed is not None and not isinstance(parsed, dict):
                errors += 1
                _LOG.warning(
                    "role=%s 工具调用参数不是 JSON 对象（%s），已包成 value 字段", role, name
                )
                arguments = {"value": parsed}
        if result.repaired:
            repaired += 1
        calls.append(
            ToolCallRequest(id=call_id, name=name, arguments=arguments, raw_arguments=raw)
        )
    if repaired and not errors:
        _LOG.info("role=%s 参数容错修复成功 %d 个工具调用（未计解析错误）", role, repaired)
    return calls, errors


def _try_json_loads(raw: str) -> Any:
    """宽容地尝试一次标准 ``json.loads``（仅用于「非对象」结果的判别）。

    Args:
        raw: 原始参数串。

    Returns:
        解析结果；非法 JSON 时返回 ``None``。
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def _tool_schemas(tools: Any) -> dict[str, dict[str, Any]] | None:
    """从发往 Provider 的 tools 列表里抽出「工具名 → 参数 JSON Schema」。

    形如 ``{"type": "function", "function": {"name": ..., "parameters": {...}}}``；
    形状不合法的条目**跳过**（schema 校验是可选增强，不该影响主流程）。

    Args:
        tools: ``stream_turn`` 收到的 tools 实参。

    Returns:
        工具名 → schema；无法抽取时返回 ``None``。
    """
    if not isinstance(tools, Sequence):
        return None
    out: dict[str, dict[str, Any]] = {}
    for item in tools:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if isinstance(name, str) and name and isinstance(parameters, Mapping):
            out[name] = dict(parameters)
    return out or None


def _backoff_delay(attempt: int) -> float:
    """取第 ``attempt`` 次重试的退避秒数（§13.3：0.5s, 1s, 2s）。

    Args:
        attempt: 已发生的重试次数（0 起）。

    Returns:
        退避秒数；超出 ``LLM_BACKOFF_S`` 长度时取最后一个值。
    """
    if not D.LLM_BACKOFF_S:
        return 0.0
    return D.LLM_BACKOFF_S[min(attempt, len(D.LLM_BACKOFF_S) - 1)]


def _as_messages(raw: Any) -> list[Message]:
    """把 Provider 实参里的消息字典还原为 ``Message``（估算用）。

    Args:
        raw: ``request["messages"]``（已净化的字典列表）。

    Returns:
        ``Message`` 列表；形状不合法时尽量容忍。
    """
    out: list[Message] = []
    for item in raw or ():
        if isinstance(item, Message):
            out.append(item)
        elif isinstance(item, Mapping):
            role = str(item.get("role", "user"))
            if role not in ("system", "user", "assistant", "tool"):
                role = "user"
            try:
                out.append(
                    Message(
                        role=role,  # type: ignore[arg-type]
                        content=str(item.get("content") or ""),
                        tool_calls=[
                            ToolCallRequest(
                                id=str(call.get("id", "")),
                                name=str((call.get("function") or {}).get("name", "")),
                                raw_arguments=str((call.get("function") or {}).get("arguments", "")),
                            )
                            for call in item.get("tool_calls") or ()
                        ],
                        tool_call_id=item.get("tool_call_id"),
                    )
                )
            except (TypeError, ValueError):  # pragma: no cover - 防御性
                continue
    return out


def _endpoint_model(config: Config) -> str:
    """取当前生效的模型名（用于选择 tokenizer encoding）。

    Args:
        config: 完整配置。

    Returns:
        模型名；端点未配置时为空串（tokenizer 会回退到默认 encoding）。
    """
    endpoint = config.llm.endpoints.get(config.llm.endpoint)
    return endpoint.model if endpoint is not None else ""


# --------------------------------------------------------------------------------------
# 日志（I-5：不含密钥；全文仅 DEBUG）
# --------------------------------------------------------------------------------------


def _log_call(
    config: Config,
    *,
    role: str,
    mode: str,
    request: Mapping[str, Any],
    text: str,
    tool_calls: int,
    finish_reason: str,
    usage: Mapping[str, int],
    latency_ms: int,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """写请求 / 响应摘要到 ``<myagent_home>/logs/llm-YYYY-MM-DD.log``。

    摘要行只含角色、模式、模型、消息条数、工具数、耗时与用量——**绝不含**密钥材料；
    提示词全文与响应全文只在 DEBUG 级别记录（§13.3「日志」）。

    Args:
        config: 完整配置（用于定位日志目录与端点）。
        role: 角色名。
        mode: ``complete`` / ``stream``。
        request: Provider 实参（只读取非敏感字段）。
        text: 响应文本（仅 DEBUG 记录）。
        tool_calls: 工具调用数量。
        finish_reason: 结束原因。
        usage: 用量字典。
        latency_ms: 耗时毫秒。
        extra: 额外的非敏感字段（解析错误数、截断标记等）。
    """
    logger = _ensure_file_logger(config)
    if logger is None:
        return
    endpoint = config.llm.endpoints.get(config.llm.endpoint)
    summary = (
        f"role={role} mode={mode} model={_endpoint_model(config) or '<unset>'} "
        f"messages={len(request.get('messages') or ())} "
        f"tools={len(request.get('tools') or ())} "
        f"max_tokens={request.get('max_tokens')} temperature={request.get('temperature')} "
        f"timeout_s={request.get('timeout_s')} "
        f"finish={finish_reason or '-'} tool_calls={tool_calls} "
        f"latency_ms={latency_ms} usage={dict(usage)}"
    )
    if endpoint is not None:
        summary += f" base_url={endpoint.base_url}"
    if extra:
        summary += " " + " ".join(f"{key}={value}" for key, value in extra.items())
    logger.info(summary)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("role=%s system=%r", role, request.get("system"))
        logger.debug("role=%s response_text=%r", role, text)


def _ensure_file_logger(config: Config) -> logging.Logger | None:
    """确保 LLM 日志文件 handler 已挂载（目录或日期变化时自动换文件）。

    Args:
        config: 完整配置；``llm.log_dir`` 可覆盖默认目录。

    Returns:
        可用的 logger；目录不可写时返回 ``None``（日志失败绝不阻断主流程）。
    """
    logger = _LOG
    path = log_file_path(config)
    if getattr(logger, "_myagent_log_path", None) == str(path):
        return logger
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = getattr(logger, "_myagent_file_handler", None)
        if previous is not None:
            logger.removeHandler(previous)
            previous.close()
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger._myagent_file_handler = handler  # type: ignore[attr-defined]
        logger._myagent_log_path = str(path)  # type: ignore[attr-defined]
    except OSError:
        return None
    return logger


def log_file_path(config: Config) -> Path:
    """返回当日 LLM 日志文件路径（测试与 ``/cost`` 排查用）。

    Args:
        config: 完整配置。

    Returns:
        当日日志文件的绝对路径。
    """
    base = Path(config.llm.log_dir) if config.llm.log_dir else myagent_home() / "logs"
    return base / f"{_LOG_PREFIX}{date.today().isoformat()}{_LOG_SUFFIX}"
