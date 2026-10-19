"""工具层网络重试基础设施（§10.5、§14）。

模型层的重试在 :mod:`my_agent.llm.client`（``LLM_MAX_RETRIES`` / ``LLM_BACKOFF_S``）；
本模块补齐**工具层**：``WebFetch`` / ``WebSearch`` / MCP HTTP 的网络异常一律走
指数退避重试，次数上限 :data:`~my_agent.config.defaults.TOOL_RETRY_MAX_ATTEMPTS`。

三条纪律（全部由 :func:`is_retryable_network_error` 收口）：

* **可重试**：连接/读写/池超时、传输层错误（连接失败、读失败、协议错乱），以及模型层
  已归并好的 :class:`~my_agent.errors.LLMUnavailableError`；
* **绝不重试 4xx**：``httpx.HTTPStatusError`` 的 4xx 是**客户端错误**（URL 写错、鉴权失败、
  被限流以外的拒绝），重试不能改变结果，只会拖慢失败反馈；
* **绝不重试 MCP 超时**：:class:`~my_agent.errors.McpTimeoutError` 按 §10.5 是终态
  （用户/配置显式设定的 ``server.timeout_s`` 已到），**不重试**——本模块把它放在判定链最前面，
  避免它被任何宽泛的 httpx 判定误伤。

状态码口径见 :func:`retryable_status`：408 / 429 / 全部 5xx 可重试，其余不可。

无向上依赖（L1）：只依赖 ``my_agent.config.defaults`` 与 ``my_agent.errors``，
任何工具/客户端（L3+）都可直接调用。

规格：``my_agent-开发文档.md`` §10.5、§14；假定登记：``TOOL_RETRY_*``。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

import httpx

from my_agent.config.defaults import TOOL_RETRY_BACKOFF_S, TOOL_RETRY_MAX_ATTEMPTS
from my_agent.errors import LLMUnavailableError, McpTimeoutError

__all__ = [
    "RETRYABLE_STATUS_CODES",
    "backoff_delay",
    "is_retryable_network_error",
    "retryable_status",
    "with_network_retry",
]

_LOG = logging.getLogger(__name__)

#: 返回类型变量（``with_network_retry`` 原样透传 ``fn`` 的返回值）。
T = TypeVar("T")

#: 可重试的「非 5xx」状态码：408 请求超时、429 请求过多。
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429})

#: 5xx 区间（含端点）：服务端错误一律可重试。
_SERVER_ERROR_RANGE = (500, 600)


async def _default_sleep(seconds: float) -> None:
    """默认退避实现：``asyncio.sleep``（生产路径）。

    Args:
        seconds: 退避秒数。
    """
    await asyncio.sleep(seconds)


def is_retryable_network_error(exc: BaseException) -> bool:
    """判断一个异常是否属于「重试有意义」的网络异常。

    Args:
        exc: 待判定的异常。

    Returns:
        ``True`` = 可重试：

        * ``httpx.TimeoutException``（含 ``ConnectTimeout`` / ``ReadTimeout`` /
          ``WriteTimeout`` / ``PoolTimeout``）；
        * ``httpx.TransportError``（含 ``ConnectError`` / ``ReadError`` /
          ``RemoteProtocolError``）；
        * ``httpx.HTTPStatusError`` 且状态码可重试（408 / 429 / 5xx）；
        * :class:`~my_agent.errors.LLMUnavailableError`（模型层已归并的网络/5xx/429）。

        ``False`` = 不可重试：

        * ``httpx.HTTPStatusError`` 的 4xx（客户端错误，重试无用）；
        * :class:`~my_agent.errors.McpTimeoutError`（§10.5：超时是终态）；
        * :class:`~my_agent.errors.ConfigError` 等**非** ``LLMUnavailableError`` 的
          ``MyAgentError`` 子类；
        * 其它任何异常（含 ``ValueError`` / ``asyncio.CancelledError``）。
    """
    # §10.5：MCP 超时绝不重试。放在最前面，避免被任何宽泛判定误伤。
    if isinstance(exc, McpTimeoutError):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return retryable_status(exc.response.status_code)
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPError):
        # 其余 httpx 错误（如 TooManyRedirects / InvalidURL）：请求本身有问题，重试无用。
        return False
    # 模型层已归并的网络/5xx/429 可重试；ConfigError 等 MyAgentError 子类
    # 以及一切未知异常：保守判为不可重试。
    return isinstance(exc, LLMUnavailableError)


def retryable_status(status_code: int) -> bool:
    """判断 HTTP 状态码是否值得重试。

    Args:
        status_code: HTTP 状态码。

    Returns:
        ``True`` = 408（请求超时）/ 429（请求过多）/ 5xx（服务端错误）；
        ``False`` = 其它（含全部 4xx 客户端错误）。
    """
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    return _SERVER_ERROR_RANGE[0] <= status_code < _SERVER_ERROR_RANGE[1]


def backoff_delay(
    attempt: int,
    *,
    schedule: Sequence[float] = TOOL_RETRY_BACKOFF_S,
) -> float:
    """取第 ``attempt`` 次重试的退避秒数。

    Args:
        attempt: 已发生的重试次数（0 起，即第一次重试传 0）。
        schedule: 退避表；默认 :data:`~my_agent.config.defaults.TOOL_RETRY_BACKOFF_S`。

    Returns:
        退避秒数；``attempt`` 超出表长时**保持返回最后一个元素**（上限收敛），
        空表时返回 ``0.0``。
    """
    if not schedule:
        return 0.0
    return float(schedule[min(max(attempt, 0), len(schedule) - 1)])


async def with_network_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = TOOL_RETRY_MAX_ATTEMPTS,
    schedule: Sequence[float] = TOOL_RETRY_BACKOFF_S,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    label: str = "",
) -> T:
    """调用 ``fn()``，网络异常按指数退避重试。

    行为：

    1. 调用 ``fn()`` 并返回其结果；
    2. 捕获到 :func:`is_retryable_network_error` 判为可重试的异常且重试次数未用尽时，
       记一条 WARNING 日志、回调 ``on_retry(第几次重试, 退避秒数, 异常)``、
       ``await sleep(退避秒数)``，然后重试；
    3. 重试次数用尽 → **重新抛出最后一个异常**（调用方翻译成
       ``ToolResult(ok=False)`` 或 MCP 异常）；
    4. 不可重试的异常**立即原样抛出**（不做任何包装，也不睡）。

    Args:
        fn: 无参可等待工厂（如 ``lambda: self._fetch_once(url)``）。每次重试都会重新调用，
            因此内部应能安全地重新发起请求。
        attempts: **总调用次数上限**（含首次）。``3`` = 首次 + 最多 2 次重试。
        schedule: 退避表；默认 :data:`~my_agent.config.defaults.TOOL_RETRY_BACKOFF_S`。
        on_retry: 可选回调，签名 ``(attempt, delay, exc)``；``attempt`` 为重试序号（0 起），
            便于调用方审计「实际重试了几次」。回调自身抛异常会照常向上传播。
        sleep: 可注入的睡眠函数（测试注入以免真实等待）；``None`` = ``asyncio.sleep``。
        label: 日志用的来源标签（如 ``"WebFetch"`` / ``"MCP tools/call"``）。

    Returns:
        ``fn()`` 的返回值。

    Raises:
        BaseException: ``fn()`` 抛出的**最后一个**异常（可重试但已用尽，或不可重试）。
    """
    pause = sleep or _default_sleep
    tried = 0
    while True:
        try:
            return await fn()
        except BaseException as exc:  # noqa: BLE001 - 逐个判定，非重试异常原样抛出
            if tried + 1 >= max(attempts, 1) or not is_retryable_network_error(exc):
                raise
            delay = backoff_delay(tried, schedule=schedule)
            tried += 1
            _LOG.warning(
                "%s：第 %d/%d 次网络重试（%.1fs 后）：%s: %s",
                label or "网络请求",
                tried,
                max(attempts, 1) - 1,
                delay,
                type(exc).__name__,
                exc,
            )
            if on_retry is not None:
                on_retry(tried - 1, delay, exc)
            await pause(delay)
