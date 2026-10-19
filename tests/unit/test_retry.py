"""``retry.py`` 单测：工具层网络重试基础设施。

覆盖三块公共 API：

* :func:`~my_agent.retry.is_retryable_network_error`：可重试/不可重试的边界
  （**MCP 超时绝不重试**，4xx 绝不重试）；
* :func:`~my_agent.retry.retryable_status`：408 / 429 / 5xx 与 4xx 的分界；
* :func:`~my_agent.retry.backoff_delay` 与 :func:`~my_agent.retry.with_network_retry`：
  指数退避表、重试次数上限、注入式 ``sleep``（**测试不真实等待**）。

断言口径：调用**次数**（不是「差不多」）与注入 ``sleep`` 记录的**精确退避序列**。
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from my_agent.config import defaults as D
from my_agent.errors import (
    ConfigError,
    LLMUnavailableError,
    McpTimeoutError,
    MyAgentError,
    ToolExecutionError,
)
from my_agent.retry import (
    RETRYABLE_STATUS_CODES,
    backoff_delay,
    is_retryable_network_error,
    retryable_status,
    with_network_retry,
)

#: 一个稳定的假请求对象（构造 ``HTTPStatusError`` 需要 ``request`` 与 ``response``）。
_REQUEST = httpx.Request("GET", "http://retry.test/resource")


def http_status_error(status_code: int) -> httpx.HTTPStatusError:
    """构造一个带真实 ``response`` 的 ``HTTPStatusError``。

    Args:
        status_code: 响应状态码。

    Returns:
        :class:`httpx.HTTPStatusError` 实例。
    """
    response = httpx.Response(status_code, request=_REQUEST, text="boom")
    return httpx.HTTPStatusError(f"HTTP {status_code}", request=_REQUEST, response=response)


class SleepRecorder:
    """记录退避调用但**不真实等待**的注入式 ``sleep``。

    Attributes:
        delays: 依次收到的退避秒数。
    """

    def __init__(self) -> None:
        """初始化空记录。"""
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        """记录一次退避（立即返回，不睡眠）。

        Args:
            seconds: 退避秒数。
        """
        self.delays.append(seconds)


class FlakyCallable:
    """按剧本依次返回结果或抛异常的假可等待工厂（记录调用次数）。

    Attributes:
        calls: 被调用的次数。
    """

    def __init__(self, script: list[Any]) -> None:
        """初始化假工厂。

        Args:
            script: 每次调用的剧本元素；异常实例 = 抛出，其余 = 作为返回值。
        """
        self._script = list(script)
        self.calls = 0

    async def __call__(self) -> Any:
        """执行剧本中的下一条（越界则重复最后一条）。

        Returns:
            剧本中的返回值。

        Raises:
            BaseException: 剧本元素是异常实例时原样抛出。
        """
        index = min(self.calls, len(self._script) - 1)
        self.calls += 1
        item = self._script[index]
        if isinstance(item, BaseException):
            raise item
        return item


# --------------------------------------------------------------------------------------
# is_retryable_network_error
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        httpx.TimeoutException("timeout"),
        httpx.ConnectTimeout("connect timeout"),
        httpx.ReadTimeout("read timeout"),
        httpx.WriteTimeout("write timeout"),
        httpx.PoolTimeout("pool timeout"),
        httpx.TransportError("transport"),
        httpx.ConnectError("connect error"),
        httpx.ReadError("read error"),
        httpx.RemoteProtocolError("remote protocol error"),
    ],
)
def test_retryable_for_timeouts_and_transport_errors(exc: BaseException) -> None:
    """httpx 超时族与传输层错误 → 可重试。"""
    assert is_retryable_network_error(exc) is True


def test_retryable_for_llm_unavailable_error() -> None:
    """模型层已归并的 ``LLMUnavailableError`` → 可重试（工具层与模型层同纪律）。"""
    assert is_retryable_network_error(LLMUnavailableError("网络不可用")) is True


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422, 499])
def test_not_retryable_for_4xx_status_error(status_code: int) -> None:
    """4xx 是客户端错误：重试无法改变结果 → 不可重试。"""
    assert is_retryable_network_error(http_status_error(status_code)) is False


@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
def test_retryable_for_retryable_status_error(status_code: int) -> None:
    """408 / 429 / 5xx 的状态错误 → 可重试。"""
    assert is_retryable_network_error(http_status_error(status_code)) is True


def test_mcp_timeout_is_never_retryable() -> None:
    """**MCP 超时绝不重试**（§10.5 已批准的例外）。"""
    exc = McpTimeoutError(server="fs", timeout=30, method="tools/call")
    assert is_retryable_network_error(exc) is False


def test_config_error_is_not_retryable() -> None:
    """``ConfigError``（密钥缺失/鉴权失败）→ 不可重试。"""
    assert is_retryable_network_error(ConfigError("缺少 API Key")) is False


@pytest.mark.parametrize(
    "exc",
    [
        ToolExecutionError("工具失败"),
        MyAgentError("基类错误"),
        ValueError("参数非法"),
        RuntimeError("未知错误"),
    ],
)
def test_other_errors_are_not_retryable(exc: BaseException) -> None:
    """其它 ``MyAgentError`` 子类与未知异常 → 不可重试（保守）。"""
    assert is_retryable_network_error(exc) is False


def test_cancellation_is_not_retryable() -> None:
    """``asyncio.CancelledError`` 不作为网络异常重试（否则取消会被吞掉）。"""
    assert is_retryable_network_error(asyncio.CancelledError()) is False


# --------------------------------------------------------------------------------------
# retryable_status
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503])
def test_retryable_status_true(status_code: int) -> None:
    """408 / 429 / 500 / 502 / 503 → 可重试。"""
    assert retryable_status(status_code) is True


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_retryable_status_false(status_code: int) -> None:
    """400 / 401 / 403 / 404 → 不可重试。"""
    assert retryable_status(status_code) is False


def test_retryable_status_boundaries() -> None:
    """边界：201 / 301 / 499 不可重试；500 与 599 可重试；600 不在 5xx 区间内。"""
    assert retryable_status(201) is False
    assert retryable_status(301) is False
    assert retryable_status(499) is False
    assert retryable_status(500) is True
    assert retryable_status(599) is True
    assert retryable_status(600) is False
    assert frozenset({408, 429}) == RETRYABLE_STATUS_CODES


# --------------------------------------------------------------------------------------
# backoff_delay
# --------------------------------------------------------------------------------------


def test_backoff_delay_follows_tool_schedule_and_clamps() -> None:
    """0→0.5、1→1.0、2→2.0、3+→2.0（**收敛到表尾**，不外推成 4s/8s）。"""
    assert D.TOOL_RETRY_BACKOFF_S == (0.5, 1.0, 2.0)
    assert backoff_delay(0) == 0.5
    assert backoff_delay(1) == 1.0
    assert backoff_delay(2) == 2.0
    assert backoff_delay(3) == 2.0
    assert backoff_delay(99) == 2.0


def test_backoff_delay_with_custom_schedule() -> None:
    """自定义退避表同样生效，且空表退化为 ``0.0``（不抛异常）；负数按 0 处理。"""
    assert backoff_delay(0, schedule=(0.1,)) == 0.1
    assert backoff_delay(5, schedule=(0.1,)) == 0.1
    assert backoff_delay(0, schedule=()) == 0.0
    assert backoff_delay(-1) == 0.5


# --------------------------------------------------------------------------------------
# with_network_retry
# --------------------------------------------------------------------------------------


async def test_succeeds_on_third_attempt_after_two_transient_failures() -> None:
    """两次瞬时失败后第 3 次成功：**恰好 3 次调用**，退避序列 ``[0.5, 1.0]``。"""
    sleeps = SleepRecorder()
    flaky = FlakyCallable([httpx.ConnectError("boom"), httpx.ReadTimeout("slow"), "ok"])
    seen: list[tuple[int, float, BaseException]] = []

    def record(attempt: int, delay: float, exc: BaseException) -> None:
        """记录一次重试回调。

        Args:
            attempt: 重试序号（0 起）。
            delay: 退避秒数。
            exc: 触发重试的异常。
        """
        seen.append((attempt, delay, exc))

    result = await with_network_retry(flaky, sleep=sleeps, label="unit", on_retry=record)

    assert result == "ok"
    assert flaky.calls == 3
    assert sleeps.delays == [0.5, 1.0]
    assert [(attempt, delay) for attempt, delay, _exc in seen] == [(0, 0.5), (1, 1.0)]
    assert all(isinstance(exc, httpx.HTTPError) for _a, _d, exc in seen)


async def test_exhausted_attempts_raises_last_error() -> None:
    """重试耗尽 → 抛**最后一个**异常：3 次调用（不是 4 次），退避 ``[0.5, 1.0]``。"""
    sleeps = SleepRecorder()
    errors = [httpx.ConnectError(f"boom-{index}") for index in range(4)]
    flaky = FlakyCallable(list(errors))

    with pytest.raises(httpx.ConnectError) as raised:
        await with_network_retry(flaky, sleep=sleeps, label="unit")

    assert str(raised.value) == "boom-2"  # 最后一次尝试（第 3 次）的异常
    assert flaky.calls == D.TOOL_RETRY_MAX_ATTEMPTS == 3
    assert sleeps.delays == [0.5, 1.0]


async def test_retries_retryable_status_error() -> None:
    """可重试状态码（503）走同一条退避路径，最终成功。"""
    sleeps = SleepRecorder()
    flaky = FlakyCallable([http_status_error(503), http_status_error(502), "ok"])

    assert await with_network_retry(flaky, sleep=sleeps, label="unit") == "ok"
    assert flaky.calls == 3
    assert sleeps.delays == [0.5, 1.0]


async def test_non_retryable_error_raises_immediately_with_single_call() -> None:
    """不可重试的错误 → **立即抛出**，只调用 1 次，且不产生任何退避。"""
    sleeps = SleepRecorder()
    flaky = FlakyCallable([http_status_error(404)])

    with pytest.raises(httpx.HTTPStatusError) as raised:
        await with_network_retry(flaky, sleep=sleeps, label="unit")

    assert raised.value.response.status_code == 404
    assert flaky.calls == 1
    assert sleeps.delays == []


async def test_mcp_timeout_is_not_retried_by_helper() -> None:
    """``McpTimeoutError`` 穿过重试包装层 → 只调用 1 次（§10.5）。"""
    sleeps = SleepRecorder()
    flaky = FlakyCallable([McpTimeoutError(server="fs", timeout=30, method="tools/call")])

    with pytest.raises(McpTimeoutError):
        await with_network_retry(flaky, sleep=sleeps, label="MCP tools/call")

    assert flaky.calls == 1
    assert sleeps.delays == []


async def test_attempts_one_means_no_retry() -> None:
    """``attempts=1`` → 单次尝试，不重试。"""
    sleeps = SleepRecorder()
    flaky = FlakyCallable([httpx.ConnectError("boom"), "ok"])

    with pytest.raises(httpx.ConnectError):
        await with_network_retry(flaky, attempts=1, schedule=(9.0,), sleep=sleeps)

    assert flaky.calls == 1
    assert sleeps.delays == []


async def test_custom_schedule_and_attempts_are_used() -> None:
    """自定义退避表被逐次消费，超出表长后 clamp 到表尾。"""
    sleeps = SleepRecorder()
    flaky = FlakyCallable(
        [
            httpx.ConnectError("1"),
            httpx.ConnectError("2"),
            httpx.ConnectError("3"),
            httpx.ConnectError("4"),
            "ok",
        ]
    )

    result = await with_network_retry(flaky, attempts=5, schedule=(0.25, 0.5), sleep=sleeps)

    assert result == "ok"
    assert flaky.calls == 5
    assert sleeps.delays == [0.25, 0.5, 0.5, 0.5]


async def test_default_sleep_path_is_used_when_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未注入 ``sleep`` 时走生产路径 ``asyncio.sleep``（此处替换为记录器，不真实等待）。"""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        """替代 ``asyncio.sleep``：只记录，不等待。

        Args:
            seconds: 退避秒数。
        """
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    flaky = FlakyCallable([httpx.ConnectError("boom"), "ok"])

    assert await with_network_retry(flaky, label="unit") == "ok"
    assert slept == [D.TOOL_RETRY_BACKOFF_S[0]]
