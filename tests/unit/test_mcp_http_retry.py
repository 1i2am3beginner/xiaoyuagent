"""``mcp/client_http.py`` 重试行为单测：JSON-RPC ``POST`` 的网络退避。

被测纪律：

* 连接/传输层故障（``httpx.ConnectError`` 等）与 408 / 429 / 5xx → 按
  ``TOOL_RETRY_BACKOFF_S`` 退避重试，上限 ``TOOL_RETRY_MAX_ATTEMPTS``；
* 4xx（如 404）→ **立即失败**（只 1 次请求）；
* **MCP 超时绝不重试**（§10.5，用户已批准的例外）：
  ``server.timeout_s`` 到点即终态 —— ``McpTimeoutError`` 恰好尝试 **1 次**；
* ``close()`` / 流停止路径**不包装重试**。

实现手法：把所有 ``httpx.AsyncClient`` 换成带 ``httpx.MockTransport`` 的实例（不触网），
并把 ``client_http`` 的模块级 ``_retry_sleep`` 注入为记录器（**测试不真实等待退避**），
请求次数由假服务端自己统计。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from my_agent.config import defaults as D
from my_agent.config.schema import McpServerConfig
from my_agent.errors import McpTimeoutError, McpValidationError
from my_agent.mcp import client_http as mcp_module
from my_agent.mcp.client_http import McpHttpClient

SERVER_URL = "http://mcp.test/rpc"

#: 剧本元素：表示「按默认的合法 JSON-RPC 应答」（区别于「无限重复最后一条」）。
DEFAULT = object()


class SleepRecorder:
    """记录退避调用但**不真实等待**的注入式 ``sleep``。

    Attributes:
        delays: 依次收到的退避秒数。
    """

    def __init__(self) -> None:
        """初始化空记录。"""
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        """记录一次退避（立即返回）。

        Args:
            seconds: 退避秒数。
        """
        self.delays.append(seconds)


class FakeServer:
    """假 JSON-RPC 服务端：按剧本回放响应/异常，并做最小 JSON-RPC 应答。

    ``initialize`` 需要合法响应（``protocolVersion``），其余方法原样回 ``result``。

    Attributes:
        requests: 收到的请求数（**断言重试次数的唯一口径**）。
        methods: 依次收到的方法名。
    """

    def __init__(self, script: list[Any] | None = None) -> None:
        """初始化假服务端。

        Args:
            script: 每次请求的剧本元素；异常实例 = 抛出，``httpx.Response`` = 返回，
                可调用对象 = 以 ``(request, payload)`` 调用后返回其响应，
                :data:`DEFAULT`（或剧本用尽）= 走默认的合法 JSON-RPC 应答。
        """
        self._script = list(script or [])
        self.requests = 0
        self.methods: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """处理一次 JSON-RPC POST。

        Args:
            request: 收到的请求。

        Returns:
            JSON-RPC 响应。

        Raises:
            BaseException: 剧本元素是异常实例时原样抛出（模拟连接/超时故障）。
        """
        index = self.requests
        self.requests += 1
        if request.method == "GET":  # 旧版 SSE 形态的事件流
            self.methods.append("GET")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=f"event: endpoint\ndata: {SERVER_URL}\n\n",
                request=request,
            )
        payload = json.loads(request.content or b"{}")
        method = payload.get("method")
        self.methods.append(str(method) if method else request.method)
        item = self._script[index] if index < len(self._script) else DEFAULT
        if item is not DEFAULT:
            if isinstance(item, BaseException):
                raise item
            if callable(item):
                return item(request, payload)
            return item
        if method == "initialize":
            result: Any = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "0"},
            }
        elif method == "tools/list":
            result = {"tools": []}
        else:
            result = {}
        return json_rpc_response(payload.get("id"), result, request=request)


def json_rpc_response(
    request_id: Any,
    result: Any = None,
    *,
    status_code: int = 200,
    request: httpx.Request | None = None,
) -> httpx.Response:
    """构造一个合法 JSON-RPC 响应。

    Args:
        request_id: 回填的请求 id。
        result: ``result`` 字段。
        status_code: HTTP 状态码。
        request: 关联的请求（可选）。

    Returns:
        :class:`httpx.Response`（``content-type: application/json``）。
    """
    body = {"jsonrpc": "2.0", "id": request_id, "result": result}
    return httpx.Response(
        status_code,
        headers={"content-type": "application/json"},
        json=body,
        request=request or httpx.Request("POST", SERVER_URL),
    )


def error_response(status_code: int, text: str = "boom") -> httpx.Response:
    """构造一个非 2xx 响应（4xx/5xx 重试分界测试用）。

    Args:
        status_code: 状态码。
        text: 响应体。

    Returns:
        :class:`httpx.Response`。
    """
    return httpx.Response(
        status_code,
        headers={"content-type": "text/plain"},
        text=text,
        request=httpx.Request("POST", SERVER_URL),
    )


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> SleepRecorder:
    """把 ``client_http`` 的退避睡眠注入为记录器。

    Args:
        monkeypatch: pytest 的 monkeypatch。

    Returns:
        记录器（断言精确退避序列）。
    """
    recorder = SleepRecorder()
    monkeypatch.setattr(mcp_module, "_retry_sleep", recorder)
    return recorder


def make_client(
    server: FakeServer, *, timeout_s: float = 30.0, url: str = SERVER_URL
) -> McpHttpClient:
    """构造一个走 MockTransport 的 Streamable HTTP 客户端（不触网）。

    Args:
        server: 假服务端。
        timeout_s: ``server.timeout_s``。
        url: 服务地址。

    Returns:
        :class:`~my_agent.mcp.client_http.McpHttpClient`。
    """
    config = McpServerConfig(name="fs", transport="http", url=url, timeout_s=timeout_s)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(server), headers={"content-type": "application/json"}
    )
    return McpHttpClient(config, client=client)


# --------------------------------------------------------------------------------------
# 可重试：传输层故障
# --------------------------------------------------------------------------------------


async def test_transport_error_is_retried_then_succeeds(sleeps: SleepRecorder) -> None:
    """连接失败 1 次后成功 → 重试生效：2 次 ``initialize`` + 1 次通知，退避 ``[0.5]``。"""
    server = FakeServer([httpx.ConnectError("connection refused"), DEFAULT])
    client = make_client(server)

    result = await client.initialize()

    assert result["protocolVersion"] == "2025-06-18"
    assert server.requests == 3
    assert server.methods == ["initialize", "initialize", "notifications/initialized"]
    assert sleeps.delays == [0.5]
    await client.close()


async def test_exhausted_transport_errors_raise_mcp_validation_error(
    sleeps: SleepRecorder,
) -> None:
    """连接一直失败 → ``initialize`` 恰好 3 次（不是 4 次），最终 ``McpValidationError``。"""
    server = FakeServer([httpx.ConnectError("boom")] * D.TOOL_RETRY_MAX_ATTEMPTS)
    client = make_client(server)

    with pytest.raises(McpValidationError) as raised:
        await client.initialize()

    assert server.requests == D.TOOL_RETRY_MAX_ATTEMPTS == 3
    assert server.methods == ["initialize", "initialize", "initialize"]
    assert sleeps.delays == [0.5, 1.0]
    assert "initialize 请求失败" in raised.value.detail
    await client.close()


async def test_server_error_status_is_retried_then_succeeds(sleeps: SleepRecorder) -> None:
    """503 一次后成功 → 重试生效：2 次 ``initialize`` + 1 次通知，退避 ``[0.5]``。"""
    server = FakeServer([error_response(503, "unavailable"), DEFAULT])
    client = make_client(server)

    result = await client.initialize()

    assert result["protocolVersion"] == "2025-06-18"
    assert server.methods == ["initialize", "initialize", "notifications/initialized"]
    assert sleeps.delays == [0.5]
    await client.close()


async def test_exhausted_server_errors_raise_mcp_validation_error(
    sleeps: SleepRecorder,
) -> None:
    """一直 503 → 3 次请求后 ``McpValidationError``，文案带状态码。"""
    server = FakeServer([error_response(503, "unavailable")] * D.TOOL_RETRY_MAX_ATTEMPTS)
    client = make_client(server)

    with pytest.raises(McpValidationError) as raised:
        await client.initialize()

    assert server.requests == 3
    assert sleeps.delays == [0.5, 1.0]
    assert "HTTP 503" in raised.value.detail
    await client.close()


# --------------------------------------------------------------------------------------
# 不可重试：4xx
# --------------------------------------------------------------------------------------


async def test_client_error_status_is_not_retried(sleeps: SleepRecorder) -> None:
    """404 是客户端错误 → **只 1 次请求**，立即 ``McpValidationError``。"""
    server = FakeServer([error_response(404, "no such method")])
    client = make_client(server)

    with pytest.raises(McpValidationError) as raised:
        await client.initialize()

    assert server.requests == 1
    assert sleeps.delays == []
    assert "HTTP 404" in raised.value.detail
    await client.close()


# --------------------------------------------------------------------------------------
# 不可重试：MCP 超时（§10.5，用户已批准的例外）
# --------------------------------------------------------------------------------------


async def test_mcp_timeout_is_not_retried_spec_10_5(sleeps: SleepRecorder) -> None:
    """**规格 §10.5：MCP 超时不重试** —— ``httpx.ReadTimeout`` 恰好只尝试 1 次。

    ``_post`` 在异常分派的第一支就把 ``httpx.TimeoutException`` 翻成
    ``McpTimeoutError``，而该异常在 ``is_retryable_network_error`` 判定链最前面即被排除，
    因此重试包装层拿不到任何可重试信号：**请求次数必须是 1，退避必须是空**。
    """
    server = FakeServer([httpx.ReadTimeout("server too slow")] * D.TOOL_RETRY_MAX_ATTEMPTS)
    client = make_client(server, timeout_s=30.0)

    with pytest.raises(McpTimeoutError) as raised:
        await client.initialize()

    assert server.requests == 1  # ← 超时是终态：不重试
    assert sleeps.delays == []  # ← 连一次退避都没有
    assert raised.value.server == "fs"
    assert raised.value.timeout == 30.0
    assert raised.value.method == "initialize"
    assert "超时" in raised.value.user_message()
    await client.close()


async def test_connect_timeout_is_not_retried_spec_10_5(sleeps: SleepRecorder) -> None:
    """``ConnectTimeout`` 同属超时族 → 同样只尝试 1 次（§10.5）。"""
    server = FakeServer([httpx.ConnectTimeout("connect timeout")] * D.TOOL_RETRY_MAX_ATTEMPTS)
    client = make_client(server)

    with pytest.raises(McpTimeoutError):
        await client.initialize()

    assert server.requests == 1
    assert sleeps.delays == []
    await client.close()


async def test_timeout_after_transport_retry_stops_immediately(
    sleeps: SleepRecorder,
) -> None:
    """先连接失败（已重试 1 次）再超时 → 超时**立刻终止**，共 2 次请求、1 次退避。"""
    server = FakeServer(
        [
            httpx.ConnectError("boom"),
            httpx.ReadTimeout("slow"),
            DEFAULT,
        ]
    )
    client = make_client(server)

    with pytest.raises(McpTimeoutError):
        await client.initialize()

    assert server.requests == 2
    assert sleeps.delays == [0.5]
    await client.close()


async def test_legacy_sse_post_timeout_is_not_retried(sleeps: SleepRecorder) -> None:
    """旧版 SSE 形态（``transport: sse``）：POST 阶段超时同样**不重试**。

    ``McpServerConfig.timeout_s`` 是整数，故直接把客户端生效超时改小，
    免得为了触发超时真的等 1 秒。
    """
    config = McpServerConfig(name="fs", transport="sse", url=SERVER_URL, timeout_s=1)
    server = FakeServer([httpx.ReadTimeout("too slow")] * D.TOOL_RETRY_MAX_ATTEMPTS)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(server), headers={"content-type": "application/json"}
    )
    legacy = McpHttpClient(config, client=client)
    try:
        with pytest.raises(McpTimeoutError) as raised:
            await legacy.initialize()

        assert raised.value.method == "initialize"
        assert sleeps.delays == []
    finally:
        await legacy.close()


# --------------------------------------------------------------------------------------
# 收尾路径不重试
# --------------------------------------------------------------------------------------


async def test_close_path_is_not_retried(sleeps: SleepRecorder) -> None:
    """``close()`` 走收尾路径：不发请求、不重试、不抛异常（可重复调用）。"""
    server = FakeServer()
    client = make_client(server)
    await client.initialize()
    calls_after_init = server.requests

    await client.close()
    await client.close()  # 幂等：不抛

    assert server.requests == calls_after_init
    assert sleeps.delays == []
    assert client.closed is True


async def test_request_after_close_fails_without_retry(sleeps: SleepRecorder) -> None:
    """已关闭的连接：立即 ``McpValidationError``，**不重试**（不自动重连，§10.5）。"""
    server = FakeServer()
    client = make_client(server)
    await client.close()
    before = server.requests

    with pytest.raises(McpValidationError) as raised:
        await client.initialize()

    assert server.requests == before
    assert sleeps.delays == []
    assert "连接已关闭" in raised.value.detail


async def test_session_header_recorded_on_first_successful_post(
    sleeps: SleepRecorder,
) -> None:
    """重试后仍记录 ``mcp-session-id``（重试不破坏会话记忆）。"""

    def session_reply(request: httpx.Request, payload: dict[str, Any]) -> httpx.Response:
        """回一个成功响应并下发会话 id（id 与本次请求配对）。

        Args:
            request: 收到的请求。
            payload: 请求体。

        Returns:
            带 ``mcp-session-id`` 的 JSON-RPC 响应。
        """
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "mcp-session-id": "sess-42"},
            json={
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "0"},
                },
            },
            request=request,
        )

    server = FakeServer([error_response(503, "unavailable"), session_reply])
    client = make_client(server)

    result = await client.initialize()

    assert result["protocolVersion"] == "2025-06-18"
    assert server.requests == 3
    assert sleeps.delays == [0.5]
    assert client.session_id == "sess-42"
    await client.close()
