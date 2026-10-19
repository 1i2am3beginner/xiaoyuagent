"""HTTP MCP 客户端（§10.5）。

支持两种远程形态（§10.1）：

* ``transport: http``（Streamable HTTP，推荐）：``POST <url>``，请求与响应同一连接，
  响应体可为 ``application/json``，也可为 ``text/event-stream``（SSE 单事件）；
* ``transport: sse``（旧版 HTTP+SSE，**兼容读取**）：``GET <url>`` 打开事件流，
  服务端先推 ``event: endpoint`` 告知 POST 地址，之后的响应从事件流回来。

两种形态共用同一套纪律：``httpx.AsyncClient``、``server.timeout_s`` 超时 →
:class:`~my_agent.errors.McpTimeoutError`（**不重试**）、``close()`` 关闭连接池与后台流任务、
进程/连接失效后**不自动重连**。

网络异常重试（工具层纪律，§14）：JSON-RPC ``POST`` 路径的连接失败 / 读失败 / 408 / 429 / 5xx
交给 :func:`~my_agent.retry.with_network_retry` 指数退避重试
（:data:`~my_agent.config.defaults.TOOL_RETRY_MAX_ATTEMPTS` 次）。**唯一的例外是超时**：

* ``httpx.TimeoutException`` 在 :meth:`McpHttpClient._post` 内就被翻译成
  :class:`~my_agent.errors.McpTimeoutError`，而该异常在
  :func:`~my_agent.retry.is_retryable_network_error` 判定链的**最前面**即被排除（§10.5），
  所以 MCP 超时**恰好只尝试 1 次**；
* ``_ensure_stream`` 等待 SSE ``endpoint`` 的超时同理（不经过重试包装）；
* ``close()`` / ``_stop_stream()`` 是收尾路径，**不包装**重试。

规格：``my_agent-开发文档.md`` §10.1、§10.5。
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from urllib.parse import urljoin

import httpx

from my_agent.config.defaults import TOOL_RETRY_BACKOFF_S, TOOL_RETRY_MAX_ATTEMPTS
from my_agent.config.schema import McpServerConfig
from my_agent.errors import McpTimeoutError, McpValidationError
from my_agent.mcp.schema import (
    MCP_CLIENT_NAME,
    MCP_CLIENT_VERSION,
    MCP_JSONRPC_VERSION,
    MCP_MAX_TOOL_PAGES,
    MCP_PROTOCOL_VERSION,
    McpTool,
    expand_server,
    map_mcp_tools,
    render_content,
    short_json,
)
from my_agent.models.tools import ToolResult
from my_agent.retry import retryable_status, with_network_retry

__all__ = ["McpHttpClient", "parse_sse_text"]

_LOG = logging.getLogger(__name__)

#: 模块级可注入睡眠（测试 monkeypatch 本对象即可跳过真实退避等待）。
_retry_sleep: Callable[[float], Awaitable[None]] | None = None


def parse_sse_text(text: str) -> list[tuple[str, str]]:
    """解析完整的 SSE 文本为 ``(event, data)`` 列表。

    Args:
        text: SSE 文本（``event:`` / ``data:`` / 空行分隔；``:`` 开头为注释）。

    Returns:
        事件列表；``event`` 缺省为 ``"message"``，多行 ``data`` 以 ``\\n`` 连接。
    """
    events: list[tuple[str, str]] = []
    event = ""
    data_lines: list[str] = []

    def flush() -> None:
        if data_lines:
            events.append((event or "message", "\n".join(data_lines)))

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith(":"):
            continue
        if not line:
            flush()
            event = ""
            data_lines = []
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    flush()
    return events


def decode_jsonrpc_response(
    response: httpx.Response, request_id: Any, *, method: str
) -> dict[str, Any]:
    """从 HTTP 响应里取出与 ``request_id`` 匹配的 JSON-RPC 消息。

    Args:
        response: ``httpx`` 响应对象。
        request_id: 本端请求 id。
        method: 方法名（仅用于错误消息）。

    Returns:
        JSON-RPC 消息字典（含 ``result`` 或 ``error``）。

    Raises:
        McpValidationError: HTTP 状态码异常、响应体非法或缺少匹配的响应。
    """
    status = response.status_code
    content_type = response.headers.get("content-type", "")
    if status >= 400:
        raise McpValidationError(
            detail=f"{method} 请求失败（HTTP {status}）：{short_json(response.text)}"
        )
    if status == 202:
        raise McpValidationError(
            detail=f"{method} 未返回响应体（HTTP 202 Accepted），该服务需改用 SSE 事件流"
        )
    messages: list[dict[str, Any]] = []
    if "text/event-stream" in content_type:
        for _event, data in parse_sse_text(response.text):
            message = _loads(data)
            if message is not None:
                messages.append(message)
    else:
        message = _loads(response.text)
        if message is not None:
            messages.append(message)
    if not messages:
        raise McpValidationError(
            detail=f"{method} 返回非法结构：无法解析 JSON-RPC 响应（{short_json(response.text)}）"
        )
    for message in messages:
        if message.get("id") == request_id and ("result" in message or "error" in message):
            return message
    single = messages[0]
    if len(messages) == 1 and ("result" in single or "error" in single) and "id" not in single:
        return single
    raise McpValidationError(detail=f"{method} 响应缺少匹配的 id={request_id}")


def _loads(text: str) -> dict[str, Any] | None:
    """把文本解析为 JSON 对象（失败或非对象 → ``None``）。

    Args:
        text: 待解析文本。

    Returns:
        解析到的对象；失败时为 ``None``。
    """
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class McpHttpClient:
    """远程 MCP 客户端（Streamable HTTP / 旧版 HTTP+SSE）。

    Args:
        server: 服务器配置（``transport`` 为 ``http`` 或 ``sse``）。
        env: ``${VAR}`` 展开用的环境变量映射（默认 ``os.environ``）。
        client: 复用外部 ``httpx.AsyncClient``（如带代理/自定义 CA 的实例）；
            ``None`` 时内部创建并负责关闭。

    Attributes:
        server_name: 服务器名。
        protocol_version: ``initialize`` 协商到的协议版本（未握手时为空串）。
        server_info: ``initialize`` 返回的 ``serverInfo``。
        session_id: Streamable HTTP 返回的 ``mcp-session-id``（若有）。
    """

    def __init__(
        self,
        server: McpServerConfig,
        *,
        env: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._server = server
        self._expanded, self._missing_vars = expand_server(server, env=env)
        self._url = (self._expanded.url or "").strip()
        self._legacy = self._expanded.transport == "sse"
        self._timeout = float(self._expanded.timeout_s)
        self._owns_client = client is None
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            **self._expanded.headers,
        }
        self._headers = headers
        self._client = (
            httpx.AsyncClient(timeout=self._timeout, headers=headers, follow_redirects=True)
            if client is None
            else client
        )
        self._ids = itertools.count(1)
        self._pending: dict[Any, asyncio.Future[dict[str, Any]]] = {}
        self._stream_task: asyncio.Task[None] | None = None
        self._endpoint_ready = asyncio.Event()
        self._post_url: str | None = None
        self._protocol_version = ""
        self._server_info: dict[str, Any] = {}
        self._session_id: str | None = None
        self._init_result: dict[str, Any] | None = None
        self._closed = False

    # ------------------------------------------------------------------ 只读属性

    @property
    def server(self) -> McpServerConfig:
        """展开 ``${VAR}`` 之后的服务器配置。"""
        return self._expanded

    @property
    def server_name(self) -> str:
        """服务器名。"""
        return self._expanded.name

    @property
    def protocol_version(self) -> str:
        """``initialize`` 协商到的协议版本。"""
        return self._protocol_version

    @property
    def server_info(self) -> dict[str, Any]:
        """``initialize`` 返回的 ``serverInfo``。"""
        return dict(self._server_info)

    @property
    def session_id(self) -> str | None:
        """Streamable HTTP 会话 id（服务端下发时记录）。"""
        return self._session_id

    @property
    def missing_env_vars(self) -> list[tuple[str, str]]:
        """未能解析的 ``${VAR}`` 位置：``[(变量名, 位置), ...]``。"""
        return list(self._missing_vars)

    @property
    def closed(self) -> bool:
        """是否已关闭（关闭后不再发起请求，**不自动重连**）。"""
        return self._closed

    def is_running(self) -> bool:
        """兼容 stdio 客户端的同名查询：未关闭即 ``True``。

        Returns:
            未关闭时为 ``True``。
        """
        return not self._closed

    # ------------------------------------------------------------------ 协议方法

    async def initialize(self) -> dict[str, Any]:
        """执行 MCP ``initialize`` 并发送 ``notifications/initialized``。

        Returns:
            ``initialize`` 的 ``result``。

        Raises:
            McpTimeoutError: 超过 ``server.timeout_s`` 未响应（不重试）。
            McpValidationError: HTTP/结构错误，或缺少 ``protocolVersion``。
        """
        if self._init_result is not None:
            return self._init_result
        params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "clientInfo": {"name": MCP_CLIENT_NAME, "version": MCP_CLIENT_VERSION},
        }
        result = await self._request("initialize", params)
        if not isinstance(result, dict):
            raise McpValidationError(detail="initialize 返回非法结构：result 不是对象")
        version = result.get("protocolVersion")
        if not isinstance(version, str) or not version:
            raise McpValidationError(detail="initialize 返回非法结构：缺少 protocolVersion")
        self._protocol_version = version
        info = result.get("serverInfo")
        self._server_info = info if isinstance(info, dict) else {}
        await self._notify("notifications/initialized", {})
        self._init_result = result
        return result

    async def list_tools(self) -> list[McpTool]:
        """执行 ``tools/list``（跟随 ``nextCursor`` 分页）并映射为 :class:`McpTool`。

        Returns:
            映射后的工具列表（§10.4）。

        Raises:
            McpTimeoutError: 超过 ``server.timeout_s`` 未响应。
            McpValidationError: 返回结构非法（缺少 ``result.tools``）。
        """
        raws: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(MCP_MAX_TOOL_PAGES):
            params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
            result = await self._request("tools/list", params)
            if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
                raise McpValidationError(detail="tools/list 返回非法结构：缺少 result.tools")
            raws.extend(item for item in result["tools"] if isinstance(item, dict))
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            cursor = next_cursor
        return map_mcp_tools(self._expanded, raws)

    async def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """执行 ``tools/call``。

        Args:
            name: MCP 侧原始工具名。
            args: 工具参数。

        Returns:
            :class:`ToolResult`（``call_id`` 由 manager 回填）。

        Raises:
            McpTimeoutError: 超过 ``server.timeout_s`` 未响应（不重试）。
            McpValidationError: HTTP/JSON-RPC 错误。
        """
        result = await self._request("tools/call", {"name": name, "arguments": args})
        if not isinstance(result, dict):
            raise McpValidationError(detail=f'工具 "{name}" 返回非法结构：result 不是对象')
        text = render_content(result.get("content"))
        is_error = bool(result.get("isError"))
        return ToolResult(
            call_id="",
            tool=name,
            ok=not is_error,
            content="" if is_error else text,
            error=text if is_error else None,
        )

    async def close(self) -> None:
        """关闭后台事件流任务与自建连接池（可重复调用；**不自动重连**）。"""
        self._closed = True
        self._fail_pending(self._closed_error())
        await self._stop_stream()
        if self._owns_client:
            with contextlib.suppress(httpx.HTTPError, RuntimeError):
                await self._client.aclose()

    # ------------------------------------------------------------------ 内部实现

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        """发送 JSON-RPC 请求并等待响应（两种传输形态统一入口）。

        网络异常（连接失败 / 读失败 / 408 / 429 / 5xx）按 ``TOOL_RETRY_*`` 指数退避重试；
        **超时例外**：``_post`` 把 ``httpx.TimeoutException`` 翻成
        :class:`~my_agent.errors.McpTimeoutError`，该异常被
        :func:`~my_agent.retry.is_retryable_network_error` 排除，因此只尝试 1 次（§10.5）。

        Args:
            method: MCP 方法名。
            params: 方法参数。

        Returns:
            响应中的 ``result``。

        Raises:
            McpTimeoutError: 超时（**不重试**，§10.5）。
            McpValidationError: HTTP/JSON-RPC 错误、重试耗尽的网络错误，或连接已关闭。
        """
        if self._closed:
            raise self._closed_error()
        # 请求 id 在**重试之外**分配：重试是同一个逻辑请求的重发，
        # 复用同一 id 才能与服务端回包正确配对（也是 JSON-RPC 的语义）。
        request_id = next(self._ids)
        try:
            return await with_network_retry(
                lambda: self._request_once(method, params, request_id),
                attempts=TOOL_RETRY_MAX_ATTEMPTS,
                schedule=TOOL_RETRY_BACKOFF_S,
                sleep=_retry_sleep,
                label=f"MCP {method}",
            )
        except httpx.TimeoutException as exc:
            # 兜底：``_post`` 已把超时翻成 McpTimeoutError；若仍有 httpx 超时逃出，
            # 一并按 §10.5 收敛成 McpTimeoutError（**不重试**）。
            raise McpTimeoutError(
                server=self.server_name, timeout=self._timeout, method=method
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise McpValidationError(
                detail=(
                    f"{method} 请求失败（HTTP {exc.response.status_code}）："
                    f"{short_json(exc.response.text)}"
                )
            ) from exc
        except httpx.HTTPError as exc:
            raise McpValidationError(
                detail=f"{method} 请求失败（{self.server_name}）：{exc}"
            ) from exc

    async def _request_once(self, method: str, params: dict[str, Any], request_id: Any) -> Any:
        """发送一次 JSON-RPC 请求（不做重试；供 :meth:`_request` 包装）。

        Args:
            method: MCP 方法名。
            params: 方法参数。
            request_id: 本次逻辑请求的 id（重试时保持不变）。

        Returns:
            响应中的 ``result``。

        Raises:
            McpTimeoutError: 超时（不重试）。
            McpValidationError: JSON-RPC 错误。
            httpx.HTTPError: 网络/协议错误或可重试的非 2xx 状态（由重试包装层处理）。
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        payload = {
            "jsonrpc": MCP_JSONRPC_VERSION,
            "id": request_id,
            "method": method,
            "params": params,
        }
        try:
            if self._legacy:
                self._pending[request_id] = future
                await self._ensure_stream()
                await self._post(self._post_url or self._url, payload, method)
                try:
                    message = await asyncio.wait_for(future, timeout=self._timeout)
                except TimeoutError:
                    raise McpTimeoutError(
                        server=self.server_name, timeout=self._timeout, method=method
                    ) from None
            else:
                response = await self._post(self._url, payload, method)
                message = decode_jsonrpc_response(response, request_id, method=method)
        finally:
            self._pending.pop(request_id, None)
        error = message.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            text = error.get("message") or ""
            detail = f"{method} 返回错误 {code}：{text}"
            if error.get("data") is not None:
                detail = f"{detail}（{short_json(error.get('data'))}）"
            raise McpValidationError(detail=detail)
        return message.get("result")

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """发送 JSON-RPC 通知（无 id；服务端通常回 202）。

        Args:
            method: 通知方法名。
            params: 通知参数。
        """
        if self._closed:
            raise self._closed_error()
        payload = {"jsonrpc": MCP_JSONRPC_VERSION, "method": method, "params": params}
        if self._legacy:
            await self._ensure_stream()
            await self._post(self._post_url or self._url, payload, method)
            return
        await self._post(self._url, payload, method)

    async def _post(self, url: str, payload: dict[str, Any], method: str) -> httpx.Response:
        """POST 一条 JSON-RPC 消息。

        异常分派是重试纪律的关键：

        * ``httpx.TimeoutException`` → :class:`~my_agent.errors.McpTimeoutError`：
          **本方法内就定案**，绝不交给重试层（§10.5）；
        * ``httpx.TransportError``（连接失败 / 读失败 / 协议错乱）→ **原样抛出**，
          由 :meth:`_request` 的退避包装层重试；
        * 可重试的非 2xx（408 / 429 / 5xx）→ 抛 ``httpx.HTTPStatusError``（含 response），
          同样由重试层处理；其它非 2xx（4xx）→ :class:`McpValidationError`（立即失败）。

        Args:
            url: 目标地址。
            payload: 消息体。
            method: 方法名（错误消息用）。

        Returns:
            HTTP 响应。

        Raises:
            McpTimeoutError: 超时（**不重试**，§10.5）。
            McpValidationError: 不可重试的网络/HTTP 错误（4xx、无效 URL 等）。
            httpx.HTTPStatusError: 可重试的非 2xx 状态码（408 / 429 / 5xx）。
            httpx.TransportError: 可重试的连接/读取故障。
        """
        try:
            response = await self._client.post(
                url, json=payload, headers=self._headers, timeout=self._timeout
            )
        except httpx.TimeoutException:
            # §10.5：MCP 超时是终态，无论重试层如何判定都只尝试 1 次。
            raise McpTimeoutError(
                server=self.server_name, timeout=self._timeout, method=method
            ) from None
        except httpx.TransportError:
            # 交给 with_network_retry 决定重试；耗尽后由 _request 翻成 McpValidationError。
            raise
        except httpx.HTTPError as exc:
            raise McpValidationError(
                detail=f"{method} 请求失败（{self.server_name}）：{exc}"
            ) from exc
        self._remember_session(response)
        if response.status_code >= 400:
            if retryable_status(response.status_code):
                response.raise_for_status()
            raise McpValidationError(
                detail=(
                    f"{method} 请求失败（HTTP {response.status_code}）：{short_json(response.text)}"
                )
            )
        return response

    def _remember_session(self, response: httpx.Response) -> None:
        """记录 Streamable HTTP 的会话 id 与协议版本（后续请求回传）。

        Args:
            response: 服务端响应。
        """
        session = response.headers.get("mcp-session-id")
        if isinstance(session, str) and session:
            self._session_id = session
            self._headers["mcp-session-id"] = session
        if self._protocol_version:
            self._headers["mcp-protocol-version"] = self._protocol_version

    async def _ensure_stream(self) -> None:
        """确保旧版 SSE 事件流已建立并拿到 ``endpoint``。

        Raises:
            McpTimeoutError: 等待 ``endpoint`` 超时。
            McpValidationError: 事件流未返回 ``endpoint`` 或已断开。
        """
        if self._stream_task is not None and not self._stream_task.done():
            if self._post_url:
                return
        else:
            self._endpoint_ready.clear()
            self._post_url = None
            self._stream_task = asyncio.create_task(self._read_stream())
        try:
            await asyncio.wait_for(self._endpoint_ready.wait(), timeout=self._timeout)
        except TimeoutError:
            raise McpTimeoutError(
                server=self.server_name, timeout=self._timeout, method="SSE endpoint"
            ) from None
        if not self._post_url:
            raise McpValidationError(
                detail=f'服务器 "{self.server_name}" 的 SSE 事件流未返回 endpoint'
            )

    async def _read_stream(self) -> None:
        """后台读取旧版 SSE 事件流（``endpoint`` 与 ``message`` 事件）。"""
        stream_timeout = httpx.Timeout(
            connect=self._timeout, read=None, write=self._timeout, pool=self._timeout
        )
        headers = {**self._headers, "accept": "text/event-stream"}
        try:
            async with self._client.stream(
                "GET", self._url, headers=headers, timeout=stream_timeout
            ) as response:
                if response.status_code >= 400:
                    self._fail_pending(
                        McpValidationError(
                            detail=(
                                f'服务器 "{self.server_name}" 的 SSE 连接失败'
                                f"（HTTP {response.status_code}）"
                            )
                        )
                    )
                    return
                event = ""
                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if line.startswith(":"):
                        continue
                    if not line:
                        if data_lines:
                            self._handle_sse_event(event or "message", "\n".join(data_lines))
                        event, data_lines = "", []
                        continue
                    field, _, value = line.partition(":")
                    value = value[1:] if value.startswith(" ") else value
                    if field == "event":
                        event = value
                    elif field == "data":
                        data_lines.append(value)
        except httpx.HTTPError as exc:
            self._fail_pending(
                McpValidationError(detail=f'服务器 "{self.server_name}" 的 SSE 连接中断：{exc}')
            )
        finally:
            self._endpoint_ready.set()

    def _handle_sse_event(self, event: str, data: str) -> None:
        """处理一个 SSE 事件（``endpoint`` 记地址，``message`` 分发响应）。

        Args:
            event: 事件名。
            data: 事件数据。
        """
        if event == "endpoint":
            self._post_url = urljoin(self._url, data.strip())
            self._endpoint_ready.set()
            _LOG.debug("MCP 服务器 %s SSE endpoint=%s", self.server_name, self._post_url)
            return
        message = _loads(data)
        if message is None:
            _LOG.debug("MCP 服务器 %s SSE 非 JSON 事件：%s", self.server_name, data[:200])
            return
        if "id" in message and ("result" in message or "error" in message):
            future = self._pending.get(message.get("id"))
            if future is not None and not future.done():
                future.set_result(message)
            return
        _LOG.debug("MCP 服务器 %s SSE 忽略事件：%s", self.server_name, event)

    async def _stop_stream(self) -> None:
        """取消并回收后台事件流任务。"""
        task = self._stream_task
        if task is None:
            return
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError):
            await task

    def _fail_pending(self, exc: McpValidationError) -> None:
        """把异常广播给所有等待中的请求。

        Args:
            exc: 广播的异常。
        """
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()

    def _closed_error(self) -> McpValidationError:
        """构造「连接已关闭」错误。

        Returns:
            连接已关闭的 :class:`McpValidationError`。
        """
        return McpValidationError(detail=f'MCP 服务器 "{self.server_name}" 的连接已关闭')
