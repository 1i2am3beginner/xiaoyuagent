"""stdio MCP 客户端（§10.5）。

要点（逐条对应硬性要求）：

* ``asyncio.create_subprocess_exec`` 启动子进程，JSON-RPC 逐行（newline-delimited JSON）；
* **stderr 由独立任务消费**：子进程写满 stderr 管道会死锁，故 :meth:`McpStdioClient._drain_stderr`
  持续读取，日志转 ``logging.debug``，内存只保留尾部 2k；
* ``initialize`` 协商并记录服务端 ``protocolVersion`` / ``serverInfo``；
* Windows 下 ``CREATE_NO_WINDOW``，避免弹出控制台窗口；
* 单次请求受 ``server.timeout_s`` 约束：超时 → :class:`~my_agent.errors.McpTimeoutError`
  （**不重试**）；
* 进程退出 → 记录 ``exit_code`` + stderr 尾部 2k，**不自动重启**（避免崩溃循环）；
* :meth:`close` = ``terminate()`` → 等待 ``MCP_TERMINATE_GRACE_S`` → ``kill()``。

规格：``my_agent-开发文档.md`` §10.5。
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import os
import subprocess
import time
from collections.abc import Mapping
from typing import Any

from my_agent.config.defaults import MCP_TERMINATE_GRACE_S
from my_agent.config.schema import McpServerConfig
from my_agent.errors import McpTimeoutError, McpValidationError
from my_agent.mcp.schema import (
    MCP_CLIENT_NAME,
    MCP_CLIENT_VERSION,
    MCP_JSONRPC_VERSION,
    MCP_MAX_TOOL_PAGES,
    MCP_PROTOCOL_VERSION,
    MCP_STDERR_CHUNK_BYTES,
    MCP_STDERR_TAIL_CHARS,
    MCP_STREAM_LIMIT_BYTES,
    McpTool,
    expand_server,
    map_mcp_tools,
    render_content,
    short_json,
)
from my_agent.models.tools import ToolResult

__all__ = ["McpStdioClient"]

_LOG = logging.getLogger(__name__)


class McpStdioClient:
    """通过 stdin/stdout 与本地子进程通信的 MCP 客户端（§10.5）。

    Args:
        server: 服务器配置（``transport`` 必须为 ``stdio``；``${VAR}`` 会先展开）。
        env: ``${VAR}`` 展开用的环境变量映射（默认 ``os.environ``）；子进程环境固定为
            ``os.environ`` 叠加 ``server.env``。
        stderr_tail_chars: 保留的 stderr 尾部字符数（默认 §10.5 的 2k）。

    Attributes:
        server_name: 服务器名（错误消息用）。
        protocol_version: ``initialize`` 协商到的服务端协议版本（未握手时为空串）。
        server_info: ``initialize`` 返回的 ``serverInfo``。
        exit_code: 子进程退出码（未退出时 ``None``）。
        stderr_tail: stderr 尾部文本（最多 ``stderr_tail_chars`` 字符）。
    """

    def __init__(
        self,
        server: McpServerConfig,
        *,
        env: Mapping[str, str] | None = None,
        stderr_tail_chars: int = MCP_STDERR_TAIL_CHARS,
    ) -> None:
        self._server = server
        self._expand_source = env
        self._expanded, self._missing_vars = expand_server(server, env=env)
        self._stderr_tail_chars = stderr_tail_chars
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[Any, asyncio.Future[dict[str, Any]]] = {}
        self._ids = itertools.count(1)
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._protocol_version = ""
        self._server_info: dict[str, Any] = {}
        self._init_result: dict[str, Any] | None = None
        self._exit_code: int | None = None
        self._stderr_tail = ""
        self._closing = False
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
        """``initialize`` 协商到的服务端协议版本。"""
        return self._protocol_version

    @property
    def server_info(self) -> dict[str, Any]:
        """``initialize`` 返回的 ``serverInfo``。"""
        return dict(self._server_info)

    @property
    def exit_code(self) -> int | None:
        """子进程退出码（仍在运行或尚未启动时为 ``None``）。"""
        return self._exit_code

    @property
    def stderr_tail(self) -> str:
        """stderr 尾部文本（最多 2k，供错误消息与审计）。"""
        return self._stderr_tail

    @property
    def missing_env_vars(self) -> list[tuple[str, str]]:
        """未能解析的 ``${VAR}`` 位置：``[(变量名, 位置), ...]``。"""
        return list(self._missing_vars)

    @property
    def pid(self) -> int | None:
        """子进程 pid（未启动时为 ``None``）。"""
        return None if self._process is None else self._process.pid

    def is_running(self) -> bool:
        """子进程是否仍在运行。

        Returns:
            已启动且未退出时为 ``True``。
        """
        return self._process is not None and self._process.returncode is None

    # ------------------------------------------------------------------ 协议方法

    async def initialize(self) -> dict[str, Any]:
        """执行 MCP ``initialize`` 并发送 ``notifications/initialized``。

        Returns:
            ``initialize`` 的 ``result``。

        Raises:
            McpTimeoutError: 超过 ``server.timeout_s`` 未响应（不重试）。
            McpValidationError: 进程已退出，或返回结构非法（缺 ``protocolVersion``）。
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
        _LOG.debug(
            "MCP 服务器 %s 握手完成：protocolVersion=%s serverInfo=%s",
            self.server_name,
            version,
            self._server_info,
        )
        return result

    async def list_tools(self) -> list[McpTool]:
        """执行 ``tools/list``（跟随 ``nextCursor`` 分页）并映射为 :class:`McpTool`。

        Returns:
            映射后的工具列表（§10.4）。

        Raises:
            McpTimeoutError: 超过 ``server.timeout_s`` 未响应。
            McpValidationError: 返回结构非法（缺少 ``result.tools``）或进程已退出。
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
            name: MCP 侧**原始**工具名。
            args: 工具参数。

        Returns:
            :class:`ToolResult`（``call_id`` 由 manager 回填；``isError=true`` →
            ``ok=False``、``error`` 为内容文本）。

        Raises:
            McpTimeoutError: 超过 ``server.timeout_s`` 未响应（不重试）。
            McpValidationError: 进程已退出或 JSON-RPC 返回错误。
        """
        started = time.perf_counter()
        result = await self._request("tools/call", {"name": name, "arguments": args})
        duration_ms = int((time.perf_counter() - started) * 1000)
        if not isinstance(result, dict):
            raise McpValidationError(detail=f'工具 "{name}" 返回非法结构：result 不是对象')
        text = render_content(result.get("content"))
        if result.get("structuredContent") is not None:
            text = _append_structured(text, result.get("structuredContent"))
        is_error = bool(result.get("isError"))
        return ToolResult(
            call_id="",
            tool=name,
            ok=not is_error,
            content="" if is_error else text,
            error=text if is_error else None,
            duration_ms=duration_ms,
        )

    async def close(self) -> None:
        """清理子进程：``terminate()`` → 等待 ``MCP_TERMINATE_GRACE_S`` → ``kill()``。

        可重复调用；关闭后本客户端不再启动新进程（**不做自动重启**）。
        """
        self._closing = True
        self._closed = True
        self._fail_pending(self._closed_error())
        proc = self._process
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(OSError):
                proc.terminate()
            await self._wait_or_kill(proc)
        if proc is not None:
            self._exit_code = proc.returncode
        await self._stop_tasks()

    # ------------------------------------------------------------------ 内部实现

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        """发送 JSON-RPC 请求并等待同 id 的响应。

        Args:
            method: MCP 方法名。
            params: 方法参数。

        Returns:
            响应中的 ``result``。

        Raises:
            McpTimeoutError: 超时（不重试）。
            McpValidationError: 进程退出、写入失败或 JSON-RPC 错误响应。
        """
        await self._ensure_process()
        request_id = next(self._ids)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        timeout_s = float(self._server.timeout_s)
        try:
            await self._send(
                {"jsonrpc": MCP_JSONRPC_VERSION, "id": request_id, "method": method, "params": params}
            )
            message = await asyncio.wait_for(future, timeout=timeout_s)
        except TimeoutError:
            raise McpTimeoutError(
                server=self.server_name, timeout=timeout_s, method=method
            ) from None
        finally:
            self._pending.pop(request_id, None)
        error = message.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            text = error.get("message") or ""
            data = error.get("data")
            detail = f"{method} 返回错误 {code}：{text}"
            if data is not None:
                detail = f"{detail}（{short_json(data)}）"
            raise McpValidationError(detail=detail)
        return message.get("result")

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """发送 JSON-RPC 通知（无 id、不等响应）。

        Args:
            method: 通知方法名。
            params: 通知参数。
        """
        await self._send({"jsonrpc": MCP_JSONRPC_VERSION, "method": method, "params": params})

    async def _send(self, payload: dict[str, Any]) -> None:
        """把一条 JSON-RPC 消息写入子进程 stdin（带锁，按行分隔）。

        Args:
            payload: JSON-RPC 消息。

        Raises:
            McpValidationError: 连接已关闭或管道已断开。
        """
        proc = self._process
        if self._closed or proc is None or proc.stdin is None:
            raise self._closed_error()
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
        async with self._write_lock:
            try:
                proc.stdin.write(data)
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError, RuntimeError, OSError) as exc:
                raise self._exited_error(f"写入 stdin 失败：{exc}") from exc

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        """确保子进程已启动（**仅启动一次**，已退出则报错，不自动重启）。

        Returns:
            运行中的子进程。

        Raises:
            McpValidationError: 客户端已关闭、进程已退出或无法启动。
        """
        if self._closed:
            raise self._closed_error()
        proc = self._process
        if proc is not None:
            if proc.returncode is None:
                return proc
            raise self._exited_error()
        self._process = await self._spawn()
        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        return self._process

    async def _spawn(self) -> asyncio.subprocess.Process:
        """启动 stdio 子进程。

        Returns:
            新建子进程。

        Raises:
            McpValidationError: 命令缺失或无法启动。
        """
        command = self._expanded.command or ""
        if not command:
            raise McpValidationError(detail=f'服务器 "{self.server_name}" 缺少 command')
        argv = [command, *self._expanded.args]
        child_env = dict(os.environ)
        child_env.update(self._expanded.env)
        kwargs: dict[str, Any] = {
            "stdin": asyncio.subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "env": child_env,
            "limit": MCP_STREAM_LIMIT_BYTES,
        }
        if self._expanded.cwd:
            kwargs["cwd"] = self._expanded.cwd
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        kwargs["creationflags"] = flags
        try:
            proc = await asyncio.create_subprocess_exec(*argv, **kwargs)
        except (OSError, ValueError) as exc:
            raise McpValidationError(
                detail=f'无法启动 MCP 服务器 "{self.server_name}"（{command}）：{exc}'
            ) from exc
        _LOG.debug(
            "MCP 服务器 %s 已启动 pid=%s cmd=%s", self.server_name, proc.pid, " ".join(argv)
        )
        return proc

    async def _read_stdout(self) -> None:
        """读取 stdout 的 NDJSON 响应并分发（独立任务，EOF 即进程结束）。"""
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        while True:
            try:
                line = await proc.stdout.readline()
            except (asyncio.LimitOverrunError, ValueError) as exc:
                self._fail_pending(
                    McpValidationError(
                        detail=f'服务器 "{self.server_name}" 返回的单行消息超过上限：{exc}'
                    )
                )
                return
            except (BrokenPipeError, ConnectionResetError, OSError):
                break
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                _LOG.debug("MCP 服务器 %s stdout 非 JSON 行：%s", self.server_name, text[:200])
                continue
            if isinstance(message, dict):
                await self._dispatch(message)
        await self._on_stream_end()

    async def _dispatch(self, message: dict[str, Any]) -> None:
        """分发一条 JSON-RPC 消息（响应 / 服务端请求 / 通知）。

        Args:
            message: 已解析的消息。
        """
        if "id" in message and ("result" in message or "error" in message):
            future = self._pending.get(message.get("id"))
            if future is not None and not future.done():
                future.set_result(message)
            return
        method = message.get("method")
        if "id" in message and isinstance(method, str):
            await self._handle_server_request(message["id"], method)
            return
        if isinstance(method, str):
            _LOG.debug("MCP 服务器 %s 通知：%s", self.server_name, method)

    async def _handle_server_request(self, request_id: Any, method: str) -> None:
        """回应服务端发来的请求（当前只支持 ``ping``，其余回 -32601）。

        Args:
            request_id: 服务端请求 id。
            method: 方法名。
        """
        if method == "ping":
            payload: dict[str, Any] = {"jsonrpc": MCP_JSONRPC_VERSION, "id": request_id, "result": {}}
        else:
            payload = {
                "jsonrpc": MCP_JSONRPC_VERSION,
                "id": request_id,
                "error": {"code": -32601, "message": f"方法未实现：{method}"},
            }
        with contextlib.suppress(McpValidationError):
            await self._send(payload)

    async def _drain_stderr(self) -> None:
        """**独立**消费 stderr（防死锁，§10.5）：日志转 debug，内存只留尾部 2k。"""
        proc = self._process
        if proc is None or proc.stderr is None:
            return
        while True:
            try:
                chunk = await proc.stderr.read(MCP_STDERR_CHUNK_BYTES)
            except (BrokenPipeError, ConnectionResetError, OSError):
                break
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            self._stderr_tail = (self._stderr_tail + text)[-self._stderr_tail_chars :]
            _LOG.debug("MCP 服务器 %s stderr：%s", self.server_name, text.rstrip())

    async def _on_stream_end(self) -> None:
        """stdout EOF：等待进程收敛，记录退出码并唤醒所有等待者。"""
        proc = self._process
        if proc is None:
            return
        with contextlib.suppress(OSError):
            await proc.wait()
        await self._settle_exit(proc)

    async def _settle_exit(self, proc: asyncio.subprocess.Process) -> None:
        """记录退出码；非主动关闭时把退出作为错误广播给等待者。

        Args:
            proc: 已结束的子进程。
        """
        self._exit_code = proc.returncode
        if self._closing:
            return
        _LOG.warning(
            "MCP 服务器 %s 进程意外退出（exit_code=%s），标记 unhealthy 且不自动重启",
            self.server_name,
            proc.returncode,
        )
        self._fail_pending(self._exited_error())

    async def _wait_or_kill(self, proc: asyncio.subprocess.Process) -> None:
        """``terminate`` 后等待宽限期，超时则 ``kill``（§10.5）。

        Args:
            proc: 待回收的子进程。
        """
        try:
            await asyncio.wait_for(proc.wait(), timeout=MCP_TERMINATE_GRACE_S)
            return
        except TimeoutError:
            _LOG.debug("MCP 服务器 %s 未在宽限期内退出，改为 kill", self.server_name)
        with contextlib.suppress(OSError):
            proc.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=MCP_TERMINATE_GRACE_S)

    async def _stop_tasks(self) -> None:
        """取消并回收 stdout / stderr 读取任务。"""
        tasks = [task for task in (self._stdout_task, self._stderr_task) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, OSError):
                await task

    def _fail_pending(self, exc: McpValidationError, *, cancel: bool = False) -> None:
        """把异常（或取消）广播给所有等待中的请求。

        Args:
            exc: 广播的异常。
            cancel: ``True`` 时改为取消（主动关闭路径）。
        """
        for future in list(self._pending.values()):
            if future.done():
                continue
            if cancel:
                future.cancel()
            else:
                future.set_exception(exc)
        self._pending.clear()

    def _exited_error(self, prefix: str = "") -> McpValidationError:
        """构造「进程已退出」错误（含 exit_code 与 stderr 尾部 2k，§10.5）。

        Args:
            prefix: 可选的补充前缀。

        Returns:
            带退出码与 stderr 尾部的 :class:`McpValidationError`。
        """
        code = self._exit_code
        if code is None and self._process is not None:
            code = self._process.returncode
        parts = [prefix] if prefix else []
        parts.append(f'MCP 服务器 "{self.server_name}" 进程已退出（exit_code={code}）')
        tail = self._stderr_tail.strip()
        if tail:
            parts.append(f"stderr 尾部：{tail}")
        return McpValidationError(detail="；".join(parts))

    def _closed_error(self) -> McpValidationError:
        """构造「连接已关闭」错误。

        Returns:
            连接已关闭的 :class:`McpValidationError`。
        """
        return McpValidationError(detail=f'MCP 服务器 "{self.server_name}" 的连接已关闭')


def _append_structured(text: str, structured: Any) -> str:
    """把 ``structuredContent`` 追加到文本结果后（保持模型可见性）。

    Args:
        text: 已渲染的文本。
        structured: ``structuredContent`` 值。

    Returns:
        追加后的文本。
    """
    blob = short_json(structured)
    return f"{text}\n{blob}" if text else blob

