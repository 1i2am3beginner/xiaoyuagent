#!/usr/bin/env python3
"""测试用的最小 MCP stdio 服务器（无第三方依赖、可离线运行）。

用于 ``tests/integration/test_mcp_stdio.py`` 覆盖 §10.5 的硬性要求：
stderr 洪泛不死锁（F-5）、调用超时后进程被清理（F-6）、进程退出错误信息等。

用法（由测试通过 ``McpServerConfig.args`` 传入）::

    python fake_mcp_server.py [--name NAME] [--tools a,b,c]
        [--protocol-version V] [--stderr-bytes N] [--stderr-marker TEXT]
        [--hang-call] [--bad-tools-list] [--bad-initialize]
        [--crash-on-call CODE] [--exit-after N] [--exit-code CODE]
        [--delay-call S] [--log PATH]

协议：JSON-RPC 2.0 逐行（newline-delimited JSON），实现
``initialize`` / ``notifications/initialized`` / ``tools/list`` / ``tools/call`` / ``ping``。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

JSONRPC = "2.0"
DEFAULT_TOOLS = ("read_file", "delete_all", "weird_tool")


class Options:
    """命令行选项（解析结果）。"""

    def __init__(self, argv: list[str]) -> None:
        self.name = "fake"
        self.tools = list(DEFAULT_TOOLS)
        self.protocol_version = "2024-11-05"
        self.stderr_bytes = 0
        self.stderr_marker = ""
        self.hang_call = False
        self.bad_tools_list = False
        self.bad_initialize = False
        self.crash_on_call: int | None = None
        self.exit_after = 0
        self.exit_code = 0
        self.delay_call = 0.0
        self.log: Path | None = None
        index = 0
        while index < len(argv):
            flag = argv[index]
            value = argv[index + 1] if index + 1 < len(argv) else ""
            if flag == "--name":
                self.name = value
                index += 2
            elif flag == "--tools":
                self.tools = [item for item in value.split(",") if item]
                index += 2
            elif flag == "--protocol-version":
                self.protocol_version = value
                index += 2
            elif flag == "--stderr-bytes":
                self.stderr_bytes = int(value)
                index += 2
            elif flag == "--stderr-marker":
                self.stderr_marker = value
                index += 2
            elif flag == "--hang-call":
                self.hang_call = True
                index += 1
            elif flag == "--bad-tools-list":
                self.bad_tools_list = True
                index += 1
            elif flag == "--bad-initialize":
                self.bad_initialize = True
                index += 1
            elif flag == "--crash-on-call":
                self.crash_on_call = int(value)
                index += 2
            elif flag == "--exit-after":
                self.exit_after = int(value)
                index += 2
            elif flag == "--exit-code":
                self.exit_code = int(value)
                index += 2
            elif flag == "--delay-call":
                self.delay_call = float(value)
                index += 2
            elif flag == "--log":
                self.log = Path(value)
                index += 2
            else:
                index += 1


def flood_stderr(total: int, marker: str) -> None:
    """向 stderr 写 ``total`` 字节（父进程不消费时必然阻塞 → 死锁场景）。

    Args:
        total: 总字节数。
        marker: 洪泛结束后写入的标记文本（用于断言「尾部 2k」捕获）。
    """
    chunk = ("stderr-flood-line-" + "x" * 84 + "\n").encode("utf-8")
    written = 0
    while written < total:
        size = min(len(chunk), total - written)
        sys.stderr.buffer.write(chunk[:size])
        written += size
    if marker:
        sys.stderr.buffer.write(marker.encode("utf-8"))
    sys.stderr.buffer.flush()


def tool_specs(names: list[str]) -> list[dict[str, Any]]:
    """构造 ``tools/list`` 的工具声明。

    Args:
        names: 工具名列表。

    Returns:
        MCP 工具对象列表。
    """
    return [
        {
            "name": name,
            "description": f"fake MCP tool {name}",
            "inputSchema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": [],
            },
        }
        for name in names
    ]


def send(payload: dict[str, Any]) -> None:
    """写出一条 JSON-RPC 消息并 flush。

    Args:
        payload: 消息体。
    """
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log_call(path: Path | None, method: str) -> None:
    """把方法名追加到日志文件（测试据此断言「不重试」）。

    Args:
        path: 日志路径；``None`` 时跳过。
        method: 方法名。
    """
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(method + "\n")


def main(argv: list[str]) -> int:
    """服务器主循环。

    Args:
        argv: 命令行参数（不含脚本名）。

    Returns:
        进程退出码。
    """
    options = Options(argv)
    if options.stderr_bytes or options.stderr_marker:
        flood_stderr(options.stderr_bytes, options.stderr_marker)

    handled = 0
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict):
            continue
        method = message.get("method")
        request_id = message.get("id")
        if method == "notifications/initialized" or request_id is None:
            continue
        handled += 1
        log_call(options.log, str(method))

        if method == "initialize":
            result: dict[str, Any] = {
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": options.name, "version": "1.0.0"},
            }
            if not options.bad_initialize:
                result["protocolVersion"] = options.protocol_version
            send({"jsonrpc": JSONRPC, "id": request_id, "result": result})
        elif method == "tools/list":
            if options.bad_tools_list:
                send({"jsonrpc": JSONRPC, "id": request_id, "result": {"nope": []}})
            else:
                send(
                    {
                        "jsonrpc": JSONRPC,
                        "id": request_id,
                        "result": {"tools": tool_specs(options.tools)},
                    }
                )
        elif method == "tools/call":
            if options.hang_call:
                while True:
                    time.sleep(3600)
            if options.crash_on_call is not None:
                sys.stderr.write("fake-mcp: tool call failed\n")
                sys.stderr.flush()
                return options.crash_on_call
            if options.delay_call:
                time.sleep(options.delay_call)
            params = message.get("params") or {}
            name = str(params.get("name", ""))
            send(
                {
                    "jsonrpc": JSONRPC,
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": f"called:{name}"}],
                        "isError": False,
                    },
                }
            )
        elif method == "ping":
            send({"jsonrpc": JSONRPC, "id": request_id, "result": {}})
        else:
            send(
                {
                    "jsonrpc": JSONRPC,
                    "id": request_id,
                    "error": {"code": -32601, "message": f"method not found: {method}"},
                }
            )
        if options.exit_after and handled >= options.exit_after:
            return options.exit_code
    return options.exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
