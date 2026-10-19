"""``mcp/validator.py`` 单测：§10.3.1 校验清单 V1–V13（验收 F-1）与 F-2。

约定：
* 每个规则至少一个用例，用例名带验收编号（``test_F1_V4_...``），并断言**具体消息文本**；
* V10/V11/V12/V13 需要「工具清单」，故静态阶段用调用方提供的 ``tools=`` 声明，
  deep 阶段用注入的假客户端（不 spawn 真实进程）。
"""

from __future__ import annotations

import sys
from typing import Any

import httpx
import pytest

from my_agent.config.schema import McpServerConfig
from my_agent.errors import McpTimeoutError, McpValidationError
from my_agent.mcp.client_http import decode_jsonrpc_response, parse_sse_text
from my_agent.mcp.config_store import parse_mcp_yaml
from my_agent.mcp.schema import McpTool, map_mcp_tools
from my_agent.mcp.validator import (
    McpValidator,
    ValidationReport,
    validate_document,
    validate_static,
)

BUILTIN = "Read"


# --------------------------------------------------------------------------- 工具


def make_server(**overrides: Any) -> McpServerConfig:
    """构造一个默认合法的 stdio 服务器配置。

    Args:
        **overrides: 覆盖字段。

    Returns:
        :class:`McpServerConfig` 实例。
    """
    data: dict[str, Any] = {
        "name": "fs",
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-c", "pass"],
    }
    data.update(overrides)
    return McpServerConfig(**data)


def issues(report: ValidationReport, rule: str) -> list[str]:
    """取某规则的全部消息。

    Args:
        report: 校验报告。
        rule: 规则编号。

    Returns:
        消息文本列表。
    """
    return [issue.message for issue in (*report.errors, *report.warnings) if issue.rule == rule]


def errors_of(report: ValidationReport, rule: str) -> list[str]:
    """取某规则的阻塞错误消息。

    Args:
        report: 校验报告。
        rule: 规则编号。

    Returns:
        消息文本列表。
    """
    return [issue.message for issue in report.errors if issue.rule == rule]


class FakeClient:
    """最小的 :class:`~my_agent.mcp.schema.McpClient` 假实现（不 spawn 进程）。"""

    def __init__(
        self,
        server: McpServerConfig,
        *,
        init_result: dict[str, Any] | None = None,
        init_error: Exception | None = None,
        list_error: Exception | None = None,
        tools: list[dict[str, Any]] | None = None,
        protocol_version: str = "2024-11-05",
    ) -> None:
        self._server = server
        self._init_result = init_result
        self._init_error = init_error
        self._list_error = list_error
        self._raws = tools if tools is not None else []
        self.protocol_version = protocol_version
        self.server_info: dict[str, Any] = {"name": "fake", "version": "1.0"}
        self.closed = False

    @property
    def server_name(self) -> str:
        """服务器名。"""
        return self._server.name

    async def initialize(self) -> dict[str, Any]:
        """返回脚本化的 ``initialize`` 结果。"""
        if self._init_error is not None:
            raise self._init_error
        if self._init_result is not None:
            return self._init_result
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": dict(self.server_info),
        }

    async def list_tools(self) -> list[McpTool]:
        """返回脚本化的工具清单。"""
        if self._list_error is not None:
            raise self._list_error
        return map_mcp_tools(self._server, list(self._raws))

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """假实现不参与校验用例。"""
        raise NotImplementedError

    async def close(self) -> None:
        """记录关闭。"""
        self.closed = True


def tool_entry(
    name: str, *, description: Any = "desc", input_schema: Any = None, **extra: Any
) -> dict[str, Any]:
    """构造一条工具声明（可作为静态 ``tools=`` 或假客户端返回）。

    Args:
        name: 工具名。
        description: 描述（``None`` 表示省略该键）。
        input_schema: ``inputSchema``（``None`` 表示省略该键）。
        **extra: 额外字段。

    Returns:
        工具声明字典。
    """
    entry: dict[str, Any] = {"name": name}
    if description is not None:
        entry["description"] = description
    if input_schema is not None:
        entry["inputSchema"] = input_schema
    entry.update(extra)
    return entry


# --------------------------------------------------------------------- V1 解析/顶层


def test_F1_V1_yaml_parse_error_reports_line_number() -> None:
    """V1：YAML 不可解析时必须给出带行号的消息。"""
    bad_yaml = "servers:\n  - name: fs\n   transport: stdio\n"
    with pytest.raises(McpValidationError) as excinfo:
        parse_mcp_yaml(bad_yaml, source="mcp.yaml")
    detail = excinfo.value.detail
    assert "mcp.yaml 解析失败" in detail
    assert "行" in detail
    assert excinfo.value.context["rule"] == "V1"


def test_F1_V1_parse_error_keeps_spec_style_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """V1：消息以「mcp.yaml 解析失败：」开头（§10.3.1 示例风格）。"""
    with pytest.raises(McpValidationError) as excinfo:
        parse_mcp_yaml("servers: [\n", source="/tmp/mcp.yaml")
    assert excinfo.value.detail.startswith("mcp.yaml 解析失败：")


def test_F1_V1_top_level_must_be_servers_list() -> None:
    """V1：顶层不是 ``servers`` 列表时给出明确消息。"""
    servers, report = validate_document({"server": ["fs"]})
    assert servers == []
    assert report.ok is False
    assert "mcp.yaml 顶层必须是 servers 列表" in errors_of(report, "V1")[0]
    # 消息必须说明当前值的实际类型，便于用户定位
    assert "null" in errors_of(report, "V1")[0]


def test_F1_V1_servers_items_must_be_mappings() -> None:
    """V1：``servers`` 里出现非对象条目时报错并跳过。"""
    servers, report = validate_document({"servers": ["fs"]})
    assert servers == []
    assert "servers 第 1 项必须是对象" in errors_of(report, "V1")[0]


# --------------------------------------------------------------------------- V2 名称


def test_F1_V2_name_must_match_pattern() -> None:
    """V2：非法服务器名 → 明确消息（含允许字符说明）。"""
    report = validate_static(make_server(name="my server"))
    assert report.ok is False
    assert errors_of(report, "V2") == [
        '服务器名 "my server" 非法：只允许字母数字下划线连字符（1–64 个字符）'
    ]


def test_F1_V2_name_is_required() -> None:
    """V2：缺失 ``name`` → 明确消息。"""
    report = validate_static({"transport": "stdio", "command": sys.executable})
    assert "服务器缺少必需字段 name" in errors_of(report, "V2")


def test_F1_V2_name_must_be_unique() -> None:
    """V2：服务器名全局唯一（重名 → 明确消息）。"""
    report = validate_static(make_server(name="filesystem"), existing_servers={"filesystem"})
    assert errors_of(report, "V2") == ['服务器名 "filesystem" 重复：已存在同名服务器']


# ---------------------------------------------------------------------- V3 transport


def test_F1_V3_transport_must_be_supported() -> None:
    """V3：非法 transport → 明确列出支持项。"""
    report = validate_static({"name": "x", "transport": "ws"})
    assert errors_of(report, "V3") == ['不支持的 transport "ws"；支持：stdio / http / sse']


# ------------------------------------------------------------------------ V4 command


def test_F1_V4_command_must_be_resolvable() -> None:
    """V4：PATH 中找不到命令 → 明确消息。"""
    report = validate_static(make_server(command="nvpx-does-not-exist-xyz"))
    assert errors_of(report, "V4") == [
        '找不到命令 "nvpx-does-not-exist-xyz"；请检查拼写或安装'
    ]


def test_F1_V4_command_required_for_stdio() -> None:
    """V4：stdio 缺 ``command`` → 明确消息。"""
    report = validate_static(make_server(command=None))
    assert errors_of(report, "V4") == ["transport=stdio 必须提供 command"]


def test_F1_V4_absolute_path_is_accepted() -> None:
    """V4：绝对路径存在即通过（不依赖 PATH）。"""
    report = validate_static(make_server(command=sys.executable))
    assert report.ok is True


# --------------------------------------------------------------------------- V5 args


def test_F1_V5_args_must_be_string_list() -> None:
    """V5：``args`` 含非字符串项 → 指出第几项（1 基，§10.3.1 示例口径）。"""
    report = validate_static(
        {"name": "fs", "transport": "stdio", "command": sys.executable, "args": ["-y", 3]}
    )
    assert errors_of(report, "V5") == ["args 必须是字符串列表（第 2 项是数字）"]


def test_F1_V5_args_must_be_a_list() -> None:
    """V5：``args`` 不是列表 → 明确消息。"""
    report = validate_static(
        {"name": "fs", "transport": "stdio", "command": sys.executable, "args": "-y"}
    )
    assert errors_of(report, "V5") == ["args 必须是字符串列表（当前是 字符串）"]


# ---------------------------------------------------------------------------- V6 url


def test_F1_V6_remote_must_use_https() -> None:
    """V6：非本地 http 地址 → 强制 HTTPS 消息。"""
    report = validate_static(
        {"name": "docs", "transport": "http", "url": "http://example.com/mcp"}
    )
    assert errors_of(report, "V6") == ["远程 MCP 必须使用 https（当前 http://example.com/mcp）"]


def test_F1_V6_localhost_exempt_from_https() -> None:
    """V6：localhost/127.0.0.1 豁免 HTTPS。"""
    for url in ("http://localhost:8080/mcp", "http://127.0.0.1:8080/mcp"):
        assert validate_static({"name": "docs", "transport": "http", "url": url}).ok is True


def test_F1_V6_url_scheme_must_be_http_or_https() -> None:
    """V6：非法 scheme → 明确消息。"""
    report = validate_static({"name": "docs", "transport": "http", "url": "ftp://example.com/mcp"})
    assert errors_of(report, "V6") == [
        'url 非法：必须是 http/https 绝对地址（当前 "ftp://example.com/mcp"）'
    ]


def test_F1_V6_url_required_for_remote() -> None:
    """V6：http/sse 缺 ``url`` → 明确消息。"""
    report = validate_static({"name": "docs", "transport": "sse"})
    assert errors_of(report, "V6") == ["transport=sse 必须提供 url"]


# ------------------------------------------------------------------------- V7 ${VAR}


def test_F1_V7_env_var_must_resolve() -> None:
    """V7：``headers`` 里的 ``${VAR}`` 未设置 → 消息含变量名与位置。"""
    server = McpServerConfig(
        name="docs",
        transport="http",
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer ${DOCS_TOKEN}"},
    )
    report = validate_static(server, env={})
    assert errors_of(report, "V7") == ["环境变量 DOCS_TOKEN 未设置（headers.Authorization）"]


def test_F1_V7_resolved_env_var_passes() -> None:
    """V7：变量存在时通过。"""
    server = McpServerConfig(
        name="docs",
        transport="http",
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer ${DOCS_TOKEN}"},
    )
    assert validate_static(server, env={"DOCS_TOKEN": "s3cr3t"}).ok is True


def test_F1_V7_stdio_env_placeholder_is_checked_too() -> None:
    """V7：stdio 的 ``env`` 值同样检查（超集口径）。"""
    report = validate_static(make_server(env={"API_KEY": "${MISSING_KEY}"}), env={})
    assert errors_of(report, "V7") == ["环境变量 MISSING_KEY 未设置（env.API_KEY）"]


# ----------------------------------------------------------------------- V8 timeout


def test_F1_V8_timeout_range() -> None:
    """V8：``timeout_s`` 越界 → 明确区间消息（区间取 MCP_TIMEOUT_RANGE）。"""
    report = validate_static(make_server(timeout_s=601))
    assert errors_of(report, "V8") == ["timeout_s 必须在 1–600 秒之间（当前 601）"]
    report_low = validate_static({"name": "fs", "transport": "stdio", "timeout_s": "30s"})
    assert "timeout_s 必须在 1–600 秒之间" in errors_of(report_low, "V8")[0]


def test_F1_V8_boundaries_accepted() -> None:
    """V8：区间端点 1 / 600 合法。"""
    assert validate_static(make_server(timeout_s=1)).ok is True
    assert validate_static(make_server(timeout_s=600)).ok is True


# ------------------------------------------------------------------ V9 tool_prefix


def test_F1_V9_tool_prefix_pattern() -> None:
    """V9：``tool_prefix`` 非法字符 → 明确消息。"""
    report = validate_static(make_server(tool_prefix="fs-1"))
    assert errors_of(report, "V9") == [
        'tool_prefix "fs-1" 非法：只允许字母数字下划线（1–32 个字符）'
    ]


def test_F1_V9_valid_prefix_passes() -> None:
    """V9：合法前缀通过。"""
    assert validate_static(make_server(tool_prefix="fs_1")).ok is True


# ---------------------------------------------------------------------- V10 握手


async def test_F1_V10_initialize_timeout_message() -> None:
    """V10：``initialize`` 超时 → 规格示例文案。"""
    server = make_server(timeout_s=30)
    client = FakeClient(
        server, init_error=McpTimeoutError(server=server.name, timeout=server.timeout_s)
    )
    validator = McpValidator(client_factory=lambda _server: client)
    report = await validator.validate(server, deep=True)
    assert errors_of(report, "V10") == ["initialize 超时（30s）：服务未响应"]
    assert client.closed is True


async def test_F1_V10_tools_list_structure_error() -> None:
    """V10：``tools/list`` 结构非法 → 规格示例文案。"""
    server = make_server()
    client = FakeClient(
        server, list_error=McpValidationError(detail="tools/list 返回非法结构：缺少 result.tools")
    )
    validator = McpValidator(client_factory=lambda _server: client)
    report = await validator.validate(server, deep=True)
    assert errors_of(report, "V10") == ["tools/list 返回非法结构：缺少 result.tools"]


async def test_F1_V10_deep_success_fills_report() -> None:
    """V10：握手成功 → ``discovered_tools`` / ``server_info`` / ``protocol_version`` 填充。"""
    server = make_server()
    client = FakeClient(server, tools=[tool_entry("read_file")], protocol_version="2024-11-05")
    validator = McpValidator(client_factory=lambda _server: client)
    report = await validator.validate(server, deep=True)
    assert report.ok is True
    assert [spec.name for spec in report.discovered_tools] == ["read_file"]
    assert report.protocol_version == "2024-11-05"
    assert report.server_info["name"] == "fake"
    assert client.closed is True


async def test_F1_V10_missing_protocol_version_is_error() -> None:
    """V10：``initialize`` 缺少 ``protocolVersion`` → 非法结构错误。"""
    server = make_server()
    # 必须显式清空客户端的协议版本，否则 FakeClient 的默认值会掩盖「缺失」场景
    client = FakeClient(
        server,
        init_result={"serverInfo": {"name": "fake"}},
        protocol_version="",
    )
    validator = McpValidator(client_factory=lambda _server: client)
    report = await validator.validate(server, deep=True)
    assert errors_of(report, "V10") == ["initialize 返回非法结构：缺少 protocolVersion"]


# ----------------------------------------------------------------- V11 工具名冲突


def test_F1_V11_conflict_with_builtin_tool() -> None:
    """V11：与内置工具重名 → 报错并提示 ``tool_prefix``（验收 F-4）。"""
    report = validate_static(
        make_server(),
        existing_names={BUILTIN: "builtin"},
        tools=[tool_entry(BUILTIN)],
    )
    assert report.ok is False
    message = errors_of(report, "V11")[0]
    assert message == '工具名 "Read" 与内置工具冲突；请设置 tool_prefix'


def test_F1_V11_conflict_with_other_mcp_server() -> None:
    """V11：与其它 MCP 服务器的工具重名 → 报错并点名来源服务器。"""
    report = validate_static(
        make_server(name="docs"),
        existing_names={"search": "filesystem"},
        tools=[tool_entry("search")],
    )
    assert errors_of(report, "V11") == [
        '工具名 "search" 与 MCP 服务器 "filesystem" 的工具冲突；请设置 tool_prefix'
    ]


def test_F1_V11_prefix_resolves_conflict() -> None:
    """V11：设置 ``tool_prefix`` 后名字不再冲突。"""
    report = validate_static(
        make_server(tool_prefix="fs"),
        existing_names={BUILTIN: "builtin"},
        tools=[tool_entry(BUILTIN)],
    )
    assert report.ok is True


async def test_F1_V11_deep_reports_conflict_from_handshake() -> None:
    """V11：deep 阶段用真实工具清单做冲突检测。"""
    server = make_server()
    client = FakeClient(server, tools=[tool_entry(BUILTIN)])
    validator = McpValidator(
        client_factory=lambda _server: client, existing_names={BUILTIN: "builtin"}
    )
    report = await validator.validate(server, deep=True)
    assert '与内置工具冲突；请设置 tool_prefix' in errors_of(report, "V11")[0]


# ---------------------------------------------------------------- V12 schema 合法性


def test_F1_V12_input_schema_must_be_object() -> None:
    """V12：``inputSchema`` 不是对象 → 规格示例文案。"""
    report = validate_static(
        make_server(), tools=[tool_entry("query", input_schema="not-an-object")]
    )
    assert errors_of(report, "V12") == ['工具 "query" 的 inputSchema 不是对象（当前是 字符串）']


def test_F1_V12_description_required() -> None:
    """V12：缺少 ``description`` → 报错；空描述由映射兜底、缺 ``inputSchema`` 容忍为 ``{}``。"""
    report = validate_static(make_server(), tools=[tool_entry("query", description=None)])
    assert errors_of(report, "V12") == ['工具 "query" 缺少 description']
    tolerant = validate_static(
        make_server(), tools=[tool_entry("query", description="", input_schema={"type": "object"})]
    )
    assert errors_of(tolerant, "V12") == []
    no_schema = validate_static(make_server(), tools=[tool_entry("query", description="desc")])
    assert errors_of(no_schema, "V12") == []


def test_F1_V12_valid_tool_passes() -> None:
    """V12：合法声明通过。"""
    report = validate_static(
        make_server(),
        tools=[tool_entry("query", input_schema={"type": "object", "properties": {}})],
    )
    assert report.ok is True


def test_F1_V11_V12_deferred_when_no_tool_list() -> None:
    """V11/V12 在静态阶段没有工具清单时延后到 deep（不误报）。"""
    report = validate_static(make_server(), existing_names={BUILTIN: "builtin"})
    assert report.errors == []


# ------------------------------------------------------------- V13 危险能力（警告）


def test_F1_V13_danger_capability_is_warning_only() -> None:
    """V13：命中危险词 → 只警告、不阻塞（非阻塞提示）。"""
    report = validate_static(
        make_server(), tools=[tool_entry("delete_all", description="删除全部数据")]
    )
    assert report.ok is True
    assert report.errors == []
    assert [issue.rule for issue in report.warnings] == ["V13"]
    assert "该 MCP 工具可能具备写/执行能力" not in report.warnings[0].message  # 见下方断言口径
    assert "可能具备写/执行能力，权限流水线会按 write/execute 类别处理" in report.warnings[0].message


def test_F1_V13_description_hit_also_warns() -> None:
    """V13：描述命中同样告警（工具名不命中时）。"""
    report = validate_static(
        make_server(), tools=[tool_entry("harmless", description="will publish a release")]
    )
    assert [issue.rule for issue in report.warnings] == ["V13"]


async def test_F1_V13_deep_warning_non_blocking() -> None:
    """V13：deep 阶段照样只警告（``ok`` 保持 ``True``）。"""
    server = make_server()
    client = FakeClient(server, tools=[tool_entry("exec_cmd")])
    validator = McpValidator(client_factory=lambda _server: client)
    report = await validator.validate(server, deep=True)
    assert report.ok is True
    assert [issue.rule for issue in report.warnings] == ["V13"]


# ------------------------------------------------------------------- F-2 启动不 deep


async def test_F2_validate_deep_false_never_creates_client() -> None:
    """F-2：``deep=False`` 时不构造客户端（不 spawn 握手进程）。"""
    created: list[str] = []

    def factory(server: McpServerConfig) -> FakeClient:
        created.append(server.name)
        return FakeClient(server)

    validator = McpValidator(client_factory=factory)
    report = await validator.validate(make_server(), deep=False)
    assert report.ok is True
    assert created == []


async def test_F2_validate_deep_false_skips_handshake_even_with_static_errors() -> None:
    """F-2：静态检查已失败时，deep=False 直接返回报告，不触碰网络/进程。"""
    created: list[str] = []
    validator = McpValidator(
        client_factory=lambda server: created.append(server.name) or FakeClient(server)
    )
    report = await validator.validate(make_server(command="nvpx-nope-xyz"), deep=False)
    assert report.ok is False
    assert created == []


# ----------------------------------------------------------------- 报告与文档辅助


def test_report_helpers_render_and_merge() -> None:
    """报告辅助方法：``messages`` / ``error_text`` / ``summary`` / ``merge``。"""
    first = ValidationReport()
    first.add_error("V4", '找不到命令 "x"；请检查拼写或安装')
    first.add_warning("V13", "工具 \"wipe\" 可能具备写/执行能力")
    second = ValidationReport()
    second.add_error("V2", "服务器缺少必需字段 name")
    assert first.ok is False
    assert first.error_text().startswith("[V4] ")
    assert first.summary() == "校验失败：1 项错误"
    assert len(first.messages()) == 2
    first.merge(second)
    assert len(first.errors) == 2
    assert first.warnings[0].render().endswith("）") is False
    assert ValidationReport().summary() == "校验通过"


def test_validate_document_builds_servers_and_aggregates_conflicts() -> None:
    """``validate_document``：构造合法服务器，并把跨服务器重名报为 V11/V2。"""
    document = {
        "servers": [
            {
                "name": "one",
                "transport": "stdio",
                "command": sys.executable,
                "tools": [{"name": "a", "description": "读 a"}],
            },
            {"name": "one", "transport": "stdio", "command": sys.executable},
            {
                "name": "two",
                "transport": "stdio",
                "command": sys.executable,
                "tools": [{"name": "a", "description": "读 a"}],
            },
        ]
    }
    servers, report = validate_document(document, existing_names={"Read": "builtin"})
    # 第 2 项是重名（V2 阻塞）→ 不构造；第 3 项的工具名与第 1 项冲突（V11 阻塞）
    # → 按 §10.3.1「拒绝加载」也不构造。因此只剩第一个服务器。
    assert [server.name for server in servers] == ["one"]
    assert any("重复" in message for message in errors_of(report, "V2"))
    # 同一份文档内两个服务器都声明了工具 a：第二个与第一个（MCP 服务器 one）冲突
    assert any("MCP 服务器 \"one\"" in message for message in errors_of(report, "V11"))


# ------------------------------------------------------------------ HTTP/SSE 解码


def test_parse_sse_text_handles_events_and_comments() -> None:
    """HTTP 客户端：SSE 文本解析（含注释、多行 data、缺省事件名）。"""
    text = ": keepalive\nevent: endpoint\ndata: /messages\n\n" 'data: {"a": 1}\ndata: \n\n'
    events = parse_sse_text(text)
    assert events[0] == ("endpoint", "/messages")
    assert events[1][0] == "message"
    assert events[1][1].startswith('{"a": 1}')


def test_decode_jsonrpc_response_from_json_body() -> None:
    """HTTP 客户端：``application/json`` 响应体解码。"""
    response = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}},
    )
    message = decode_jsonrpc_response(response, 1, method="tools/list")
    assert message["result"] == {"tools": []}


def test_decode_jsonrpc_response_from_sse_body() -> None:
    """HTTP 客户端：``text/event-stream`` 单事件响应解码（streamable HTTP）。"""
    body = 'event: message\ndata: {"jsonrpc": "2.0", "id": 7, "result": {"ok": true}}\n\n'
    response = httpx.Response(
        200, headers={"content-type": "text/event-stream"}, text=body
    )
    message = decode_jsonrpc_response(response, 7, method="tools/call")
    assert message["result"] == {"ok": True}


def test_decode_jsonrpc_response_rejects_bad_shapes() -> None:
    """HTTP 客户端：202/HTTP 错误/缺 id 都给出明确错误。"""
    accepted = httpx.Response(202, headers={"content-type": "text/plain"}, text="")
    with pytest.raises(McpValidationError) as excinfo:
        decode_jsonrpc_response(accepted, 1, method="tools/call")
    assert "202 Accepted" in excinfo.value.detail

    failed = httpx.Response(500, headers={"content-type": "text/plain"}, text="boom")
    with pytest.raises(McpValidationError) as excinfo2:
        decode_jsonrpc_response(failed, 1, method="initialize")
    assert "HTTP 500" in excinfo2.value.detail

    wrong_id = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={"jsonrpc": "2.0", "id": 99, "result": {}},
    )
    with pytest.raises(McpValidationError) as excinfo3:
        decode_jsonrpc_response(wrong_id, 1, method="initialize")
    assert "缺少匹配的 id=1" in excinfo3.value.detail
