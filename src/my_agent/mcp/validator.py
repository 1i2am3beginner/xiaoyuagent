"""MCP 配置校验（§10.3 / §10.3.1，V1–V13）。

**只在用户添加时执行**（上游原文）：``/mcp add`` 走 :meth:`McpValidator.validate`
（``deep=True``，跑 V10 真实握手）；启动加载只做「配置能被解析」的最小检查，
绝不调用 deep 校验（F-2）。

校验清单与消息风格**逐条对齐** §10.3.1 的示例文案：每条规则一个 ``rule`` 标签
（``"V1"`` … ``"V13"``）与一条具体中文消息。V10/V13 依赖工具清单，故：

* V11/V12（冲突、schema 合法性）在**静态**阶段只对「调用方提供的工具声明」执行，
  未提供时延后到 deep 阶段（握手拿到的真实工具清单）执行；
* V13 只在 deep 阶段产生（非阻塞警告）。

规格：``my_agent-开发文档.md`` §10.3；契约：``CONTRACTS.md`` §11。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError

from my_agent.config.defaults import MCP_NAME_RE, MCP_PREFIX_RE, MCP_TIMEOUT_RANGE
from my_agent.config.schema import McpServerConfig
from my_agent.errors import McpTimeoutError, McpValidationError
from my_agent.mcp.schema import (
    DANGER_NAME_RE,
    ENV_PLACEHOLDER_RE,
    McpClient,
    McpTool,
    expand_server,
)
from my_agent.models.tools import ToolSpec

__all__ = [
    "LOCAL_HTTP_HOSTS",
    "McpValidator",
    "ValidationIssue",
    "ValidationReport",
    "default_client_factory",
    "validate_static",
]

_LOG = logging.getLogger(__name__)

#: 允许免 HTTPS 的本地主机（§10.3.1 V6："除 localhost/127.0.0.1"）。
LOCAL_HTTP_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_NAME_PATTERN = re.compile(MCP_NAME_RE)
_PREFIX_PATTERN = re.compile(MCP_PREFIX_RE)

#: 校验入口既能接已构造的配置，也能接 YAML 原始条目（V1/V5 需要看原始类型）。
RawEntry = McpServerConfig | Mapping[str, Any]


class ValidationIssue(BaseModel):
    """一条校验结论（错误或警告）。

    Attributes:
        rule: 规则编号，``"V1"`` … ``"V13"``。
        message: 中文消息（对齐 §10.3.1 的示例文案）。
        server: 相关服务器名（文档级规则为空串）。
        hint: 可选的修复建议（如「请设置 tool_prefix」的完整写法）。
        level: ``"error"`` 阻塞加载；``"warning"`` 仅提示。
    """

    rule: str = Field(description="规则编号 V1–V13")
    message: str = Field(description="中文错误/警告消息")
    server: str = Field(default="", description="相关服务器名")
    hint: str = Field(default="", description="修复建议")
    level: Literal["error", "warning"] = Field(default="error", description="严重级别")

    model_config = {"extra": "ignore"}

    def render(self) -> str:
        """渲染为 UI 一行。

        Returns:
            形如 ``[V4] 找不到命令 "nvpx"；请检查拼写或安装`` 的文本（带 hint 时追加）。
        """
        text = f"[{self.rule}] {self.message}"
        return f"{text}（{self.hint}）" if self.hint else text


class ValidationReport(BaseModel):
    """校验报告（§10.3 冻结结构）。

    Attributes:
        ok: 无 ``errors`` 时为 ``True``（警告不影响）。
        errors: 阻塞项（拒绝加载）。
        warnings: 非阻塞项。
        discovered_tools: ``deep=True`` 时填充的映射后工具声明。
        server_info: ``initialize`` 返回的 ``serverInfo``。
        protocol_version: ``initialize`` 协商得到的 ``protocolVersion``。
        server: 被校验的服务器名（便于多服务器汇总时定位）。
    """

    ok: bool = True
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    discovered_tools: list[ToolSpec] = Field(default_factory=list)
    server_info: dict[str, Any] = Field(default_factory=dict)
    protocol_version: str = ""
    server: str = ""

    model_config = {"extra": "ignore"}

    def add_error(self, rule: str, message: str, *, server: str = "", hint: str = "") -> None:
        """追加一条阻塞错误（并把 :attr:`ok` 置为 ``False``）。

        Args:
            rule: 规则编号。
            message: 中文消息。
            server: 相关服务器名。
            hint: 修复建议。
        """
        self.errors.append(
            ValidationIssue(rule=rule, message=message, server=server, hint=hint, level="error")
        )
        self.ok = False

    def add_warning(self, rule: str, message: str, *, server: str = "", hint: str = "") -> None:
        """追加一条非阻塞警告。

        Args:
            rule: 规则编号。
            message: 中文消息。
            server: 相关服务器名。
            hint: 修复建议。
        """
        self.warnings.append(
            ValidationIssue(rule=rule, message=message, server=server, hint=hint, level="warning")
        )

    def merge(self, other: ValidationReport) -> None:
        """把另一份报告合并进本报告（多服务器场景）。

        Args:
            other: 待合并报告；其 ``server_info`` / ``protocol_version`` 非空时覆盖本报告。
        """
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.discovered_tools.extend(other.discovered_tools)
        if other.server_info:
            self.server_info = other.server_info
        if other.protocol_version:
            self.protocol_version = other.protocol_version
        self.ok = not self.errors

    def messages(self) -> list[str]:
        """返回逐行可打印的全部结论（错误在前，警告在后）。

        Returns:
            已渲染的文本行列表。
        """
        return [issue.render() for issue in (*self.errors, *self.warnings)]

    def error_text(self) -> str:
        """返回错误清单文本（``/mcp add`` 失败时输出）。

        Returns:
            每行一条错误的文本；无错误时为空串。
        """
        return "\n".join(issue.render() for issue in self.errors)

    def summary(self) -> str:
        """返回一行结论摘要。

        Returns:
            形如 ``校验通过（发现 3 个工具，1 条警告）`` 或 ``校验失败：2 项错误``。
        """
        if not self.ok:
            return f"校验失败：{len(self.errors)} 项错误"
        extra = f"，{len(self.warnings)} 条警告" if self.warnings else ""
        tools = f"（发现 {len(self.discovered_tools)} 个工具{extra}）" if self.discovered_tools else ""
        return f"校验通过{tools}{extra}"


# ============================================================================
# 字段读取（同时支持已构造配置与 YAML 原始条目）
# ============================================================================


def _field(source: RawEntry, key: str, default: Any = None) -> Any:
    """从配置对象或原始映射中读字段。

    Args:
        source: :class:`McpServerConfig` 或 YAML 原始条目。
        key: 字段名。
        default: 缺失时的默认值。

    Returns:
        字段值；缺失时返回 ``default``。
    """
    if isinstance(source, McpServerConfig):
        return getattr(source, key, default)
    return source.get(key, default)


def _entry_name(source: RawEntry) -> str:
    """取服务器名（非法类型转成可读字符串，供消息使用）。

    Args:
        source: 配置对象或原始条目。

    Returns:
        服务器名的字符串形式；缺失时为空串。
    """
    value = _field(source, "name")
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _resolve_command(command: str, resolver: Callable[[str], str | None] | None) -> str | None:
    """解析 stdio 命令（§10.3.1 V4）。

    Args:
        command: 命令名或路径。
        resolver: 注入的解析器（测试用）；``None`` 时用 ``shutil.which`` + 绝对路径存在性。

    Returns:
        解析到的可执行文件路径；找不到时为 ``None``。
    """
    if not command:
        return None
    if resolver is not None:
        return resolver(command)
    if "/" in command or "\\" in command:
        path = Path(command)
        return str(path) if path.exists() else None
    return shutil.which(command)


def _describe(value: Any) -> str:
    """给非法值一个中文可读的类型描述（用于消息里的「当前是 …」）。

    Args:
        value: 任意值。

    Returns:
        类型描述，如 ``数字`` / ``字符串`` / ``null`` / ``列表``。
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "布尔值"
    if isinstance(value, (int, float)):
        return "数字"
    if isinstance(value, str):
        return "字符串"
    if isinstance(value, Mapping):
        return "对象"
    if isinstance(value, (list, tuple)):
        return "列表"
    return type(value).__name__


# ============================================================================
# 静态校验（V2–V9、V11、V12；V1 见 validate_document）
# ============================================================================


def validate_static(
    server: RawEntry,
    *,
    existing_names: Mapping[str, str] | Iterable[str] | None = None,
    existing_servers: Iterable[str] | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> ValidationReport:
    """**不连接**服务器的静态校验（契约 §11：V1–V9、V11、V12）。

    Args:
        server: 已构造的 :class:`McpServerConfig`，或 YAML 原始条目（后者可校验
            V2/V3/V5 等类型问题——pydantic 模型构造期就会先拒绝非法类型）。
        existing_names: 已占用的**工具名**；``set[str]`` 时一律视为内置冲突，
            ``Mapping[str, str]`` 时 value 为来源标签（``"builtin"`` 或 MCP 服务器名）。
        existing_servers: 已存在的**服务器名**（V2 全局唯一）。
        tools: 已知的工具声明（V11/V12）；``None`` 表示延后到 deep 阶段。
        env: 环境变量映射（V7 用；默认 ``os.environ``）。
        which: 命令解析器（V4 用；默认 ``shutil.which``）。

    Returns:
        :class:`ValidationReport`（``ok=False`` 表示存在阻塞错误）。
    """
    report = ValidationReport(server=_entry_name(server))
    name = _entry_name(server)
    known_servers = set(existing_servers or ())

    _check_name(server, report, name, known_servers)
    transport = _check_transport(server, report, name)
    if transport == "stdio":
        _check_stdio(server, report, name, which)
    elif transport in ("http", "sse"):
        _check_remote_url(server, report, name, transport)
    _check_env_placeholders(server, report, name, env)
    _check_timeout(server, report, name)
    _check_prefix(server, report, name)
    _check_tools(
        _probes_from_raw(server, tools),
        report,
        name,
        existing_names=existing_names,
    )
    return report


def _check_name(
    server: RawEntry, report: ValidationReport, name: str, existing_servers: set[str]
) -> None:
    """V2：``name`` 存在、匹配 ``MCP_NAME_RE``、全局唯一。

    Args:
        server: 配置或原始条目。
        report: 报告（就地追加）。
        name: 服务器名。
        existing_servers: 已存在的服务器名集合。
    """
    raw_name = _field(server, "name")
    if raw_name is None or (isinstance(raw_name, str) and not raw_name.strip()):
        report.add_error("V2", "服务器缺少必需字段 name", server=name)
        return
    if not isinstance(raw_name, str) or not _NAME_PATTERN.match(raw_name):
        report.add_error(
            "V2",
            f'服务器名 "{raw_name}" 非法：只允许字母数字下划线连字符（1–64 个字符）',
            server=name,
        )
        return
    if raw_name in existing_servers:
        report.add_error("V2", f'服务器名 "{raw_name}" 重复：已存在同名服务器', server=name)


def _check_transport(server: RawEntry, report: ValidationReport, name: str) -> str:
    """V3：``transport ∈ {stdio, http, sse}``。

    Args:
        server: 配置或原始条目。
        report: 报告（就地追加）。
        name: 服务器名。

    Returns:
        合法的 transport；非法时返回空串。
    """
    transport = _field(server, "transport", "stdio")
    if transport not in ("stdio", "http", "sse"):
        report.add_error(
            "V3", f'不支持的 transport "{transport}"；支持：stdio / http / sse', server=name
        )
        return ""
    return str(transport)


def _check_stdio(
    server: RawEntry,
    report: ValidationReport,
    name: str,
    resolver: Callable[[str], str | None] | None,
) -> None:
    """V4 + V5：stdio 的 ``command`` 可解析、``args`` 为字符串数组。

    Args:
        server: 配置或原始条目。
        report: 报告（就地追加）。
        name: 服务器名。
        resolver: 命令解析器。
    """
    command = _field(server, "command")
    if not isinstance(command, str) or not command.strip():
        report.add_error("V4", "transport=stdio 必须提供 command", server=name)
    elif _resolve_command(command, resolver) is None:
        report.add_error("V4", f'找不到命令 "{command}"；请检查拼写或安装', server=name)

    args = _field(server, "args", [])
    if args is None:
        return
    if not isinstance(args, (list, tuple)):
        report.add_error(
            "V5", f"args 必须是字符串列表（当前是 {_describe(args)}）", server=name
        )
        return
    for index, item in enumerate(args, start=1):
        if not isinstance(item, str):
            report.add_error(
                "V5",
                f"args 必须是字符串列表（第 {index} 项是{_describe(item)}）",
                server=name,
            )
            return


def _check_remote_url(server: RawEntry, report: ValidationReport, name: str, transport: str) -> None:
    """V6：http/sse 的 ``url`` 合法 + HTTPS 强制（本地主机豁免）。

    Args:
        server: 配置或原始条目。
        report: 报告（就地追加）。
        name: 服务器名。
        transport: 已校验的传输方式。
    """
    url = _field(server, "url")
    if not isinstance(url, str) or not url.strip():
        report.add_error("V6", f"transport={transport} 必须提供 url", server=name)
        return
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        report.add_error(
            "V6", f'url 非法：必须是 http/https 绝对地址（当前 "{url}"）', server=name
        )
        return
    if parsed.scheme == "http" and (parsed.hostname or "") not in LOCAL_HTTP_HOSTS:
        report.add_error("V6", f"远程 MCP 必须使用 https（当前 {url}）", server=name)


def _check_env_placeholders(
    server: RawEntry, report: ValidationReport, name: str, env: Mapping[str, str] | None
) -> None:
    """V7：``headers`` / ``env`` 值中的 ``${VAR}`` 必须能解析到环境变量。

    Args:
        server: 配置或原始条目。
        report: 报告（就地追加）。
        name: 服务器名。
        env: 环境变量映射。
    """
    source = os.environ if env is None else env
    headers = _field(server, "headers") or {}
    child_env = _field(server, "env") or {}
    for field_name, values in (("headers", headers), ("env", child_env)):
        if not isinstance(values, Mapping):
            continue
        for key, value in values.items():
            if not isinstance(value, str):
                continue
            for match in ENV_PLACEHOLDER_RE.finditer(value):
                var = match.group(1)
                if source.get(var) is None:
                    report.add_error(
                        "V7",
                        f"环境变量 {var} 未设置（{field_name}.{key}）",
                        server=name,
                        hint=f"请先 export {var}=… 或改用明文值",
                    )


def _check_timeout(server: RawEntry, report: ValidationReport, name: str) -> None:
    """V8：``timeout_s ∈ [1, 600]``（区间取 ``MCP_TIMEOUT_RANGE``）。

    Args:
        server: 配置或原始条目。
        report: 报告（就地追加）。
        name: 服务器名。
    """
    low, high = MCP_TIMEOUT_RANGE
    timeout = _field(server, "timeout_s")
    if timeout is None:
        return
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        report.add_error(
            "V8",
            f"timeout_s 必须在 {low}–{high} 秒之间（当前是{_describe(timeout)}）",
            server=name,
        )
        return
    if not low <= timeout <= high:
        report.add_error(
            "V8", f"timeout_s 必须在 {low}–{high} 秒之间（当前 {timeout}）", server=name
        )


def _check_prefix(server: RawEntry, report: ValidationReport, name: str) -> None:
    """V9：``tool_prefix``（若给）匹配 ``MCP_PREFIX_RE``。

    Args:
        server: 配置或原始条目。
        report: 报告（就地追加）。
        name: 服务器名。
    """
    prefix = _field(server, "tool_prefix")
    if prefix is None or prefix == "":
        return
    if not isinstance(prefix, str) or not _PREFIX_PATTERN.match(prefix):
        report.add_error(
            "V9",
            f'tool_prefix "{prefix}" 非法：只允许字母数字下划线（1–32 个字符）',
            server=name,
        )


# ============================================================================
# 工具清单校验（V11 / V12 / V13）
# ============================================================================


class _ToolProbe(BaseModel):
    """V11/V12/V13 的最小输入（原始名 + 映射名 + 描述 + 原始 schema）。"""

    raw_name: str
    mapped_name: str
    description: str = ""
    has_description: bool = True
    input_schema: Any = None
    has_input_schema: bool = False


def _probes_from_raw(server: RawEntry, tools: Sequence[Mapping[str, Any]] | None) -> list[_ToolProbe]:
    """把「调用方提供的工具声明」转成探针。

    Args:
        server: 配置或原始条目（提供 ``tool_prefix``；原始条目自带的 ``tools`` 键
            在 ``tools`` 参数为 ``None`` 时作为声明来源）。
        tools: 工具声明列表；``None`` 时返回空列表（V11/V12 延后到 deep）。

    Returns:
        探针列表。
    """
    if tools is None:
        declared = _field(server, "tools")
        tools = declared if isinstance(declared, (list, tuple)) else None
    if not tools:
        return []
    probes: list[_ToolProbe] = []
    for raw in tools:
        if not isinstance(raw, Mapping):
            probes.append(
                _ToolProbe(raw_name="", mapped_name="", description="", input_schema=raw)
            )
            continue
        raw_name = raw.get("name")
        raw_name = raw_name if isinstance(raw_name, str) else ""
        description = raw.get("description")
        probes.append(
            _ToolProbe(
                raw_name=raw_name,
                mapped_name=_mapped_name(_field(server, "tool_prefix"), raw_name),
                description=description if isinstance(description, str) else "",
                has_description="description" in raw,
                input_schema=raw.get("inputSchema"),
                has_input_schema="inputSchema" in raw,
            )
        )
    return probes


def _probes_from_discovered(tools: Sequence[McpTool]) -> list[_ToolProbe]:
    """把 deep 阶段发现的工具转成探针。

    Args:
        tools: :class:`McpTool` 列表。

    Returns:
        探针列表。
    """
    return [
        _ToolProbe(
            raw_name=tool.raw_name,
            mapped_name=tool.name,
            description=tool.description,
            has_description=True,
            input_schema=tool.input_schema,
            has_input_schema=tool.input_schema is not None,
        )
        for tool in tools
    ]


def _mapped_name(prefix: Any, raw_name: str) -> str:
    """按 §10.4 计算映射名（前缀非法时退化为原名，避免消息里出现怪名）。

    Args:
        prefix: ``tool_prefix`` 原值。
        raw_name: 原始工具名。

    Returns:
        映射后的工具名。
    """
    if isinstance(prefix, str) and prefix:
        return f"{prefix}__{raw_name}"
    return raw_name


def _check_tools(
    probes: Sequence[_ToolProbe],
    report: ValidationReport,
    server: str,
    *,
    existing_names: Mapping[str, str] | Iterable[str] | None,
) -> None:
    """V11 + V12（+ V13 警告）：工具名冲突与 schema 合法性。

    Args:
        probes: 工具探针（空列表 = 跳过，延后到 deep）。
        report: 报告（就地追加）。
        server: 服务器名。
        existing_names: 已占用工具名（value 为来源标签）。
    """
    occupied = _normalize_existing_names(existing_names)
    hint = f'在 mcp.yaml 中为该服务器设置 tool_prefix（例如 tool_prefix: "{server[:8] or "srv"}"）'
    for probe in probes:
        name = probe.mapped_name or probe.raw_name
        if not probe.raw_name:
            report.add_error(
                "V12", f"工具缺少 name（当前是 {_describe(probe.input_schema)}）", server=server
            )
            continue
        if not probe.has_description:
            report.add_error("V12", f'工具 "{probe.raw_name}" 缺少 description', server=server)
        if probe.has_input_schema and not isinstance(probe.input_schema, dict):
            report.add_error(
                "V12",
                f'工具 "{probe.raw_name}" 的 inputSchema 不是对象'
                f"（当前是 {_describe(probe.input_schema)}）",
                server=server,
            )
        if name in occupied:
            source = occupied[name]
            if source == "builtin":
                report.add_error(
                    "V11", f'工具名 "{name}" 与内置工具冲突；请设置 tool_prefix', server=server, hint=hint
                )
            else:
                report.add_error(
                    "V11",
                    f'工具名 "{name}" 与 MCP 服务器 "{source}" 的工具冲突；请设置 tool_prefix',
                    server=server,
                    hint=hint,
                )
        if DANGER_NAME_RE.search(probe.raw_name) or DANGER_NAME_RE.search(probe.description):
            report.add_warning(
                "V13",
                f'工具 "{name}" 可能具备写/执行能力，权限流水线会按 write/execute 类别处理',
                server=server,
            )


def _normalize_existing_names(
    existing_names: Mapping[str, str] | Iterable[str] | None,
) -> dict[str, str]:
    """把 ``existing_names`` 归一化为 ``{工具名: 来源标签}``。

    Args:
        existing_names: ``Mapping[str, str]``（value = 来源标签）或任意字符串可迭代对象。

    Returns:
        归一化后的映射；``set[str]`` 输入的标签为 ``"builtin"``。
    """
    if existing_names is None:
        return {}
    if isinstance(existing_names, Mapping):
        return {str(key): str(value) for key, value in existing_names.items()}
    return {str(name): "builtin" for name in existing_names}


# ============================================================================
# 文档级校验（V1）
# ============================================================================


def validate_document(
    document: Any,
    *,
    source: str = "mcp.yaml",
    env: Mapping[str, str] | None = None,
    existing_names: Mapping[str, str] | Iterable[str] | None = None,
    existing_servers: Iterable[str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> tuple[list[McpServerConfig], ValidationReport]:
    """V1 + 逐服务器静态校验：校验一份已是 Python 对象的 ``mcp.yaml`` 内容。

    Args:
        document: YAML 解析结果（期望 ``{"servers": [...]}``）。
        source: 文件名（进消息文案）。
        env: 环境变量映射（V7 用）。
        existing_names: 已占用工具名。
        existing_servers: 已存在服务器名。
        which: 命令解析器（V4 用）。

    Returns:
        ``(成功构造的服务器列表, 报告)``；构造失败的服务器只出现在报告里。
    """
    report = ValidationReport()
    if not isinstance(document, Mapping):
        report.add_error(
            "V1", f"{source} 顶层必须是 servers 列表（当前是 {_describe(document)}）"
        )
        return [], report
    entries = document.get("servers")
    if not isinstance(entries, (list, tuple)):
        report.add_error(
            "V1",
            f"{source} 顶层必须是 servers 列表（当前是 {_describe(document.get('servers'))}）",
        )
        return [], report

    servers: list[McpServerConfig] = []
    taken_servers = set(existing_servers or ())
    taken_names = _normalize_existing_names(existing_names)
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping):
            report.add_error(
                "V1", f"{source} 的 servers 第 {index} 项必须是对象（当前是 {_describe(entry)}）"
            )
            continue
        entry_report = validate_static(
            entry,
            existing_names=taken_names,
            existing_servers=taken_servers,
            env=env,
            which=which,
        )
        report.merge(entry_report)
        config = _construct(entry, entry_report) if not entry_report.errors else None
        if config is None:
            continue
        servers.append(config)
        taken_servers.add(config.name)
        for probe in _probes_from_raw(entry, entry.get("tools")):
            taken_names[probe.mapped_name or probe.raw_name] = config.name
    return servers, report


def _construct(entry: Mapping[str, Any], report: ValidationReport) -> McpServerConfig | None:
    """把 YAML 原始条目构造成 :class:`McpServerConfig`（失败 → 记 V2/V3 错误）。

    Args:
        entry: 原始条目。
        report: 报告（就地追加）。

    Returns:
        构造成功时返回配置对象，否则 ``None``。
    """
    name = entry.get("name") if isinstance(entry.get("name"), str) else ""
    try:
        return McpServerConfig.model_validate(dict(entry))
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ())) or "配置"
        report.add_error(
            "V2",
            f"服务器 {location} 非法：{first.get('msg', '结构不合法')}",
            server=name,
        )
        return None


# ============================================================================
# 校验器（含 deep 握手）
# ============================================================================


def default_client_factory(
    server: McpServerConfig, *, env: Mapping[str, str] | None = None
) -> McpClient:
    """按 ``transport`` 构造真实客户端（stdio / http / sse 兼容）。

    延迟 import：静态校验路径无需加载 ``httpx`` / 子进程模块。

    Args:
        server: 服务器配置（``${VAR}`` 会被展开）。
        env: 环境变量映射。

    Returns:
        可用的 :class:`McpClient` 实例。
    """
    from my_agent.mcp.client_http import McpHttpClient
    from my_agent.mcp.client_stdio import McpStdioClient

    if server.transport == "stdio":
        return McpStdioClient(server, env=env)
    return McpHttpClient(server, env=env)


class McpValidator:
    """MCP 校验器（§10.3）。

    Args:
        client_factory: 客户端工厂（测试注入用）；``None`` 时用
            :func:`default_client_factory`。
        env: 环境变量映射（``${VAR}`` 展开与 V7 用；默认 ``os.environ``）。
        existing_names: 默认的已占用工具名（``Mapping[str, str]`` 时可区分内置/其它 MCP）。
        existing_servers: 默认的已存在服务器名（V2 唯一性）。
        which: 命令解析器（V4 用）。
    """

    def __init__(
        self,
        *,
        client_factory: Callable[[McpServerConfig], McpClient] | None = None,
        env: Mapping[str, str] | None = None,
        existing_names: Mapping[str, str] | Iterable[str] | None = None,
        existing_servers: Iterable[str] | None = None,
        which: Callable[[str], str | None] | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._env = env
        self._existing_names = existing_names
        self._existing_servers = existing_servers
        self._which = which

    def validate_static(
        self,
        server: RawEntry,
        *,
        existing_names: Mapping[str, str] | Iterable[str] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ValidationReport:
        """方法形式的 :func:`validate_static`（沿用实例默认值）。

        Args:
            server: 配置或原始条目。
            existing_names: 覆盖实例默认的已占用工具名。
            tools: 已知工具声明（V11/V12）。

        Returns:
            静态校验报告。
        """
        return validate_static(
            server,
            existing_names=self._existing_names if existing_names is None else existing_names,
            existing_servers=self._existing_servers,
            tools=tools,
            env=self._env,
            which=self._which,
        )

    async def validate(
        self,
        server: RawEntry,
        *,
        deep: bool = True,
        existing_names: Mapping[str, str] | Iterable[str] | None = None,
        existing_servers: Iterable[str] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ValidationReport:
        """完整校验：静态规则 + （``deep=True`` 时）V10 真实握手 + V11/V12/V13。

        Args:
            server: 已构造的 :class:`McpServerConfig`，或 YAML 原始条目。
            deep: ``True`` 执行 V10 连通性握手（``initialize`` + ``tools/list``）；
                ``False`` 只做静态规则（启动加载路径**必须**传 ``False``，F-2）。
            existing_names: 覆盖实例默认的已占用工具名。
            existing_servers: 覆盖实例默认的已存在服务器名。
            tools: 调用方已知的工具声明（静态 V11/V12；deep 时以握手结果为准）。

        Returns:
            :class:`ValidationReport`；``deep=True`` 时 ``discovered_tools`` /
            ``server_info`` / ``protocol_version`` 已填充。

        Raises:
            McpValidationError: 传入对象既不是配置也不是映射。
        """
        config: McpServerConfig | None
        if isinstance(server, McpServerConfig):
            report = validate_static(
                server,
                existing_names=self._existing_names if existing_names is None else existing_names,
                existing_servers=(
                    self._existing_servers if existing_servers is None else existing_servers
                ),
                tools=tools,
                env=self._env,
                which=self._which,
            )
            config = server
        elif isinstance(server, Mapping):
            raw_report = validate_static(
                server,
                existing_names=self._existing_names if existing_names is None else existing_names,
                existing_servers=(
                    self._existing_servers if existing_servers is None else existing_servers
                ),
                tools=tools,
                env=self._env,
                which=self._which,
            )
            report = raw_report
            config = None
            if not raw_report.errors:
                config = _construct(server, raw_report)
        else:
            raise McpValidationError(detail=f"无法校验的对象类型：{type(server).__name__}")

        if not deep or config is None or report.errors:
            if config is not None and report.server == "":
                report.server = config.name
            return report

        report.server = config.name
        await self._deep_check(config, report)
        return report

    async def _deep_check(self, config: McpServerConfig, report: ValidationReport) -> None:
        """V10 握手 + 基于真实工具清单的 V11/V12/V13。

        Args:
            config: 已通过静态校验的配置。
            report: 报告（就地追加）。
        """
        expanded, missing = expand_server(config, env=self._env)
        if missing:
            # 静态 V7 已报错；此处只是二次防线（例如调用方跳过了静态校验）。
            report.add_error(
                "V7",
                f"环境变量 {missing[0][0]} 未设置（{missing[0][1]}）",
                server=config.name,
            )
            return
        factory = self._client_factory or self._make_default_client
        client: McpClient | None = None
        try:
            client = factory(expanded)
        except OSError as exc:
            report.add_error("V10", f"无法启动 MCP 服务器：{exc}", server=config.name)
            return
        try:
            await self._handshake(config, client, report)
        finally:
            await self._safe_close(client)

    async def _handshake(
        self, config: McpServerConfig, client: McpClient, report: ValidationReport
    ) -> None:
        """执行 ``initialize`` + ``tools/list`` 并填充报告（V10/V11/V12/V13）。

        Args:
            config: 服务器配置。
            client: 已构造的客户端。
            report: 报告（就地追加）。
        """
        timeout = config.timeout_s
        try:
            info = await client.initialize()
        except McpTimeoutError:
            report.add_error(
                "V10", f"initialize 超时（{timeout}s）：服务未响应", server=config.name
            )
            return
        except McpValidationError as exc:
            report.add_error("V10", f"initialize 失败：{_detail_of(exc)}", server=config.name)
            return
        except OSError as exc:
            report.add_error("V10", f"无法启动 MCP 服务器：{exc}", server=config.name)
            return
        server_info = info.get("serverInfo") if isinstance(info, dict) else None
        report.server_info = server_info if isinstance(server_info, dict) else {}
        version = getattr(client, "protocol_version", "")
        if (not isinstance(version, str) or not version) and isinstance(info, dict):
            raw_version = info.get("protocolVersion")
            version = raw_version if isinstance(raw_version, str) else ""
        if not isinstance(version, str) or not version:
            report.add_error(
                "V10", "initialize 返回非法结构：缺少 protocolVersion", server=config.name
            )
            return
        report.protocol_version = version

        try:
            tools = await client.list_tools()
        except McpTimeoutError:
            report.add_error(
                "V10", f"tools/list 超时（{timeout}s）：服务未响应", server=config.name
            )
            return
        except McpValidationError as exc:
            report.add_error("V10", _detail_of(exc), server=config.name)
            return
        except OSError as exc:
            report.add_error("V10", f"tools/list 失败：{exc}", server=config.name)
            return

        report.discovered_tools = [tool.to_tool_spec() for tool in tools]
        _check_tools(
            _probes_from_discovered(tools),
            report,
            config.name,
            existing_names=self._existing_names,
        )

    async def _safe_close(self, client: McpClient) -> None:
        """关闭 deep 校验期间启动的客户端（**绝不**因关闭失败而掩盖校验结论）。

        Args:
            client: 待关闭客户端（关闭异常降级为调试日志）。
        """
        try:
            await client.close()
        except (McpValidationError, McpTimeoutError, OSError) as exc:
            _LOG.debug(
                "关闭 MCP 校验客户端失败 server=%s: %s", getattr(client, "server_name", "?"), exc
            )

    def _make_default_client(self, server: McpServerConfig) -> McpClient:
        """用包内默认工厂构造客户端（沿用实例的 ``env``）。

        Args:
            server: 服务器配置。

        Returns:
            客户端实例。
        """
        return default_client_factory(server, env=self._env)


def _detail_of(exc: McpValidationError) -> str:
    """取异常的可读消息（``detail`` 优先，兼容未实现 ``user_message`` 的场景）。

    Args:
        exc: MCP 校验异常。

    Returns:
        消息文本。
    """
    detail = getattr(exc, "detail", "")
    return detail or str(exc)
