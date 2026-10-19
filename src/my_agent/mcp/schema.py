"""MCP 领域模型与共享词表（§10.2、§10.4、§10.5）。

本模块是 ``mcp/`` 包的**最底层**：只依赖 ``models`` 与 ``config``，不依赖
``validator`` / ``client_*`` / ``manager``，因此三者都能安全地 import 它。

包含三部分：

1. 配置模型再导出：``McpServerConfig`` / ``McpConfig`` 定义在 ``config/schema.py``
   （避免 ``config`` 反向依赖 ``mcp``），此处按契约 §11 再导出，两处是**同一个类型**。
2. ``McpClient`` 协议：§10.5 冻结的四个方法（``initialize`` / ``list_tools`` /
   ``call_tool`` / ``close``）。
3. §10.4 工具映射的纯函数：名称前缀、描述兜底、类别推断、破坏性判定，
   以及 §10.2 的 ``${ENV_VAR}`` 展开。

规格：``my_agent-开发文档.md`` §10.1–§10.5；契约：``CONTRACTS.md`` §11。
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from my_agent.config.schema import McpConfig, McpServerConfig
from my_agent.models.tools import ToolCategory, ToolResult, ToolSpec

__all__ = [
    "DANGER_NAME_RE",
    "DESTRUCTIVE_NAME_RE",
    "ENV_PLACEHOLDER_RE",
    "MCP_CLIENT_NAME",
    "MCP_CLIENT_VERSION",
    "MCP_JSONRPC_VERSION",
    "MCP_JSON_PREVIEW_CHARS",
    "MCP_MAX_TOOL_PAGES",
    "MCP_PROTOCOL_VERSION",
    "MCP_STDERR_CHUNK_BYTES",
    "MCP_STDERR_TAIL_CHARS",
    "MCP_STREAM_LIMIT_BYTES",
    "MUTATING_NAME_RE",
    "READ_NAME_RE",
    "McpClient",
    "McpConfig",
    "McpServerConfig",
    "McpTool",
    "description_fallback",
    "expand_env_placeholders",
    "expand_server",
    "infer_category",
    "is_destructive_name",
    "map_mcp_tool",
    "map_mcp_tools",
    "mapped_tool_name",
    "render_content",
    "short_json",
    "tool_prefix_of",
]

# ============================================================================
# 常量
#
# 说明（需要上游确认）：契约 §0.1 冻结的 §10 常量只有 MCP_NAME_RE / MCP_PREFIX_RE /
# MCP_TIMEOUT_RANGE / MCP_DEFAULT_TIMEOUT_S / MCP_TERMINATE_GRACE_S 五个。以下协议级
# 常量与 §10.4 的推断正则**规格已逐字给出**（非可调阈值），故集中定义在 mcp 包内，
# 避免散落字面量；建议后续按 §0.1 迁入 config/defaults.py（见交付报告）。
# ============================================================================

MCP_JSONRPC_VERSION = "2.0"
"""JSON-RPC 版本（协议固定）。"""

MCP_PROTOCOL_VERSION = "2024-11-05"
"""客户端在 ``initialize`` 中声明的 MCP 协议版本（§10.1：2024-11-05 起）。"""

MCP_CLIENT_NAME = "myagent"
"""``initialize`` 的 ``clientInfo.name``。"""

MCP_CLIENT_VERSION = "0.1.0"
"""``initialize`` 的 ``clientInfo.version``。"""

MCP_STDERR_TAIL_CHARS = 2048
"""子进程退出时保留的 stderr 尾部字符数（§10.5「stderr 尾部 2k」）。"""

MCP_STDERR_CHUNK_BYTES = 65536
"""stderr 读取块大小；10MB 洪泛下内存占用仍恒定为 O(尾部 2k)。"""

MCP_STREAM_LIMIT_BYTES = 8 * 1024 * 1024
"""stdout StreamReader 的单行上限（超出即报错，避免 ``LimitOverrunError`` 击穿协议层）。"""

MCP_MAX_TOOL_PAGES = 20
"""``tools/list`` 跟随 ``nextCursor`` 的最大页数（防御服务端游标死循环）。"""

READ_NAME_RE = re.compile(r"(?i)^(read|get|list|search|query|fetch|describe)")
"""§10.4：命中 → ``category="read"``。"""

MUTATING_NAME_RE = re.compile(r"(?i)(delete|remove|drop|write|create|update|exec|run|publish)")
"""§10.4：命中 → ``category="write"``。"""

DESTRUCTIVE_NAME_RE = re.compile(r"(?i)(delete|remove|drop|destroy|purge|reset)")
"""§10.4：命中 → ``is_destructive=True``。"""

DANGER_NAME_RE = re.compile(r"(?i)(delete|remove|drop|exec|shell|write|publish)")
"""§10.3.1 V13：工具名/描述命中 → 危险能力警告（不拒绝）。"""

ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
"""§10.2 的 ``${ENV_VAR}`` 占位符（仅支持该形式，不支持 ``${VAR:-default}``）。"""

MCP_JSON_PREVIEW_CHARS = 500
"""错误消息/未知内容块里 JSON 预览的最大字符数。"""


# ============================================================================
# 领域模型
# ============================================================================


class McpTool(BaseModel):
    """一个 MCP 工具：原始声明 + 映射后的 :class:`ToolSpec`（§10.4）。

    ``list_tools()`` 返回本对象而非裸 ``ToolSpec``，原因：
    V12 需要**原始** ``inputSchema``（映射后的 ``parameters`` 已被规整为 dict）、
    V13 需要原始描述、审计需要 ``mcp_server``。映射结果本身仍是 ``ToolSpec``，
    通过 :attr:`spec` / :meth:`to_tool_spec` 取得——§10.3 的
    ``ValidationReport.discovered_tools: list[ToolSpec]`` 契约因此不受影响。

    Attributes:
        mcp_server: 来源服务器名（= §10.4 审计字段 ``mcp_server``）。
        name: 映射后的工具名（含 ``tool_prefix``）。
        raw_name: MCP 侧原始工具名（类别推断按它匹配，见 :func:`infer_category`）。
        description: 映射后的描述（空描述已替换为兜底文案）。
        input_schema: MCP ``inputSchema`` 原样（类型不限，V12 用它判合法性）。
        spec: 映射后的内置 :class:`ToolSpec`。
    """

    mcp_server: str = Field(description="来源 MCP 服务器名（审计字段 mcp_server）")
    name: str = Field(description="映射后的工具名（含 tool_prefix）")
    raw_name: str = Field(description="MCP 侧原始工具名")
    description: str = Field(default="", description="映射后的描述")
    input_schema: Any = Field(default=None, description="MCP inputSchema 原样")
    spec: ToolSpec = Field(description="映射后的内置 ToolSpec（§10.4）")

    model_config = {"extra": "ignore"}

    def to_tool_spec(self) -> ToolSpec:
        """返回映射后的内置工具声明。

        Returns:
            §10.4 映射得到的 :class:`~my_agent.models.tools.ToolSpec`。
        """
        return self.spec

    def audit_fields(self) -> dict[str, str]:
        """返回审计层需要补写的字段（§10.4 审计要求，F-8）。

        ``security/audit.py`` 组装 ``AuditRecord`` 时用本方法填充
        ``AuditRecord.mcp_server``；``ToolResult`` 因 ``extra="ignore"`` 无法承载该字段。

        Returns:
            形如 ``{"mcp_server": "filesystem", "tool": "fs__read_file"}``。
        """
        return {"mcp_server": self.mcp_server, "tool": self.name}


@runtime_checkable
class McpClient(Protocol):
    """MCP 客户端协议（§10.5，签名冻结）。

    实现：:class:`~my_agent.mcp.client_stdio.McpStdioClient`、
    :class:`~my_agent.mcp.client_http.McpHttpClient`。
    """

    @property
    def server_name(self) -> str:
        """本客户端对应的服务器名（用于错误消息与审计）。"""
        ...

    @property
    def protocol_version(self) -> str:
        """``initialize`` 协商后记录的服务端 ``protocolVersion``（未握手时为空串）。"""
        ...

    async def initialize(self) -> dict[str, Any]:
        """执行 MCP ``initialize`` 握手并发送 ``notifications/initialized``。

        Returns:
            ``initialize`` 的 ``result``（含 ``protocolVersion`` / ``serverInfo``）。

        Raises:
            McpTimeoutError: 超过 ``server.timeout_s`` 未响应。
            McpValidationError: 进程/连接失败或返回结构非法。
        """
        ...

    async def list_tools(self) -> list[McpTool]:
        """执行 MCP ``tools/list``，并映射为 :class:`McpTool`（§10.4）。

        Returns:
            映射后的工具列表（每条含 :class:`ToolSpec`）。

        Raises:
            McpTimeoutError: 超过 ``server.timeout_s`` 未响应。
            McpValidationError: 返回结构非法（如缺少 ``result.tools``）。
        """
        ...

    async def call_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        """执行 MCP ``tools/call``。

        Args:
            name: MCP 侧**原始**工具名（映射/前缀由 manager 还原）。
            args: 工具参数（原样作为 ``arguments``）。

        Returns:
            规范化后的 :class:`ToolResult`。

        Raises:
            McpTimeoutError: 超过 ``server.timeout_s`` 未响应（**不重试**）。
            McpValidationError: 进程已退出或返回结构非法。
        """
        ...

    async def close(self) -> None:
        """释放资源：stdio 为 ``terminate`` → 等待 → ``kill``；http 为关闭连接池。"""
        ...


# ============================================================================
# §10.4 工具映射（纯函数，validator / manager / client_* 共用）
# ============================================================================


def tool_prefix_of(server: McpServerConfig) -> str:
    """取服务器的 ``tool_prefix``（空串与 ``None`` 等价）。

    Args:
        server: 服务器配置。

    Returns:
        前缀字符串；未配置时返回空串。
    """
    return (server.tool_prefix or "").strip()


def mapped_tool_name(server: McpServerConfig, raw_name: str) -> str:
    """按 §10.4 计算映射后的工具名。

    Args:
        server: 服务器配置（提供 ``tool_prefix``）。
        raw_name: MCP 侧原始工具名。

    Returns:
        ``f"{tool_prefix}__{tool_name}"``；无前缀时为原名。
    """
    prefix = tool_prefix_of(server)
    return f"{prefix}__{raw_name}" if prefix else raw_name


def description_fallback(server_name: str, raw_name: str) -> str:
    """§10.4 的空描述兜底文案。

    Args:
        server_name: 服务器名。
        raw_name: MCP 侧原始工具名。

    Returns:
        ``"(MCP 工具 <server>.<tool> 未提供描述)"``。
    """
    return f"(MCP 工具 {server_name}.{raw_name} 未提供描述)"


def infer_category(raw_name: str) -> ToolCategory:
    """按 §10.4 推断工具类别（MCP 不声明类别）。

    规则：默认 ``execute``（保守）；命中 :data:`READ_NAME_RE` → ``read``；
    命中 :data:`MUTATING_NAME_RE` → ``write``。

    **实现口径**（规格未定义同时命中的情形）：写模式优先于读模式，取更保守的
    ``write``——例如 ``get_and_delete`` 判为 ``write`` 而非 ``read``。匹配对象是
    **原始工具名**：若按加前缀后的名字匹配，``fs__read_file`` 会因锚定 ``^`` 失配
    而退化成 ``execute``。

    Args:
        raw_name: MCP 侧原始工具名。

    Returns:
        ``"read"`` / ``"write"`` / ``"execute"``。
    """
    if MUTATING_NAME_RE.search(raw_name):
        return "write"
    if READ_NAME_RE.search(raw_name):
        return "read"
    return "execute"


def is_destructive_name(raw_name: str) -> bool:
    """按 §10.4 判断工具名是否具破坏性。

    Args:
        raw_name: MCP 侧原始工具名。

    Returns:
        命中 :data:`DESTRUCTIVE_NAME_RE` 时为 ``True``。
    """
    return bool(DESTRUCTIVE_NAME_RE.search(raw_name))


def map_mcp_tool(server: McpServerConfig, raw: Mapping[str, Any]) -> McpTool:
    """把一条 MCP 工具声明映射为 :class:`McpTool`（§10.4 全字段规则）。

    Args:
        server: 来源服务器配置。
        raw: MCP ``tools/list`` 返回的单个工具对象。

    Returns:
        映射结果；``inputSchema`` 非对象时 ``parameters`` 置空但仍保留原值，
        交由 V12 报错。
    """
    raw_name = str(raw.get("name") or "")
    raw_description = raw.get("description")
    description = raw_description if isinstance(raw_description, str) else ""
    if not description.strip():
        description = description_fallback(server.name, raw_name)
    input_schema = raw.get("inputSchema")
    parameters = input_schema if isinstance(input_schema, dict) else {}
    category = infer_category(raw_name)
    spec = ToolSpec(
        name=mapped_tool_name(server, raw_name),
        description=description,
        parameters=parameters,
        category=category,
        is_read_only=category == "read",
        is_destructive=is_destructive_name(raw_name),
        source="mcp",
    )
    return McpTool(
        mcp_server=server.name,
        name=spec.name,
        raw_name=raw_name,
        description=description,
        input_schema=input_schema,
        spec=spec,
    )


def map_mcp_tools(server: McpServerConfig, raws: list[Mapping[str, Any]]) -> list[McpTool]:
    """批量映射 MCP 工具声明。

    Args:
        server: 来源服务器配置。
        raws: ``tools/list`` 返回的工具对象列表。

    Returns:
        映射结果列表，顺序与入参一致。
    """
    return [map_mcp_tool(server, raw) for raw in raws]


# ============================================================================
# §10.2 ${ENV_VAR} 展开
# ============================================================================


def expand_env_placeholders(text: str, *, env: Mapping[str, str] | None = None) -> tuple[str, list[str]]:
    """展开字符串中的 ``${ENV_VAR}``（§10.2）。

    Args:
        text: 待展开字符串。
        env: 环境变量映射（默认 ``os.environ``）。

    Returns:
        ``(展开后的字符串, 未解析到的变量名列表)``；未解析的占位符**原样保留**
        （交由 V7 报错，绝不静默替换为空串）。
    """
    source = os.environ if env is None else env
    missing: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        var = match.group(1)
        value = source.get(var)
        if value is None:
            missing.append(var)
            return match.group(0)
        return value

    return ENV_PLACEHOLDER_RE.sub(_replace, text), missing


def expand_server(
    server: McpServerConfig, *, env: Mapping[str, str] | None = None
) -> tuple[McpServerConfig, list[tuple[str, str]]]:
    """展开服务器配置里 ``headers`` 与 ``env`` 的 ``${ENV_VAR}``。

    §10.2 明确要求 ``headers`` 值支持展开；stdio 的 ``env`` 值同样展开（超集，
    理由：子进程环境变量同样需要从宿主环境注入密钥），未解析到的位置一并返回，
    由 V7 报错：``环境变量 DOCS_TOKEN 未设置（headers.Authorization）``。

    Args:
        server: 服务器配置（保持原值不被修改）。
        env: 环境变量映射（默认 ``os.environ``）。

    Returns:
        ``(展开后的配置副本, [(变量名, 位置), ...])``；无占位符时返回原对象。
    """
    missing: list[tuple[str, str]] = []
    updates: dict[str, dict[str, str]] = {}
    for field in ("headers", "env"):
        values: dict[str, str] = getattr(server, field) or {}
        if not values:
            continue
        expanded: dict[str, str] = {}
        for key, value in values.items():
            if not isinstance(value, str):
                expanded[key] = value
                continue
            new_value, vars_missing = expand_env_placeholders(value, env=env)
            missing.extend((var, f"{field}.{key}") for var in vars_missing)
            expanded[key] = new_value
        updates[field] = expanded
    if not updates:
        return server, missing
    return server.model_copy(update=updates), missing


# ============================================================================
# tools/call 结果渲染（stdio / http 客户端共用）
# ============================================================================


def short_json(value: Any, limit: int = MCP_JSON_PREVIEW_CHARS) -> str:
    """把任意值压成短字符串（超长截断），用于错误消息与未知内容块。

    Args:
        value: 任意值。
        limit: 最大字符数。

    Returns:
        紧凑 JSON（不可序列化时退化为 ``str``）。
    """
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}…[已截断]"


def render_content(content: Any) -> str:
    """把 MCP ``tools/call`` 结果的 ``content`` 渲染为纯文本。

    Args:
        content: ``content`` 字段（内容块列表、字符串或 ``None``）。

    Returns:
        拼接后的文本；未知块类型退化为紧凑 JSON。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return short_json(content)
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            kind = item.get("type")
            if kind == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif kind == "image":
                parts.append(f"[图片 {item.get('mimeType', '未知类型')}]")
            elif kind == "resource":
                parts.append(f"[资源 {short_json(item.get('resource') or item.get('uri'))}]")
            else:
                parts.append(short_json(item))
    return "\n".join(parts)
