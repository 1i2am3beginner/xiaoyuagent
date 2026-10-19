"""MCP 模块（§10 模块 F）。

对外入口：

* 配置与校验：:class:`~my_agent.mcp.config_store.McpConfigStore`、
  :class:`~my_agent.mcp.validator.McpValidator`、:class:`ValidationReport`；
* 客户端：:class:`~my_agent.mcp.client_stdio.McpStdioClient`（stdio）、
  :class:`~my_agent.mcp.client_http.McpHttpClient`（Streamable HTTP / 旧版 SSE 兼容）；
* 编排：:class:`~my_agent.mcp.manager.MCPManager`、:class:`McpToolAdapter`
  （供 ``ToolRegistry.register``）、:class:`McpToolBinding`（审计字段 ``mcp_server``，F-8）。

规格：``my_agent-开发文档.md`` §10。
"""

from __future__ import annotations

from my_agent.mcp.client_http import McpHttpClient, decode_jsonrpc_response, parse_sse_text
from my_agent.mcp.client_stdio import McpStdioClient
from my_agent.mcp.config_store import (
    MCP_FILENAME,
    McpConfigStore,
    expand_env_placeholders,
    parse_mcp_yaml,
)
from my_agent.mcp.manager import (
    BUILTIN_TOOL_NAMES,
    MCPManager,
    McpToolAdapter,
    McpToolBinding,
)
from my_agent.mcp.schema import (
    MCP_PROTOCOL_VERSION,
    McpClient,
    McpConfig,
    McpServerConfig,
    McpTool,
    infer_category,
    map_mcp_tool,
    map_mcp_tools,
    mapped_tool_name,
)
from my_agent.mcp.validator import (
    McpValidator,
    ValidationIssue,
    ValidationReport,
    default_client_factory,
    validate_document,
    validate_static,
)

__all__ = [
    "BUILTIN_TOOL_NAMES",
    "MCP_FILENAME",
    "MCP_PROTOCOL_VERSION",
    "MCPManager",
    "McpClient",
    "McpConfig",
    "McpConfigStore",
    "McpHttpClient",
    "McpServerConfig",
    "McpStdioClient",
    "McpTool",
    "McpToolAdapter",
    "McpToolBinding",
    "McpValidator",
    "ValidationIssue",
    "ValidationReport",
    "decode_jsonrpc_response",
    "default_client_factory",
    "expand_env_placeholders",
    "infer_category",
    "map_mcp_tool",
    "map_mcp_tools",
    "mapped_tool_name",
    "parse_mcp_yaml",
    "parse_sse_text",
    "validate_document",
    "validate_static",
]
