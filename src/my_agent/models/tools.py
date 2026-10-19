"""工具模型（§4.2）与工具执行上下文（§14.1）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from my_agent.config.schema import Config

__all__ = ["ToolContext", "ToolInvocation", "ToolResult", "ToolSpec"]

ToolCategory = Literal["read", "write", "execute", "network", "meta"]
ToolSource = Literal["builtin", "mcp", "skill", "subagent"]


class ToolSpec(BaseModel):
    """一个工具的静态声明（进 tools 提示词，也是权限判定的输入）。"""

    name: str = Field(description="唯一名，snake_case 或 PascalCase（如 Task、Read）")
    description: str = Field(description="给模型看的描述")
    parameters: dict[str, Any] = Field(default_factory=dict, description="JSON Schema")
    category: ToolCategory
    is_read_only: bool
    is_destructive: bool = Field(default=False, description="是否破坏性（plan 模式用它判断）")
    requires_user_approval_hint: bool = False
    source: ToolSource = "builtin"

    model_config = {"extra": "ignore"}

    def to_schema(self) -> dict[str, Any]:
        """转换为发往 Provider 的 function schema。

        Returns:
            OpenAI 兼容的 ``{"type": "function", "function": {...}}``。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


class ToolInvocation(BaseModel):
    """一次待执行的工具调用（权限流水线的输入单位）。"""

    call_id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = Field(default=None, description="模型给出的调用原因（§6.5 判定输入之一）")
    spec: ToolSpec


class ToolResult(BaseModel):
    """一次工具调用的执行结果。"""

    call_id: str
    tool: str
    ok: bool
    content: str = Field(default="", description="给模型看的文本")
    error: str | None = Field(default=None, description="失败原因（给模型看）")
    exit_code: int | None = Field(default=None, description="命令类工具")
    duration_ms: int = 0
    tokens: int = 0
    truncated: bool = False
    artifact_path: str | None = Field(default=None, description="大结果落盘路径")
    denied_by_permission: bool = False

    model_config = {"extra": "ignore"}

    def as_model_text(self) -> str:
        """把结果压成回填给模型的一段文本。

        Returns:
            成功时返回 ``content``；失败时返回带原因的说明（权限拒绝会显式标注）。
        """
        if self.ok:
            return self.content
        if self.denied_by_permission:
            return f"权限被拒绝：{self.error or '未说明理由'}"
        return f"执行失败：{self.error or '未说明原因'}"


class ToolContext(BaseModel):
    """工具执行上下文（§14.1）。

    ``permission_checked`` 是 C1 双保险的载体：``ToolRegistry.execute()`` 内部断言其为
    ``True``，否则抛 :class:`~my_agent.errors.PermissionBypassedError`。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    session_id: str
    turn_index: int
    cwd: str
    project_root: str
    config: Config
    artifact_dir: str
    abort_event: Any = Field(default=None, description="asyncio.Event | None，Ctrl-C 中止信号")
    permission_checked: bool = Field(default=False, description="是否已通过权限流水线")
    call_id: str = ""
    subagent_id: str | None = None
