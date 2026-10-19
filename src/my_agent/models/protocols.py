"""跨模块协议（§4.5）与显示端口。

C6：L3 能力层**不得** import ``ui/``；需要用户输入或输出时，通过这里定义的
Protocol 回调（由 L5 注入实现）。
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from my_agent.models.permission import Risk

__all__ = [
    "AutoRejectConfirmer",
    "Confirmer",
    "ConsolePort",
    "DisplayPort",
    "NullDisplay",
    "PermissionRequest",
    "UserAnswer",
]


class UserAnswer(BaseModel):
    """用户对权限询问的回答。"""

    choice: Literal["once", "always", "reject", "reject_with_reason", "abort_turn"]
    note: str | None = None

    model_config = {"extra": "ignore"}


class PermissionRequest(BaseModel):
    """展示给用户的询问卡片内容（§6.9）。"""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    risk: Risk = "medium"
    reason: str = ""
    phase_reached: int = 0
    permanent_eligible: bool = True
    preview: str | None = Field(default=None, description="例如 diff 预览")
    subagent_id: str | None = Field(default=None, description="§12：标注来源 [子 Agent <id>]")

    model_config = {"extra": "ignore"}


@runtime_checkable
class Confirmer(Protocol):
    """权限询问回调（由 UI 层实现，L3 依赖它而非 ``ui/``）。"""

    async def ask(self, request: PermissionRequest) -> UserAnswer:
        """向用户展示询问卡片并等待回答。

        Args:
            request: 询问卡片内容。

        Returns:
            用户的回答。
        """
        ...


@runtime_checkable
class DisplayPort(Protocol):
    """最小的输出端口（L3/L4 可依赖）。"""

    def stream_text(self, delta: str) -> None:
        """输出流式文本增量。"""
        ...

    def info(self, msg: str) -> None:
        """输出提示信息。"""
        ...

    def warn(self, msg: str) -> None:
        """输出警告。"""
        ...

    def error(self, msg: str) -> None:
        """输出错误。"""
        ...

    def tool_start(self, name: str, summary: str) -> None:
        """工具开始执行。"""
        ...

    def tool_result(self, name: str, ok: bool, summary: str) -> None:
        """工具执行结束。"""
        ...


@runtime_checkable
class ConsolePort(DisplayPort, Protocol):
    """完整控制台端口（L4/L5 使用）。"""

    def render_context_report(self, report: Any) -> None:
        """渲染 ``/context`` 报告。"""
        ...

    def render_cost(self, text: str) -> None:
        """渲染 ``/cost`` 报告。"""
        ...

    def render_permission_card(self, request: PermissionRequest) -> None:
        """渲染权限询问卡片。"""
        ...


class NullDisplay:
    """不做任何输出的显示端口（测试与非交互场景的默认值）。"""

    def stream_text(self, delta: str) -> None:
        """忽略流式文本。"""

    def info(self, msg: str) -> None:
        """忽略提示。"""

    def warn(self, msg: str) -> None:
        """忽略警告。"""

    def error(self, msg: str) -> None:
        """忽略错误。"""

    def tool_start(self, name: str, summary: str) -> None:
        """忽略工具开始事件。"""

    def tool_result(self, name: str, ok: bool, summary: str) -> None:
        """忽略工具结束事件。"""


class AutoRejectConfirmer:
    """非交互模式的确认器：遇 ask 一律 reject（§6.9，**绝不默认 allow**）。"""

    def __init__(self, note: str = "非交互模式下无法询问用户") -> None:
        self.note = note

    async def ask(self, request: PermissionRequest) -> UserAnswer:
        """直接拒绝。

        Args:
            request: 询问卡片（被忽略）。

        Returns:
            固定为 ``reject_with_reason`` 的回答。
        """
        return UserAnswer(choice="reject_with_reason", note=self.note)
