"""核心数据模型（§4 全局契约）。

本包是**所有模块的共同语言**：跨模块数据必须是这里定义的 pydantic 模型，
禁止传裸 dict（C5）。规格：``my_agent-开发文档.md`` §4.1–§4.5。
"""

from __future__ import annotations

from my_agent.models.audit import CompactEvent, MemoryWriteRecord
from my_agent.models.messages import Message, Role, ToolCallRequest
from my_agent.models.permission import (
    AuditRecord,
    Confidence,
    Decision,
    JudgeOutput,
    PermissionDecision,
    Risk,
    SafetyMode,
)
from my_agent.models.protocols import (
    Confirmer,
    ConsolePort,
    DisplayPort,
    PermissionRequest,
    UserAnswer,
)
from my_agent.models.session import SessionState, TurnRecord
from my_agent.models.tools import ToolContext, ToolInvocation, ToolResult, ToolSpec

__all__ = [
    "AuditRecord",
    "CompactEvent",
    "Confidence",
    "Confirmer",
    "ConsolePort",
    "Decision",
    "DisplayPort",
    "JudgeOutput",
    "MemoryWriteRecord",
    "Message",
    "PermissionDecision",
    "PermissionRequest",
    "Risk",
    "Role",
    "SafetyMode",
    "SessionState",
    "ToolCallRequest",
    "ToolContext",
    "ToolInvocation",
    "ToolResult",
    "ToolSpec",
    "TurnRecord",
    "UserAnswer",
]
