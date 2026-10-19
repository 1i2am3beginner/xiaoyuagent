"""Hook 事件、负载与结果模型（§11.1–§11.4）。

本模块是 ``hooks/`` 的数据契约层，**不产生任何副作用**（仅 pydantic 模型与枚举）：

* :class:`HookEvent` —— 规格 §11.1 的 **7 个事件**（U11：``HOOK_MERGE_PERMISSION_EVENTS =
  false``），另加引擎使用的两个生命周期事件 ``session_start`` / ``session_end``
  （CONTRACTS §17.3）。
* :class:`HookPayload` —— 事件负载，**浅合并值域**：hook 只能改动这里出现的字段。
* :class:`HookOutcome` —— hook 执行后的聚合结果；``continue_`` 的 JSON 字段名是
  ``"continue"``（``continue`` 是 Python 关键字）。

能力矩阵（§11.4）由 :func:`mutable_fields` 与 :data:`EDITABLE_EVENTS` 表达：
``after_*`` 事件的入参修改必须被忽略并告警（§11.3），只有模型/工具**输出**可被改写。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from my_agent.models.messages import Message
from my_agent.models.permission import Decision
from my_agent.models.tools import ToolResult

__all__ = [
    "EDITABLE_EVENTS",
    "HookEvent",
    "HookOutcome",
    "HookPayload",
    "OUTPUT_EVENTS",
    "SPEC_EVENTS",
    "TOOL_EVENTS",
    "mutable_fields",
]


class HookEvent(StrEnum):
    """可挂载的 hook 事件（§11.1 的 7 个 + §17.3 的 2 个生命周期事件）。

    Attributes:
        USER_INPUT_RECEIVED: 用户输入被接收后、进入引擎处理前。
        BEFORE_MODEL_CALL: 每次调用模型前。
        AFTER_MODEL_CALL: 模型返回后（含流式结束）。
        BEFORE_TOOL_CALL: 工具执行前（权限判定**之前**）。
        AFTER_TOOL_CALL: 工具执行后。
        BEFORE_PERMISSION_CHECK: 权限流水线开始前。
        AFTER_PERMISSION_CHECK: 权限裁决产出后（**只能收紧**，§11.4 安全不变量）。
        SESSION_START: 会话启动（§17.3，引擎 ``_emit_session_start`` 使用）。
        SESSION_END: 会话结束（§17.3，引擎 ``shutdown`` 使用）。
    """

    # ---- §11.1 的 7 个事件 ----
    USER_INPUT_RECEIVED = "user_input_received"
    BEFORE_MODEL_CALL = "before_model_call"
    AFTER_MODEL_CALL = "after_model_call"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_PERMISSION_CHECK = "before_permission_check"
    AFTER_PERMISSION_CHECK = "after_permission_check"

    # ---- CONTRACTS §17.3：生命周期事件（引擎已在调用）----
    SESSION_START = "session_start"
    SESSION_END = "session_end"


#: 规格 §11.1 表中的 7 个事件（U11 默认口径，7 而非 6）。
SPEC_EVENTS: tuple[HookEvent, ...] = (
    HookEvent.USER_INPUT_RECEIVED,
    HookEvent.BEFORE_MODEL_CALL,
    HookEvent.AFTER_MODEL_CALL,
    HookEvent.BEFORE_TOOL_CALL,
    HookEvent.AFTER_TOOL_CALL,
    HookEvent.BEFORE_PERMISSION_CHECK,
    HookEvent.AFTER_PERMISSION_CHECK,
)
"""§11.1 能力矩阵覆盖的事件集合（顺序与规格表一致）。"""

#: 事件名 → 负载中**允许被 hook 改写**的字段（§11.3「仅 before_* 事件允许」+ §11.4 输出列）。
_EDITABLE_BY_EVENT: dict[HookEvent, frozenset[str]] = {
    HookEvent.USER_INPUT_RECEIVED: frozenset({"text"}),
    HookEvent.BEFORE_MODEL_CALL: frozenset({"text", "messages"}),
    HookEvent.AFTER_MODEL_CALL: frozenset({"text"}),
    HookEvent.BEFORE_TOOL_CALL: frozenset({"arguments"}),
    HookEvent.AFTER_TOOL_CALL: frozenset({"result"}),
    HookEvent.BEFORE_PERMISSION_CHECK: frozenset({"arguments"}),
    HookEvent.AFTER_PERMISSION_CHECK: frozenset(),
}

#: 允许 ``modified_payload`` 生效的事件集合（其余事件忽略并告警，§11.3）。
EDITABLE_EVENTS: frozenset[HookEvent] = frozenset(_EDITABLE_BY_EVENT)

#: 可改写「模型/工具输出」的事件（§11.4 的「可改模型输出」列）。
OUTPUT_EVENTS: frozenset[HookEvent] = frozenset(
    {HookEvent.AFTER_MODEL_CALL, HookEvent.AFTER_TOOL_CALL}
)

#: 携带单个工具名、``matcher`` 正则可过滤的事件。
TOOL_EVENTS: frozenset[HookEvent] = frozenset(
    {
        HookEvent.BEFORE_TOOL_CALL,
        HookEvent.AFTER_TOOL_CALL,
        HookEvent.BEFORE_PERMISSION_CHECK,
        HookEvent.AFTER_PERMISSION_CHECK,
    }
)


def mutable_fields(event: HookEvent) -> frozenset[str]:
    """返回该事件下允许 hook 改写的 :class:`HookPayload` 字段名。

    Args:
        event: 事件。

    Returns:
        允许改写的字段名集合；``after_permission_check`` 返回空集（裁决已定，§11.4）。
    """
    return _EDITABLE_BY_EVENT.get(event, frozenset())


class HookPayload(BaseModel):
    """发给 hook 的事件负载（§11.3 的 ``payload`` 对象）。

    浅合并值域：hook 返回的 ``modified_payload`` 只覆盖这里**非 None** 的字段，
    其余字段保持原值（由 :meth:`overlay` 实现）。

    Attributes:
        text: 用户输入文本 / 模型输出文本。
        messages: 模型输入消息（``before_model_call``）。
        tool: 工具名（工具相关事件）。
        arguments: 工具入参。
        reason: 调用原因。
        result: 工具结果（``after_tool_call``）。
        decision: 当前权限裁决（``after_permission_check`` 的输入）。
        subagent_id: 来源子 Agent（§12；事件名不变）。
    """

    model_config = ConfigDict(extra="ignore")

    text: str | None = None
    messages: list[Message] | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    reason: str | None = None
    result: ToolResult | None = None
    decision: Decision | None = None
    subagent_id: str | None = None

    def overlay(self, changes: HookPayload, allowed: frozenset[str]) -> HookPayload:
        """把 ``changes`` 中**允许**且非 None 的字段浅合并到自身之上。

        Args:
            changes: hook 返回的 ``modified_payload``。
            allowed: 本事件允许改写的字段名（见 :func:`mutable_fields`）。

        Returns:
            合并后的新负载；``self`` 不被修改。
        """
        updates: dict[str, Any] = {}
        for name in allowed:
            value = getattr(changes, name)
            if value is not None:
                updates[name] = value
        return self.model_copy(update=updates) if updates else self.model_copy(deep=True)

    def to_json_dict(self) -> dict[str, Any]:
        """序列化为 stdin JSON（剔除全部 None 字段，§11.3）。

        Returns:
            形如 ``{"tool": "Write", "arguments": {...}}`` 的字典。
        """
        return self.model_dump(exclude_none=True, mode="json")


class HookOutcome(BaseModel):
    """一次 :meth:`HookRunner.run` 的聚合结果（链式合并后）。

    Attributes:
        decision: 生效裁决；``None`` = 不改变原流程。``"ask"`` 强制转人工确认。
        modified_payload: 链式累积后的负载改写；仅 ``before_*``/输出类事件生效。
        message: 展示给用户并进审计的文案（多 hook 时按顺序拼接）。
        continue_: ``False`` → 中止整个回合（JSON 字段名 ``"continue"``）。
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    decision: Decision | None = None
    modified_payload: HookPayload | None = None
    message: str = ""
    continue_: bool = Field(default=True, alias="continue")

    @property
    def aborted(self) -> bool:
        """是否要求中止整个回合（``continue_ is False`` 的可读别名）。

        Returns:
            ``True`` 表示 :attr:`continue_` 为 ``False``。
        """
        return not self.continue_


# 解析前向引用（``ToolResult`` / ``Message`` / ``Decision`` 已在上方导入）。
HookPayload.model_rebuild()
HookOutcome.model_rebuild()
