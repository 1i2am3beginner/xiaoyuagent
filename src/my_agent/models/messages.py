"""消息模型（§4.1）。

``Message`` 同时承载「发给模型的内容」与「上下文管理所需的内部元数据」；
发往 Provider 前必须剥离内部元数据字段（见 ``llm/client.py::sanitize``）。
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

__all__ = ["Message", "Role", "ToolCallRequest", "new_message_id"]

Role = Literal["system", "user", "assistant", "tool"]

#: 不发送给模型的内部元数据字段（§4.1 不变量）。
INTERNAL_FIELDS: tuple[str, ...] = (
    "id",
    "created_at",
    "tokens",
    "kind",
    "is_large_result",
    "archive_ref",
    "cleared_marker",
    "turn_index",
    "subagent_id",
)


def new_message_id() -> str:
    """生成本地唯一消息 id。

    Returns:
        uuid4 十六进制字符串。
    """
    return uuid.uuid4().hex


class ToolCallRequest(BaseModel):
    """模型请求调用一个工具。"""

    id: str = Field(description="模型返回的 call id，用于回填结果")
    name: str = Field(description="工具名")
    arguments: dict[str, Any] = Field(default_factory=dict, description="解析后的参数")
    raw_arguments: str = Field(default="", description="原始 JSON 字符串（解析失败时保留，供审计）")

    model_config = {"extra": "ignore"}


class Message(BaseModel):
    """一条对话消息。

    不变量（§4.1）：
        ``role == "tool"`` 的消息必须带 ``tool_call_id``，且能在 history 中找到同 id 的
        assistant ``ToolCallRequest``。
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list, description="仅 assistant")
    tool_call_id: str | None = Field(default=None, description="仅 tool")
    name: str | None = None

    # ---- 内部元数据（不发送给模型，仅供上下文管理使用）----
    id: str = Field(default_factory=new_message_id, description="本地唯一 id（uuid4）")
    created_at: float = Field(default_factory=time.time, description="epoch 秒")
    tokens: int = Field(default=0, description="估算 token 数（§7.2）")
    kind: Literal["normal", "tool_result", "summary", "cleared"] = "normal"
    is_large_result: bool = Field(default=False, description="是否为大工具结果（§7.4）")
    archive_ref: str | None = Field(default=None, description="落盘/归档引用（§7.4、§7.5）")
    cleared_marker: str | None = Field(default=None, description="被裁剪时的替换文本")

    # ---- 【补齐】并发/记账辅助字段（不影响 §4.1 语义）----
    turn_index: int = Field(default=0, description="所属轮次（§7.7 摘要需带轮次号）")
    subagent_id: str | None = Field(default=None, description="来源子 Agent（§12）")

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def _check_tool_call_id(self) -> Message:
        """校验 §4.1 不变量：tool 消息必须带 tool_call_id。

        Returns:
            自身。

        Raises:
            ValueError: ``role == "tool"`` 但缺少 ``tool_call_id``。
        """
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("role='tool' 的消息必须带 tool_call_id（§4.1 不变量）")
        return self

    def to_provider_dict(self) -> dict[str, Any]:
        """转换为 OpenAI 兼容的消息结构（剥离内部元数据）。

        Returns:
            仅含 ``role`` / ``content`` / ``tool_calls`` / ``tool_call_id`` / ``name`` 的字典。
        """
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments or _dumps(call.arguments),
                    },
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.name:
            out["name"] = self.name
        return out


def _dumps(value: dict[str, Any]) -> str:
    """安全地把参数序列化为 JSON 字符串。

    Args:
        value: 待序列化字典。

    Returns:
        JSON 字符串；不可序列化时退化为 ``str(value)``。
    """
    import json

    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
