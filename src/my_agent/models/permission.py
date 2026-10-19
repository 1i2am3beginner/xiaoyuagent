"""权限模型（§4.3，最核心）。

四阶段流水线的输入/中间/最终产物，以及审计记录。所有阶段痕迹**全部保留**，
用于审计与 UI 解释。
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "AuditRecord",
    "Confidence",
    "Decision",
    "JudgeOutput",
    "PermissionDecision",
    "Risk",
    "SafetyMode",
]

Decision = Literal["allow", "deny", "ask"]
Confidence = Literal["high", "medium", "low"]
Risk = Literal["low", "medium", "high", "critical"]
SafetyMode = Literal["default", "acceptEdits", "plan", "dontAsk", "bypassPermissions"]

ALL_MODES: tuple[SafetyMode, ...] = (
    "default",
    "acceptEdits",
    "plan",
    "dontAsk",
    "bypassPermissions",
)
"""五种安全模式（§6.1）。"""


class JudgeOutput(BaseModel):
    """大模型判定（§6.6）的严格输出契约。"""

    decision: Decision
    confidence: Confidence
    reason: str = Field(default="", max_length=500)
    risk: Risk
    matched_rule: str | None = Field(
        default=None, description="可选的规则/原因标签，用于审计"
    )

    model_config = {"extra": "ignore"}


class PermissionDecision(BaseModel):
    """四阶段流水线最终产出。"""

    call_id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    final: Decision
    phase_reached: Literal[0, 1, 2, 3] = Field(description="停在哪个阶段（0=硬拦截命中）")
    mode: SafetyMode

    # ---- 各阶段痕迹（全部保留，用于审计与 UI 解释）----
    phase0: dict[str, Any] | None = None
    phase1: dict[str, Any] | None = None
    judge: JudgeOutput | None = None
    phase3: dict[str, Any] | None = None

    # ---- 元数据 ----
    permanent_eligible: bool = True
    user_choice: Literal["once", "always", "reject", "reject_with_reason"] | None = None
    user_note: str | None = None
    model_unavailable: bool = False
    decided_at: float = Field(default_factory=time.time)
    duration_ms: int = 0

    model_config = {"extra": "ignore"}

    @property
    def denied(self) -> bool:
        """是否为拒绝裁决。"""
        return self.final == "deny"

    def explain(self) -> str:
        """生成一行可读的裁决解释（UI 用）。

        Returns:
            形如 ``phase=0 deny · 命中敏感路径 .git/`` 的短句。
        """
        parts = [f"phase={self.phase_reached}", self.final]
        reason = ""
        if self.judge is not None and self.judge.reason:
            reason = self.judge.reason
        elif self.phase0 is not None:
            reason = str(self.phase0.get("detail") or self.phase0.get("hit") or "")
        elif self.phase1 is not None:
            reason = str(self.phase1.get("reason") or self.phase1.get("result") or "")
        if reason:
            parts.append(f"· {reason}")
        return " ".join(parts)


class AuditRecord(BaseModel):
    """审计日志一条（§6.10 ⇒ §6.8 写入 JSONL）。"""

    ts: float = Field(default_factory=time.time)
    session_id: str
    turn_index: int
    call_id: str
    tool: str
    arguments_redacted: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    mode: SafetyMode = "default"
    phase_reached: int = 0
    judge: JudgeOutput | None = None
    final: Decision = "ask"
    user_choice: str | None = None
    latency_ms: int = 0
    mcp_server: str | None = Field(default=None, description="§10.4：MCP 工具必须带此字段")
    subagent_id: str | None = Field(default=None, description="§12.1：子 Agent 归属")
    parent_call_id: str | None = None
    hook_messages: list[str] = Field(default_factory=list)
    note: str | None = None

    model_config = {"extra": "ignore"}
