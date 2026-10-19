"""模式定义速查（§6.10，逐字实现）。

规格：``my_agent-开发文档.md`` §6.10。本表是**模式行为的唯一权威声明**：
阶段 1 的走向、是否调用判定模型、阶段 3 的修正方式全部从这里读取，
流水线不再内联 ``if mode == ...`` 的分支表（`.git` 之外唯一例外见 §6.11 Q1/Q2
的降级表，位于 :mod:`my_agent.security.phase3_postfix`）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from my_agent.models.permission import SafetyMode

__all__ = ["MODE_SPECS", "ModeSpec", "Phase1Kind", "PostfixKind"]

Phase1Kind = Literal["direct_ask", "fast_path", "readonly_gate", "needs_llm"]
"""阶段 1 的分支种类（§6.4）。"""

PostfixKind = Literal["normalize_only", "allow_only", "ask_to_allow"]
"""阶段 3 的模式修正种类（§6.6.2）。"""


class ModeSpec(BaseModel):
    """单个安全模式的行为声明（§6.10）。

    Attributes:
        phase1: 阶段 1 的走向种类。
        calls_judge: 该模式是否可能调用判定模型（阶段 2）。
        postfix: 阶段 3 的修正种类；``None``（表中写作 ``False``）表示不进入阶段 3。
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    phase1: Phase1Kind
    calls_judge: bool
    postfix: PostfixKind | None = None

    @field_validator("postfix", mode="before")
    @classmethod
    def _false_means_none(cls, value: object) -> object:
        """把 §6.10 表中逐字写下的 ``postfix=False`` 归一为 ``None``。

        Args:
            value: 原始字段值。

        Returns:
            ``False`` → ``None``；其余原样返回供 pydantic 校验。
        """
        return None if value is False else value


MODE_SPECS: dict[SafetyMode, ModeSpec] = {
    "default": ModeSpec(phase1="direct_ask", calls_judge=False, postfix=False),
    "acceptEdits": ModeSpec(phase1="fast_path", calls_judge=True, postfix="normalize_only"),
    "plan": ModeSpec(phase1="readonly_gate", calls_judge=False, postfix=False),
    "dontAsk": ModeSpec(phase1="needs_llm", calls_judge=True, postfix="allow_only"),
    "bypassPermissions": ModeSpec(phase1="needs_llm", calls_judge=True, postfix="ask_to_allow"),
}
"""五种安全模式的行为规格（§6.10 逐字）。"""
