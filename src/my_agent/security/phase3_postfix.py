"""阶段 3：后置修正（§6.6）。

规格：``my_agent-开发文档.md`` §6.6。两小步，顺序不可颠倒：

1. :func:`normalize_confidence`：置信度归一（§6.6.1 五条规则，逐字实现）。
2. :func:`apply_mode_fix`：模式修正（§6.6.2），在归一结果之上叠加。

另含 :func:`unavailable_fallback`：判定模型不可用时的降级裁决（§6.5.4 表 +
§6.11 Q1/Q2 的**本文档选定实现**）：

* ``dontAsk`` → ``deny``（Q1：模式契约优先，保守拒绝，不问用户）；
* ``bypassPermissions`` → ``ask``（Q2：无自动放行依据时**不**自动放行）。
"""

from __future__ import annotations

from my_agent.models.permission import Decision, JudgeOutput, SafetyMode

__all__ = [
    "BYPASS_ALLOW_REASON",
    "DONT_ASK_DENY_REASON",
    "DONT_ASK_UNAVAILABLE_REASON",
    "apply_mode_fix",
    "normalize_confidence",
    "unavailable_fallback",
]

DONT_ASK_DENY_REASON = "dontAsk 模式不接受询问，已按最保守策略拒绝"
"""§6.6.2：``dontAsk`` 下 ``ask`` → ``deny`` 时追加的理由。"""

BYPASS_ALLOW_REASON = "bypassPermissions：ask 已自动放行"
"""§6.6.2：``bypassPermissions`` 下 ``ask`` → ``allow`` 时追加的理由。"""

DONT_ASK_UNAVAILABLE_REASON = "dontAsk 模式不接受询问，已拒绝"
"""§6.11 Q1：``dontAsk`` 遇判定模型不可用时的追加理由。"""


def normalize_confidence(judge: JudgeOutput) -> Decision:
    """置信度归一（§6.6.1 五条规则）。

    ====================  ==========
    输入                 归一结果
    ====================  ==========
    ``allow`` + ``high``  ``allow``
    ``allow`` + 非 high   ``ask``
    ``deny`` + ``high``   ``deny``
    ``deny`` + 非 high    ``ask``
    ``ask``               ``ask``
    ====================  ==========

    Args:
        judge: 判定模型的输出。

    Returns:
        归一后的裁决。
    """
    if judge.decision == "ask":
        return "ask"
    if judge.confidence == "high":
        return judge.decision
    return "ask"


def apply_mode_fix(normalized: Decision, mode: SafetyMode) -> tuple[Decision, str]:
    """模式修正（§6.6.2）。

    Args:
        normalized: :func:`normalize_confidence` 的输出。
        mode: 当前安全模式。

    Returns:
        ``(修正后的裁决, 追加理由)``；无需修正时追加理由为空串。
        ``default`` / ``plan`` 不进入阶段 3，原样返回。
    """
    if mode == "dontAsk":
        if normalized == "ask":
            return "deny", DONT_ASK_DENY_REASON
        return normalized, ""
    if mode == "bypassPermissions":
        if normalized == "ask":
            return "allow", BYPASS_ALLOW_REASON
        return normalized, ""
    return normalized, ""


def unavailable_fallback(mode: SafetyMode) -> Decision:
    """判定模型不可用时的降级裁决（§6.5.4 表 + §6.11 Q1/Q2）。

    本函数**不**经过 :func:`apply_mode_fix`：``bypassPermissions`` 的
    「ask 自动放行」在无判定依据时不得生效（Q2）。

    Args:
        mode: 当前安全模式。

    Returns:
        ``dontAsk`` → ``"deny"``（Q1）；其余模式 → ``"ask"``（最保守）。
    """
    if mode == "dontAsk":
        return "deny"
    return "ask"
