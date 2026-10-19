"""阶段 1：前置修正（§6.4 / §6.4.1）。

规格：``my_agent-开发文档.md`` §6.4。阶段 1 是**确定性**分支，只有
``acceptEdits`` 的快速通道会在「目标位于工作目录内」时直接放行；其余情况要么
直接终结（``default`` 直接 ask、``plan`` 只读闸门），要么交给阶段 2。

`accept_edits_fast_path` 的六条边界（§6.4.1）全部在此实现：
相对路径先按 ``cwd`` 解析、``..`` 逃逸与符号链接逃逸用 ``resolve()`` 排除、
Windows 大小写用 :func:`os.path.normcase` 归一、阶段 0 优先级由流水线顺序保证、
快速通道放行不调用模型。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from my_agent.errors import ConfigError
from my_agent.models.permission import SafetyMode
from my_agent.models.tools import ToolInvocation
from my_agent.security.modes import MODE_SPECS

__all__ = [
    "DEFAULT_MODE_REASON",
    "EDIT_TOOLS",
    "PLAN_DENY_REASON",
    "TARGET_PATH_KEYS",
    "Phase1Outcome",
    "accept_edits_fast_path",
    "extract_target_path",
    "phase1_reason",
    "precheck",
]

EDIT_TOOLS: frozenset[str] = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
"""可走 acceptEdits 快速通道的编辑类工具（§6.4.1；U27 NotebookEdit 首版不注册但保留判定）。"""

TARGET_PATH_KEYS: tuple[str, ...] = ("file_path", "path", "notebook_path")
"""参数中目标路径的键名优先级（§6.4.1）。"""

DEFAULT_MODE_REASON = "default 模式下所有工具调用需用户确认"
"""§6.4 表中 default 模式的固定理由文案。"""

PLAN_DENY_REASON = "plan 模式禁止写操作/破坏性操作"
"""§6.4 表中 plan 模式拒绝的固定理由文案。"""

_ACCEPT_EDITS_ALLOW_REASON = "acceptEdits 快速通道：目标位于工作目录内，确定性放行（未调用判定模型）"
_ACCEPT_EDITS_LLM_REASON = "acceptEdits：目标不在工作目录内或非编辑类工具，交判定模型"


class Phase1Outcome(StrEnum):
    """阶段 1 的四种走向（§6.4）。"""

    FAST_ALLOW = "fast_allow"
    FAST_DENY = "fast_deny"
    NEEDS_LLM = "needs_llm"
    DIRECT_ASK = "direct_ask"


def extract_target_path(arguments: Mapping[str, Any]) -> str | None:
    """从工具参数中取出目标路径（§6.4.1）。

    按 ``file_path`` → ``path`` → ``notebook_path`` 的优先级取第一个非空字符串值。

    Args:
        arguments: 工具调用参数。

    Returns:
        目标路径字符串；没有任何可用路径时返回 ``None``。
    """
    for key in TARGET_PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def accept_edits_fast_path(inv: ToolInvocation, cwd: str) -> bool:
    """判断本次调用是否可走 acceptEdits 快速通道（§6.4.1，确定性、不调模型）。

    判定步骤：工具必须属于 :data:`EDIT_TOOLS` → 能取出目标路径 → 目标按 ``cwd``
    解析并 ``resolve()`` 后必须位于工作目录内（含子目录）。

    Args:
        inv: 待判定调用。
        cwd: 工作目录（可为相对路径，内部会 ``resolve()``）。

    Returns:
        ``True`` 表示可确定性放行；``False`` 表示需要进入阶段 2（含路径无法
        ``resolve()`` 的异常路径，一律按「不放行」处理）。
    """
    if inv.tool not in EDIT_TOOLS:
        return False
    target = extract_target_path(inv.arguments)
    if target is None:
        return False
    base = Path(cwd) if cwd else Path.cwd()
    try:
        candidate = Path(target)
        abs_target = (candidate if candidate.is_absolute() else base / candidate).resolve()
        abs_cwd = base.resolve()
    except (OSError, RuntimeError):
        return False
    try:
        Path(os.path.normcase(str(abs_target))).relative_to(Path(os.path.normcase(str(abs_cwd))))
    except ValueError:
        return False
    else:
        return True


def precheck(inv: ToolInvocation, mode: SafetyMode, cwd: str) -> Phase1Outcome:
    """按模式给出阶段 1 的走向（§6.4）。

    Args:
        inv: 待判定调用。
        mode: 当前安全模式。
        cwd: 工作目录。

    Returns:
        :class:`Phase1Outcome` 之一。

    Raises:
        ConfigError: ``mode`` 不在 :data:`~my_agent.security.modes.MODE_SPECS` 中。
    """
    mode_spec = MODE_SPECS.get(mode)
    if mode_spec is None:  # pragma: no cover - SafetyMode 为 Literal，正常不可达
        raise ConfigError(detail=f"未知安全模式：{mode!r}")

    if mode_spec.phase1 == "direct_ask":
        return Phase1Outcome.DIRECT_ASK
    if mode_spec.phase1 == "readonly_gate":
        if inv.spec.is_read_only and not inv.spec.is_destructive:
            return Phase1Outcome.FAST_ALLOW
        return Phase1Outcome.FAST_DENY
    if mode_spec.phase1 == "fast_path":
        if accept_edits_fast_path(inv, cwd):
            return Phase1Outcome.FAST_ALLOW
        return Phase1Outcome.NEEDS_LLM
    return Phase1Outcome.NEEDS_LLM


def phase1_reason(outcome: Phase1Outcome, mode: SafetyMode) -> str:
    """给出阶段 1 走向的可读理由（进 ``phase1.reason`` 与 UI）。

    Args:
        outcome: 阶段 1 走向。
        mode: 当前安全模式。

    Returns:
        中文理由文案；``default`` 与 ``plan`` 拒绝的文案逐字取自 §6.4 表。
    """
    if outcome is Phase1Outcome.DIRECT_ASK:
        return DEFAULT_MODE_REASON
    if outcome is Phase1Outcome.FAST_DENY:
        return PLAN_DENY_REASON
    if outcome is Phase1Outcome.FAST_ALLOW:
        if mode == "plan":
            return "plan 模式：只读工具直接允许"
        return _ACCEPT_EDITS_ALLOW_REASON
    return _ACCEPT_EDITS_LLM_REASON
