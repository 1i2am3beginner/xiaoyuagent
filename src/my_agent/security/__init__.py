"""安全防护模块（§6）：四阶段权限流水线。

对外统一出口：

* :class:`~my_agent.security.pipeline.PermissionPipeline`：四阶段编排器。
* :class:`~my_agent.security.cache.ApprovalCache`：会话级精确键审批缓存（§6.7）。
* :class:`~my_agent.security.audit.AuditWriter` 与
  :func:`~my_agent.security.audit.redact`：审计写入与打码（§6.8，UI 复用）。
* :data:`~my_agent.security.modes.MODE_SPECS`：五模式行为规格（§6.10）。
* 四个阶段各自的纯函数，便于单测与复用。
"""

from __future__ import annotations

from my_agent.security.audit import AuditWriter, redact
from my_agent.security.cache import ApprovalCache, ApprovalEntry
from my_agent.security.modes import MODE_SPECS, ModeSpec
from my_agent.security.patterns import GLOBAL_ASK, GLOBAL_DENY, IRREVERSIBLE_PATTERNS
from my_agent.security.phase0_hard_block import (
    Phase0Outcome,
    hard_block,
    hits_global_list,
    hits_irreversible,
    hits_sensitive_path,
)
from my_agent.security.phase1_precheck import (
    EDIT_TOOLS,
    Phase1Outcome,
    accept_edits_fast_path,
    extract_target_path,
    precheck,
)
from my_agent.security.phase2_llm_judge import JudgeCaller, parse_judge_output
from my_agent.security.phase3_postfix import (
    apply_mode_fix,
    normalize_confidence,
    unavailable_fallback,
)
from my_agent.security.pipeline import MODE_SWITCH_TOOL, PermissionPipeline

__all__ = [
    "EDIT_TOOLS",
    "GLOBAL_ASK",
    "GLOBAL_DENY",
    "IRREVERSIBLE_PATTERNS",
    "MODE_SPECS",
    "MODE_SWITCH_TOOL",
    "ApprovalCache",
    "ApprovalEntry",
    "AuditWriter",
    "JudgeCaller",
    "ModeSpec",
    "PermissionPipeline",
    "Phase0Outcome",
    "Phase1Outcome",
    "accept_edits_fast_path",
    "apply_mode_fix",
    "extract_target_path",
    "hard_block",
    "hits_global_list",
    "hits_irreversible",
    "hits_sensitive_path",
    "normalize_confidence",
    "parse_judge_output",
    "precheck",
    "redact",
    "unavailable_fallback",
]
