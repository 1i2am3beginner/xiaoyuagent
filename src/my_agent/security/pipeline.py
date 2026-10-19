"""四阶段权限流水线编排器（§6.2）。

规格：``my_agent-开发文档.md`` §6.2。顺序与短路规则：

1. **阶段 0** 硬拦截（:func:`~my_agent.security.phase0_hard_block.hard_block`）——
   命中即终态（``deny``/``ask``，一律 ``permanent_eligible=False``）。
2. **会话缓存**（§6.7）——精确键命中即 ``allow``，审计记 ``phase_reached=3``、
   ``judge=None``、``user_choice="always(cached)"``。
3. **阶段 1** 前置修正（:func:`~my_agent.security.phase1_precheck.precheck`）——
   ``FAST_ALLOW``/``FAST_DENY``/``DIRECT_ASK`` 直接终结，``NEEDS_LLM`` 进入阶段 2。
4. **阶段 2** 大模型判定（:class:`~my_agent.security.phase2_llm_judge.JudgeCaller`）——
   受总耗时预算 :data:`~my_agent.config.defaults.PIPELINE_TOTAL_BUDGET_S` 约束；
   超时按「模型不可用」处理（§6.5.3、U14）。
5. **阶段 3** 后置修正（:mod:`my_agent.security.phase3_postfix`）。
6. 终态为 ``ask`` 时调用注入的 :class:`~my_agent.models.protocols.Confirmer`，
   按 ``once``/``always``/``reject``/``reject_with_reason``/``abort_turn`` 映射结果
   （§6.9；非交互实现由 UI 层注入 ``AutoRejectConfirmer``，绝不默认 allow）。
7. 每次判定写一条 :class:`~my_agent.models.permission.AuditRecord`（§6.8），
   四个阶段的痕迹全部保留在 :class:`~my_agent.models.permission.PermissionDecision`
   与审计 ``note`` 中。

hook 接入（§11.4，可选）：``before_permission_check`` 可拒绝/强制询问/改写入参；
``after_permission_check`` **只能收紧** allow → ask/deny（安全不变量 G-5）。
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

from my_agent.config.defaults import PIPELINE_TOTAL_BUDGET_S
from my_agent.errors import ModelUnavailableError, UserAbortError
from my_agent.models.permission import (
    AuditRecord,
    Decision,
    JudgeOutput,
    PermissionDecision,
    Risk,
    SafetyMode,
)
from my_agent.models.protocols import Confirmer, PermissionRequest, UserAnswer
from my_agent.models.tools import ToolInvocation
from my_agent.security.audit import AuditWriter, redact
from my_agent.security.cache import ApprovalCache
from my_agent.security.phase0_hard_block import Phase0Outcome, hard_block
from my_agent.security.phase1_precheck import (
    Phase1Outcome,
    phase1_reason,
    precheck,
)
from my_agent.security.phase2_llm_judge import JudgeCaller
from my_agent.security.phase3_postfix import (
    DONT_ASK_UNAVAILABLE_REASON,
    apply_mode_fix,
    normalize_confidence,
    unavailable_fallback,
)

if TYPE_CHECKING:
    from my_agent.config.schema import Config
    from my_agent.hooks.events import HookOutcome
    from my_agent.hooks.runner import HookRunner
    from my_agent.llm.client import LLMClient

__all__ = ["MODE_SWITCH_TOOL", "PermissionPipeline"]

MODE_SWITCH_TOOL = "ModeSwitch"
"""模式切换审计记录使用的伪工具名（§6.1：模式切换本身要写审计记录）。"""

CACHE_HIT_REASON = "命中会话审批缓存（永久放行）"
"""缓存命中的理由文案（§6.7）。"""


class PermissionPipeline:
    """四阶段权限流水线（§6.2）。

    Args:
        config: 全局配置（安全子配置决定阈值与规则表）。
        llm: 角色化 LLM 客户端（构造默认判定器用）。
        cache: 审批缓存（§6.7）。
        audit: 审计写入器（§6.8）。
        confirmer: 用户询问回调（§6.9）。
        judge: 判定器；``None`` 时按 ``config.safety`` 自动构造。
        hooks: hook 运行器（§11，可选）。
        cwd: 默认工作目录；``decide(cwd=...)`` 可逐次覆盖。
        session_id: 默认会话 ID（写审计用）。
        turn_index: 默认轮次号（写审计用）。
    """

    def __init__(
        self,
        *,
        config: Config,
        llm: LLMClient,
        cache: ApprovalCache,
        audit: AuditWriter,
        confirmer: Confirmer,
        judge: JudgeCaller | None = None,
        hooks: HookRunner | None = None,
        cwd: str | None = None,
        session_id: str = "",
        turn_index: int = 0,
    ) -> None:
        self._config = config
        self._llm = llm
        self._cache = cache
        self._audit = audit
        self._confirmer = confirmer
        self._judge = judge or JudgeCaller(
            llm,
            timeout_s=config.safety.judge_timeout_s,
            max_retries=config.safety.judge_max_retries,
            role=config.safety.judge_model_role,
        )
        self._hooks = hooks
        self._cwd = cwd
        self._session_id = session_id
        self._turn_index = turn_index
        self._last_mode: SafetyMode | None = None

    # ------------------------------------------------------------------ 公共属性
    @property
    def cache(self) -> ApprovalCache:
        """审批缓存（测试与 ``/clear`` 使用）。"""
        return self._cache

    @property
    def judge(self) -> JudgeCaller:
        """判定器。"""
        return self._judge

    # ------------------------------------------------------------------ 模式切换
    def on_mode_switch(self, old: SafetyMode, new: SafetyMode) -> None:
        """处理 ``/mode`` 切换（§6.1 + §6.7）。

        清空会话审批缓存并写一条审计记录。

        Args:
            old: 切换前的模式。
            new: 切换后的模式。
        """
        self._cache.clear()
        self._last_mode = new
        self._audit.submit(
            AuditRecord(
                session_id=self._session_id,
                turn_index=self._turn_index,
                call_id=f"mode-switch-{int(time.time() * 1000)}",
                tool=MODE_SWITCH_TOOL,
                arguments_redacted={},
                reason=f"模式切换：{old} → {new}",
                mode=new,
                phase_reached=0,
                final="allow",
                user_choice=None,
                note=json.dumps(
                    {"event": "mode_switch", "from": old, "to": new},
                    ensure_ascii=False,
                ),
            )
        )

    # ------------------------------------------------------------------ 主流程
    async def decide(
        self,
        inv: ToolInvocation,
        mode: SafetyMode,
        *,
        cwd: str | None = None,
        session_id: str | None = None,
        turn_index: int | None = None,
        subagent_id: str | None = None,
        parent_call_id: str | None = None,
        mcp_server: str | None = None,
        assistant_text: str | None = None,
    ) -> PermissionDecision:
        """执行四阶段流水线并返回终态裁决（§6.2）。

        Args:
            inv: 待判定调用。
            mode: 当前安全模式。
            cwd: 工作目录（缺省用构造时的 ``cwd`` 或 ``os.getcwd()``）。
            session_id: 会话 ID（审计用）。
            turn_index: 轮次号（审计用）。
            subagent_id: 子 Agent 归属（审计用，§12）。
            parent_call_id: 父调用 ID（审计用）。
            mcp_server: MCP 服务名（审计用，§10.4）。
            assistant_text: 同轮 assistant 文本（判定 reason 兜底，§6.5.1）。

        Returns:
            填充了四个阶段痕迹的裁决。

        Raises:
            UserAbortError: 用户在询问卡片中选择「中止本轮」，或 hook 要求中止回合。
            ConfigError: 安全规则正则非法，或安全模式未知。
        """
        start = time.perf_counter()
        work_cwd = cwd or self._cwd or os.getcwd()
        session = self._session_id if session_id is None else session_id
        turn = self._turn_index if turn_index is None else turn_index
        if self._last_mode is not None and self._last_mode != mode:
            self.on_mode_switch(self._last_mode, mode)
        self._last_mode = mode

        hook_messages: list[str] = []
        invocation = inv
        forced: Decision | None = None
        before = await self._run_before_hook(invocation, session, turn, hook_messages)
        if before is not None:
            invocation, forced = before

        phase0: Phase0Outcome | None = None
        phase1_trace: dict[str, Any] | None = None
        judge_output: JudgeOutput | None = None
        phase3_trace: dict[str, Any] | None = None
        model_unavailable = False
        permanent_eligible = True
        risk: Risk = "medium"
        user_choice: str | None = None
        user_note: str | None = None

        if forced is not None:
            final = forced
            phase_reached = 1
            reason = hook_messages[-1] if hook_messages else "hook 干预（before_permission_check）"
        else:
            phase0 = hard_block(invocation, config=self._config, cwd=work_cwd)
            if phase0 is not None:
                final = phase0.decision
                phase_reached = 0
                reason = phase0.detail
                permanent_eligible = phase0.permanent_eligible
                risk = phase0.risk
            else:
                cached = self._cache.get(
                    invocation.tool,
                    invocation.arguments,
                    category=invocation.spec.category,
                )
                if cached is not None:
                    final, phase_reached, reason = "allow", 3, CACHE_HIT_REASON
                    user_choice = "always(cached)"
                    phase3_trace = {
                        "normalized": "allow",
                        "mode_adjusted": None,
                        "reason": CACHE_HIT_REASON,
                        "cached": True,
                    }
                else:
                    outcome = precheck(invocation, mode, work_cwd)
                    phase1_trace = {
                        "result": outcome.value,
                        "reason": phase1_reason(outcome, mode),
                    }
                    if outcome is Phase1Outcome.NEEDS_LLM:
                        (
                            final,
                            reason,
                            judge_output,
                            phase3_trace,
                            risk,
                            model_unavailable,
                            phase_reached,
                        ) = await self._run_judge_phase(
                            invocation, mode, work_cwd, start, assistant_text
                        )
                    elif outcome is Phase1Outcome.DIRECT_ASK:
                        final, phase_reached, reason = "ask", 1, phase1_trace["reason"]
                    elif outcome is Phase1Outcome.FAST_DENY:
                        final, phase_reached, reason = "deny", 1, phase1_trace["reason"]
                    else:
                        final, phase_reached = "allow", 1
                        reason = phase1_trace["reason"]

        # ---- 询问用户（§6.9）----
        if final == "ask":
            answer = await self._confirmer.ask(
                PermissionRequest(
                    tool=invocation.tool,
                    arguments=dict(invocation.arguments),
                    risk=risk,
                    reason=reason,
                    phase_reached=phase_reached,
                    permanent_eligible=permanent_eligible,
                    subagent_id=subagent_id,
                )
            )
            (
                final,
                user_choice,
                user_note,
                abort,
            ) = self._map_user_answer(invocation, answer, permanent_eligible)
            if phase3_trace is None:
                phase3_trace = {"normalized": None, "mode_adjusted": None, "reason": reason}
            phase3_trace["user_choice"] = user_choice
            if abort:
                self._submit_audit(
                    invocation,
                    mode=mode,
                    final="deny",
                    phase_reached=phase_reached,
                    reason=reason,
                    judge=judge_output,
                    user_choice="abort_turn",
                    user_note=user_note,
                    session_id=session,
                    turn_index=turn,
                    subagent_id=subagent_id,
                    parent_call_id=parent_call_id,
                    mcp_server=mcp_server,
                    hook_messages=hook_messages,
                    phase0=phase0,
                    phase1=phase1_trace,
                    phase3=phase3_trace,
                    model_unavailable=model_unavailable,
                    latency_ms=_elapsed_ms(start),
                )
                raise UserAbortError(detail="用户选择中止本轮")

        # ---- hook：只能收紧（G-5）----
        tightened = await self._run_after_hook(
            invocation, session, turn, final, reason, hook_messages
        )
        if tightened is not None:
            final, reason = tightened

        duration_ms = _elapsed_ms(start)
        public_choice = (
            user_choice
            if user_choice in {"once", "always", "reject", "reject_with_reason"}
            else None
        )
        decision = PermissionDecision(
            call_id=invocation.call_id,
            tool=invocation.tool,
            arguments=dict(invocation.arguments),
            final=final,
            phase_reached=phase_reached,
            mode=mode,
            phase0=phase0.as_trace() if phase0 is not None else None,
            phase1=phase1_trace,
            judge=judge_output,
            phase3=phase3_trace,
            permanent_eligible=permanent_eligible,
            user_choice=public_choice,
            user_note=user_note,
            model_unavailable=model_unavailable,
            duration_ms=duration_ms,
        )
        self._submit_audit(
            invocation,
            mode=mode,
            final=final,
            phase_reached=phase_reached,
            reason=reason,
            judge=judge_output,
            user_choice=user_choice,
            user_note=user_note,
            session_id=session,
            turn_index=turn,
            subagent_id=subagent_id,
            parent_call_id=parent_call_id,
            mcp_server=mcp_server,
            hook_messages=hook_messages,
            phase0=phase0,
            phase1=phase1_trace,
            phase3=phase3_trace,
            model_unavailable=model_unavailable,
            latency_ms=duration_ms,
        )
        return decision

    # ------------------------------------------------------------------ 阶段 2+3
    def _budget_s(self) -> float:
        """总耗时预算（§6.2：默认 :data:`PIPELINE_TOTAL_BUDGET_S`）。

        Returns:
            配置覆盖值（``safety.pipeline_budget_s``）或内置常量。
        """
        return float(getattr(self._config.safety, "pipeline_budget_s", PIPELINE_TOTAL_BUDGET_S))

    async def _run_judge_phase(
        self,
        inv: ToolInvocation,
        mode: SafetyMode,
        cwd: str,
        start: float,
        assistant_text: str | None,
    ) -> tuple[Decision, str, JudgeOutput | None, dict[str, Any], Risk, bool, int]:
        """执行阶段 2（判定）与阶段 3（后置修正）。

        Args:
            inv: 待判定调用。
            mode: 安全模式。
            cwd: 工作目录。
            start: 本次判定的起始时刻（``time.perf_counter()``，用于总预算）。
            assistant_text: 同轮 assistant 文本。

        Returns:
            ``(终态裁决, 理由, 判定输出, phase3 痕迹, risk, 是否模型不可用, phase_reached)``。
        """
        remaining = self._budget_s() - (time.perf_counter() - start)
        timeout_s = min(self._config.safety.judge_timeout_s, remaining)
        try:
            judge_output = await self._judge.judge(
                inv, mode, cwd, assistant_text=assistant_text, timeout_s=timeout_s
            )
        except ModelUnavailableError as exc:
            final = unavailable_fallback(mode)
            if mode == "dontAsk":
                reason = f"判定模型不可用（{exc.detail}）；{DONT_ASK_UNAVAILABLE_REASON}"
            else:
                reason = exc.user_message()
            trace: dict[str, Any] = {
                "normalized": None,
                "mode_adjusted": final,
                "reason": reason,
                "model_unavailable": True,
            }
            return final, reason, None, trace, "medium", True, 2

        normalized = normalize_confidence(judge_output)
        adjusted, appended = apply_mode_fix(normalized, mode)
        reason = judge_output.reason
        if appended:
            reason = f"{reason}；{appended}" if reason else appended
        trace = {
            "normalized": normalized,
            "mode_adjusted": adjusted if adjusted != normalized else None,
            "reason": appended,
            "judge_reason": judge_output.reason,
        }
        return adjusted, reason, judge_output, trace, judge_output.risk, False, 3

    # ------------------------------------------------------------------ 用户回答
    def _map_user_answer(
        self,
        inv: ToolInvocation,
        answer: UserAnswer,
        permanent_eligible: bool,
    ) -> tuple[Decision, str | None, str | None, bool]:
        """把用户回答映射为裁决（§6.9）。

        Args:
            inv: 待判定调用。
            answer: 用户回答。
            permanent_eligible: 是否允许永久放行。

        Returns:
            ``(裁决, user_choice, user_note, 是否中止回合)``。``always`` 且允许永久放行时
            写入会话缓存（通配参数与阶段 0 命中项不写，§6.7）。
        """
        choice = answer.choice
        note = answer.note
        if choice == "once":
            return "allow", "once", note, False
        if choice == "always":
            if permanent_eligible and self._cache.is_cacheable(
                inv.tool, inv.arguments, permanent_eligible=permanent_eligible
            ):
                self._cache.put(
                    inv.tool,
                    inv.arguments,
                    permanent_eligible=permanent_eligible,
                    category=inv.spec.category,
                )
            return "allow", "always", note, False
        if choice == "abort_turn":
            return "deny", "abort_turn", note, True
        if choice == "reject_with_reason":
            return "deny", "reject_with_reason", note, False
        return "deny", "reject", note, False

    # ------------------------------------------------------------------ hook
    async def _run_before_hook(
        self,
        inv: ToolInvocation,
        session_id: str,
        turn_index: int,
        messages: list[str],
    ) -> tuple[ToolInvocation, Decision | None] | None:
        """运行 ``before_permission_check``（§11.4）。

        Args:
            inv: 待判定调用。
            session_id: 会话 ID。
            turn_index: 轮次号。
            messages: 收集 hook 消息（就地追加）。

        Returns:
            ``(可能被改写的调用, 被强制指定的裁决或 None)``；无 hook 时返回 ``None``。

        Raises:
            UserAbortError: hook 返回 ``continue=false``。
        """
        if self._hooks is None:
            return None
        from my_agent.hooks.events import HookEvent, HookPayload  # noqa: PLC0415 - 可选依赖

        outcome: HookOutcome = await self._hooks.run(
            HookEvent.BEFORE_PERMISSION_CHECK,
            session_id=session_id,
            turn_index=turn_index,
            payload=HookPayload(
                tool=inv.tool, arguments=dict(inv.arguments), reason=inv.reason
            ),
        )
        if outcome.message:
            messages.append(outcome.message)
        if not outcome.continue_:
            raise UserAbortError(detail=outcome.message or "hook 要求中止回合")
        updated = inv
        if outcome.modified_payload is not None and outcome.modified_payload.arguments is not None:
            updated = inv.model_copy(
                update={"arguments": dict(outcome.modified_payload.arguments)}
            )
        forced: Decision | None = None
        if outcome.decision == "deny":
            forced = "deny"
        elif outcome.decision == "ask":
            forced = "ask"
        return updated, forced

    async def _run_after_hook(
        self,
        inv: ToolInvocation,
        session_id: str,
        turn_index: int,
        final: Decision,
        reason: str,
        messages: list[str],
    ) -> tuple[Decision, str] | None:
        """运行 ``after_permission_check``（§11.4，安全不变量 G-5：只能收紧）。

        Args:
            inv: 待判定调用。
            session_id: 会话 ID。
            turn_index: 轮次号。
            final: 当前裁决。
            reason: 当前理由。
            messages: 收集 hook 消息（就地追加）。

        Returns:
            ``(收紧后的裁决, 理由)``；未收紧时返回 ``None``。
        """
        if self._hooks is None:
            return None
        from my_agent.hooks.events import HookEvent, HookPayload  # noqa: PLC0415 - 可选依赖

        outcome: HookOutcome = await self._hooks.run(
            HookEvent.AFTER_PERMISSION_CHECK,
            session_id=session_id,
            turn_index=turn_index,
            payload=HookPayload(
                tool=inv.tool, arguments=dict(inv.arguments), decision=final, reason=reason
            ),
        )
        if outcome.message:
            messages.append(outcome.message)
        if final != "allow" or outcome.decision not in {"ask", "deny"}:
            return None
        tightened: Decision = "deny" if outcome.decision == "deny" else "ask"
        extra = outcome.message or "hook 收紧了该裁决"
        return tightened, f"{reason}；{extra}" if reason else extra

    # ------------------------------------------------------------------ 审计
    def _submit_audit(
        self,
        inv: ToolInvocation,
        *,
        mode: SafetyMode,
        final: Decision,
        phase_reached: int,
        reason: str,
        judge: JudgeOutput | None,
        user_choice: str | None,
        user_note: str | None,
        session_id: str,
        turn_index: int,
        subagent_id: str | None,
        parent_call_id: str | None,
        mcp_server: str | None,
        hook_messages: list[str],
        phase0: Phase0Outcome | None,
        phase1: dict[str, Any] | None,
        phase3: dict[str, Any] | None,
        model_unavailable: bool,
        latency_ms: int,
    ) -> None:
        """组装并提交一条审计记录（§6.8 + B-17：四个阶段字段齐备）。

        Args:
            inv: 待判定调用。
            mode: 安全模式。
            final: 终态裁决。
            phase_reached: 停在第几阶段。
            reason: 终态理由。
            judge: 阶段 2 判定输出。
            user_choice: 用户选择。
            user_note: 用户备注。
            session_id: 会话 ID。
            turn_index: 轮次号。
            subagent_id: 子 Agent 归属。
            parent_call_id: 父调用 ID。
            mcp_server: MCP 服务名。
            hook_messages: hook 消息。
            phase0: 阶段 0 命中。
            phase1: 阶段 1 痕迹。
            phase3: 阶段 3 痕迹。
            model_unavailable: 是否模型不可用。
            latency_ms: 耗时（毫秒）。
        """
        note = json.dumps(
            {
                "phase0": phase0.as_trace() if phase0 is not None else None,
                "phase1": phase1,
                "phase2": judge.model_dump() if judge is not None else None,
                "phase3": phase3,
                "model_unavailable": model_unavailable,
                "user_note": user_note,
            },
            ensure_ascii=False,
            default=str,
        )
        self._audit.submit(
            AuditRecord(
                session_id=session_id,
                turn_index=turn_index,
                call_id=inv.call_id,
                tool=inv.tool,
                arguments_redacted=redact(inv.arguments),
                reason=reason,
                mode=mode,
                phase_reached=phase_reached,
                judge=judge,
                final=final,
                user_choice=user_choice,
                latency_ms=latency_ms,
                mcp_server=mcp_server,
                subagent_id=subagent_id,
                parent_call_id=parent_call_id,
                hook_messages=list(hook_messages),
                note=note,
            )
        )


def _elapsed_ms(start: float) -> int:
    """计算从 ``start`` 到现在的毫秒数。

    Args:
        start: ``time.perf_counter()`` 的取值。

    Returns:
        毫秒整数（非负）。
    """
    return max(0, int((time.perf_counter() - start) * 1000))
