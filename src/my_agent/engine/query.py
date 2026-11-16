"""内层 ReAct 循环（§5.3）。

与「一个用户回合」同寿：每次用户输入创建一个实例，``await query.run(text)`` 跑完即弃。

状态机：``ASSEMBLE → THINK → (ACT → OBSERVE)* → ANSWER | LIMIT``。
`/` 命令**不在此处理**（外层 ``QueryEngine`` 负责）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from my_agent.config.defaults import (
    CONSECUTIVE_FAILURE_LIMIT,
    EMPTY_RESPONSE_RETRIES,
    MAX_DISPATCH_JUDGE_PER_TURN,
    MAX_TOOL_CALLS_PER_TURN,
    MAX_TURNS_PER_REQUEST,
)
from my_agent.engine.loop_guard import LoopCheck, LoopGuard, LoopVerdict
from my_agent.errors import (
    EmptyModelResponseError,
    LLMUnavailableError,
    MyAgentError,
)
from my_agent.models.messages import Message, ToolCallRequest
from my_agent.models.permission import PermissionDecision
from my_agent.models.protocols import NullDisplay
from my_agent.models.session import SessionState
from my_agent.models.tools import ToolContext, ToolInvocation, ToolResult

if TYPE_CHECKING:
    from my_agent.config.schema import Config
    from my_agent.context.assembler import ContextAssembler
    from my_agent.engine.query_engine import EngineResult
    from my_agent.hooks.events import HookPayload
    from my_agent.hooks.runner import HookRunner
    from my_agent.llm.client import AssistantTurn, LLMClient
    from my_agent.memory.writer import MemoryWriter
    from my_agent.models.protocols import Confirmer, DisplayPort
    from my_agent.security.pipeline import PermissionPipeline
    from my_agent.tools.registry import ToolRegistry

__all__ = ["Query"]

_LOG = logging.getLogger(__name__)

TURN_LIMIT_NOTICE = (
    "已达单回合工具调用上限（{n} 轮）。请立即停止调用工具，"
    "用现有信息给出总结性回答，并说明还有哪些未完成。"
)
CONSECUTIVE_FAILURE_NOTICE = (
    "连续 {n} 轮工具调用全部失败。请停止重试，直接向用户解释失败原因与建议的下一步。"
)
INTERRUPTED_NOTICE = "用户中断了本次工具调用（Ctrl-C）。"
TRUNCATED_NOTICE = "上一次回答因输出过长被截断，请分批处理或缩短输出。"
EMPTY_RESPONSE_NOTICE = "上一次模型返回了空响应，请重新给出回答。"
LOOP_ABORT_NOTICE = "检测到工具调用死循环，已终止本回合。"
LOOP_RESULT_ERROR_NOTICE = "请立即停止以相同参数重复调用该工具，改用其它方式或直接向用户说明。"


class Query:
    """内层 ReAct 循环本体。"""

    def __init__(
        self,
        *,
        session: SessionState,
        config: Config,
        llm: LLMClient,
        tools: ToolRegistry,
        context: ContextAssembler,
        pipeline: PermissionPipeline,
        hooks: HookRunner,
        confirmer: Confirmer,
        display: DisplayPort | None = None,
        memory_writer: MemoryWriter | None = None,
        system_prompt: str = "",
        repomap: str = "",
        memory_block: str = "",
        skills_block: str = "",
        subagent_id: str | None = None,
        loop_guard: LoopGuard | None = None,
        subagent_runner: Any = None,
        dispatch_judge: Any = None,
    ) -> None:
        """装配内层循环的全部依赖（依赖注入，便于测试）。

        Args:
            session: 会话状态（history 是唯一真相源）。
            config: 生效配置。
            llm: 模型客户端。
            tools: 工具注册表。
            context: 上下文组装器。
            pipeline: 权限流水线（唯一裁决者）。
            hooks: hook 执行器。
            confirmer: 权限询问回调。
            display: 输出端口（缺省静默）。
            memory_writer: 回合结束后的记忆写入器（子 Agent 传 None）。
            system_prompt: 主对话系统提示词。
            repomap: 工作目录摘要块。
            memory_block: 记忆块。
            skills_block: skills/mcp 索引块。
            subagent_id: 子 Agent 标识（§12，用于审计与提示）。
            loop_guard: 死循环守卫（【待确认假设 U33】）；``None`` = 按配置构造真实守卫，
                注入点供测试替换（测试不得触网）。
            subagent_runner: 子 Agent 运行时（§12.2）。给出时，每轮模型调用前会
                ``drain_background()`` 把已完成的后台子 Agent 结论注入主上下文；
                ``None`` = 不处理后台结论（子 Agent 内部构造的 Query 即为此情形）。
            dispatch_judge: 任务分发审判器（【待确认假设 U35】）。给出时，每轮 ``Task``
                批次执行**之前**先审一遍分发计划：过粗由审判器亲自补发、过细返回补丁
                让模型重新分发；``None`` = 不审判。
        """
        self.session = session
        self.config = config
        self.llm = llm
        self.tools = tools
        self.context = context
        self.pipeline = pipeline
        self.hooks = hooks
        self.confirmer = confirmer
        self.display: DisplayPort = display or NullDisplay()
        self.memory_writer = memory_writer
        self.system_prompt = system_prompt
        self.repomap = repomap
        self.memory_block = memory_block
        self.skills_block = skills_block
        self.subagent_id = subagent_id
        self.subagent_runner = subagent_runner
        self.dispatch_judge = dispatch_judge
        self._dispatch_judge_count = 0
        self.loop_guard = loop_guard if loop_guard is not None else self._build_loop_guard()

        self.abort_event = asyncio.Event()
        self._turn_calls: list[str] = []
        self._turn_results: list[ToolResult] = []
        self._tokens_in = 0
        self._tokens_out = 0
        self._loop_aborted = False
        self._loop_message = ""
        self.compacted = False

    def _build_loop_guard(self) -> LoopGuard:
        """按配置构造死循环守卫（后端故障绝不阻断 Agent 启动）。

        ``get_backend(config)`` 可能是 embedding 后端（配置错误时抛 :class:`MyAgentError`），
        一旦构造失败就回退到离线 trigram 后端——死循环检测是**保护性**能力，
        不允许它反而把整个回合弄挂。

        Returns:
            可用的 :class:`~my_agent.engine.loop_guard.LoopGuard`。
        """
        from my_agent.skills.similarity import TrigramCosineBackend, get_backend

        try:
            backend = get_backend(self.config)
        except MyAgentError as exc:
            _LOG.warning("相似度后端构造失败，死循环检测回退 trigram：%s", exc.user_message())
            backend = TrigramCosineBackend()
        return LoopGuard(backend=backend)

    # ------------------------------------------------------------------ 主循环
    async def run(self, user_text: str) -> EngineResult:
        """执行一次完整的 ReAct 循环。

        Args:
            user_text: 用户输入（已确认不是 ``/`` 命令）。

        Returns:
            :class:`~my_agent.engine.query_engine.EngineResult`。

        Raises:
            EmptyModelResponseError: 模型连续返回空响应且无工具调用（B12）。
            LLMUnavailableError: 主对话模型不可用且重试耗尽（§13.3 降级）。
        """
        from my_agent.engine.query_engine import EngineResult  # 延迟导入避免循环

        self.loop_guard.reset()  # 每个用户回合从零开始（U33）
        self._dispatch_judge_count = 0  # 分发审判的回合预算（U35）
        self._loop_aborted = False
        self._loop_message = ""
        self.session.turn_index += 1
        turn = self.session.turn_index
        user_msg = Message(role="user", content=user_text, turn_index=turn)
        self.session.history.append(user_msg)

        final_text = ""
        try:
            final_text = await self._loop(turn)
        except asyncio.CancelledError:
            raise
        except MyAgentError as exc:
            _LOG.warning("回合异常终止：%s", exc)
            result = EngineResult(
                kind="error", text=exc.user_message(), turn_index=turn, compacted=self.compacted
            )
            self._finish_turn(turn, user_text, result.text)
            return result

        result = EngineResult(
            kind="assistant_text",
            text=final_text,
            usage={
                "prompt_tokens": self._tokens_in,
                "completion_tokens": self._tokens_out,
                "total": self._tokens_in + self._tokens_out,
            },
            compacted=self.compacted,
            turn_index=turn,
        )
        self._finish_turn(turn, user_text, final_text)
        return result

    async def _loop(self, turn: int) -> str:
        failures = 0
        empty_retries = 0
        for attempt in range(MAX_TURNS_PER_REQUEST):
            self._drain_background(turn)
            messages = await self._assemble(turn)
            await self._hook_before_model(messages)

            turn_result: AssistantTurn = await self.llm.assistant_turn(
                "main",
                messages=messages,
                tools=self.tools.schema() if self._tools_enabled() else None,
                system=self._system_text(),
                display=self.display,
                timeout_s=None,
            )
            self._account(turn_result)

            await self._hook_after_model(turn_result)

            if not turn_result.tool_calls:
                if not turn_result.text.strip() and empty_retries < EMPTY_RESPONSE_RETRIES:
                    empty_retries += 1
                    self.session.history.append(
                        Message(role="user", content=EMPTY_RESPONSE_NOTICE, turn_index=turn)
                    )
                    continue
                if not turn_result.text.strip():
                    raise EmptyModelResponseError(
                        f"已重试 {empty_retries} 次", retries=empty_retries
                    )
                return turn_result.text

            if turn_result.truncated:
                self.session.history.append(
                    Message(role="user", content=TRUNCATED_NOTICE, turn_index=turn)
                )

            calls = self._trim_calls(turn_result.tool_calls)
            results = await self._step_act(calls, turn)
            await self._step_observe(results, turn)

            if self._loop_aborted:
                return self._loop_message or LOOP_ABORT_NOTICE

            failures = failures + 1 if results and all(not r.ok for r in results) else 0
            if failures >= CONSECUTIVE_FAILURE_LIMIT:
                self.session.history.append(
                    Message(
                        role="user",
                        content=CONSECUTIVE_FAILURE_NOTICE.format(n=failures),
                        turn_index=turn,
                    )
                )
                failures = 0
            empty_retries = 0

            if attempt == MAX_TURNS_PER_REQUEST - 1:
                self.display.warn(TURN_LIMIT_NOTICE.format(n=MAX_TURNS_PER_REQUEST))
                return await self._forced_summary(turn)

        return await self._forced_summary(turn)

    async def _forced_summary(self, turn: int) -> str:
        """达到轮次上限后强制要一段总结（B11）。

        Args:
            turn: 当前轮次。

        Returns:
            最后一条 assistant 文本，或固定文案。
        """
        messages = await self._assemble(turn)
        messages.append(
            Message(
                role="user",
                content=TURN_LIMIT_NOTICE.format(n=MAX_TURNS_PER_REQUEST),
                turn_index=turn,
            )
        )
        try:
            result = await self.llm.assistant_turn(
                "main",
                messages=messages,
                tools=None,
                system=self._system_text(),
                display=self.display,
                timeout_s=None,
            )
        except LLMUnavailableError:
            return f"已达单次请求的工具调用上限（{MAX_TURNS_PER_REQUEST} 轮），未能生成总结。"
        self._account(result)
        text = result.text.strip()
        if text:
            self.session.history.append(
                Message(
                    role="assistant",
                    content=text,
                    turn_index=turn,
                    subagent_id=self.subagent_id,
                )
            )
        return text or f"已达单次请求的工具调用上限（{MAX_TURNS_PER_REQUEST} 轮）。"

    # ------------------------------------------------------------------ 后台子 Agent
    def _drain_background(self, turn: int) -> None:
        """把已完成的后台子 Agent 结论注入主上下文（§12.2）。

        每次模型调用前调用一次：``SubagentRunner.drain_background()`` 返回尚未投递的
        ``[后台子 Agent 完成] …`` 结论块，这里以 ``role="user"`` 消息追加进 history，
        使模型在**本轮**就能看到（这正是「后台并行」的意义）。

        设计要点：
        - 只在**主 Query** 生效：子 Agent 内部构造的 ``Query`` 不传 ``subagent_runner``，
          因此深度 1 的隔离不被破坏；
        - 任何异常都只记日志、不阻断回合——后台结论是**增强信息**，不能反过来弄挂主流程；
        - 取走即清空（``drain_background`` 语义），不会重复注入。

        Args:
            turn: 当前轮次（写入消息元数据，供摘要/检索定位）。
        """
        runner = self.subagent_runner
        if runner is None:
            return
        try:
            blocks = runner.drain_background()
        except Exception as exc:  # 后台结论是增强信息：任何故障都不得阻断主回合
            _LOG.warning("取回后台子 Agent 结论失败（已忽略）：%s", exc)
            return
        for block in blocks:
            if not block:
                continue
            self.session.history.append(
                Message(role="user", content=block, turn_index=turn, subagent_id=None)
            )
            _LOG.info("已注入后台子 Agent 结论（%d 字符）", len(block))

    # ------------------------------------------------------------------ 三步
    async def _assemble(self, turn: int) -> list[Message]:
        """组装发往模型的消息（§7.6）。

        Args:
            turn: 当前轮次。

        Returns:
            消息列表。
        """
        build = self.context.build(
            session=self.session,
            tools_schema=self.tools.schema(),
            memory_block=self.memory_block,
            skills_block=self.skills_block,
            repomap=self.repomap,
        )
        self.session.context_watermark = build.watermark
        self.compacted = self.compacted or build.compacted
        return list(build.messages)

    async def _step_think(self, messages: list[Message]) -> AssistantTurn:
        """一次模型调用（供外部/测试直接驱动）。

        Args:
            messages: 发往模型的消息。

        Returns:
            :class:`~my_agent.llm.client.AssistantTurn`。
        """
        return await self.llm.assistant_turn(
            "main",
            messages=messages,
            tools=self.tools.schema() if self._tools_enabled() else None,
            system=self._system_text(),
            display=self.display,
            timeout_s=None,
        )

    async def _review_dispatch(
        self, calls: list[ToolCallRequest], turn: int
    ) -> dict[str, str]:
        """对整批 ``Task`` 调用做分发审判（U35）。

        两条派生路径（用户明确要求）：

        - **过粗**：审判器**亲自补发**遗漏的子任务，结论以 ``[审判 Agent 补发的…]``
          的 user 消息注入主上下文；原批次照常执行；
        - **过细**：审判器**不派生**，把「结论/补丁/影响分析/理由」作为反馈打回，
          由模型在下一轮用修订后的任务重新分发。

        审判模型不可用时：审判器已按指数退避重试并**告警用户**，此处按放行处理。

        Args:
            calls: 本轮全部工具调用。
            turn: 当前轮次。

        Returns:
            ``call_id -> 打回反馈``；空字典表示不需要打回（放行）。
        """
        judge = self.dispatch_judge
        if judge is None:
            return {}
        task_calls = [call for call in calls if call.name == "Task"]
        if not task_calls:
            return {}
        if self._dispatch_judge_count >= MAX_DISPATCH_JUDGE_PER_TURN:
            _LOG.info("本回合审判次数已达上限（%d），跳过审判", MAX_DISPATCH_JUDGE_PER_TURN)
            return {}
        self._dispatch_judge_count += 1

        from my_agent.agents.dispatch_judge import (
            DispatchVerdict,
            SubtaskSpec,
            build_context,
            supplement_text,
        )

        subtasks = [
            SubtaskSpec(
                description=str(call.arguments.get("description") or "").strip() or "子任务",
                prompt=str(call.arguments.get("prompt") or "").strip(),
                agent_type=str(call.arguments.get("agent_type") or "general"),
            )
            for call in task_calls
        ]
        root = Path(self.session.project_root or self.session.cwd)
        constraints = (self.memory_block or "")[:2_000]
        try:
            context = build_context(
                user_requirement=self._current_user_requirement(),
                subtasks=subtasks,
                constraints=constraints,
                root=root,
            )
            judgment, supplements = await judge.review(context, parent=self.session)
        except MyAgentError as exc:
            # 审判失败绝不阻断分发（fail-open）；审判器内部已尝试退避重试并告警
            _LOG.warning("分发审判失败（放行）：%s", exc.user_message())
            return {}

        if supplements:
            block = supplement_text(supplements)
            if block:
                self.session.history.append(
                    Message(role="user", content=block, turn_index=turn)
                )
                self.display.info(f"审判 Agent 已补发 {len(supplements)} 个子任务")

        if judgment.verdict is DispatchVerdict.TOO_FINE:
            feedback = judgment.feedback_text()
            self.display.warn(f"审判 Agent 判定分发过细：{judgment.reason}")
            return {call.id: feedback for call in task_calls}

        if judgment.verdict is DispatchVerdict.TOO_COARSE:
            _LOG.info("审判判定分发过粗：%s", judgment.reason)

        return {}

    def _current_user_requirement(self) -> str:
        """取「原始用户需求」——最近一条真实用户输入（排除注入块）。

        Returns:
            用户输入文本；找不到时返回空串。
        """
        skip_prefixes = (
            "[后台子 Agent 完成]",
            "[审判 Agent 补发的子任务结论]",
        )
        for message in reversed(self.session.history):
            if message.role != "user" or message.kind != "normal":
                continue
            content = (message.content or "").strip()
            if not content or content.startswith(skip_prefixes):
                continue
            return content
        return ""

    async def _step_act(self, calls: list[ToolCallRequest], turn: int) -> list[ToolResult]:
        """对每个 call 走权限流水线并执行（§5.3.4 B3/B4/B5/B6）。

        Args:
            calls: 本轮的模型工具调用（顺序即回填顺序）。
            turn: 当前轮次。

        Returns:
            与 ``calls`` 顺序一致的执行结果。
        """
        assistant_msg = Message(
            role="assistant",
            content="",
            tool_calls=calls,
            turn_index=turn,
            subagent_id=self.subagent_id,
        )
        self.session.history.append(assistant_msg)

        # 分发审判（U35）：整批 Task 在执行前先审一遍。返回值 = 需要打回的 call_id → 反馈
        rejected = await self._review_dispatch(calls, turn)

        plans: list[PermissionDecision | ToolResult] = []

        for call in calls:
            if call.id in rejected:
                plans.append(
                    ToolResult(
                        call_id=call.id,
                        tool=call.name,
                        ok=False,
                        error=rejected[call.id],
                    )
                )
                continue
            spec = self.tools.get_spec(call.name)
            if spec is None:
                plans.append(
                    ToolResult(
                        call_id=call.id,
                        tool=call.name,
                        ok=False,
                        error=f"工具 {call.name} 不存在；请从可用工具中选择。",
                    )
                )
                continue
            inv = ToolInvocation(
                call_id=call.id,
                tool=call.name,
                arguments=dict(call.arguments),
                reason=self._reason_for(call),
                spec=spec,
            )
            inv = await self._hook_before_tool(inv, turn)
            if inv is None:
                plans.append(
                    ToolResult(
                        call_id=call.id,
                        tool=call.name,
                        ok=False,
                        denied_by_permission=True,
                        error="被 hook 拒绝",
                    )
                )
                continue
            inv = await self._hook_before_permission(inv, turn)
            if inv is None:
                plans.append(
                    ToolResult(
                        call_id=call.id,
                        tool=call.name,
                        ok=False,
                        denied_by_permission=True,
                        error="被 hook 拒绝（权限检查前）",
                    )
                )
                continue

            decision = await self.pipeline.decide(inv, self.session.mode)
            decision = await self._hook_after_permission(decision, turn)
            plans.append(decision)
            if decision.user_choice == "abort_turn":
                self._append_results(self._results_from_plans(plans, calls))
                raise _UserAbort()

        return await self._execute_plans(plans, calls, turn)

    async def _execute_plans(
        self,
        plans: list[PermissionDecision | ToolResult],
        calls: list[ToolCallRequest],
        turn: int,
    ) -> list[ToolResult]:
        """执行已裁决的调用（读并发、写串行；B2 顺序保证）。

        Args:
            plans: 每个 call 的裁决或直接结果。
            calls: 原始调用列表。
            turn: 当前轮次。

        Returns:
            与 ``calls`` 顺序一致的结果列表。
        """
        results: list[ToolResult] = []
        executable: list[tuple[str, ToolInvocation]] = []

        for call, plan in zip(calls, plans, strict=False):
            if isinstance(plan, ToolResult):
                results.append(plan)
                continue
            if plan.final == "deny":
                self.display.tool_result(call.name, False, "权限被拒绝")
                results.append(
                    ToolResult(
                        call_id=call.id,
                        tool=call.name,
                        ok=False,
                        denied_by_permission=True,
                        error=plan.user_note or (plan.judge.reason if plan.judge else "权限被拒绝"),
                    )
                )
                continue
            spec = self.tools.get_spec(call.name)
            if spec is None:  # pragma: no cover - 上面已过滤
                continue
            executable.append(
                (
                    call.id,
                    ToolInvocation(
                        call_id=call.id,
                        tool=call.name,
                        arguments=dict(call.arguments),
                        reason=self._reason_for(call),
                        spec=spec,
                    ),
                )
            )
            results.append(
                ToolResult(call_id=call.id, tool=call.name, ok=False, error="__pending__")
            )

        ctx = self._tool_context(turn)
        executed = await self.tools.execute_many(
            [inv for _, inv in executable], ctx.model_copy(update={"permission_checked": True})
        )
        by_call = {result.call_id: result for result in executed}

        final: list[ToolResult] = []
        for call, item in zip(calls, results, strict=False):
            if item.error == "__pending__":
                result = by_call.get(call.id) or ToolResult(
                    call_id=call.id, tool=call.name, ok=False, error="工具未返回结果"
                )
                await self._hook_after_tool(result, turn)
                final.append(result)
            else:
                final.append(item)
        return final

    async def _step_observe(self, results: list[ToolResult], turn: int) -> None:
        """把工具结果回填为 ``role="tool"`` 消息（B2 顺序一致）并做死循环检测。

        每回填一条结果就交给 :class:`~my_agent.engine.loop_guard.LoopGuard` 记录一次：
        「同一 (tool, 参数) 指纹重复 ≥ N 次 **且** 结果内容向量相似度 ≥ 阈值」即判死循环
        （【待确认假设 U33】）。判定成立时**先补完本批全部结果**（保持 assistant
        ``tool_calls`` 与 tool 消息一一对应，不破坏 §4.1 不变量），再回填一条说明原因的
        tool 消息并置位 ``_loop_aborted``，由 :meth:`_loop` 终止本回合。

        Args:
            results: 执行结果（与请求顺序一致）。
            turn: 当前轮次。
        """
        detected: tuple[LoopCheck, str] | None = None
        for result in results:
            self.display.tool_result(result.tool, result.ok, _summarize(result))
            content = result.as_model_text()
            msg = Message(
                role="tool",
                content=content,
                tool_call_id=result.call_id,
                name=result.tool,
                kind="tool_result" if result.ok else "normal",
                is_large_result=bool(result.artifact_path),
                archive_ref=result.artifact_path,
                tokens=result.tokens,
                turn_index=turn,
                subagent_id=self.subagent_id,
            )
            self.session.history.append(msg)
            self._turn_calls.append(result.call_id)
            self._turn_results.append(result)

            check = self.loop_guard.record(
                tool=result.tool,
                arguments=self._arguments_for(result.call_id),
                result_text=content,
            )
            if check.verdict is LoopVerdict.LOOP_DETECTED and detected is None:
                detected = (check, result.call_id)

        if detected is not None:
            self._abort_loop(detected[0], detected[1], turn)

    def _abort_loop(self, check: LoopCheck, call_id: str, turn: int) -> None:
        """终止本回合：向用户告警 + 回填一条错误 tool 消息说明检测到重复调用。

        Args:
            check: 触发终止的判定（``message`` 为中文说明）。
            call_id: 触发判定的调用 id（tool 消息必须挂在某个 assistant ``tool_calls`` 上）。
            turn: 当前轮次。
        """
        self._loop_aborted = True
        self._loop_message = check.message
        self.display.warn(check.message)
        self.session.history.append(
            Message(
                role="tool",
                content=f"执行失败：{check.message}{LOOP_RESULT_ERROR_NOTICE}",
                tool_call_id=call_id,
                name=check.tool,
                turn_index=turn,
                subagent_id=self.subagent_id,
            )
        )

    def _arguments_for(self, call_id: str) -> dict[str, Any]:
        """从最近一条 assistant 消息的 ``tool_calls`` 里取回本次调用的参数。

        :meth:`_step_observe` 只拿到结果，参数在 :meth:`_step_act` 追加的 assistant
        消息上；取不到时返回空字典（死循环检测退化为「同工具但参数未知」，
        **不**影响消息回填）。

        Args:
            call_id: 工具调用 id。

        Returns:
            参数副本；未找到时为空字典。
        """
        for msg in reversed(self.session.history):
            if msg.role != "assistant" or not msg.tool_calls:
                continue
            for call in msg.tool_calls:
                if call.id == call_id:
                    return dict(call.arguments)
            break
        return {}

    # ------------------------------------------------------------------ hook
    async def _hook_before_model(self, messages: list[Message]) -> None:
        payload = self._hook_payload()
        payload.messages = list(messages)
        outcome = await self.hooks.run(
            _event("BEFORE_MODEL_CALL"),
            session_id=self.session.session_id,
            turn_index=self.session.turn_index,
            payload=payload,
        )
        if outcome.modified_payload and outcome.modified_payload.messages is not None:
            messages[:] = outcome.modified_payload.messages
        if outcome.decision == "deny":
            self.display.warn(f"hook 拒绝了本次模型调用：{outcome.message}")

    async def _hook_after_model(self, turn_result: AssistantTurn) -> None:
        payload = self._hook_payload()
        payload.text = turn_result.text
        outcome = await self.hooks.run(
            _event("AFTER_MODEL_CALL"),
            session_id=self.session.session_id,
            turn_index=self.session.turn_index,
            payload=payload,
        )
        if outcome.modified_payload and outcome.modified_payload.text is not None:
            turn_result.text = outcome.modified_payload.text

    async def _hook_before_tool(self, inv: ToolInvocation, turn: int) -> ToolInvocation | None:
        payload = self._hook_payload()
        payload.tool = inv.tool
        payload.arguments = dict(inv.arguments)
        payload.reason = inv.reason
        outcome = await self.hooks.run(
            _event("BEFORE_TOOL_CALL"),
            session_id=self.session.session_id,
            turn_index=turn,
            payload=payload,
        )
        if outcome.decision == "deny":
            self.display.warn(f"hook 拒绝 {inv.tool}：{outcome.message}")
            return None
        if outcome.modified_payload and outcome.modified_payload.arguments:
            inv = inv.model_copy(update={"arguments": dict(outcome.modified_payload.arguments)})
        return inv

    async def _hook_before_permission(
        self, inv: ToolInvocation, turn: int
    ) -> ToolInvocation | None:
        payload = self._hook_payload()
        payload.tool = inv.tool
        payload.arguments = dict(inv.arguments)
        payload.reason = inv.reason
        outcome = await self.hooks.run(
            _event("BEFORE_PERMISSION_CHECK"),
            session_id=self.session.session_id,
            turn_index=turn,
            payload=payload,
        )
        if outcome.decision == "deny":
            self.display.warn(f"hook 在权限检查前拒绝 {inv.tool}：{outcome.message}")
            return None
        if outcome.modified_payload and outcome.modified_payload.arguments:
            inv = inv.model_copy(update={"arguments": dict(outcome.modified_payload.arguments)})
        return inv

    async def _hook_after_permission(
        self, decision: PermissionDecision, turn: int
    ) -> PermissionDecision:
        """``after_permission_check``：只能收紧，不能放宽（§11.4 安全不变量）。"""
        payload = self._hook_payload()
        payload.tool = decision.tool
        payload.arguments = dict(decision.arguments)
        payload.decision = decision.final
        outcome = await self.hooks.run(
            _event("AFTER_PERMISSION_CHECK"),
            session_id=self.session.session_id,
            turn_index=turn,
            payload=payload,
        )
        if outcome.decision in {"deny", "ask"} and decision.final == "allow":
            self.display.warn(
                f"hook 收紧了 {decision.tool} 的裁决：allow → {outcome.decision}（{outcome.message}）"
            )
            return decision.model_copy(update={"final": outcome.decision})
        return decision

    async def _hook_after_tool(self, result: ToolResult, turn: int) -> None:
        payload = self._hook_payload()
        payload.tool = result.tool
        payload.result = result
        outcome = await self.hooks.run(
            _event("AFTER_TOOL_CALL"),
            session_id=self.session.session_id,
            turn_index=turn,
            payload=payload,
        )
        if outcome.modified_payload and outcome.modified_payload.result is not None:
            result.content = outcome.modified_payload.result.content

    # ------------------------------------------------------------------ 辅助
    def _hook_payload(self) -> HookPayload:
        from my_agent.hooks.events import HookPayload

        return HookPayload(subagent_id=self.subagent_id)

    def _tool_context(self, turn: int) -> ToolContext:
        return ToolContext(
            session_id=self.session.session_id,
            turn_index=turn,
            cwd=self.session.cwd,
            project_root=self.session.project_root or self.session.cwd,
            config=self.config,
            artifact_dir=str(self._artifact_dir()),
            abort_event=self.abort_event,
            permission_checked=False,
            subagent_id=self.subagent_id,
        )

    def _artifact_dir(self) -> Any:
        from pathlib import Path

        root = Path(self.session.project_root or self.session.cwd)
        return root / ".myagent" / "artifacts" / self.session.session_id

    def _system_text(self) -> str:
        return self.system_prompt

    def _tools_enabled(self) -> bool:
        return True

    def _reason_for(self, call: ToolCallRequest) -> str:
        """取调用原因（§6.5.1 第 3 条：优先模型给出的原因，否则取同一轮文本末 3 行）。"""
        if call.arguments.get("reason"):
            return str(call.arguments["reason"])
        for msg in reversed(self.session.history):
            if msg.role == "assistant" and msg.content.strip():
                tail = [line for line in msg.content.strip().splitlines() if line.strip()][-3:]
                if tail:
                    return "\n".join(tail)
                break
        return "(模型未说明)"

    def _trim_calls(self, calls: list[ToolCallRequest]) -> list[ToolCallRequest]:
        if len(calls) <= MAX_TOOL_CALLS_PER_TURN:
            return calls
        kept = calls[:MAX_TOOL_CALLS_PER_TURN]
        skipped = calls[MAX_TOOL_CALLS_PER_TURN:]
        self.display.warn(
            f"单轮工具调用超过上限（{MAX_TOOL_CALLS_PER_TURN}），已截断 {len(skipped)} 个："
            + ", ".join(call.name for call in skipped)
        )
        return kept

    def _results_from_plans(
        self, plans: list[PermissionDecision | ToolResult], calls: list[ToolCallRequest]
    ) -> list[ToolResult]:
        out: list[ToolResult] = []
        for call, plan in zip(calls, plans, strict=False):
            if isinstance(plan, ToolResult):
                out.append(plan)
            else:
                out.append(
                    ToolResult(
                        call_id=call.id,
                        tool=call.name,
                        ok=False,
                        denied_by_permission=plan.final == "deny",
                        error="用户中止了本轮",
                    )
                )
        return out

    def _account(self, turn_result: AssistantTurn) -> None:
        usage = turn_result.usage or {}
        self._tokens_in += int(usage.get("prompt_tokens", 0))
        self._tokens_out += int(usage.get("completion_tokens", 0))

    def _finish_turn(self, turn: int, user_input: str, final_text: str) -> None:
        from my_agent.models.session import TurnRecord

        record = TurnRecord(
            index=turn,
            user_input=user_input,
            assistant_final=final_text,
            tool_calls=list(self._turn_calls),
            started_at=self.session.created_at,
            ended_at=time.time(),
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
        )
        self.session.turns.append(record)
        totals = self.session.usage_totals
        totals["prompt_tokens"] = totals.get("prompt_tokens", 0) + self._tokens_in
        totals["completion_tokens"] = totals.get("completion_tokens", 0) + self._tokens_out
        totals["total"] = totals.get("total", 0) + self._tokens_in + self._tokens_out
        if final_text:
            self.session.history.append(
                Message(
                    role="assistant",
                    content=final_text,
                    turn_index=turn,
                    subagent_id=self.subagent_id,
                )
            )

    def _append_results(self, results: list[ToolResult]) -> None:
        for result in results:
            self.session.history.append(
                Message(
                    role="tool",
                    content=result.as_model_text(),
                    tool_call_id=result.call_id,
                    name=result.tool,
                    turn_index=self.session.turn_index,
                )
            )


class _UserAbort(Exception):
    """内部信号：用户选择 abort_turn（B5）。"""


def _event(name: str) -> Any:
    """按名取 hook 事件枚举成员（延迟导入避免循环依赖）。"""
    from my_agent.hooks import events

    return getattr(events.HookEvent, name)


def _summarize(result: ToolResult) -> str:
    """生成工具结果的一行摘要（UI 用）。

    Args:
        result: 工具结果。

    Returns:
        截断到 120 字符的摘要。
    """
    text = (result.content or result.error or "").strip().replace("\n", " ")
    return text[:120]
