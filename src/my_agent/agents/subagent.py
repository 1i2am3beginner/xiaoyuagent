"""子 Agent 运行时（§12）。

设计要点（§12.1）：

* **派生方式**：主 Agent 通过内置工具 ``Task`` 派生（见 ``task_tool.py``）。
* **上下文**：子 Agent 全新独立上下文——**不含**父 history，只注入子 Agent 系统提示词、
  工作目录摘要、项目记忆、工具 schema 与任务描述（H-1）。
* **回传**：只回传最终文本结论，中间工具调用**不写入**主上下文。
* **并行**：支持后台并行，上限 ``MAX_CONCURRENT_SUBAGENTS``，超出**排队**而非报错（H-4）。
* **权限**：继承父会话安全模式，每次工具调用走**同一个** ``PermissionPipeline``（H-2）；
  询问转交主 UI 并标注来源 ``[子 Agent <id>]``。
* **记忆**：子 Agent **不写**记忆（H-6）。
* **深度**：``MAX_SUBAGENT_DEPTH = 1``，子 Agent 内不可再用 ``Task``（H-5）。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from my_agent.config.defaults import (
    MAX_CONCURRENT_SUBAGENTS,
    MAX_SUBAGENT_DEPTH,
    SUBAGENT_POLL_INTERVAL_S,
)
from my_agent.errors import MyAgentError, SubagentLimitError
from my_agent.models.messages import Message
from my_agent.models.permission import SafetyMode
from my_agent.models.protocols import DisplayPort, NullDisplay, PermissionRequest, UserAnswer
from my_agent.models.session import SessionState
from my_agent.models.tools import ToolSpec

if TYPE_CHECKING:
    from my_agent.config.schema import Config
    from my_agent.context.archive import ArchiveStore
    from my_agent.hooks.runner import HookRunner
    from my_agent.llm.client import LLMClient
    from my_agent.llm.tokenizer import TokenCounter
    from my_agent.models.protocols import Confirmer
    from my_agent.security.pipeline import PermissionPipeline
    from my_agent.tools.registry import ToolRegistry

__all__ = ["AgentType", "SubagentResult", "SubagentRunner"]

_LOG = logging.getLogger(__name__)

#: 允许的 agent_type（§12.2）。
AgentType = str

AGENT_TYPES: tuple[str, ...] = ("general", "explore", "review")

#: 只读工具类别（``explore`` 预设）。
READ_ONLY_CATEGORIES: frozenset[str] = frozenset({"read", "network", "meta"})

#: ``review`` 预设允许的额外工具（Bash 仅只读命令由提示词约束 + 权限流水线兜底）。
REVIEW_EXTRA_TOOLS: frozenset[str] = frozenset({"Bash"})

BACKGROUND_PREFIX = "[后台子 Agent 完成]"
DONE_TEMPLATE = (
    "[子 Agent 完成] {description}（{duration:.1f}s, {calls} 次工具调用, {tokens} tokens）\n"
    "结论：\n{text}"
)
FAIL_TEMPLATE = "[子 Agent 失败] {reason}"
STARTED_TEMPLATE = "[子 Agent 已在后台启动] id={sid}"


class SubagentResult(BaseModel):
    """一次子 Agent 执行的结论。"""

    id: str
    description: str
    ok: bool
    text: str = ""
    error: str | None = None
    duration_ms: int = 0
    tool_calls: int = 0
    tokens: int = 0
    depth: int = 1
    agent_type: str = "general"

    model_config = {"extra": "ignore"}

    def as_block(self, *, background: bool = False) -> str:
        """渲染成注入主上下文的文本块（§12.2）。

        Args:
            background: 是否为后台完成的通知（加 ``[后台子 Agent 完成]`` 前缀）。

        Returns:
            Markdown 文本块。
        """
        if not self.ok:
            body = FAIL_TEMPLATE.format(reason=self.error or "未知原因")
        else:
            body = DONE_TEMPLATE.format(
                description=self.description,
                duration=self.duration_ms / 1000,
                calls=self.tool_calls,
                tokens=self.tokens,
                text=self.text.strip() or "（子 Agent 未给出结论）",
            )
        return f"{BACKGROUND_PREFIX}\n{body}" if background else body


class _InheritedConfirmer:
    """把子 Agent 的权限询问转交主 UI，并标注来源（§12.1）。"""

    def __init__(self, parent: Confirmer, subagent_id: str) -> None:
        self._parent = parent
        self._id = subagent_id

    async def ask(self, request: PermissionRequest) -> UserAnswer:
        """转交询问并补齐 ``subagent_id``。

        Args:
            request: 询问卡片。

        Returns:
            父确认器的回答。
        """
        forwarded = request.model_copy(
            update={
                "subagent_id": self._id,
                "reason": f"[子 Agent {self._id}] {request.reason}",
            }
        )
        return await self._parent.ask(forwarded)


class SubagentRunner:
    """子 Agent 的创建、限额与结论回传。"""

    def __init__(
        self,
        *,
        config: Config,
        llm: LLMClient,
        tools: ToolRegistry,
        pipeline: PermissionPipeline | None,
        hooks: HookRunner | None,
        archive: ArchiveStore | None = None,
        counter: TokenCounter | None = None,
        cwd: Path | None = None,
        project_root: Path | None = None,
        display: DisplayPort | None = None,
        confirmer: Confirmer | None = None,
        repomap: str = "",
        memory_block: str = "",
        depth: int = 0,
    ) -> None:
        """初始化。

        Args:
            config: 生效配置。
            llm: 模型客户端。
            tools: 父级工具注册表（子 Agent 的工具集会从中过滤）。
            pipeline: 权限流水线（**与父共用**，H-2）；缺省时子 Agent 无法执行工具。
            hooks: hook 执行器。
            archive: 归档存储（当前仅用于上下文视图，子 Agent 不写归档）。
            counter: token 计数器。
            cwd: 工作目录。
            project_root: 项目根。
            display: 输出端口。
            confirmer: 父确认器（用于转交权限询问）。
            repomap: 工作目录摘要块。
            memory_block: 项目记忆块（**只注入项目记忆**，子 Agent 不写记忆）。
            depth: 当前深度（0 = 顶层主 Agent）。
        """
        self.config = config
        self.llm = llm
        self.tools = tools
        self.pipeline = pipeline
        self.hooks = hooks
        self.archive = archive
        self.counter = counter
        self.cwd = Path(cwd or Path.cwd())
        self.project_root = Path(project_root or self.cwd)
        self.display: DisplayPort = display or NullDisplay()
        self.confirmer = confirmer
        self.repomap = repomap
        self.memory_block = memory_block
        self.depth = depth

        self._semaphore = asyncio.Semaphore(config.agents.max_concurrent_subagents)
        self._background: dict[str, asyncio.Task[SubagentResult]] = {}
        self._completed: list[str] = []
        self._results: dict[str, SubagentResult] = {}
        self._active = 0

    # ------------------------------------------------------------------ 派生
    async def spawn(
        self,
        *,
        description: str,
        prompt: str,
        agent_type: str = "general",
        mode: str = "inherit",
        run_in_background: bool = False,
        parent: SessionState | None = None,
        parent_call_id: str = "",
    ) -> SubagentResult:
        """派生一个子 Agent（§12.2）。

        Args:
            description: 3-5 词短标题。
            prompt: 完整自包含的任务说明。
            agent_type: ``general`` / ``explore`` / ``review``。
            mode: ``inherit`` / ``plan`` / ``default``。
            run_in_background: 是否后台并行。
            parent: 父会话（用于继承安全模式与工作目录）。
            parent_call_id: 父级调用 id（审计用）。

        Returns:
            :class:`SubagentResult`；后台模式立即返回 ``ok=True`` 的启动回执。

        Raises:
            SubagentLimitError: 超过深度上限。
        """
        if self.depth + 1 > MAX_SUBAGENT_DEPTH:
            raise SubagentLimitError(
                f"子 Agent 深度已达上限（{MAX_SUBAGENT_DEPTH}）",
                n=MAX_SUBAGENT_DEPTH,
                d=MAX_SUBAGENT_DEPTH,
            )
        if agent_type not in AGENT_TYPES:
            agent_type = "general"
        sub_id = f"sub-{uuid.uuid4().hex[:8]}"

        if run_in_background:
            task = asyncio.create_task(
                self._guarded(sub_id, description, prompt, agent_type, mode, parent, parent_call_id)
            )
            self._background[sub_id] = task
            self.display.info(STARTED_TEMPLATE.format(sid=sub_id))
            return SubagentResult(
                id=sub_id,
                description=description,
                ok=True,
                text=STARTED_TEMPLATE.format(sid=sub_id),
                agent_type=agent_type,
            )

        return await self._guarded(
            sub_id, description, prompt, agent_type, mode, parent, parent_call_id
        )

    async def _guarded(
        self,
        sub_id: str,
        description: str,
        prompt: str,
        agent_type: str,
        mode: str,
        parent: SessionState | None,
        parent_call_id: str,
    ) -> SubagentResult:
        """在并发上限内执行子 Agent，任何异常都转成失败结论（H-4 排队语义）。

        Args:
            sub_id: 子 Agent id。
            description: 短标题。
            prompt: 任务说明。
            agent_type: 预设类型。
            mode: 安全模式策略。
            parent: 父会话。
            parent_call_id: 父调用 id。

        Returns:
            :class:`SubagentResult`。
        """
        async with self._semaphore:  # 超出上限时在此排队，不报错
            self._active += 1
            started = time.monotonic()
            try:
                text, calls, tokens = await self._execute(
                    sub_id, description, prompt, agent_type, mode, parent, parent_call_id
                )
                result = SubagentResult(
                    id=sub_id,
                    description=description,
                    ok=True,
                    text=text,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    tool_calls=calls,
                    tokens=tokens,
                    agent_type=agent_type,
                )
            except MyAgentError as exc:
                result = SubagentResult(
                    id=sub_id,
                    description=description,
                    ok=False,
                    error=exc.user_message(),
                    duration_ms=int((time.monotonic() - started) * 1000),
                    agent_type=agent_type,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # 子 Agent 隔离：任何异常都不拖垮主流程
                _LOG.exception("子 Agent %s 内部错误", sub_id)
                result = SubagentResult(
                    id=sub_id,
                    description=description,
                    ok=False,
                    error=f"内部错误（{type(exc).__name__}）：{exc}",
                    duration_ms=int((time.monotonic() - started) * 1000),
                    agent_type=agent_type,
                )
            finally:
                self._active -= 1
            self._results[sub_id] = result
            self._completed.append(sub_id)
            return result

    async def _execute(
        self,
        sub_id: str,
        description: str,
        prompt: str,
        agent_type: str,
        mode: str,
        parent: SessionState | None,
        parent_call_id: str,
    ) -> tuple[str, int, int]:
        """跑一次子 Agent 的 ReAct 循环。

        Args:
            sub_id: 子 Agent id。
            description: 短标题。
            prompt: 任务说明。
            agent_type: 预设类型。
            mode: 安全模式策略。
            parent: 父会话。
            parent_call_id: 父调用 id。

        Returns:
            ``(最终文本, 工具调用数, 消耗 token 数)``。
        """
        from my_agent.context.assembler import ContextAssembler
        from my_agent.engine.query import Query
        from my_agent.llm.prompts import SUBAGENT_SYSTEM_PROMPT

        session = self._fresh_session(parent, mode, sub_id, description)
        tools = self._filtered_registry(agent_type)
        assert self.counter is not None, "SubagentRunner 需要 TokenCounter"
        context = ContextAssembler(config=self.config, counter=self.counter)

        system = SUBAGENT_SYSTEM_PROMPT.format(
            agent_type=agent_type, description=description, prompt=prompt
        )
        query = Query(
            session=session,
            config=self.config,
            llm=self.llm,
            tools=tools,
            context=context,
            pipeline=self._pipeline_for(sub_id, parent_call_id),
            hooks=self._hooks_for(sub_id),
            confirmer=self._confirmer_for(sub_id),
            display=self.display,
            memory_writer=None,  # H-6：子 Agent 不写记忆
            system_prompt=system,
            repomap=self.repomap,
            memory_block=self.memory_block,
            skills_block="",
            subagent_id=sub_id,
        )
        result = await query.run(prompt)
        usage = session.usage_totals
        return (
            result.text,
            len([call for turn in session.turns for call in turn.tool_calls]),
            int(usage.get("total", 0)),
        )

    # ------------------------------------------------------------------ 装配
    def _fresh_session(
        self, parent: SessionState | None, mode: str, sub_id: str, description: str
    ) -> SessionState:
        """构造子 Agent 的**全新**会话（不含父 history，H-1）。

        Args:
            parent: 父会话。
            mode: 安全模式策略。
            sub_id: 子 Agent id。
            description: 短标题。

        Returns:
            :class:`~my_agent.models.session.SessionState`。
        """
        safety: SafetyMode = "default"
        if parent is not None:
            safety = parent.mode if mode == "inherit" else _mode_from(mode, parent.mode)
        elif mode in {"plan", "default"}:
            safety = mode  # type: ignore[assignment]
        now = time.time()
        return SessionState(
            session_id=sub_id,
            cwd=str(parent.cwd if parent else self.cwd),
            mode=safety,
            created_at=now,
            updated_at=now,
            project_root=str(parent.project_root if parent else self.project_root),
            title=description,
        )

    def _filtered_registry(self, agent_type: str) -> ToolRegistry:
        """按 ``agent_type`` 过滤工具集（§12.2 预设表）。

        Args:
            agent_type: 预设类型。

        Returns:
            新的 :class:`~my_agent.tools.registry.ToolRegistry`（父注册表不被修改）。
        """
        from my_agent.tools.registry import ToolRegistry

        registry = ToolRegistry()
        for spec in self.tools.to_specs():
            if self._allowed(spec, agent_type):
                tool = self.tools.get(spec.name)
                if tool is not None:
                    registry.register(tool)
        return registry

    @staticmethod
    def _allowed(spec: ToolSpec, agent_type: str) -> bool:
        """判断某工具在该 ``agent_type`` 下是否可用。

        Args:
            spec: 工具声明。
            agent_type: 预设类型。

        Returns:
            可用时 ``True``。
        """
        if spec.name == "Task":
            return False  # H-5：深度上限 1，子 Agent 内无 Task
        if agent_type == "general":
            return True
        if agent_type == "explore":
            return spec.is_read_only or spec.category in READ_ONLY_CATEGORIES
        if agent_type == "review":
            return (
                spec.is_read_only
                or spec.category in READ_ONLY_CATEGORIES
                or spec.name in REVIEW_EXTRA_TOOLS
            )
        return True

    def _pipeline_for(self, sub_id: str, parent_call_id: str) -> Any:
        """返回子 Agent 使用的权限流水线（**与父共用**，H-2）。

        Args:
            sub_id: 子 Agent id。
            parent_call_id: 父调用 id。

        Returns:
            权限流水线；父级未提供时返回一个拒绝一切的占位对象。
        """
        if self.pipeline is None:
            return _DenyAllPipeline()
        return _TracingPipeline(self.pipeline, sub_id=sub_id, parent_call_id=parent_call_id)

    def _hooks_for(self, sub_id: str) -> Any:
        """返回子 Agent 使用的 hook 运行器（``subagent_isolation`` 时禁用）。

        Args:
            sub_id: 子 Agent id。

        Returns:
            hook 运行器或空运行器。
        """
        if self.hooks is None or self.config.hooks.subagent_isolation:
            return _NoHooks()
        return self.hooks

    def _confirmer_for(self, sub_id: str) -> Any:
        """返回转发权限询问的确认器。

        Args:
            sub_id: 子 Agent id。

        Returns:
            转发确认器；父级未提供时返回自动拒绝的确认器。
        """
        if self.confirmer is None:
            from my_agent.models.protocols import AutoRejectConfirmer

            return AutoRejectConfirmer("子 Agent 无可用的用户交互通道")
        return _InheritedConfirmer(self.confirmer, sub_id)

    # ------------------------------------------------------------------ 后台
    def drain_background(self) -> list[str]:
        """取回已完成的后台子 Agent 结论块（主循环每轮调用一次，§12.2）。

        Returns:
            形如 ``[后台子 Agent 完成] …`` 的文本块列表（已取走的不会重复返回）。
        """
        blocks: list[str] = []
        pending: list[str] = []
        for sub_id in self._completed:
            result = self._results.get(sub_id)
            if result is None:
                continue
            blocks.append(result.as_block(background=True))
            pending.append(sub_id)
        for sub_id in pending:
            self._completed.remove(sub_id)
        for sub_id, task in list(self._background.items()):
            if task.done():
                self._background.pop(sub_id, None)
        return blocks

    def background_ids(self) -> list[str]:
        """返回仍在运行的后台子 Agent id。

        Returns:
            id 列表。
        """
        return [sid for sid, task in self._background.items() if not task.done()]

    def active_count(self) -> int:
        """返回当前正在执行的子 Agent 数量。

        Returns:
            数量。
        """
        return self._active

    async def aclose(self) -> None:
        """取消全部后台任务（会话退出时调用）。"""
        for task in self._background.values():
            task.cancel()
        for task in self._background.values():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._background.clear()


def _mode_from(value: str, fallback: SafetyMode) -> SafetyMode:
    """把 ``Task`` 的 ``mode`` 参数解析为安全模式。

    Args:
        value: ``inherit`` / ``plan`` / ``default``。
        fallback: ``inherit`` 时的回退模式。

    Returns:
        安全模式。
    """
    if value == "inherit":
        return fallback
    if value in {"plan", "default"}:
        return value  # type: ignore[return-value]
    return fallback


class _TracingPipeline:
    """给权限流水线补写 ``subagent_id`` / ``parent_call_id`` 的装饰器（§12.1 审计）。"""

    def __init__(self, inner: Any, *, sub_id: str, parent_call_id: str) -> None:
        self._inner = inner
        self._sub_id = sub_id
        self._parent = parent_call_id

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    async def decide(self, inv: Any, mode: Any) -> Any:
        """调用真实流水线（审计归属由 ``parent_call_id``/``subagent_id`` 传给流水线上游记录）。

        Args:
            inv: 待裁决调用。
            mode: 安全模式。

        Returns:
            :class:`~my_agent.models.permission.PermissionDecision`。
        """
        return await self._inner.decide(inv, mode)


class _DenyAllPipeline:
    """父级未提供流水线时的占位：一律 deny（子 Agent 不得绕过权限，C1）。"""

    async def decide(self, inv: Any, mode: Any) -> Any:
        """返回 deny 裁决。

        Args:
            inv: 待裁决调用。
            mode: 安全模式。

        Returns:
            deny 的 :class:`~my_agent.models.permission.PermissionDecision`。
        """
        from my_agent.models.permission import PermissionDecision

        return PermissionDecision(
            call_id=inv.call_id,
            tool=inv.tool,
            arguments=dict(inv.arguments),
            final="deny",
            phase_reached=0,
            mode=mode,
            phase0={"hit": "no_pipeline", "detail": "子 Agent 无权限流水线，按最保守策略拒绝"},
            permanent_eligible=False,
        )


class _NoHooks:
    """``subagent_isolation=true`` 时的空 hook 运行器。"""

    async def run(self, event: Any, **kwargs: Any) -> Any:
        """返回空结果。

        Args:
            event: 事件（忽略）。
            **kwargs: 其余参数（忽略）。

        Returns:
            空的 :class:`~my_agent.hooks.events.HookOutcome` 等价物。
        """
        from my_agent.hooks.events import HookOutcome

        return HookOutcome(modified_payload=kwargs.get("payload"))


def build_subagent_result(
    *, description: str, text: str, tool_calls: int = 0, tokens: int = 0
) -> SubagentResult:
    """构造一个成功的子 Agent 结论（测试与工具回执复用）。

    Args:
        description: 短标题。
        text: 结论文本。
        tool_calls: 工具调用数。
        tokens: 消耗 token。

    Returns:
        :class:`SubagentResult`。
    """
    return SubagentResult(
        id=f"sub-{uuid.uuid4().hex[:8]}",
        description=description,
        ok=True,
        text=text,
        tool_calls=tool_calls,
        tokens=tokens,
    )


def idle_poll_interval() -> float:
    """后台结论轮询间隔（主循环使用）。

    Returns:
        秒数。
    """
    return SUBAGENT_POLL_INTERVAL_S


def concurrency_cap(config: Config | None = None) -> int:
    """返回并发上限（配置优先）。

    Args:
        config: 生效配置。

    Returns:
        并发上限。
    """
    if config is not None:
        return config.agents.max_concurrent_subagents
    return MAX_CONCURRENT_SUBAGENTS


def available_agent_types() -> tuple[str, ...]:
    """返回可用的 ``agent_type`` 取值。

    Returns:
        取值元组。
    """
    return AGENT_TYPES


def task_messages_note() -> str:
    """返回子 Agent 上下文注入说明（自检与文档用）。

    Returns:
        说明文本。
    """
    return (
        "子 Agent 上下文仅含：子 Agent 系统提示词 + 工作目录摘要 + 项目记忆 + "
        "工具 schema + 任务描述；不含父 history，且不写记忆（§12.1）。"
    )


def detail_block(result: SubagentResult) -> str:
    """返回结论的紧凑单行摘要。

    Args:
        result: 子 Agent 结论。

    Returns:
        单行摘要。
    """
    status = "ok" if result.ok else "fail"
    return (
        f"{result.id} [{status}] {result.description} — "
        f"{result.tool_calls} calls / {result.tokens} tokens / {result.duration_ms}ms"
    )


def fresh_message(role: str, content: str) -> Message:
    """构造一条子 Agent 上下文的初始消息（便于测试断言 H-1）。

    Args:
        role: 角色。
        content: 内容。

    Returns:
        :class:`~my_agent.models.messages.Message`。
    """
    from typing import cast

    return Message(role=cast(Any, role), content=content)
