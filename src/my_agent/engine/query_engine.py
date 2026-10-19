"""外层 QueryEngine（§5.2）。

与进程同寿（一个 CLI 会话一个实例）：负责上下文**初始化**、``/`` 命令分发、
压缩与归档决策、会话持久化、回合记账、记忆写入触发。

**不做**单步工具执行，**不做**单次模型调用的重试（那是内层 ``Query`` 与 ``LLMClient`` 的事）。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from my_agent.config.loader import load_config, myagent_home, project_root_for
from my_agent.errors import ConfigError, MyAgentError
from my_agent.models.permission import SafetyMode
from my_agent.models.protocols import AutoRejectConfirmer, ConsolePort, NullDisplay
from my_agent.models.session import SessionState

if TYPE_CHECKING:
    from my_agent.config.schema import Config
    from my_agent.context.archive import ArchiveStore
    from my_agent.context.assembler import ContextAssembler
    from my_agent.context.compactor import Compactor
    from my_agent.engine.session_store import SessionStore
    from my_agent.engine.slash import SlashRegistry
    from my_agent.hooks.runner import HookRunner
    from my_agent.llm.client import LLMClient, UsageTracker
    from my_agent.llm.tokenizer import TokenCounter
    from my_agent.memory.writer import MemoryWriter
    from my_agent.models.protocols import Confirmer
    from my_agent.security.audit import AuditWriter
    from my_agent.security.pipeline import PermissionPipeline
    from my_agent.skills.registry import SkillRegistry
    from my_agent.tools.registry import ToolRegistry

__all__ = ["EngineResult", "QueryEngine"]

_LOG = logging.getLogger(__name__)


class EngineResult(BaseModel):
    """一次 ``handle_input`` 的结果。"""

    kind: Literal["assistant_text", "slash_output", "aborted", "error"]
    text: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    compacted: bool = False
    turn_index: int = 0

    model_config = {"extra": "ignore"}


class QueryEngine:
    """会话级编排器。"""

    def __init__(
        self,
        config: Config,
        confirmer: Confirmer,
        console: ConsolePort | NullDisplay | None = None,
        *,
        cwd: Path | None = None,
        resume: str | None = None,
        non_interactive: bool = False,
    ) -> None:
        """初始化（不做 I/O；装配发生在 :meth:`start`）。

        Args:
            config: 生效配置。
            confirmer: 权限询问回调。
            console: 输出端口（缺省静默）。
            cwd: 工作目录（缺省进程当前目录）。
            resume: 要恢复的会话 id（``--resume`` 或 ``--continue`` 的结果）。
            non_interactive: 非交互模式（``--print``），遇 ask 一律拒绝。
        """
        self.config = config
        self.confirmer = confirmer if not non_interactive else AutoRejectConfirmer()
        self.console = console or NullDisplay()
        self.cwd = Path(cwd or Path.cwd()).resolve()
        self.resume_id = resume
        self.non_interactive = non_interactive

        self.project_root = project_root_for(self.cwd)
        self.home = myagent_home()

        self.session: SessionState | None = None
        self.started = False

        # 以下依赖在 start() 中装配
        self.store: SessionStore | None = None
        self.llm: LLMClient | None = None
        self.usage: UsageTracker | None = None
        self.counter: TokenCounter | None = None
        self.tools: ToolRegistry | None = None
        self.pipeline: PermissionPipeline | None = None
        self.hooks: HookRunner | None = None
        self.audit: AuditWriter | None = None
        self.context: ContextAssembler | None = None
        self.compactor: Compactor | None = None
        self.archive: ArchiveStore | None = None
        self.skills: SkillRegistry | None = None
        self.memory_writer: MemoryWriter | None = None
        self.slash: SlashRegistry | None = None
        self.mcp: Any = None

        self.system_prompt = ""
        self.repomap = ""
        self.memory_block = ""
        self.skills_block = ""
        self.tool_summaries: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ 启动
    async def start(self) -> None:
        """按 §5.2 的十步顺序初始化（顺序有依赖，不可乱序）。

        Raises:
            ConfigError: 必需配置缺失（如 API Key 未设置）。
        """
        from my_agent.context.archive import ArchiveStore
        from my_agent.context.assembler import ContextAssembler
        from my_agent.context.compactor import Compactor
        from my_agent.engine.session_store import SessionStore
        from my_agent.engine.slash import SlashRegistry
        from my_agent.hooks.loader import load_hooks
        from my_agent.hooks.runner import HookRunner
        from my_agent.llm.client import LLMClient, UsageTracker

        # 1. 校验必需配置（端点缺失/API Key 未设置 → ConfigError 并提示如何配置）
        from my_agent.llm.provider import OpenAISDKClient
        from my_agent.llm.tokenizer import get_counter
        from my_agent.security.audit import AuditWriter
        from my_agent.security.cache import ApprovalCache
        from my_agent.security.pipeline import PermissionPipeline

        endpoint = self.config.llm.resolved_endpoint()
        provider = OpenAISDKClient(endpoint)  # 缺密钥时在 provider 内抛 ConfigError

        # 2. 载入 Hooks（后续所有步骤都可被 hook 观测）
        specs = load_hooks(self.project_root, self.home, self.config)
        self.hooks = HookRunner(
            specs,
            cwd=self.cwd,
            config=self.config,
            display=self.console,
            project_root=self.project_root,
            home=self.home,
        )

        # 3. LLM 层
        self.usage = UsageTracker()
        self.counter = get_counter(endpoint.model)
        self.llm = LLMClient(provider, self.config, self.usage)

        # 4. 审计与审批缓存
        self.audit = AuditWriter(
            self.project_root / ".myagent" / "audit",
            enabled=self.config.safety.audit_enabled,
        )
        cache = ApprovalCache(
            project_root=self.project_root,
            persist=self.config.safety.persist_approvals,
        )

        # 5. 会话状态（恢复或新建）
        self.store = SessionStore(self.project_root)
        self.session = self._restore_or_create()
        self.archive = ArchiveStore(
            self.project_root / ".myagent" / "archive", counter=self.counter
        )

        # 6. 权限流水线（先于工具装配：TaskTool 需要它）
        self.pipeline = PermissionPipeline(
            config=self.config,
            llm=self.llm,
            cache=cache,
            audit=self.audit,
            confirmer=self.confirmer,
            hooks=self.hooks,
            cwd=str(self.cwd),
            session_id=self.session.session_id,
        )

        # 7. Skills 注册表 + 去重/建图 +（必要时）路由层生成
        self.skills = self._build_skills()

        # 8. 工具注册表 = 内置 + 记忆/skill/归档 + MCP + 子 Agent
        self.tools = await self._build_tools()

        # 9. 工作目录摘要（命中缓存则直接用）与记忆注入
        self.repomap = await self._load_repomap()
        self.memory_block = self._load_memory()

        # 10. 上下文组装器、系统提示词、slash 注册表
        self.context = ContextAssembler(config=self.config, counter=self.counter)
        self.compactor = Compactor(config=self.config, counter=self.counter)
        self.system_prompt = self._build_system_prompt()
        self.skills_block = self._build_skills_block()
        self.slash = SlashRegistry(self)

        self.started = True
        await self._emit_session_start()
        _LOG.info("myagent 已启动：session=%s mode=%s", self.session.session_id, self.session.mode)

    def _restore_or_create(self) -> SessionState:
        assert self.store is not None
        if self.resume_id:
            try:
                session = self.store.load(self.resume_id)
                session.cwd = str(self.cwd)
                self.console.info(
                    f"已恢复会话 {session.session_id}（{session.turn_index} 轮，mode={session.mode}）"
                )
                return session
            except ConfigError as exc:
                self.console.warn(f"恢复会话失败，已新建：{exc.user_message()}")
        return self.store.new_session(
            cwd=str(self.cwd),
            mode=self.config.safety.mode,
            project_root=str(self.project_root),
        )

    def _build_skills(self) -> SkillRegistry | None:
        from my_agent.skills.registry import SkillRegistry

        try:
            registry = SkillRegistry(
                project_root=self.project_root,
                home=self.home,
                config=self.config,
                llm=self.llm,
            )
            registry.load_all()
            return registry
        except MyAgentError as exc:
            self.console.warn(f"skills 加载失败（已跳过）：{exc.user_message()}")
            return None

    async def _build_tools(self) -> ToolRegistry:
        from my_agent.agents.task_tool import TaskTool
        from my_agent.context.archive_tool import ArchiveSearchTool
        from my_agent.skills.skill_tool import SkillListTool, SkillTool
        from my_agent.tools import build_default_registry
        from my_agent.tools.backup import BackupManager

        assert self.session is not None
        backups = BackupManager(self.project_root, self.session.session_id)
        registry: ToolRegistry = build_default_registry(
            config=self.config,
            backups=backups,
            todo_path=self.project_root / ".myagent" / "todos.json",
        )
        extra: list[Any] = []
        if self.skills is not None:
            extra.extend([SkillTool(self.skills), SkillListTool(self.skills)])
        if self.archive is not None:
            extra.append(ArchiveSearchTool(self.archive))
        extra.extend(self._memory_tools())
        if self.config.agents.enabled:
            extra.append(TaskTool(self._build_subagent_runner(registry)))
        for tool in extra:
            try:
                registry.register(tool)
            except MyAgentError as exc:
                self.console.warn(f"工具注册失败：{exc.user_message()}")
        await self._register_mcp_tools(registry)
        return registry

    def _memory_tools(self) -> list[Any]:
        from my_agent.config.loader import PROJECT_DIR_NAME
        from my_agent.memory.store import MemoryStore
        from my_agent.memory.tools import MemoryForgetTool, MemoryReadTool, MemoryWriteTool

        global_store = MemoryStore(self.home / "memory.md", scope="global")
        assert self.session is not None
        project_store = MemoryStore(
            Path(self.session.project_root or self.cwd) / PROJECT_DIR_NAME / "memory.md",
            scope="project",
        )
        assert self.llm is not None
        return [
            MemoryReadTool(global_store, project_store),
            MemoryWriteTool(self._build_memory_writer(global_store, project_store)),
            MemoryForgetTool(global_store, project_store),
        ]

    def _build_memory_writer(self, global_store: Any, project_store: Any) -> Any:
        from my_agent.memory.writer import MemoryWriter

        assert self.llm is not None
        assert self.session is not None
        writer = MemoryWriter(
            llm=self.llm,
            config=self.config,
            global_store=global_store,
            project_store=project_store,
            audit_dir=self.project_root / ".myagent" / "audit",
            session_id=self.session.session_id,
        )
        self.memory_writer = writer
        return writer

    def _build_subagent_runner(self, registry: Any) -> Any:
        from my_agent.agents.subagent import SubagentRunner

        assert self.llm is not None
        assert self.pipeline is not None, "权限流水线必须先于工具装配完成（§5.2 步骤 5 → 7）"
        return SubagentRunner(
            config=self.config,
            llm=self.llm,
            tools=registry,
            pipeline=self.pipeline,
            hooks=self.hooks,
            archive=self.archive,
            counter=self.counter,
            cwd=self.cwd,
            project_root=self.project_root,
            display=self.console,
        )

    async def _register_mcp_tools(self, registry: Any) -> None:
        """加载 MCP 服务器并注册其工具（§5.2 步骤 4；启动不做 deep 校验，F-2）。"""
        from my_agent.mcp.manager import MCPManager, McpToolAdapter

        try:
            manager = MCPManager(self.config, project_root=self.project_root, home=self.home)
            self.mcp = manager
            specs = await manager.load_all()
        except MyAgentError as exc:
            self.console.warn(f"MCP 加载失败（已跳过）：{exc.user_message()}")
            return
        for spec in specs:
            try:
                registry.register(McpToolAdapter(manager, spec))
            except MyAgentError as exc:
                self.console.warn(f"MCP 工具注册失败（已跳过）：{exc.user_message()}")

    async def _load_repomap(self) -> str:
        from my_agent.context.repomap import RepoMap

        try:
            repo = RepoMap(config=self.config, counter=self.counter)
            return await repo.load_or_generate(cwd=self.cwd, llm=self.llm)
        except MyAgentError as exc:
            self.console.warn(f"工作目录摘要生成失败（已跳过）：{exc.user_message()}")
            return ""

    def _load_memory(self) -> str:
        from my_agent.config.loader import PROJECT_DIR_NAME
        from my_agent.memory.injector import build_block
        from my_agent.memory.store import MemoryStore

        assert self.counter is not None
        assert self.session is not None
        try:
            global_store = MemoryStore(self.home / "memory.md", scope="global")
            project_store = MemoryStore(
                Path(self.session.project_root or self.cwd) / PROJECT_DIR_NAME / "memory.md",
                scope="project",
            )
            external = self._read_external_memory()
            return build_block(
                global_store.read(),
                project_store.read(),
                max_tokens=self.config.memory.inject_max_tokens,
                counter=self.counter,
                external=external,
                project_name=Path(self.session.project_root or self.cwd).name,
            )
        except MyAgentError as exc:
            self.console.warn(f"记忆注入失败（已跳过）：{exc.user_message()}")
            return ""

    def _read_external_memory(self) -> str:
        """只读兼容外部记忆文件（§8.8 / U16）——永不写入这些文件。"""
        if not self.config.memory.read_external:
            return ""
        candidates = [
            self.home / "CLAUDE.md",
            Path.home() / ".claude" / "CLAUDE.md",
            self.project_root / "MYAGENT.md",
            self.project_root / "CLAUDE.md",
        ]
        chunks: list[str] = []
        for path in candidates:
            try:
                if path.is_file():
                    text = path.read_text(encoding="utf-8", errors="replace").strip()
                    if text:
                        chunks.append(f"### {path.name}（{path.parent}）\n{text}")
            except OSError:
                continue
        return "\n\n".join(chunks)

    def _build_system_prompt(self) -> str:
        from my_agent.llm.prompts import MAIN_SYSTEM_PROMPT

        return MAIN_SYSTEM_PROMPT

    def _build_skills_block(self) -> str:
        if self.skills is None:
            return ""
        try:
            return self.skills.block_for_context()
        except MyAgentError:
            return ""

    async def _emit_session_start(self) -> None:
        if self.hooks is None or self.session is None:
            return
        from my_agent.hooks.events import HookEvent, HookPayload

        await self.hooks.run(
            HookEvent.SESSION_START,
            session_id=self.session.session_id,
            turn_index=self.session.turn_index,
            payload=HookPayload(),
        )

    # ------------------------------------------------------------------ 输入
    async def handle_input(self, text: str) -> EngineResult:
        """单次用户输入入口。

        以 ``/`` 开头 → 外层 slash 分发（**绝不**进入内层循环）；否则创建内层 ``Query``。

        Args:
            text: 用户原始输入。

        Returns:
            :class:`EngineResult`。
        """
        if not self.started:
            raise ConfigError("引擎尚未 start()", detail="请先 await engine.start()")

        await self._maybe_compact()

        stripped = text.strip()
        if stripped.startswith("/"):
            assert self.slash is not None
            return await self.slash.dispatch(stripped)

        rewritten = await self._hook_user_input(stripped)
        if rewritten is None:
            return EngineResult(kind="aborted", text="输入被 hook 拒绝", turn_index=self._turn())

        assert self.session is not None
        if not self.session.title:
            self.session.title = stripped[:40]

        query = self._new_query()
        result = await query.run(rewritten)
        await self._after_turn(query, result)
        return result

    def _new_query(self) -> Any:
        from my_agent.engine.query import Query

        assert self.session is not None
        assert self.config is not None
        assert self.llm is not None
        assert self.tools is not None
        assert self.context is not None
        assert self.pipeline is not None
        assert self.hooks is not None
        return Query(
            session=self.session,
            config=self.config,
            llm=self.llm,
            tools=self.tools,
            context=self.context,
            pipeline=self.pipeline,
            hooks=self.hooks,
            confirmer=self.confirmer,
            display=self.console,
            memory_writer=self.memory_writer,
            system_prompt=self.system_prompt,
            repomap=self.repomap,
            memory_block=self.memory_block,
            skills_block=self.skills_block,
        )

    async def _hook_user_input(self, text: str) -> str | None:
        if self.hooks is None or self.session is None:
            return text
        from my_agent.hooks.events import HookEvent, HookPayload

        outcome = await self.hooks.run(
            HookEvent.USER_INPUT_RECEIVED,
            session_id=self.session.session_id,
            turn_index=self.session.turn_index + 1,
            payload=HookPayload(text=text),
        )
        if outcome.decision == "deny":
            self.console.warn(f"hook 拒绝了本次输入：{outcome.message}")
            return None
        if outcome.modified_payload and outcome.modified_payload.text is not None:
            return outcome.modified_payload.text
        return text

    async def _after_turn(self, query: Any, result: EngineResult) -> None:
        """回合收尾：记忆写入触发 + 会话持久化。"""
        assert self.session is not None
        if (
            self.memory_writer is not None
            and self.config.memory.write_per_turn
            and result.kind == "assistant_text"
        ):
            turn = self.session.turns[-1] if self.session.turns else None
            try:
                await self.memory_writer.after_turn(
                    turn=turn, result=result, tool_summary=self._tool_summary(query)
                )
            except MyAgentError as exc:
                self.console.warn(f"记忆写入失败（已跳过）：{exc.user_message()}")
        self._persist()

    def _tool_summary(self, query: Any) -> str:
        lines: list[str] = []
        for result in getattr(query, "_turn_results", []) or []:
            status = "成功" if result.ok else "失败"
            lines.append(f"- {result.tool}（{status}）")
        return "\n".join(lines) or "（本轮无工具调用）"

    def _persist(self) -> None:
        if self.store is None or self.session is None:
            return
        try:
            self.store.persist(self.session)
        except ConfigError as exc:
            _LOG.warning("会话持久化失败：%s", exc.user_message())

    # ------------------------------------------------------------------ 压缩
    async def _maybe_compact(self) -> bool:
        """水位 ≥ 触发比例时执行全文压缩（§7.5/§7.7）。

        Returns:
            本次是否真的发生了压缩。
        """
        assert self.session is not None
        if self.compactor is None:
            return False
        ratio = self.config.context.compress_trigger_ratio
        if self.session.context_watermark < ratio:
            return False
        return await self.compact()

    async def compact(self, focus: str | None = None) -> bool:
        """执行一次全文压缩（``/compact`` 与自动触发共用）。

        Args:
            focus: 可选的聚焦主题。

        Returns:
            是否压缩成功。
        """
        assert self.compactor is not None
        assert self.session is not None
        before = self.session.context_watermark
        try:
            event = await self.compactor.compact(session=self.session, focus=focus)
        except MyAgentError as exc:
            self.console.warn(f"压缩失败：{exc.user_message()}")
            return False
        self.session.compact_count += 1
        self.session.last_compact_turn = self.session.turn_index
        self.session.context_watermark = event.watermark_after
        if event.mode == "rule_based":
            self.console.warn(
                "⚠ 本次压缩未使用模型，采用规则化摘要；可用 ArchiveSearch 回读原文"
            )
        _LOG.info(
            "压缩完成：轮 %d-%d，归档 %s，水位 %.3f → %.3f",
            event.turn_start,
            event.turn_end,
            event.archive_path,
            before,
            event.watermark_after,
        )
        self._persist()
        return True

    def _turn(self) -> int:
        return self.session.turn_index if self.session else 0

    # ------------------------------------------------------------------ 模式
    def set_mode(self, mode: SafetyMode) -> None:
        """切换安全模式并清空会话级审批缓存（B-18）。

        Args:
            mode: 新的安全模式。
        """
        assert self.session is not None
        old = self.session.mode
        self.session.mode = mode
        self.session.mode_switches += 1
        if self.pipeline is not None:
            self.pipeline.clear_cache()
        _LOG.info("安全模式切换：%s → %s", old, mode)
        self._persist()

    # ------------------------------------------------------------------ 关闭
    async def shutdown(self, reason: str = "user_exit") -> None:
        """持久化会话、写记忆、关闭 MCP 子进程、flush 审计日志。

        Args:
            reason: 退出原因（写入审计）。
        """
        if self.hooks is not None and self.session is not None:
            from my_agent.hooks.events import HookEvent, HookPayload

            try:
                await self.hooks.run(
                    HookEvent.SESSION_END,
                    session_id=self.session.session_id,
                    turn_index=self.session.turn_index,
                    payload=HookPayload(text=reason),
                )
            except MyAgentError as exc:  # pragma: no cover - 退出路径尽力而为
                _LOG.warning("session_end hook 失败：%s", exc.user_message())

        self._persist()
        if self.mcp is not None:
            try:
                await self.mcp.aclose()
            except MyAgentError as exc:  # pragma: no cover
                _LOG.warning("MCP 关闭失败：%s", exc.user_message())
        if self.audit is not None:
            await self.audit.aclose()
        if self.llm is not None:
            await self.llm.aclose()
        self.started = False

    # ------------------------------------------------------------------ 报告
    def context_report(self) -> str:
        """生成 ``/context`` 报告文本（§7.7.5）。

        Returns:
            多行报告。
        """
        from my_agent.context.assembler import format_context_report

        assert self.session is not None
        assert self.context is not None
        build = self.context.build(
            session=self.session,
            tools_schema=self.tools.schema() if self.tools else [],
            memory_block=self.memory_block,
            skills_block=self.skills_block,
            repomap=self.repomap,
        )
        return format_context_report(build, self.session, self.config)

    def cost_report(self) -> str:
        """生成 ``/cost`` 报告文本（§13.3）。

        Returns:
            多行报告。
        """
        assert self.session is not None
        totals = self.session.usage_totals
        usage = self.usage.snapshot() if self.usage else {}
        lines = [
            f"会话 {self.session.session_id} · {self.session.turn_index} 轮",
            f"  累计 prompt      {totals.get('prompt_tokens', 0):>10,}",
            f"  累计 completion  {totals.get('completion_tokens', 0):>10,}",
            f"  累计 total       {totals.get('total', 0):>10,}",
        ]
        if usage and usage != totals:
            lines.append(f"  本次进程 total   {usage.get('total', 0):>10,}")
        lines.append(
            f"  压缩次数 {self.session.compact_count} · 水位 {self.session.context_watermark:.1%}"
        )
        return "\n".join(lines)

    def status_line(self) -> str:
        """生成顶部状态行（§16.2）。

        Returns:
            形如 ``myagent v0.1 · proj · mode=default · ctx=42%`` 的单行文本。
        """
        from my_agent import __version__

        assert self.session is not None
        ctx = f"{self.session.context_watermark:.0%}"
        tokens = self.session.usage_totals.get("total", 0)
        return (
            f"myagent v{__version__} · {self.project_root.name} · "
            f"mode={self.session.mode} · ctx={ctx} · tokens={tokens / 1000:.1f}k"
        )


def build_engine(
    *,
    cwd: Path | None = None,
    mode: SafetyMode | None = None,
    resume: str | None = None,
    config_path: Path | None = None,
    non_interactive: bool = False,
    confirmer: Confirmer | None = None,
    console: ConsolePort | NullDisplay | None = None,
) -> QueryEngine:
    """按 CLI 参数装配一个 :class:`QueryEngine`（不启动）。

    Args:
        cwd: 工作目录。
        mode: ``--mode`` 覆盖。
        resume: ``--resume`` / ``--continue`` 结果。
        config_path: ``--config``。
        non_interactive: 是否为 ``--print`` 模式。
        confirmer: 权限询问回调（缺省自动拒绝，由 CLI 层注入真实实现）。
        console: 输出端口。

    Returns:
        装配好的引擎（需自行 ``await engine.start()``）。
    """
    workdir = Path(cwd or Path.cwd())
    config = load_config(
        cwd=workdir,
        cli_overrides={"mode": mode} if mode else None,
        config_path=config_path,
    )
    if mode:
        config.safety.mode = mode
    return QueryEngine(
        config,
        confirmer or AutoRejectConfirmer(),
        console,
        cwd=workdir,
        resume=resume,
        non_interactive=non_interactive,
    )


def uptime_line(started_at: float) -> str:
    """格式化会话已运行时长（状态行辅助）。

    Args:
        started_at: 起始 epoch 秒。

    Returns:
        形如 ``00:03:12`` 的时长。
    """
    elapsed = max(0, int(time.time() - started_at))
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


async def run_once(engine: QueryEngine, prompt: str) -> EngineResult:
    """启动引擎、跑一次输入、优雅关闭（``--print`` 一条龙）。

    Args:
        engine: 未启动的引擎。
        prompt: 用户输入。

    Returns:
        :class:`EngineResult`。
    """
    await engine.start()
    try:
        return await engine.handle_input(prompt)
    finally:
        await engine.shutdown("print_mode")


def collect_messages(session: SessionState) -> list[Any]:
    """返回会话历史的浅拷贝（供导出/调试，不修改真相源）。

    Args:
        session: 会话状态。

    Returns:
        :class:`~my_agent.models.messages.Message` 列表的浅拷贝。
    """
    return list(session.history)
