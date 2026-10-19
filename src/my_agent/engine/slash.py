"""``/`` 命令注册与分发（§5.4）。

**全部在 QueryEngine 处理**，不消耗模型 token（``/compact``、``/init`` 等本身需要 LLM
的命令除外）。未知命令 → 提示「未知命令」且**不**把它当普通输入发给模型。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from my_agent.config.defaults import assumption_report
from my_agent.errors import MyAgentError
from my_agent.models.permission import ALL_MODES, SafetyMode

if TYPE_CHECKING:
    from my_agent.engine.query_engine import EngineResult, QueryEngine

__all__ = ["SlashCommand", "SlashRegistry"]

_LOG = logging.getLogger(__name__)

Handler = Callable[[list[str]], Awaitable["EngineResult"]]
"""/ 命令处理函数：接收参数列表，返回 EngineResult。"""

UNKNOWN_COMMAND = "未知命令：{name}。输入 /help 查看全部命令。"


class SlashCommand(BaseModel):
    """一条 ``/`` 命令的声明。"""

    name: str = Field(description="不含前导斜杠的命令名")
    usage: str = Field(description="用法字符串，如 ``/mode [name]``")
    help: str = Field(description="一行说明")
    needs_llm: bool = False

    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}


class SlashRegistry:
    """``/`` 命令注册表与分发器。"""

    def __init__(self, engine: QueryEngine) -> None:
        """初始化并注册 §5.4 的全部命令。

        Args:
            engine: 所属的外层引擎。
        """
        self.engine = engine
        self._commands: dict[str, SlashCommand] = {}
        self._handlers: dict[str, Handler] = {}
        self._register_builtins()

    # ------------------------------------------------------------------ 注册
    def register(self, cmd: SlashCommand, handler: Handler) -> None:
        """注册一条命令（同名后注册者覆盖）。

        Args:
            cmd: 命令声明。
            handler: 异步处理函数。
        """
        self._commands[cmd.name] = cmd
        self._handlers[cmd.name] = handler

    def get(self, name: str) -> SlashCommand | None:
        """按名取命令声明。

        Args:
            name: 不含斜杠的命令名。

        Returns:
            :class:`SlashCommand` 或 ``None``。
        """
        return self._commands.get(name.lstrip("/"))

    def names(self) -> list[str]:
        """列出全部已注册命令名。

        Returns:
            升序命令名列表。
        """
        return sorted(self._commands)

    # ------------------------------------------------------------------ 分发
    async def dispatch(self, text: str) -> EngineResult:
        """分发一条 ``/`` 输入。

        Args:
            text: 以 ``/`` 开头的原始输入。

        Returns:
            :class:`~my_agent.engine.query_engine.EngineResult`（``kind="slash_output"``）。
        """
        from my_agent.engine.query_engine import EngineResult

        body = text[1:].strip()
        if not body:
            return self._out(self.help_text())
        parts = body.split()
        name, args = parts[0], parts[1:]
        handler = self._handlers.get(name)
        if handler is None:
            return self._out(UNKNOWN_COMMAND.format(name=f"/{name}"))
        try:
            return await handler(args)
        except MyAgentError as exc:
            return EngineResult(
                kind="error",
                text=exc.user_message(),
                turn_index=self.engine.session.turn_index if self.engine.session else 0,
            )

    # ------------------------------------------------------------------ 帮助
    def help_text(self) -> str:
        """生成 ``/help`` 文本。

        Returns:
            多行命令清单 + 当前模式 + 待确认假设提示。
        """
        lines = ["可用命令：", ""]
        for name in self.names():
            cmd = self._commands[name]
            suffix = "  (需要模型)" if cmd.needs_llm else ""
            lines.append(f"  {cmd.usage:<34} {cmd.help}{suffix}")
        mode = self.engine.session.mode if self.engine.session else self.engine.config.safety.mode
        lines.extend(
            [
                "",
                f"当前安全模式：{mode}",
                f"可用模式：{' / '.join(ALL_MODES)}",
                "运行 `myagent --assumptions` 查看全部【待确认假设】的当前实现口径。",
            ]
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ 内置
    def _out(self, text: str) -> EngineResult:
        from my_agent.engine.query_engine import EngineResult

        return EngineResult(
            kind="slash_output",
            text=text,
            turn_index=self.engine.session.turn_index if self.engine.session else 0,
        )

    def _register_builtins(self) -> None:
        """注册 §5.4 表中的全部命令。"""
        self.register(
            SlashCommand(name="help", usage="/help", help="列出全部命令与用法"),
            self._cmd_help,
        )
        self.register(
            SlashCommand(name="mode", usage="/mode [name]", help="显示或切换安全模式"),
            self._cmd_mode,
        )
        self.register(
            SlashCommand(name="clear", usage="/clear", help="清空当前会话上下文（归档后新建）"),
            self._cmd_clear,
        )
        self.register(
            SlashCommand(
                name="compact",
                usage="/compact [focus]",
                help="手动触发全文压缩",
                needs_llm=True,
            ),
            self._cmd_compact,
        )
        self.register(
            SlashCommand(name="context", usage="/context", help="显示上下文水位与各块占比"),
            self._cmd_context,
        )
        self.register(
            SlashCommand(name="cost", usage="/cost", help="本会话 token/费用统计"),
            self._cmd_cost,
        )
        self.register(
            SlashCommand(
                name="memory",
                usage="/memory show|edit|add <text>|forget <id>",
                help="记忆查看/编辑/新增/删除",
                needs_llm=True,
            ),
            self._cmd_memory,
        )
        self.register(
            SlashCommand(
                name="skills",
                usage="/skills list|add <path>|dedupe|route",
                help="skills 管理与相似度维护",
                needs_llm=True,
            ),
            self._cmd_skills,
        )
        self.register(
            SlashCommand(
                name="mcp",
                usage="/mcp list|add <path>|remove <name>|reload",
                help="MCP 管理（add 触发完整校验）",
            ),
            self._cmd_mcp,
        )
        self.register(
            SlashCommand(name="hook", usage="/hook list|reload", help="hook 状态"),
            self._cmd_hook,
        )
        self.register(
            SlashCommand(
                name="session",
                usage="/session list|resume <id>|export",
                help="会话管理",
            ),
            self._cmd_session,
        )
        self.register(
            SlashCommand(
                name="run",
                usage="/run <command>",
                help="代替用户执行 shell 命令（走权限流水线）",
            ),
            self._cmd_run,
        )
        self.register(
            SlashCommand(name="init", usage="/init", help="刷新项目记忆与工作目录摘要", needs_llm=True),
            self._cmd_init,
        )
        self.register(
            SlashCommand(name="undo", usage="/undo", help="撤销最近一次写类工具的文件修改"),
            self._cmd_undo,
        )
        self.register(
            SlashCommand(name="exit", usage="/exit", help="优雅退出（等价 Ctrl-D）"),
            self._cmd_exit,
        )
        self.register(
            SlashCommand(
                name="assumptions",
                usage="/assumptions",
                help="打印全部【待确认假设】的当前实现口径",
            ),
            self._cmd_assumptions,
        )

    # ------------------------------------------------------------------ 实现
    async def _cmd_help(self, args: list[str]) -> EngineResult:
        return self._out(self.help_text())

    async def _cmd_mode(self, args: list[str]) -> EngineResult:
        engine = self.engine
        if not args:
            mode = engine.session.mode if engine.session else engine.config.safety.mode
            return self._out(f"当前安全模式：{mode}\n可用：{' / '.join(ALL_MODES)}")
        name = args[0]
        if name not in ALL_MODES:
            return self._out(f"未知模式：{name}；可用：{' / '.join(ALL_MODES)}")
        engine.set_mode(name)  # type: ignore[arg-type]
        return self._out(f"已切换到 {name} 模式（会话审批缓存已清空）")

    async def _cmd_clear(self, args: list[str]) -> EngineResult:
        engine = self.engine
        if engine.store is None:
            return self._out("会话存储不可用")
        old = engine.session
        archived = ""
        if engine.archive is not None and old is not None and old.history:
            try:
                path = engine.archive.archive_messages(
                    old.session_id, list(old.history), turn_start=1, turn_end=old.turn_index
                )
                archived = f"\n旧历史已归档：{path}"
            except MyAgentError as exc:
                archived = f"\n（归档失败：{exc.user_message()}）"
        engine.session = engine.store.new_session(
            cwd=str(engine.cwd),
            mode=old.mode if old else engine.config.safety.mode,
            project_root=str(engine.project_root),
        )
        engine._persist()
        return self._out(f"上下文已清空（新会话 {engine.session.session_id}）{archived}")

    async def _cmd_compact(self, args: list[str]) -> EngineResult:
        focus = " ".join(args) if args else None
        ok = await self.engine.compact(focus)
        if not ok:
            return self._out("压缩未执行（模型不可用或无需压缩）")
        session = self.engine.session
        watermark = session.context_watermark if session else 0.0
        return self._out(f"压缩完成，当前水位 {watermark:.1%}")

    async def _cmd_context(self, args: list[str]) -> EngineResult:
        return self._out(self.engine.context_report())

    async def _cmd_cost(self, args: list[str]) -> EngineResult:
        return self._out(self.engine.cost_report())

    async def _cmd_assumptions(self, args: list[str]) -> EngineResult:
        return self._out(assumption_report())

    async def _cmd_memory(self, args: list[str]) -> EngineResult:
        from pathlib import Path

        from my_agent.config.loader import PROJECT_DIR_NAME
        from my_agent.memory.injector import build_block
        from my_agent.memory.store import MemoryStore

        engine = self.engine
        sub = args[0] if args else "show"
        global_store = MemoryStore(engine.home / "memory.md", scope="global")
        project_store = MemoryStore(
            Path(engine.project_root) / PROJECT_DIR_NAME / "memory.md", scope="project"
        )

        if sub == "show":
            strip = self._out("")
            block = build_block(
                global_store.read(),
                project_store.read(),
                max_tokens=engine.config.memory.inject_max_tokens,
                counter=engine.counter,
            )
            lines = [
                f"全局：{global_store.path}（{len(global_store.read())} 条）",
                f"项目：{project_store.path}（{len(project_store.read())} 条）",
                "",
                block or "（暂无记忆）",
            ]
            strip.text = "\n".join(lines)
            return strip

        if sub == "edit":
            paths = [str(global_store.path), str(project_store.path)]
            return self._out("请直接编辑以下文件（人类可读 Markdown）：\n  " + "\n  ".join(paths))

        if sub == "add":
            text = " ".join(args[1:]).strip()
            if not text:
                return self._out("用法：/memory add <text>")
            writer = self._memory_writer(global_store, project_store)
            if writer is None:
                return self._out("记忆写入不可用（模型未就绪）")
            records = await writer.write_candidates([(text, "high")], turn=0)
            if not records:
                return self._out("未写入（可能重复或无可写内容）")
            rec = records[0]
            return self._out(
                f"{rec.action} → {rec.scope}/{rec.section}（id={rec.target_id or '新条目'}）\n理由：{rec.why}"
            )

        if sub == "forget":
            entry_id = args[1] if len(args) > 1 else ""
            if not entry_id:
                return self._out("用法：/memory forget <id>")
            removed = global_store.delete(entry_id) or project_store.delete(entry_id)
            if removed is None:
                return self._out(f"未找到条目 {entry_id}")
            return self._out(f"已删除 {entry_id}：{removed.text}")

        return self._out("用法：/memory show|edit|add <text>|forget <id>")

    def _memory_writer(self, global_store: object, project_store: object) -> object | None:
        engine = self.engine
        if engine.llm is None or engine.session is None:
            return None
        if engine.memory_writer is not None:
            return engine.memory_writer
        from my_agent.memory.writer import MemoryWriter

        return MemoryWriter(
            llm=engine.llm,
            config=engine.config,
            global_store=global_store,
            project_store=project_store,
            audit_dir=engine.project_root / ".myagent" / "audit",
            session_id=engine.session.session_id,
        )

    async def _cmd_skills(self, args: list[str]) -> EngineResult:
        engine = self.engine
        if engine.skills is None:
            return self._out("skills 注册表不可用")
        sub = args[0] if args else "list"

        if sub == "list":
            skills = engine.skills.load_all()
            if not skills:
                return self._out(f"暂无 skills（目录：{engine.skills.skills_root()}）")
            lines = [f"共 {len(skills)} 个 skills：", ""]
            for meta in skills:
                flag = " [router]" if meta.is_router else ""
                related = f" · 强相关 {len(meta.related)}" if meta.related else ""
                lines.append(f"  {meta.name}{flag} — {meta.description[:60]}{related}")
            return self._out("\n".join(lines))

        if sub == "add":
            if len(args) < 2:
                return self._out("用法：/skills add <path>")
            from pathlib import Path

            result = engine.skills.add(Path(args[1]))
            return self._out(
                f"added={result.added} · {result.reason}"
                + (f" · 路由重建={result.router_rebuilt}" if result.router_rebuilt else "")
            )

        if sub in {"dedupe", "route"}:
            from my_agent.skills.router import rebuild

            built = await rebuild(
                engine.skills, engine.skills.load_all(), llm=engine.llm, force=True
            )
            names = ", ".join(meta.name for meta in built) or "（无）"
            return self._out(f"已重建路由层：{names}")

        return self._out("用法：/skills list|add <path>|dedupe|route")

    async def _cmd_mcp(self, args: list[str]) -> EngineResult:
        engine = self.engine
        sub = args[0] if args else "list"
        if engine.mcp is None:
            return self._out("MCP 未初始化（检查 .myagent/mcp.yaml）")
        if sub == "list":
            return self._out(engine.mcp.describe())
        if sub == "add":
            if len(args) < 2:
                return self._out("用法：/mcp add <path>")
            report = await engine.mcp.add(args[1])
            if report.ok:
                return self._out(f"MCP 校验通过，已写入配置。\n{engine.mcp.describe()}")
            issues = "\n".join(f"  ✗ {item.message}" for item in report.errors)
            return self._out(f"MCP 校验失败（配置未修改）：\n{issues}")
        if sub == "reload":
            report = await engine.mcp.reload()
            return self._out(f"已重载：{report}")
        if sub == "remove":
            if len(args) < 2:
                return self._out("用法：/mcp remove <name>")
            engine.mcp.remove(args[1])
            return self._out(f"已移除 {args[1]}")
        return self._out("用法：/mcp list|add <path>|remove <name>|reload")

    async def _cmd_hook(self, args: list[str]) -> EngineResult:
        engine = self.engine
        if engine.hooks is None:
            return self._out("hook 未加载")
        sub = args[0] if args else "list"
        if sub == "reload":
            engine.hooks.reload()
        return self._out(engine.hooks.describe())

    async def _cmd_session(self, args: list[str]) -> EngineResult:
        engine = self.engine
        if engine.store is None:
            return self._out("会话存储不可用")
        sub = args[0] if args else "list"

        if sub == "list":
            metas = engine.store.list_sessions()
            if not metas:
                return self._out("暂无历史会话")
            lines = ["历史会话（按更新时间倒序）：", ""]
            for meta in metas[:20]:
                stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(meta.updated_at))
                marker = " ←当前" if engine.session and meta.session_id == engine.session.session_id else ""
                lines.append(
                    f"  {meta.session_id}  {stamp}  {meta.turn_index:>3} 轮  "
                    f"mode={meta.mode}{marker}"
                )
            return self._out("\n".join(lines))

        if sub == "resume":
            if len(args) < 2:
                return self._out("用法：/session resume <id>")
            session = engine.store.load(args[1])
            session.cwd = str(engine.cwd)
            engine.session = session
            return self._out(
                f"已恢复会话 {session.session_id}（{session.turn_index} 轮，mode={session.mode}）"
            )

        if sub == "export":
            if engine.session is None:
                return self._out("无活动会话")
            path = engine.store.export(engine.session.session_id)
            return self._out(f"已导出：{path}")

        return self._out("用法：/session list|resume <id>|export")

    async def _cmd_run(self, args: list[str]) -> EngineResult:
        """``/run <cmd>``：代替用户执行命令，**必须**走权限流水线（§5.3.5）。"""
        engine = self.engine
        command = " ".join(args).strip()
        if not command:
            return self._out("用法：/run <command>")
        if engine.tools is None or engine.pipeline is None or engine.session is None:
            return self._out("引擎未就绪")
        import asyncio

        from my_agent.models.tools import ToolContext, ToolInvocation

        spec = engine.tools.get_spec("Bash")
        if spec is None:
            return self._out("Bash 工具不可用")
        inv = ToolInvocation(
            call_id=f"slash-{int(time.time() * 1000)}",
            tool="Bash",
            arguments={"command": command, "description": "用户通过 /run 显式请求"},
            reason="用户通过 /run 显式请求执行",
            spec=spec,
        )
        decision = await engine.pipeline.decide(inv, engine.session.mode)
        if decision.final != "allow":
            return self._out(f"命令未执行（{decision.final}）：{decision.explain()}")
        ctx = ToolContext(
            session_id=engine.session.session_id,
            turn_index=engine.session.turn_index,
            cwd=engine.session.cwd,
            project_root=engine.session.project_root or engine.session.cwd,
            config=engine.config,
            artifact_dir=str(engine.project_root / ".myagent" / "artifacts"),
            abort_event=asyncio.Event(),
            permission_checked=True,
            call_id=inv.call_id,
        )
        result = await engine.tools.execute(inv, ctx)
        body = result.content if result.ok else f"失败：{result.error}"
        return self._out(f"$ {command}\n{body}")

    async def _cmd_init(self, args: list[str]) -> EngineResult:
        engine = self.engine
        from my_agent.context.repomap import RepoMap

        repo = RepoMap(config=engine.config, counter=engine.counter)
        engine.repomap = await repo.load_or_generate(cwd=engine.cwd, llm=engine.llm, force=True)
        return self._out(
            f"工作目录摘要已刷新（缓存：{engine.project_root / '.myagent' / 'repomap.md'}）"
        )

    async def _cmd_undo(self, args: list[str]) -> EngineResult:
        engine = self.engine
        if engine.session is None:
            return self._out("无活动会话")
        from my_agent.tools.backup import BackupManager

        manager = BackupManager(engine.project_root, engine.session.session_id)
        outcome = manager.undo_last()
        return self._out(outcome.message)

    async def _cmd_exit(self, args: list[str]) -> EngineResult:
        from my_agent.engine.query_engine import EngineResult

        return EngineResult(
            kind="aborted",
            text="__exit__",
            turn_index=self.engine.session.turn_index if self.engine.session else 0,
        )


def parse_mode(value: str) -> SafetyMode:
    """把字符串解析为安全模式（CLI 与 ``/mode`` 共用）。

    Args:
        value: 模式名。

    Returns:
        对应的 :data:`~my_agent.models.permission.SafetyMode`。

    Raises:
        ValueError: 非法模式名。
    """
    if value not in ALL_MODES:
        raise ValueError(f"未知模式：{value}；可用：{' / '.join(ALL_MODES)}")
    return value  # type: ignore[return-value]
