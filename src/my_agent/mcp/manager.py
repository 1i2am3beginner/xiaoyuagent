"""MCP 管理器：服务器生命周期、工具映射与调用转发（§10.3、§10.4、§10.5）。

本文件同时满足两份冻结：

* 契约 §11（``mcp/manager.py``【补齐】）：``load_all`` / ``add`` / ``remove`` /
  ``call_tool`` / ``tools`` / ``aclose``；
* 契约 §17.5（已落地调用点，``engine/query_engine.py`` 与 ``engine/slash.py`` 写死）：
  ``MCPManager(config, *, project_root, home)``、``load_all() -> list[ToolSpec]``、
  ``add(path) -> ValidationReport``、``reload() -> str``、``remove(name) -> None``、
  ``describe() -> str``、``McpToolAdapter(manager, spec)``。

行为要点：

* **启动路径不 deep 校验**（F-2）：只做静态最小检查 + 首次连接；连接失败只告警并跳过，
  绝不阻塞启动。
* **``/mcp add`` 失败绝不写配置**（F-3）：deep 校验（V10 握手）全部通过才落地，
  且写入前先把目标文件既有条目读全，避免丢条目。
* **工具映射**（§10.4）：前缀、描述兜底、类别推断、只读/破坏性、``source="mcp"``；
  与内置工具/其它 MCP 工具重名 → V11 报错并提示 ``tool_prefix``（F-4）。
* **审计字段**（F-8）：:class:`McpToolBinding` 与 :class:`McpToolAdapter` 均公开
  ``mcp_server``，供 ``security`` 层填 ``AuditRecord.mcp_server``。
* **不自动重启**：进程意外退出/调用超时 → 标记 ``unhealthy`` 并清理进程，
  由用户 ``/mcp`` 处理，绝不进入崩溃循环。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from my_agent.config.schema import Config, McpServerConfig
from my_agent.errors import McpTimeoutError, McpValidationError
from my_agent.mcp.config_store import McpConfigStore
from my_agent.mcp.schema import McpClient, McpTool
from my_agent.mcp.validator import McpValidator, ValidationReport, default_client_factory
from my_agent.models.tools import ToolContext, ToolInvocation, ToolResult, ToolSpec

__all__ = ["BUILTIN_TOOL_NAMES", "MCPManager", "McpToolAdapter", "McpToolBinding"]

_LOG = logging.getLogger(__name__)

#: §14.2 内置工具名（**冻结**，契约 §5）。
#: 与 ``tools/`` 注册表一致；engine 若已装配注册表，可用 ``existing_names=`` 覆盖，
#: 以保证 V11 冲突检测用的是运行时的真实工具集。
BUILTIN_TOOL_NAMES: tuple[str, ...] = (
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Glob",
    "Grep",
    "Bash",
    "TodoWrite",
    "WebSearch",
    "WebFetch",
    "Task",
    "Skill",
    "SkillList",
    "MemoryRead",
    "MemoryWrite",
    "MemoryForget",
    "ArchiveSearch",
)


class McpToolBinding(BaseModel):
    """一个 MCP 工具的运行期绑定：模型可见声明 + 来源服务器 + 所属客户端。

    **审计消费点（F-8）**：``security`` 层用 :meth:`audit_fields` 填充
    ``AuditRecord.mcp_server``，追溯「哪个外部服务的调用被放行」。
    ``ToolResult`` 的 ``model_config`` 为 ``extra="ignore"``，无法承载该字段，
    故来源信息由本对象显式携带。

    Attributes:
        spec: 映射后的内置工具声明（进 tools schema）。
        mcp_server: 来源 MCP 服务器名（= 审计字段 ``mcp_server``）。
        tool: 完整映射记录（含原始工具名与原始 ``inputSchema``）。
        client: 承载该工具的客户端（不参与序列化）。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: ToolSpec = Field(description="映射后的内置工具声明（§10.4）")
    mcp_server: str = Field(description="来源 MCP 服务器名（审计字段 mcp_server）")
    tool: McpTool = Field(description="完整映射记录（原工具名 + 原 inputSchema）")
    client: McpClient = Field(exclude=True, description="承载该工具的客户端")

    @property
    def name(self) -> str:
        """映射后的工具名。"""
        return self.spec.name

    @property
    def raw_name(self) -> str:
        """MCP 侧原始工具名（调用时使用）。"""
        return self.tool.raw_name

    def audit_fields(self) -> dict[str, str]:
        """返回审计层要补写的字段（§10.4，F-8）。

        Returns:
            形如 ``{"mcp_server": "filesystem", "tool": "fs__read_file"}``。
        """
        return {"mcp_server": self.mcp_server, "tool": self.spec.name}


@dataclass
class _ServerState:
    """一个服务器的运行期状态（内部结构）。"""

    server: McpServerConfig
    client: McpClient | None = None
    tools: list[McpTool] = field(default_factory=list)
    healthy: bool = True
    error: str = ""
    connected: bool = False


class MCPManager:
    """MCP 服务器的装载、管理与调用转发（契约 §11 / §17.5）。

    Args:
        config: 生效配置（读取 ``mcp.validate_on_add_only``；``False`` = 启动也 deep 校验）。
        project_root: 项目根目录；``None`` 时由 ``cwd`` 推导。
        home: 全局目录（``~/.myagent``）；``None`` 时取 ``$MYAGENT_HOME`` 或 ``~/.myagent``。
        env: 环境变量映射（``${VAR}`` 展开用；缺省 ``os.environ``）。
        store: 配置读写器（测试注入用）。
        validator: 校验器（测试注入用）。
        client_factory: 客户端工厂（测试注入用，可统计 spawn 次数）。
        existing_names: 已占用工具名（默认 :data:`BUILTIN_TOOL_NAMES` 视为内置工具）。
        cwd: 推导项目根用的目录。
    """

    def __init__(
        self,
        config: Config,
        project_root: Path | None = None,
        home: Path | None = None,
        *,
        env: Mapping[str, str] | None = None,
        store: McpConfigStore | None = None,
        validator: McpValidator | None = None,
        client_factory: Callable[[McpServerConfig], McpClient] | None = None,
        existing_names: Mapping[str, str] | Iterable[str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.config = config
        self.env = env
        self.store = store or McpConfigStore(
            project_root=project_root, home=home, cwd=cwd, env=env
        )
        self.project_root = Path(project_root) if project_root else self.store.project_root
        self.home = Path(home) if home else self.store.home
        self.validator = validator or McpValidator(client_factory=client_factory, env=env)
        self._client_factory = client_factory
        self._builtin_names = _normalize_names(existing_names)
        self._states: dict[str, _ServerState] = {}
        self._tool_index: dict[str, str] = {}
        self._load_warnings: list[str] = []

    # ------------------------------------------------------------------ 装载

    async def load_all(self, *, connect: bool = True) -> list[ToolSpec]:
        """装载全部启用的服务器并返回映射后的工具声明。

        **启动路径不做 deep 校验**（V10 不执行，F-2）；连接失败只记录告警并跳过，
        绝不阻塞启动，也不自动重试/重启。

        Args:
            connect: ``True``（默认）为每个 ``enabled`` 服务器建立首次连接
                （``initialize`` + ``tools/list``）以取得工具清单；``False`` 则
                不启动任何子进程/网络连接（仅登记，用于「零 spawn」约束测试）。

        Returns:
            可用工具的 :class:`~my_agent.models.tools.ToolSpec` 列表。
        """
        for server in self.store.load(existing_names=self._builtin_names):
            self._register(server)
        if connect and self._states:
            await asyncio.gather(
                *(self._connect_state(state) for state in self._states.values()),
                return_exceptions=True,
            )
        specs = self.tools()
        _LOG.info(
            "MCP 装载完成：服务器 %d 个，工具 %d 个，不健康 %d 个",
            len(self._states),
            len(specs),
            len(self.unhealthy()),
        )
        return specs

    async def reload(self) -> str:
        """关闭现有连接并重新装载（``/mcp reload``：重新读取配置）。

        Returns:
            一行结果摘要（``engine/slash.py`` 直接输出）。
        """
        await self._close_all()
        self._states.clear()
        self._tool_index.clear()
        self._load_warnings.clear()
        specs = await self.load_all()
        return (
            f"{len(self._states)} 个服务器 / {len(specs)} 个工具"
            + (f"（不健康 {len(self.unhealthy())}）" if self.unhealthy() else "")
        )

    async def connect(self, name: str) -> list[ToolSpec]:
        """按需连接一个已登记的服务器（首次连接 / 懒连接路径）。

        Args:
            name: 服务器名。

        Returns:
            该服务器贡献的工具声明（未登记或连接失败 → 空列表）。
        """
        state = self._states.get(name)
        if state is None:
            return []
        if not state.connected or not state.healthy:
            await self._connect_state(state)
        return [tool.spec for tool in state.tools] if state.healthy else []

    # ------------------------------------------------------------------ 管理

    async def add(self, path: str | Path, *, target: Path | None = None) -> ValidationReport:
        """``/mcp add <path>``：deep 校验（V1–V13）通过后才写配置（F-3）。

        Args:
            path: 待导入的 mcp 配置文件路径。
            target: 写入目标；``None`` → 项目级 ``<project>/.myagent/mcp.yaml``。

        Returns:
            :class:`~my_agent.mcp.validator.ValidationReport`；``ok=False`` 时
            配置文件**字节级未变**（且不会新建目录/文件）。
        """
        report = ValidationReport()
        source = Path(path)
        try:
            document = self.store.read_document(source)
            entries = self.store.entries_of(document, source=source.name)
        except McpValidationError as exc:
            report.add_error(
                str(exc.context.get("rule", "V1")), exc.detail, hint=f"来源：{source}"
            )
            return report
        if not entries:
            report.add_error("V1", f"{source.name} 顶层缺少非空的 servers 列表")
            return report

        taken_names = dict(self._builtin_names)
        taken_names.update({name: state.server.name for name, state in self._states.items()})
        taken_servers = set(self._states)
        constructed: list[McpServerConfig] = []
        for entry in entries:
            sub = await self.validator.validate(
                entry,
                deep=True,
                existing_names=taken_names,
                existing_servers=taken_servers,
            )
            report.merge(sub)
            if not sub.ok:
                continue
            try:
                server = self.store.build(entry, source=source.name)
            except McpValidationError as exc:
                report.add_error(str(exc.context.get("rule", "V2")), exc.detail)
                continue
            constructed.append(server)
            taken_servers.add(server.name)
            for spec in sub.discovered_tools:
                taken_names[spec.name] = server.name

        if report.errors:
            _LOG.warning("MCP 校验失败，未写入配置：%s", report.error_text())
            return report  # F-3：校验失败，配置文件一字未动

        write_target = Path(target) if target else self.store.default_write_path()
        existing = self._existing_servers_for_write(write_target, report)
        if report.errors:
            return report
        merged = self.store.merge_servers(existing, constructed)
        self.store.save(merged, path=write_target, validation=report)
        _LOG.info("MCP 配置已写入 %s（新增/更新 %d 个服务器）", write_target, len(constructed))
        for server in constructed:
            self._register(server)
            await self._connect_state(self._states[server.name])
        return report

    def remove(self, name: str) -> None:
        """``/mcp remove <name>``：从配置文件移除（并断开连接、清理状态）。

        Args:
            name: 服务器名。
        """
        self._states.pop(name, None)
        self._sync_tool_index()
        for path in (self.store.project_path, self.store.home_path):
            if not path.exists():
                continue
            try:
                servers, _report = self.store.read_file(path)
            except McpValidationError as exc:
                self._load_warnings.append(f"移除时跳过无法解析的配置 {path}：{exc.detail}")
                _LOG.warning("移除时跳过无法解析的配置 %s：%s", path, exc.detail)
                continue
            if not any(server.name == name for server in servers):
                continue
            self.store.save(self.store.remove_server(servers, name), path=path)

    async def remove_async(self, name: str) -> None:
        """``remove`` 的异步版本：先关闭客户端再改写配置。

        Args:
            name: 服务器名。
        """
        state = self._states.pop(name, None)
        if state is not None:
            await self._close_state(state)
        self._sync_tool_index()
        self.remove(name)

    # ------------------------------------------------------------------ 调用

    async def call_tool(
        self, name: str, args: dict[str, Any], *, call_id: str = ""
    ) -> ToolResult:
        """把一个工具调用转发给所属服务器。

        Args:
            name: 映射后的工具名。
            args: 调用参数。
            call_id: 引擎侧调用 id（回填 :class:`ToolResult`）。

        Returns:
            :class:`~my_agent.models.tools.ToolResult`；工具未注册、服务器不健康、
            进程已退出或 JSON-RPC 错误 → ``ok=False`` 且 ``error`` 说明原因（给模型看）。

        Raises:
            McpTimeoutError: 超过 ``server.timeout_s``（**不重试**）；此时服务器被标记
                ``unhealthy`` 并**清理进程**（F-6）。
        """
        binding = self.binding(name)
        if binding is None:
            return ToolResult(
                call_id=call_id, tool=name, ok=False, error=f"未注册的 MCP 工具：{name}"
            )
        state = self._states.get(binding.mcp_server)
        if state is None or state.client is None or not state.healthy:
            reason = state.error if state is not None and state.error else "未连接或已不健康"
            return ToolResult(
                call_id=call_id,
                tool=name,
                ok=False,
                error=(
                    f'MCP 服务器 "{binding.mcp_server}" 不可用（{reason}）；'
                    "不会自动重启，请用 /mcp 检查后重载"
                ),
            )
        try:
            result = await state.client.call_tool(binding.raw_name, args)
        except McpTimeoutError:
            self._mark_unhealthy(state, f"调用超时（{state.server.timeout_s}s）")
            await self._close_state(state)
            raise
        except (McpValidationError, OSError) as exc:
            detail = _detail(exc)
            self._mark_unhealthy(state, detail)
            await self._close_state(state)
            return ToolResult(call_id=call_id, tool=name, ok=False, error=detail)
        return result.model_copy(update={"call_id": call_id, "tool": name})

    # ------------------------------------------------------------------ 查询

    def tools(self) -> list[ToolSpec]:
        """返回全部可用 MCP 工具的 :class:`ToolSpec`（按名字排序，保证稳定）。

        Returns:
            工具声明列表（仅含健康且有工具清单的服务器）。
        """
        return [binding.spec for binding in self.bindings()]

    def mcp_tools(self) -> list[McpTool]:
        """返回全部可用工具的映射记录（含原始工具名与原始 ``inputSchema``）。

        Returns:
            :class:`~my_agent.mcp.schema.McpTool` 列表。
        """
        return [tool for state in self._states.values() if state.healthy for tool in state.tools]

    def bindings(self) -> list[McpToolBinding]:
        """返回全部可用工具的运行期绑定（含 ``mcp_server``，审计/UI 用）。

        Returns:
            绑定列表，按工具名排序。
        """
        out: list[McpToolBinding] = []
        for state in self._states.values():
            if state.client is None or not state.healthy:
                continue
            out.extend(
                McpToolBinding(
                    spec=tool.spec,
                    mcp_server=state.server.name,
                    tool=tool,
                    client=state.client,
                )
                for tool in state.tools
            )
        return sorted(out, key=lambda binding: binding.spec.name)

    def binding(self, name: str) -> McpToolBinding | None:
        """按映射后的工具名取绑定对象。

        Args:
            name: 映射后的工具名。

        Returns:
            绑定对象；不存在时为 ``None``。
        """
        server_name = self._tool_index.get(name)
        if server_name is None:
            return None
        state = self._states.get(server_name)
        if state is None or state.client is None:
            return None
        tool = next((item for item in state.tools if item.spec.name == name), None)
        if tool is None:
            return None
        return McpToolBinding(
            spec=tool.spec, mcp_server=state.server.name, tool=tool, client=state.client
        )

    def servers(self) -> list[McpServerConfig]:
        """返回已登记的服务器配置（含未连接/不健康）。

        Returns:
            服务器配置列表。
        """
        return [state.server for state in self._states.values()]

    def server_names(self) -> list[str]:
        """返回已登记的服务器名。

        Returns:
            服务器名列表。
        """
        return list(self._states)

    def unhealthy(self) -> dict[str, str]:
        """返回不健康服务器及其原因（UI 提示用；**不自动重启**）。

        Returns:
            服务器名 → 原因。
        """
        return {
            name: state.error or "未知原因"
            for name, state in self._states.items()
            if not state.healthy
        }

    def is_healthy(self, name: str) -> bool:
        """查询服务器健康状态。

        Args:
            name: 服务器名。

        Returns:
            已登记且健康时为 ``True``。
        """
        state = self._states.get(name)
        return bool(state is not None and state.healthy)

    def client_for(self, name: str) -> McpClient | None:
        """返回服务器当前使用的客户端（``/mcp`` 状态与测试用）。

        Args:
            name: 服务器名。

        Returns:
            客户端；未连接时为 ``None``。
        """
        state = self._states.get(name)
        return None if state is None else state.client

    def describe(self) -> str:
        """生成 ``/mcp list`` 文本。

        Returns:
            多行清单：服务器/协议版本/工具、不健康列表、加载告警、配置文件路径。
        """
        lines: list[str] = []
        connected = [
            (name, state) for name, state in self._states.items() if state.client is not None
        ]
        if not connected:
            lines.append("已连接的 MCP 服务器：无")
        else:
            lines.append(f"已连接的 MCP 服务器（{len(connected)}）：")
            for name, state in connected:
                version = getattr(state.client, "protocol_version", "") or "-"
                lines.append(f"  {name}  protocol={version}  工具 {len(state.tools)} 个")
                lines.extend(f"    - {tool.spec.name}" for tool in state.tools)
        disabled = [state.server.name for state in self._states.values() if state.client is None]
        if disabled:
            lines.append(f"未连接：{', '.join(disabled)}")
        bad = self.unhealthy()
        if bad:
            lines.append("")
            lines.append("不健康（不会自动重启）：")
            lines.extend(f"  ! {name}：{reason}" for name, reason in bad.items())
        if self._load_warnings:
            lines.append("")
            lines.extend(f"  ~ {warning}" for warning in self._load_warnings)
        lines.append("")
        lines.append(f"配置文件：{self.store.project_path} / {self.store.home_path}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ 关闭

    async def aclose(self) -> None:
        """关闭全部客户端（stdio：``terminate`` → ``MCP_TERMINATE_GRACE_S`` → ``kill``）。"""
        await self._close_all()
        self._states.clear()
        self._tool_index.clear()

    # ------------------------------------------------------------------ 内部实现

    def _register(self, server: McpServerConfig) -> None:
        """登记服务器（``enabled=false`` 跳过；同名更新配置）。

        Args:
            server: 服务器配置。
        """
        if not server.enabled:
            _LOG.debug("MCP 服务器 %s 已禁用，跳过", server.name)
            return
        existing = self._states.get(server.name)
        if existing is not None:
            existing.server = server
            return
        self._states[server.name] = _ServerState(server=server)

    def _sync_tool_index(self) -> None:
        """重建「工具名 → 服务器名」索引（过滤不健康服务器）。"""
        index: dict[str, str] = {}
        for name, state in self._states.items():
            if state.client is None or not state.healthy:
                continue
            for tool in state.tools:
                index.setdefault(tool.spec.name, name)
        self._tool_index = index

    async def _connect_state(self, state: _ServerState) -> None:
        """连接服务器并登记工具（失败 → 标记 unhealthy，绝不抛出）。

        Args:
            state: 服务器状态（就地更新）。
        """
        if state.connected and state.healthy and state.client is not None:
            return
        server = state.server
        static = self.validator.validate_static(
            server, existing_names=self._taken_names()
        )
        if not static.ok:
            self._warn(f"MCP 服务器 {server.name} 配置非法（已跳过）：{static.error_text()}")
            self._mark_unhealthy(state, static.error_text() or "配置非法")
            return
        try:
            client = self._make_client(server)
        except (McpValidationError, OSError) as exc:
            self._warn(f"MCP 服务器 {server.name} 启动失败（已跳过）：{_detail(exc)}")
            self._mark_unhealthy(state, _detail(exc))
            return
        try:
            await client.initialize()
            tools = await client.list_tools()
        except (McpTimeoutError, McpValidationError, OSError) as exc:
            self._warn(f"MCP 服务器 {server.name} 连接失败（已跳过）：{_detail(exc)}")
            self._mark_unhealthy(state, _detail(exc))
            await self._safe_close(client)
            return

        taken = self._taken_names()
        accepted: list[McpTool] = []
        for tool in tools:
            if tool.spec.name in taken:
                reason = taken[tool.spec.name]
                where = "内置工具" if reason == "builtin" else f'MCP 服务器 "{reason}"'
                self._warn(
                    f'工具名 "{tool.spec.name}" 与{where}冲突，已跳过该工具；'
                    f"请为服务器 {server.name} 设置 tool_prefix"
                )
                continue
            accepted.append(tool)
            taken[tool.spec.name] = server.name
        state.client = client
        state.tools = accepted
        state.connected = True
        state.healthy = True
        state.error = ""
        self._sync_tool_index()
        _LOG.info("MCP 服务器 %s 已连接，发现 %d 个工具", server.name, len(accepted))

    def _make_client(self, server: McpServerConfig) -> McpClient:
        """按 transport 创建客户端（stdio / http / sse）。

        Args:
            server: 服务器配置。

        Returns:
            :class:`~my_agent.mcp.schema.McpClient` 实现。
        """
        if self._client_factory is not None:
            return self._client_factory(server)
        return default_client_factory(server, env=self.env)

    def _taken_names(self) -> dict[str, str]:
        """返回「已占用工具名 → 来源标签」（V11 用；含内置工具与已连接 MCP 工具）。

        Returns:
            工具名 → 来源（``"builtin"`` 或 MCP 服务器名）。
        """
        taken = dict(self._builtin_names)
        for state in self._states.values():
            if state.client is None or not state.healthy:
                continue
            for tool in state.tools:
                taken.setdefault(tool.spec.name, state.server.name)
        return taken

    def _mark_unhealthy(self, state: _ServerState, reason: str) -> None:
        """标记服务器不健康（**不自动重启**）。

        Args:
            state: 服务器状态。
            reason: 原因文本。
        """
        state.healthy = False
        state.error = reason
        state.connected = False
        state.client = None
        state.tools = []
        self._sync_tool_index()
        _LOG.warning("MCP 服务器 %s 标记 unhealthy：%s（不会自动重启）", state.server.name, reason)

    def _warn(self, message: str) -> None:
        """记录一条加载/连接告警（同时进 :meth:`describe` 输出）。

        Args:
            message: 告警文本。
        """
        self._load_warnings.append(message)
        _LOG.warning("%s", message)

    async def _close_state(self, state: _ServerState) -> None:
        """关闭一个服务器的客户端并清空其工具缓存。

        Args:
            state: 服务器状态。
        """
        client = state.client
        state.client = None
        state.tools = []
        state.connected = False
        if client is not None:
            await self._safe_close(client)

    async def _close_all(self) -> None:
        """关闭全部服务器的客户端（并发，尽力而为）。"""
        await asyncio.gather(
            *(self._close_state(state) for state in self._states.values()),
            return_exceptions=True,
        )

    async def _safe_close(self, client: McpClient) -> None:
        """关闭客户端（关闭失败只记日志，绝不上抛）。

        Args:
            client: 客户端实例。
        """
        try:
            await client.close()
        except (McpValidationError, McpTimeoutError, OSError) as exc:
            _LOG.debug("关闭 MCP 客户端失败：%s", exc)

    def _existing_servers_for_write(
        self, target: Path, report: ValidationReport
    ) -> list[McpServerConfig]:
        """读取写入目标里已有的服务器（避免丢条目；已有非法条目则拒绝写入）。

        Args:
            target: 写入目标路径。
            report: 报告（就地追加错误）。

        Returns:
            既有服务器列表；目标不存在 → 空列表。
        """
        if not target.exists():
            return []
        try:
            document = self.store.read_document(target)
            entries = self.store.entries_of(document, source=target.name)
        except McpValidationError as exc:
            report.add_error(
                str(exc.context.get("rule", "V1")),
                f"写入目标已存在但无法解析：{exc.detail}",
                hint="请先修复该文件再执行 /mcp add",
            )
            return []
        existing: list[McpServerConfig] = []
        for entry in entries:
            try:
                existing.append(self.store.build(entry, source=target.name))
            except McpValidationError as exc:
                raw_name = entry.get("name")
                name = raw_name if isinstance(raw_name, str) else "?"
                report.add_error(
                    str(exc.context.get("rule", "V2")),
                    f'写入目标里的服务器 "{name}" 非法：{exc.detail}',
                    hint="为避免丢失既有条目，本次未写入任何改动",
                )
        return existing


class McpToolAdapter:
    """把 MCP 工具适配成内置 ``Tool`` 协议（§10.4，契约 §17.5）。

    审计所需字段（``mcp_server``）通过 :attr:`mcp_server` 公开，
    权限流水线与审计层据此追溯「哪个外部服务的调用被放行」（F-8）。

    Args:
        manager: 所属 MCP 管理器。
        spec: 映射后的工具声明（``source="mcp"``）。
    """

    def __init__(self, manager: MCPManager, spec: ToolSpec) -> None:
        self.manager = manager
        self.spec = spec
        self.mcp_server = self._server_of(spec)
        self.binding = manager.binding(spec.name)

    def _server_of(self, spec: ToolSpec) -> str:
        """从管理器中反查该工具的来源服务器名。

        Args:
            spec: 工具声明。

        Returns:
            服务器名；查不到时返回空串。
        """
        binding = self.manager.binding(spec.name)
        return "" if binding is None else binding.mcp_server

    def audit_fields(self) -> dict[str, str]:
        """返回审计层要补写的字段（§10.4，F-8）。

        Returns:
            形如 ``{"mcp_server": "filesystem", "tool": "fs__read_file"}``。
        """
        return {"mcp_server": self.mcp_server, "tool": self.spec.name}

    def schema(self) -> dict[str, Any]:
        """返回发往 Provider 的 function schema。

        Returns:
            OpenAI 兼容 schema。
        """
        return self.spec.to_schema()

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """转发调用（``ctx`` 仅用于回填 ``call_id``，MCP 侧不需要工作目录等）。

        Args:
            args: 调用参数。
            ctx: 执行上下文。

        Returns:
            :class:`~my_agent.models.tools.ToolResult`。

        Raises:
            McpTimeoutError: 服务器超时（由工具注册表转成失败结果）。
        """
        result = await self.manager.call_tool(
            self.spec.name, args, call_id=ctx.call_id
        )
        if not result.tool:
            result.tool = self.spec.name
        return result


def map_tools_for(server: McpServerConfig, raws: list[Mapping[str, Any]]) -> list[McpTool]:
    """便捷函数：把原始工具声明映射为 :class:`McpTool`（委托 ``schema.map_mcp_tools``）。

    Args:
        server: 服务器配置。
        raws: 原始 ``tools/list`` 条目。

    Returns:
        映射后的工具列表。
    """
    from my_agent.mcp.schema import map_mcp_tools

    return map_mcp_tools(server, raws)


def tool_invocation_for(spec: ToolSpec, call_id: str, arguments: dict[str, Any]) -> ToolInvocation:
    """构造一个 MCP 工具的调用对象（便于 ``/`` 命令或测试直接驱动）。

    Args:
        spec: 工具声明。
        call_id: 调用 id。
        arguments: 参数。

    Returns:
        :class:`~my_agent.models.tools.ToolInvocation`。
    """
    return ToolInvocation(call_id=call_id, tool=spec.name, arguments=arguments, spec=spec)


def _normalize_names(
    existing_names: Mapping[str, str] | Iterable[str] | None,
) -> dict[str, str]:
    """把注入的已占用工具名归一化为 ``{工具名: 来源标签}``。

    Args:
        existing_names: ``None`` → :data:`BUILTIN_TOOL_NAMES`（标签 ``builtin``）；
            ``Mapping`` → value 作来源标签；其它可迭代对象 → 标签 ``builtin``。

    Returns:
        归一化映射。
    """
    if existing_names is None:
        return dict.fromkeys(BUILTIN_TOOL_NAMES, "builtin")
    if isinstance(existing_names, Mapping):
        return {str(key): str(value) for key, value in existing_names.items()}
    return {str(name): "builtin" for name in existing_names}


def _detail(exc: BaseException) -> str:
    """取异常的可读详情（``MyAgentError.detail`` 优先）。

    Args:
        exc: 任意异常。

    Returns:
        详情文本。
    """
    detail = getattr(exc, "detail", "")
    return detail or str(exc) or type(exc).__name__
