"""工具层（模块 J，§14）：工具协议、注册表、内置工具与写前备份。

对外出口：

- :class:`~my_agent.tools.base.Tool` / :class:`~my_agent.tools.base.BaseTool`：工具协议与基类。
- :class:`~my_agent.tools.registry.ToolRegistry`：注册 + 受控执行（C1 双保险）。
- :class:`~my_agent.tools.backup.BackupManager`：写前备份与 ``/undo``（§14.5）。
- :func:`build_default_registry`：装配引擎用的默认工具集。

本模块**不注册** ``NotebookEdit``（【待确认假设 U27】首版不做），也不实现
``Task`` / ``Skill`` / ``SkillList`` / ``MemoryRead`` / ``MemoryWrite`` / ``MemoryForget`` /
``ArchiveSearch``（由对应模块提供，通过 ``extra=`` 注入）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from my_agent.models.tools import ToolContext
from my_agent.tools.backup import BackupEntry, BackupManager, UndoResult
from my_agent.tools.base import BaseTool, Tool
from my_agent.tools.fs_edit import EditTool, MultiEditTool
from my_agent.tools.fs_glob import GlobTool
from my_agent.tools.fs_grep import GrepTool
from my_agent.tools.fs_read import ReadTool
from my_agent.tools.fs_write import WriteTool
from my_agent.tools.registry import PermissionGate, ToolRegistry
from my_agent.tools.shell import BashTool
from my_agent.tools.todo_tool import TodoWriteTool
from my_agent.tools.web import WebFetchTool, WebSearchTool

__all__ = [
    "BaseTool",
    "BackupEntry",
    "BackupManager",
    "BashTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "MultiEditTool",
    "PermissionGate",
    "ReadTool",
    "TodoWriteTool",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "UndoResult",
    "WebFetchTool",
    "WebSearchTool",
    "WriteTool",
    "build_default_registry",
    "ensure_tool_context_ready",
]


def ensure_tool_context_ready() -> bool:
    """尽力让 :class:`~my_agent.models.tools.ToolContext` 的前向引用 ``Config`` 解析成功。

    ``models/tools.py`` 只在 ``TYPE_CHECKING`` 下引用 ``Config``，因此 pydantic 无法在
    运行时自动完成类构建（构造 ``ToolContext(...)`` 会抛 ``PydanticUserError``）。
    这里用真实的 ``Config`` 命名空间显式重建一次；属于对冻结模型的**外部补救**，
    不修改 ``models/`` 任何文件。

    Returns:
        ``True`` = ``ToolContext`` 已可正常构造；``False`` = ``config.schema`` 尚不可用
        （此时构造会由 pydantic 明确报错，不会静默产生坏对象）。
    """
    if ToolContext.__pydantic_complete__:
        return True
    try:
        from my_agent.config.schema import Config
    except Exception:  # 配置层尚未就绪/存在导入环时不应拖垮工具层导入
        return False
    try:
        ToolContext.model_rebuild(_types_namespace={"Config": Config})
    except Exception:  # pragma: no cover - schema 定义异常时保持原状
        return False
    return ToolContext.__pydantic_complete__


def build_default_registry(
    *,
    config: Any,
    backups: BackupManager,
    todo_path: Path | None = None,
    extra: list[Tool] | None = None,
    permission_gate: PermissionGate | None = None,
) -> ToolRegistry:
    """装配默认工具集（§14.2 最小可用集）。

    注册顺序：``Read``、``Write``、``Edit``、``MultiEdit``、``Glob``、``Grep``、``Bash``、
    ``TodoWrite``、``WebSearch``、``WebFetch``，随后追加 ``extra``（Task/Skill/Memory/
    ArchiveSearch 等由各自模块提供）。``NotebookEdit`` **不注册**（U27）。

    Args:
        config: 全局配置（WebSearch/WebFetch 读取其中的后端设置；可为 ``None``）。
        backups: 备份管理器（Write/Edit/MultiEdit 写前备份，§14.5）。
        todo_path: ``TodoWrite`` 的落盘路径；缺省为
            ``<project_root>/.myagent/todos.json``。
        extra: 额外工具（其他 agent 提供的工具、MCP 工具等）。
        permission_gate: 可选的批量权限询问回调（§14.4 串行询问）。

    Returns:
        已装配好的 :class:`~my_agent.tools.registry.ToolRegistry`。

    Raises:
        ToolNameConflictError: ``extra`` 与内置工具重名（J-7 的同一条规则）。
        ValueError: ``todo_path`` 不在 ``.myagent/`` 内。
    """
    registry = ToolRegistry(permission_gate=permission_gate)
    ensure_tool_context_ready()
    resolved_todo = (
        Path(todo_path)
        if todo_path is not None
        else Path(backups.project_root) / ".myagent" / "todos.json"
    )
    for tool in (
        ReadTool(),
        WriteTool(backups),
        EditTool(backups),
        MultiEditTool(backups),
        GlobTool(),
        GrepTool(),
        BashTool(),
        TodoWriteTool(resolved_todo),
        WebSearchTool(config),
        WebFetchTool(config),
    ):
        registry.register(tool)
    for tool in extra or []:
        registry.register(tool)
    return registry
