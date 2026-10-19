"""``TodoWrite`` 工具：任务清单持久化 + 渲染（§14.2）。

- 落盘位置由构造参数给出（CLI 装配时传 ``<project>/.myagent/todos.json``）。
- **原子写**：临时文件 + ``os.replace``，避免进程被中断时留下半截 JSON。
- 内部状态文件**只允许**位于 ``.myagent/`` 内（构造时强制校验，越界直接拒绝）。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal

from my_agent.models.tools import ToolContext, ToolResult, ToolSpec
from my_agent.tools.backup import MYAGENT_DIR_NAME
from my_agent.tools.base import BaseTool, atomic_write_text, display_path

__all__ = ["TodoItem", "TodoWriteTool", "render_todos"]

#: 允许的任务状态（与前端渲染一一对应）。
TodoStatus = Literal["pending", "in_progress", "completed"]

#: 状态 → checklist 标记。
STATUS_MARKS: dict[str, str] = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}

_DESCRIPTION = """创建或更新当前任务清单（持久化到 .myagent/todos.json，并渲染成 checklist）。

何时用：任务需要 3 个以上步骤、或要跨多轮工具调用才能完成时，用它记录进度、保持不跑偏。
何时不用：① 不要用它记录与当前任务无关的待办事项；② 不要用它写代码或文件内容——用 Write/Edit；
③ 不要用它维护第二份清单（同一时刻只保留一份最新清单，重复调用会整体覆盖）。
副作用：**会修改磁盘文件**（仅 .myagent/todos.json 一个内部状态文件，不触碰源码）。"""


class TodoItem:
    """一条任务项（轻量数据结构，避免为工具入参再引入 pydantic 模型）。

    Args:
        content: 任务描述。
        status: ``pending`` / ``in_progress`` / ``completed``。
        active_form: 进行中时展示的措辞（如「正在写测试」）。

    Attributes:
        content: 任务描述。
        status: 任务状态。
        active_form: 进行中的措辞。
    """

    def __init__(self, content: str, status: TodoStatus, active_form: str = "") -> None:
        self.content = content
        self.status = status
        self.active_form = active_form

    def to_dict(self) -> dict[str, str]:
        """转换为落盘用的字典。

        Returns:
            ``{"content":..., "status":..., "activeForm":...}``。
        """
        return {"content": self.content, "status": self.status, "activeForm": self.active_form}


def render_todos(items: list[TodoItem]) -> str:
    """把任务清单渲染成 checklist 文本。

    Args:
        items: 任务项列表。

    Returns:
        多行文本；空清单返回 ``（空清单）``。
    """
    if not items:
        return "任务清单已清空（空清单）"
    done = sum(1 for item in items if item.status == "completed")
    running = sum(1 for item in items if item.status == "in_progress")
    lines = [f"任务清单已更新（完成 {done}/{len(items)}，进行中 {running}）："]
    for index, item in enumerate(items, start=1):
        mark = STATUS_MARKS.get(item.status, "[ ]")
        suffix = f"（正在：{item.active_form}）" if item.status == "in_progress" and item.active_form else ""
        lines.append(f"  {index}. {mark} {item.content}{suffix}")
    return "\n".join(lines)


class TodoWriteTool(BaseTool):
    """写入任务清单（原子持久化 + 渲染）。

    Args:
        todo_path: 清单文件路径（必须位于 ``.myagent/`` 目录内）。

    Raises:
        ValueError: 路径不在 ``.myagent/`` 内时（防止内部状态污染用户源码目录）。
    """

    spec = ToolSpec(
        name="TodoWrite",
        description=_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "完整任务清单（整体覆盖旧清单）",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "任务描述"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "任务状态",
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "进行中时的措辞（可选）",
                            },
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
        category="meta",
        is_read_only=False,
        is_destructive=False,
    )

    def __init__(self, todo_path: Path) -> None:
        resolved = Path(todo_path)
        if MYAGENT_DIR_NAME not in resolved.parts:
            raise ValueError(f"TodoWrite 的清单路径必须位于 {MYAGENT_DIR_NAME}/ 内：{resolved}")
        self.todo_path = resolved

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """写入任务清单。

        Args:
            args: ``todos``（数组，元素含 ``content``/``status``/可选 ``activeForm``）。
            ctx: 执行上下文（仅用于结果记账与展示）。

        Returns:
            ``ok=True`` 且 ``content`` 为渲染后的 checklist；参数非法或写盘失败时 ``ok=False``。
        """
        raw_todos = args.get("todos")
        if not isinstance(raw_todos, list):
            return self.fail(ctx, "缺少必填参数 todos（应为任务数组；清空清单请传 []）")
        items: list[TodoItem] = []
        for index, raw in enumerate(raw_todos):
            if not isinstance(raw, dict):
                return self.fail(ctx, f"todos[{index}] 非法：应为对象 {{content, status}}")
            content = raw.get("content")
            if not isinstance(content, str) or not content.strip():
                return self.fail(ctx, f"todos[{index}].content 非法：应为非空字符串")
            status = raw.get("status", "pending")
            if status not in STATUS_MARKS:
                return self.fail(
                    ctx,
                    f"todos[{index}].status 非法：{status!r}；"
                    "允许值 pending / in_progress / completed",
                )
            active_form = raw.get("activeForm", raw.get("active_form", ""))
            if not isinstance(active_form, str):
                return self.fail(ctx, f"todos[{index}].activeForm 非法：应为字符串")
            items.append(TodoItem(content=content.strip(), status=status, active_form=active_form))

        error = await asyncio.to_thread(self._persist, items, ctx)
        if error is not None:
            return self.fail(ctx, error)
        return self.ok(ctx, render_todos(items))

    def _persist(self, items: list[TodoItem], ctx: ToolContext) -> str | None:
        """原子写入清单文件。

        Args:
            items: 任务项。
            ctx: 执行上下文（写入 ``session_id`` 便于追溯）。

        Returns:
            失败原因；成功返回 ``None``。
        """
        payload = {
            "session_id": ctx.session_id,
            "updated_at": time.time(),
            "todos": [item.to_dict() for item in items],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        try:
            self.todo_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.todo_path, text)
        except OSError as exc:
            return f"写入任务清单失败：{display_path(self.todo_path, ctx)}（{exc}）"
        return None
