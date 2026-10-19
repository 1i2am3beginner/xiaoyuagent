"""``Task`` 工具：主 Agent 派生子 Agent 的唯一入口（§12.2）。

参数契约**逐字**采用 §12.2 的 JSON Schema；描述按 §14.3 三段式撰写
（做什么 / 何时用何时不用（含反例）/ 副作用声明）。
"""

from __future__ import annotations

from typing import Any

from my_agent.agents.subagent import AGENT_TYPES, SubagentRunner
from my_agent.errors import MyAgentError, SubagentLimitError
from my_agent.models.tools import ToolContext, ToolResult, ToolSpec

__all__ = ["TASK_PARAMETERS", "TaskTool"]

TASK_DESCRIPTION = (
    "派生一个子 Agent 完成独立子任务（搜索、分析、批量修改调研等）。"
    "子 Agent 有独立上下文，只回传结论。\n"
    "何时用：需要大量探索但只需要结论的任务；需要并行推进多个互不依赖的调查；"
    "希望把大段中间过程挡在主上下文之外时。\n"
    "何时不用：需要与用户交互的任务（子 Agent 无法提问）；需要读取当前对话细节的任务"
    "（子 Agent 看不到主对话历史）；一句话就能答完的小事——直接自己做，不要为省事派生。\n"
    "副作用声明：子 Agent 会按继承的安全模式实际调用工具，可能修改磁盘文件；"
    "其内部的写操作同样要经过权限流水线确认。"
)

#: §12.2 的参数 schema（逐字采用）。
TASK_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "description": "3-5 词的短标题"},
        "prompt": {
            "type": "string",
            "description": "完整、自包含的任务说明（子 Agent 看不到当前对话）",
        },
        "agent_type": {
            "type": "string",
            "enum": list(AGENT_TYPES),
            "default": "general",
        },
        "mode": {"type": "string", "enum": ["inherit", "plan", "default"], "default": "inherit"},
        "run_in_background": {"type": "boolean", "default": False},
    },
    "required": ["description", "prompt"],
}


class TaskTool:
    """``Task`` 工具的 Tool 协议实现。"""

    def __init__(self, runner: SubagentRunner) -> None:
        """初始化。

        Args:
            runner: 子 Agent 运行时。
        """
        self.runner = runner
        self.spec = ToolSpec(
            name="Task",
            description=TASK_DESCRIPTION,
            parameters=TASK_PARAMETERS,
            category="meta",
            is_read_only=False,
            is_destructive=False,
            requires_user_approval_hint=False,
            source="subagent",
        )

    def schema(self) -> dict[str, Any]:
        """返回发往 Provider 的 function schema。

        Returns:
            OpenAI 兼容 schema。
        """
        return self.spec.to_schema()

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """派生一个子 Agent 并返回其结论（§12.2）。

        Args:
            args: ``description`` / ``prompt`` / ``agent_type`` / ``mode`` /
                ``run_in_background``。
            ctx: 执行上下文（提供父会话 id 与轮次，用于审计归属）。

        Returns:
            :class:`~my_agent.models.tools.ToolResult`：``content`` 为结论块，
            ``ok=False`` 时 ``error`` 为失败原因。

        Raises:
            SubagentLimitError: 深度超限（由运行时抛出，此处转成失败结果）。
        """
        description = str(args.get("description") or "").strip()
        prompt = str(args.get("prompt") or "").strip()
        if not description or not prompt:
            return ToolResult(
                call_id=ctx.call_id,
                tool="Task",
                ok=False,
                error="Task 需要非空的 description 与 prompt（prompt 必须自包含）",
            )
        agent_type = str(args.get("agent_type") or "general")
        mode = str(args.get("mode") or "inherit")
        run_in_background = bool(args.get("run_in_background", False))

        parent = _parent_session(ctx)
        try:
            result = await self.runner.spawn(
                description=description,
                prompt=prompt,
                agent_type=agent_type,
                mode=mode,
                run_in_background=run_in_background,
                parent=parent,
                parent_call_id=ctx.call_id,
            )
        except SubagentLimitError as exc:
            return ToolResult(
                call_id=ctx.call_id, tool="Task", ok=False, error=exc.user_message()
            )
        except MyAgentError as exc:
            return ToolResult(
                call_id=ctx.call_id, tool="Task", ok=False, error=exc.user_message()
            )

        if run_in_background:
            return ToolResult(
                call_id=ctx.call_id,
                tool="Task",
                ok=True,
                content=(
                    f"{result.text}\n"
                    "（其结论将在完成后的下一轮以 [后台子 Agent 完成] 形式出现在上下文中；"
                    "现在不要等待，继续其它工作）"
                ),
            )
        if not result.ok:
            return ToolResult(
                call_id=ctx.call_id,
                tool="Task",
                ok=False,
                error=result.error or "子 Agent 失败",
                content=result.as_block(),
            )
        return ToolResult(
            call_id=ctx.call_id,
            tool="Task",
            ok=True,
            content=result.as_block(),
            tokens=result.tokens,
        )


def _parent_session(ctx: ToolContext) -> Any:
    """从执行上下文里取回父会话（缺省构造一个最小会话）。

    Args:
        ctx: 执行上下文。

    Returns:
        :class:`~my_agent.models.session.SessionState`。
    """
    from my_agent.models.session import SessionState

    session = getattr(ctx.config, "_session", None)
    if isinstance(session, SessionState):
        return session
    return SessionState(
        session_id=ctx.session_id,
        cwd=ctx.cwd,
        project_root=ctx.project_root or ctx.cwd,
    )
