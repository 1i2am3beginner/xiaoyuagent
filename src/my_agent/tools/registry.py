"""工具注册表与并发执行策略（§14.1、§14.4）。

职责：
- **注册/查询**：``register`` / ``get`` / ``get_spec`` / ``names`` / ``schema`` / ``to_specs``；
  重名注册直接抛 :class:`~my_agent.errors.ToolNameConflictError`（J-7）。
- **执行闸门**：:meth:`ToolRegistry.execute` 做 **C1 双保险**——``ctx.permission_checked``
  必须为 ``True``，否则抛 :class:`~my_agent.errors.PermissionBypassedError`（J-1）。
- **批量执行**：:meth:`ToolRegistry.execute_many` 实现 §14.4：

  ==================== ==========================================================
  情形                 策略
  ==================== ==========================================================
  全只读               并发（``asyncio.Semaphore(MAX_CONCURRENT_TOOLS)``）
  含写/执行            串行，按模型给出的顺序
  权限询问             串行、按调用顺序排队（一次性问完再执行）
  结果顺序             无论并发与否，都与 ``invs`` 顺序一致（B2）
  失败隔离             单个工具抛异常 → 转 ``ToolResult(ok=False)``，不影响其它工具
  ==================== ==========================================================
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from my_agent.config.defaults import MAX_CONCURRENT_TOOLS
from my_agent.errors import MyAgentError, PermissionBypassedError, ToolNameConflictError
from my_agent.models.tools import ToolContext, ToolInvocation, ToolResult, ToolSpec
from my_agent.tools.base import Tool

__all__ = ["PermissionGate", "ToolRegistry"]

#: 批量执行前的权限询问回调：返回 ``True`` 放行、``False`` 拒绝（由 L4 装配时注入）。
PermissionGate = Callable[[ToolInvocation, ToolContext], Awaitable[bool]]


class ToolRegistry:
    """工具注册表（注册、查询、受控执行）。

    Args:
        permission_gate: 可选的批量权限询问回调。注入后 :meth:`execute_many` 会先按调用顺序
            **串行**询问（一次只问一个）；未注入时认为权限已由上游流水线完成
            （此时 ``ctx.permission_checked`` 必须已是 ``True``，否则 C1 拦截）。
    """

    def __init__(self, *, permission_gate: PermissionGate | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._permission_gate = permission_gate
        self._ask_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # 注册与查询
    # ------------------------------------------------------------------ #
    def register(self, tool: Tool) -> None:
        """注册一个工具。

        Args:
            tool: 实现 :class:`~my_agent.tools.base.Tool` 协议的对象。

        Raises:
            ToolNameConflictError: 工具名已被占用（J-7；含 MCP/子 Agent 工具撞名的情况）。
        """
        name = tool.spec.name
        existing = self._tools.get(name)
        if existing is not None:
            raise ToolNameConflictError(
                name=name,
                source=existing.spec.source,
                detail=f"工具名 {name} 已被注册（来源 {existing.spec.source}）",
            )
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        """按名字取工具。

        Args:
            name: 工具名。

        Returns:
            工具对象；未注册时 ``None``。
        """
        return self._tools.get(name)

    def get_spec(self, name: str) -> ToolSpec | None:
        """按名字取工具声明（ReAct 循环在构造 ``ToolInvocation`` 前用它）。

        Args:
            name: 工具名。

        Returns:
            工具声明；未注册时 ``None``。
        """
        tool = self._tools.get(name)
        return tool.spec if tool is not None else None

    def names(self) -> list[str]:
        """返回全部工具名（按字典序，保证提示词缓存前缀稳定）。

        Returns:
            工具名列表。
        """
        return sorted(self._tools)

    def schema(self) -> list[dict[str, Any]]:
        """返回发往 Provider 的 ``tools`` 数组。

        Returns:
            每个元素形如 ``{"type": "function", "function": {...}}``。
        """
        return [self._tools[name].schema() for name in self.names()]

    def to_specs(self) -> list[ToolSpec]:
        """返回全部工具声明。

        Returns:
            按工具名字典序排列的 :class:`~my_agent.models.tools.ToolSpec` 列表。
        """
        return [self._tools[name].spec for name in self.names()]

    def __len__(self) -> int:
        """返回已注册工具数量。

        Returns:
            工具数量。
        """
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        """判断工具是否已注册。

        Args:
            name: 工具名。

        Returns:
            已注册时 ``True``。
        """
        return name in self._tools

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #
    async def execute(self, inv: ToolInvocation, ctx: ToolContext) -> ToolResult:
        """执行单个工具调用（C1 双保险闸门）。

        Args:
            inv: 待执行的调用（``arguments`` 直接透传给工具）。
            ctx: 执行上下文；``permission_checked`` 必须为 ``True``。

        Returns:
            工具结果。单个工具抛出的异常会被隔离成 ``ok=False`` 的结果（§14.4），
            未知工具返回 ``ok=False`` 并列出可用工具。

        Raises:
            PermissionBypassedError: ``ctx.permission_checked is not True``（未过权限流水线）。
        """
        if not ctx.permission_checked:
            raise PermissionBypassedError(name=inv.tool)
        tool = self._tools.get(inv.tool)
        if tool is None:
            return ToolResult(
                call_id=inv.call_id,
                tool=inv.tool,
                ok=False,
                error=f"未知工具：{inv.tool}（可用工具：{', '.join(self.names())}）",
            )
        started = time.monotonic()
        try:
            result = await tool.run(inv.arguments, ctx)
        except PermissionBypassedError:
            raise
        except MyAgentError as exc:
            result = ToolResult(call_id=inv.call_id, tool=inv.tool, ok=False, error=exc.user_message())
        except Exception as exc:  # §14.4 失败隔离：任何工具异常都转成失败结果
            result = ToolResult(
                call_id=inv.call_id,
                tool=inv.tool,
                ok=False,
                error=f"工具内部错误（{type(exc).__name__}）：{exc}",
            )
        # 以注册表的请求为准回填标识，保证 B2「回填顺序与请求一一对应」
        result.call_id = inv.call_id
        result.tool = inv.tool
        if result.duration_ms == 0:
            result.duration_ms = int((time.monotonic() - started) * 1000)
        return result

    async def execute_many(
        self, invs: list[ToolInvocation], ctx: ToolContext
    ) -> list[ToolResult]:
        """按 §14.4 策略批量执行，并保证结果顺序与请求顺序一致（B2）。

        Args:
            invs: 同一轮的多个调用（保持模型给出的顺序）。
            ctx: 执行上下文；含写/执行工具时由上游流水线先置 ``permission_checked=True``。

        Returns:
            与 ``invs`` 等长、且第 i 个结果对应第 i 个请求的结果列表。

        Raises:
            PermissionBypassedError: 未注入 ``permission_gate`` 且 ``ctx.permission_checked``
                不为 ``True``（C1；不会被失败隔离吞掉）。
        """
        if not invs:
            return []

        denied = await self._ask_all(invs, ctx)

        results: list[ToolResult | None] = [None] * len(invs)
        pending: list[tuple[int, ToolInvocation]] = []
        for index, inv in enumerate(invs):
            blocked = denied.get(index)
            if blocked is not None:
                results[index] = blocked
            else:
                pending.append((index, inv))

        if self._needs_serial([inv for _index, inv in pending]):
            for index, inv in pending:
                results[index] = await self.execute(inv, ctx)
        else:
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_TOOLS)

            async def run_one(inv: ToolInvocation) -> ToolResult:
                async with semaphore:
                    return await self.execute(inv, ctx)

            gathered = await asyncio.gather(*(run_one(inv) for _index, inv in pending))
            for (index, _inv), result in zip(pending, gathered, strict=True):
                results[index] = result

        ordered: list[ToolResult] = []
        for index, inv in enumerate(invs):
            item = results[index]
            ordered.append(item if item is not None else self._missing_result(inv))
        return ordered

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    async def _ask_all(
        self, invs: list[ToolInvocation], ctx: ToolContext
    ) -> dict[int, ToolResult]:
        """按调用顺序串行询问权限（§14.4「一次只问一个」）。

        Args:
            invs: 本轮全部调用。
            ctx: 执行上下文；放行后会把 ``permission_checked`` 置为 ``True``。

        Returns:
            ``下标 -> 被拒绝的结果``；放行的调用不出现在其中。
        """
        if self._permission_gate is None:
            return {}
        denied: dict[int, ToolResult] = {}
        for index, inv in enumerate(invs):
            async with self._ask_lock:
                allowed = await self._permission_gate(inv, ctx)
            if allowed:
                ctx.permission_checked = True
                continue
            denied[index] = ToolResult(
                call_id=inv.call_id,
                tool=inv.tool,
                ok=False,
                error=f"权限被拒绝：{inv.tool} 未获得批准",
                denied_by_permission=True,
            )
        return denied

    def _needs_serial(self, invs: list[ToolInvocation]) -> bool:
        """判断本轮是否必须串行执行（存在写/执行类工具或未知工具）。

        Args:
            invs: 待执行的调用。

        Returns:
            需要串行时 ``True``（只读工具集返回 ``False``，可并发）。
        """
        for inv in invs:
            spec = self.get_spec(inv.tool)
            if spec is None:
                return True
            if not spec.is_read_only:
                return True
        return False

    @staticmethod
    def _missing_result(inv: ToolInvocation) -> ToolResult:  # pragma: no cover - 兜底
        """构造兜底失败结果（正常流程不会走到）。

        Args:
            inv: 对应调用。

        Returns:
            ``ok=False`` 的结果。
        """
        return ToolResult(
            call_id=inv.call_id,
            tool=inv.tool,
            ok=False,
            error="内部错误：该调用没有产生结果",
        )
