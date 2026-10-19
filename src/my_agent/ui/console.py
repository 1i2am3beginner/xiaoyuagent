"""rich Console 封装（§16.2、§16.3）。

硬性要求：
- 工具参数展示**必须**脱敏（复用 ``security.audit.redact``）；
- 颜色/emoji **必须**可关闭（``NO_COLOR`` 环境变量 + ``ui.emoji=false``）；
- 长输出折叠而非截断。
"""

from __future__ import annotations

import os
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from my_agent.models.permission import PermissionDecision, Risk
from my_agent.models.protocols import PermissionRequest

__all__ = ["RichConsole"]

RISK_STYLE: dict[Risk, str] = {
    "low": "green",
    "medium": "yellow",
    "high": "red",
    "critical": "bold red",
}
"""风隩徽章配色（§6.9 要求展示 risk 徽章）。"""
RISK_LABEL: dict[Risk, str] = {
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
    "critical": "严重风险",
}

NO_COLOR_ENV = "NO_COLOR"


class RichConsole:
    """rich 渲染的 ConsolePort 实现。"""

    def __init__(
        self,
        *,
        no_color: bool | None = None,
        emoji: bool = True,
        stream: bool = True,
        preview_lines: int = 10,
    ) -> None:
        """初始化。

        Args:
            no_color: 是否禁用颜色；None = 依据 ``NO_COLOR`` 环境变量自动判断。
            emoji: 是否允许 emoji（``ui.emoji=false`` 关闭）。
            stream: 是否流式输出文本增量。
            preview_lines: 工具结果折叠展示的行数。
        """
        if no_color is None:
            no_color = bool(os.environ.get(NO_COLOR_ENV))
        self.no_color = no_color
        self.emoji = emoji and not no_color
        self.stream_enabled = stream
        self.preview_lines = preview_lines
        self.console = Console(
            no_color=no_color,
            highlight=False,
            soft_wrap=False,
            emoji=self.emoji,
        )
        self._stream_open = False

    # ------------------------------------------------------------------ 图标
    def _icon(self, name: str, fallback: str) -> str:
        """按 emoji 开关返回图标文本。

        Args:
            name: emoji 字符。
            fallback: 关闭 emoji 时的等价 ASCII 记号。

        Returns:
            可用图标。
        """
        return name if self.emoji else fallback

    # ------------------------------------------------------------------ 输出
    def stream_text(self, delta: str) -> None:
        """输出流式文本增量。

        Args:
            delta: 文本增量。
        """
        if not self.stream_enabled or not delta:
            return
        self.console.print(delta, end="", markup=False, highlight=False)
        self._stream_open = True

    def _close_stream(self) -> None:
        if self._stream_open:
            self.console.print()
            self._stream_open = False

    def info(self, msg: str) -> None:
        """输出提示信息。

        Args:
            msg: 文本。
        """
        self._close_stream()
        self.console.print(msg, style=None, markup=False, highlight=False)

    def warn(self, msg: str) -> None:
        """输出警告。

        Args:
            msg: 文本。
        """
        self._close_stream()
        icon = self._icon("⚠️ ", "[!] ")
        self.console.print(f"{icon}{msg}", style="yellow", markup=False, highlight=False)

    def error(self, msg: str) -> None:
        """输出错误。

        Args:
            msg: 文本。
        """
        self._close_stream()
        icon = self._icon("✗ ", "[x] ")
        self.console.print(f"{icon}{msg}", style="red", markup=False, highlight=False)

    def print_markdown(self, text: str) -> None:
        """渲染 Markdown 文本。

        Args:
            text: Markdown 源文本。
        """
        self._close_stream()
        self.console.print(Markdown(text))

    def stream_done(self) -> None:
        """结束一次流式输出（补换行）。"""
        self._close_stream()
        self.console.print()

    # ------------------------------------------------------------------ 工具
    def tool_start(self, name: str, summary: str) -> None:
        """工具开始执行。

        Args:
            name: 工具名。
            summary: 关键参数摘要（**调用方须已脱敏**）。
        """
        self._close_stream()
        icon = self._icon("▸ ", "> ")
        self.console.print(
            Text(f"{icon}{name}", style="cyan") + Text(f"  {summary}", style="dim"),
            highlight=False,
        )

    def tool_result(self, name: str, ok: bool, summary: str) -> None:
        """工具执行结束（默认折叠展示，§16.2）。

        Args:
            name: 工具名。
            ok: 是否成功。
            summary: 结果摘要（**调用方须已脱敏**）。
        """
        self._close_stream()
        style = "green" if ok else "red"
        icon = self._icon("✓ ", "[ok] ") if ok else self._icon("✗ ", "[fail] ")
        lines = summary.splitlines()
        shown = lines[: self.preview_lines]
        body = "\n".join(f"    {line}" for line in shown)
        extra = len(lines) - len(shown)
        text = Text(f"  {icon}", style=style) + Text(body)
        if extra > 0:
            text.append(f"\n    … 另有 {extra} 行（输出过长时已落盘，可用 Read/Grep 回读）", style="dim")
        self.console.print(text, highlight=False)

    # ------------------------------------------------------------------ 卡片
    def render_permission_card(self, request: PermissionRequest) -> None:
        """渲染权限询问卡片（§6.9）。

        展示：工具名、参数（脱敏 + 截断）、risk 徽章、理由、判定阶段来源，
        以及按 ``permanent_eligible`` 动态生成的选项。

        Args:
            request: 询问卡片内容。
        """
        from my_agent.security.audit import redact

        self._close_stream()
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold")
        table.add_column()

        badge = RISK_LABEL.get(request.risk, request.risk)
        style = RISK_STYLE.get(request.risk, "yellow")
        table.add_row("工具", request.tool)
        if request.subagent_id:
            table.add_row("来源", f"[子 Agent {request.subagent_id}]")
        table.add_row("风险", Text(badge, style=style))
        table.add_row("判定阶段", f"phase {request.phase_reached}")
        table.add_row("理由", request.reason or "（模型未给出理由）")
        args = redact(dict(request.arguments))
        table.add_row("参数", _short(_format_args(args), 1200))
        if request.preview:
            table.add_row("预览", _short(request.preview, 1200))

        options = self._options(request)
        table.add_row("选项", options)
        if not request.permanent_eligible:
            table.add_row("", Text("该操作在任何模式下都需确认", style="dim"))

        title = self._icon("🔐 需要确认", "[?] 需要确认")
        self.console.print(Panel(table, title=title, border_style=style))

    def _options(self, request: PermissionRequest) -> str:
        """生成选项行（按 permanent_eligible 动态生成，§6.9）。

        Args:
            request: 询问卡片。

        Returns:
            选项说明文本。
        """
        parts = ["[y] 允许一次"]
        if request.permanent_eligible:
            parts.append("[a] 永久允许（本会话）")
        parts.extend(["[n] 拒绝", "[r] 拒绝并说明", "[q] 中止本轮"])
        return "  ".join(parts)

    def render_decision(self, decision: PermissionDecision) -> None:
        """渲染一条裁决解释（审计可见性）。

        Args:
            decision: 裁决结果。
        """
        style = "green" if decision.final == "allow" else "red"
        self.console.print(Text(f"  → {decision.explain()}", style=style), highlight=False)

    # ------------------------------------------------------------------ 报告
    def render_context_report(self, report: Any) -> None:
        """渲染 ``/context`` 报告。

        Args:
            report: 已格式化的报告文本。
        """
        self._close_stream()
        self.console.print(str(report), markup=False, highlight=False)

    def render_cost(self, text: str) -> None:
        """渲染 ``/cost`` 报告。

        Args:
            text: 报告文本。
        """
        self._close_stream()
        self.console.print(text, markup=False, highlight=False)

    def render_status(self, line: str) -> None:
        """渲染顶部状态行（§16.2）。

        Args:
            line: 状态行文本。
        """
        self.console.print(Text(line, style="dim"), highlight=False)

    def render_help(self, text: str) -> None:
        """渲染 ``/help`` 输出。

        Args:
            text: 帮助文本。
        """
        self.console.print(text, markup=False, highlight=False)

    def rule(self) -> None:
        """输出一条分隔线。"""
        self.console.print(Text("─" * 60, style="dim"))


def _format_args(args: dict[str, Any]) -> str:
    """把参数字典压成紧凑的展示文本。

    Args:
        args: 已脱敏的参数。

    Returns:
        每行一个键值对的文本。
    """
    import json

    lines: list[str] = []
    for key, value in args.items():
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines) or "（无参数）"


def _short(text: str, limit: int) -> str:
    """截断过长文本并标注。

    Args:
        text: 原文。
        limit: 最大字符数。

    Returns:
        截断后的文本。
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…[已截断 {len(text) - limit} 字符]"
