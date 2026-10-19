"""交互层（§16）：rich 渲染、权限问答、diff 预览。"""

from __future__ import annotations

from my_agent.ui.console import RichConsole
from my_agent.ui.prompt import TerminalConfirmer, read_choice, read_line
from my_agent.ui.render import (
    diff_preview,
    permission_preview,
    summarize_arguments,
    tool_result_summary,
    write_preview,
)

__all__ = [
    "RichConsole",
    "TerminalConfirmer",
    "diff_preview",
    "permission_preview",
    "read_choice",
    "read_line",
    "summarize_arguments",
    "tool_result_summary",
    "write_preview",
]
