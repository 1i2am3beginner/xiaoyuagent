"""权限询问交互（§6.9）。

询问卡片必须展示：工具名、参数（长值截断）、risk 徽章、reason、**基于哪个阶段**的判定。

非交互模式（``--print``）下遇到 ``ask``：视为 ``reject``，结果中附「非交互模式下无法
询问用户」；**绝不默认 allow**。
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING

from my_agent.models.protocols import PermissionRequest, UserAnswer

if TYPE_CHECKING:
    from my_agent.ui.console import RichConsole

__all__ = ["TerminalConfirmer", "read_choice"]

_LOG = logging.getLogger(__name__)

PROMPT_HINT = "请选择 [y/a/n/r/q]: "
NOTE_PROMPT = "拒绝理由（回车跳过）: "
NON_INTERACTIVE_NOTE = "非交互模式下无法询问用户"

CHOICE_MAP: dict[str, str] = {
    "y": "once",
    "yes": "once",
    "a": "always",
    "always": "always",
    "n": "reject",
    "no": "reject",
    "r": "reject_with_reason",
    "q": "abort_turn",
    "quit": "abort_turn",
}


class TerminalConfirmer:
    """终端询问回调（实现 :class:`~my_agent.models.protocols.Confirmer`）。"""

    def __init__(
        self,
        console: RichConsole | None = None,
        *,
        interactive: bool = True,
        default_note: str = "用户拒绝",
    ) -> None:
        """初始化。

        Args:
            console: rich 控制台（用于渲染卡片）。
            interactive: 是否可交互；False 时遇 ask 一律 reject。
            default_note: 未填写理由时的默认拒绝理由。
        """
        self.console = console
        self.interactive = interactive
        self.default_note = default_note

    async def ask(self, request: PermissionRequest) -> UserAnswer:
        """展示询问卡片并等待用户回答。

        Args:
            request: 询问卡片内容。

        Returns:
            用户的回答；非交互模式下固定为 ``reject_with_reason``。
        """
        if not self.interactive or not sys.stdin.isatty():
            if self.console is not None:
                self.console.warn(
                    f"⚠ {NON_INTERACTIVE_NOTE}，已按最保守策略拒绝：{request.tool}"
                )
            return UserAnswer(choice="reject_with_reason", note=NON_INTERACTIVE_NOTE)

        if self.console is not None:
            self.console.render_permission_card(request)

        choice, note = await asyncio.to_thread(read_choice, request.permanent_eligible)
        if choice == "reject_with_reason" and not note:
            note = self.default_note
        return UserAnswer(choice=choice, note=note)  # type: ignore[arg-type]


def read_choice(permanent_eligible: bool) -> tuple[str, str | None]:
    """同步读取用户选择（供 ``asyncio.to_thread`` 调用）。

    Args:
        permanent_eligible: 是否允许「永久放行」（False 时不接受 ``a``）。

    Returns:
        形如 ``("once", None)`` 的 ``(choice, note)``。

    Raises:
        KeyboardInterrupt: 透传（由上层转 abort_turn）。
    """
    while True:
        try:
            raw = input(PROMPT_HINT).strip().lower()
        except EOFError:
            return "reject_with_reason", NON_INTERACTIVE_NOTE
        if not raw:
            continue
        choice = CHOICE_MAP.get(raw)
        if choice is None:
            print("无效输入。可用：y=允许一次 / a=永久允许 / n=拒绝 / r=拒绝并说明 / q=中止本轮")
            continue
        if choice == "always" and not permanent_eligible:
            print("该操作在任何模式下都需逐次确认，无法永久放行。请选择 y / n / r / q。")
            continue
        if choice == "reject_with_reason":
            try:
                note = input(NOTE_PROMPT).strip()
            except EOFError:
                note = ""
            return choice, note or None
        return choice, None


async def read_line(prompt: str = "› ") -> str:
    """读取一行用户输入（REPL 用，支持 Ctrl-C/Ctrl-D 语义）。

    Args:
        prompt: 提示符。

    Returns:
        用户输入；EOF 时返回空字符串。

    Raises:
        KeyboardInterrupt: 用户在输入态按下 Ctrl-C。
    """
    try:
        return await asyncio.to_thread(input, prompt)
    except EOFError:
        return ""
