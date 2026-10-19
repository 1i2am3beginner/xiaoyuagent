"""消息 / diff / 工具结果渲染（§16.3）。

- 工具参数展示**必须**脱敏（复用 :func:`my_agent.security.audit.redact`）；
- 权限询问卡中的文件写入**必须**展示 diff 预览（``Write`` 新建展示前 20 行；
  ``Edit`` 展示 old/new 对比）；
- 长输出必须折叠而非截断；颜色/emoji 必须可关闭。
"""

from __future__ import annotations

import difflib
from typing import Any

from my_agent.models.messages import Message
from my_agent.models.tools import ToolResult

__all__ = [
    "WRITE_PREVIEW_LINES",
    "diff_preview",
    "render_message_markdown",
    "summarize_arguments",
    "tool_result_summary",
    "write_preview",
]

WRITE_PREVIEW_LINES = 20
"""``Write`` 新建时展示的前 N 行（§16.3）。"""

_MAX_ARG_CHARS = 160


def summarize_arguments(tool: str, arguments: dict[str, Any]) -> str:
    """生成工具调用的一行关键参数摘要（已脱敏）。

    Args:
        tool: 工具名。
        arguments: 原始参数。

    Returns:
        单行摘要（超长截断）。
    """
    from my_agent.security.audit import redact

    safe = redact(dict(arguments))
    preferred = (
        "file_path",
        "path",
        "pattern",
        "command",
        "url",
        "query",
        "name",
        "id",
        "notebook_path",
    )
    parts: list[str] = []
    for key in preferred:
        if key in safe:
            parts.append(f"{key}={_flat(safe[key])}")
    if not parts:
        for key, value in list(safe.items())[:3]:
            parts.append(f"{key}={_flat(value)}")
    summary = " ".join(parts) or "（无参数）"
    return summary if len(summary) <= _MAX_ARG_CHARS else summary[:_MAX_ARG_CHARS] + "…"


def _flat(value: Any) -> str:
    """把参数值压成单行短文本。

    Args:
        value: 任意参数值。

    Returns:
        单行文本。
    """
    if isinstance(value, str):
        collapsed = " ".join(value.split())
        return collapsed if len(collapsed) <= 80 else collapsed[:80] + "…"
    return str(value)


def tool_result_summary(result: ToolResult, *, limit: int = 10) -> str:
    """生成工具结果的折叠摘要（长输出折叠而非截断）。

    Args:
        result: 工具结果。
        limit: 展示行数。

    Returns:
        折叠后的文本。
    """
    body = result.content if result.ok else (result.error or "未知错误")
    lines = body.splitlines() or [""]
    shown = lines[:limit]
    if len(lines) > limit:
        shown.append(f"… 另有 {len(lines) - limit} 行")
    if result.artifact_path:
        shown.append(f"[完整输出已落盘：{result.artifact_path}]")
    return "\n".join(shown)


def diff_preview(
    file_path: str,
    old_string: str,
    new_string: str,
    *,
    context_lines: int = 3,
) -> str:
    """生成 ``Edit`` 的 old/new 对比预览（§16.3）。

    Args:
        file_path: 目标文件路径。
        old_string: 被替换文本。
        new_string: 替换后的文本。
        context_lines: 统一 diff 的上下文行数。

    Returns:
        统一 diff 文本。
    """
    old_lines = old_string.splitlines(keepends=True)
    new_lines = new_string.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        n=context_lines,
    )
    text = "".join(diff).strip()
    return text or "（替换后内容一致）"


def write_preview(file_path: str, content: str, *, limit: int = WRITE_PREVIEW_LINES) -> str:
    """生成 ``Write`` 的前 N 行预览（新建文件场景）。

    Args:
        file_path: 目标文件路径。
        content: 待写入内容。
        limit: 展示行数。

    Returns:
        形如 ``+++ b/<path>`` 的预览文本。
    """
    lines = content.splitlines()
    shown = lines[:limit]
    body = "\n".join(f"+{line}" for line in shown)
    tail = f"\n… 另有 {len(lines) - limit} 行" if len(lines) > limit else ""
    return f"+++ b/{file_path}\n{body}{tail}"


def permission_preview(tool: str, arguments: dict[str, Any]) -> str | None:
    """按工具类型生成权限卡的预览文本（§16.3 要求写操作展示 diff）。

    Args:
        tool: 工具名。
        arguments: 调用参数。

    Returns:
        预览文本；不适用时返回 ``None``。
    """
    path = str(arguments.get("file_path") or arguments.get("path") or arguments.get("notebook_path") or "")
    if tool == "Write" and path:
        return write_preview(path, str(arguments.get("content", "")))
    if tool == "Edit" and path:
        return diff_preview(
            path, str(arguments.get("old_string", "")), str(arguments.get("new_string", ""))
        )
    if tool == "MultiEdit" and path:
        edits = arguments.get("edits") or []
        chunks: list[str] = []
        for index, edit in enumerate(edits if isinstance(edits, list) else []):
            if isinstance(edit, dict):
                chunks.append(
                    f"# 编辑 {index + 1}\n"
                    + diff_preview(
                        path,
                        str(edit.get("old_string", "")),
                        str(edit.get("new_string", "")),
                    )
                )
        return "\n".join(chunks) or None
    return None


def render_message_markdown(message: Message) -> str:
    """把消息渲染为可读 Markdown（``/session export`` 与调试用）。

    Args:
        message: 消息。

    Returns:
        Markdown 文本。
    """
    header = f"### {message.role}"
    if message.name:
        header += f" ({message.name})"
    if message.kind == "cleared":
        return f"{header}\n\n> {message.cleared_marker or 'Old tool result content cleared'}"
    body = message.content or ""
    if message.tool_calls:
        calls = "\n".join(
            f"- `{call.name}` {call.arguments}" for call in message.tool_calls
        )
        body = f"{body}\n\n{calls}".strip()
    return f"{header}\n\n{body}".strip()
