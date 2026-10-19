"""``Read`` 工具：读取文本文件内容并返回带行号的输出（§14.2）。

行为要点：
- 行号格式 ``{行号:>6}\\t{原文}``（对齐本环境的 ``cat -n`` 约定）。
- 默认只返回前 :data:`~my_agent.config.defaults.READ_DEFAULT_LIMIT` 行，未读完时给出
  「继续读取请用 offset=N」的提示。
- 二进制文件（头部含 NUL 或无法按候选编码解码）一律拒绝，并给出可操作建议。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from my_agent.config.defaults import READ_DEFAULT_LIMIT
from my_agent.models.tools import ToolContext, ToolResult, ToolSpec
from my_agent.tools.base import (
    BaseTool,
    as_positive_int,
    display_path,
    is_binary_data,
    read_text_normalized,
    resolve_path,
)

__all__ = ["ReadTool"]

_DESCRIPTION = """读取文本文件内容，返回带行号的输出（行号 + TAB + 原文）。

何时用：需要查看某个具体文件的内容、核对已有代码或确认符号定义时。
何时不用：① 不要用它读二进制文件（可执行文件、图片、压缩包、.pyc）——会被直接拒绝；
② 不要用它整读超大文件（默认只返回前 2000 行），先用 Grep 定位行号再用 offset/limit 精读；
③ 不要用它列目录或按文件名找文件——用 Glob；④ 不要用它搜内容——用 Grep。
副作用：无（只读，不修改磁盘文件）。
参数：file_path 必填；offset 为起始行号（从 1 开始，默认 1）；limit 为最多返回行数（默认 2000，上限 2000）。"""


class ReadTool(BaseTool):
    """读取文本文件并输出带行号的内容（§14.2）。"""

    spec = ToolSpec(
        name="Read",
        description=_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文件路径（绝对或相对 cwd）"},
                "offset": {"type": "integer", "description": "起始行号，从 1 开始，默认 1"},
                "limit": {
                    "type": "integer",
                    "description": f"最多返回行数，默认 {READ_DEFAULT_LIMIT}",
                },
            },
            "required": ["file_path"],
        },
        category="read",
        is_read_only=True,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """读取文件内容。

        Args:
            args: ``file_path``、可选 ``offset`` / ``limit``。
            ctx: 执行上下文（相对路径基于 ``ctx.cwd``）。

        Returns:
            ``ok=True`` 且 ``content`` 为带行号文本（附继续读取提示）；
            参数非法、文件不存在、是目录或疑似二进制时返回 ``ok=False``。
        """
        raw_path = args.get("file_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return self.fail(ctx, "缺少必填参数 file_path（应为文件路径字符串）")
        offset = as_positive_int(args.get("offset"), 1)
        if offset is None:
            return self.fail(ctx, "参数 offset 非法：应为 ≥1 的整数（起始行号）")
        limit = as_positive_int(args.get("limit"), READ_DEFAULT_LIMIT)
        if limit is None:
            return self.fail(ctx, "参数 limit 非法：应为 ≥1 的整数（最多返回行数）")
        limit = min(limit, READ_DEFAULT_LIMIT)

        path = resolve_path(raw_path, ctx)
        return await asyncio.to_thread(self._read, path, offset, limit, ctx)

    def _read(self, path: Path, offset: int, limit: int, ctx: ToolContext) -> ToolResult:
        """同步读取实现（由 :meth:`run` 放进线程执行）。

        Args:
            path: 已解析的绝对路径。
            offset: 起始行号（1 起）。
            limit: 最多返回行数。
            ctx: 执行上下文。

        Returns:
            读取结果。
        """
        shown = display_path(path, ctx)
        if path.is_dir():
            return self.fail(ctx, f"{shown} 是目录，不是文件；列出目录请用 Glob 或 Bash")
        if not path.is_file():
            return self.fail(ctx, f"文件不存在：{shown}")
        try:
            data = path.read_bytes()
        except OSError as exc:
            return self.fail(ctx, f"读取失败：{shown}（{exc}）")
        if is_binary_data(data):
            return self.fail(
                ctx,
                f"拒绝读取二进制文件：{shown}（检测到 NUL 字节）。"
                "如需处理二进制内容，请改用 Bash 的专用命令（如 file/xxd）。",
            )
        try:
            text, _newline = read_text_normalized(path)
        except (UnicodeDecodeError, OSError) as exc:
            return self.fail(ctx, f"无法按文本解码 {shown}（疑似二进制或非常用编码）：{exc}")

        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        total = len(lines)
        if total == 0:
            return self.ok(ctx, f"（文件为空：{shown}）")
        if offset > total:
            return self.fail(
                ctx,
                f"offset={offset} 超出文件范围：{shown} 共 {total} 行（可用 offset=1 重新读取）",
            )

        start = offset - 1
        end = min(start + limit, total)
        window = lines[start:end]
        numbered = "\n".join(
            f"{number:>6}\t{line}" for number, line in enumerate(window, start=offset)
        )
        if end < total:
            numbered += (
                f"\n\n（文件共 {total} 行，本次显示第 {offset}-{end} 行；"
                f"继续读取请用 Read file_path={shown} offset={end + 1}）"
            )
        return self.ok(ctx, numbered, truncated=end < total)
