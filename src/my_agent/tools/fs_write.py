"""``Write`` 工具：全量写入文件，写前自动备份（§14.2、§14.5）。

行为要点：
- 全量覆盖；父目录不存在时自动创建（§14.2「新建需父目录存在或自动创建」）。
- **写前备份**：调用 :meth:`~my_agent.tools.backup.BackupManager.snapshot`，
  备份路径随结果返回，可由 ``/undo`` 撤销。
- 原子写入（临时文件 + ``os.replace``），避免磁盘满/中断留下半截文件。
- 返回**简短确认**（路径、行数、字节数、备份位置），绝不回显整段内容。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from my_agent.models.tools import ToolContext, ToolResult, ToolSpec
from my_agent.tools.backup import BackupManager
from my_agent.tools.base import (
    BaseTool,
    atomic_write_text,
    count_lines,
    display_path,
    resolve_path,
)

__all__ = ["WriteTool"]

_DESCRIPTION = """把内容全量写入指定文件：文件已存在则整体覆盖，不存在则新建（父目录自动创建）。

何时用：新建文件，或需要整文件重写时。
何时不用：① 只想改几行时不要用它——用 Edit（否则会把文件中模型没看到的其它内容一起覆盖掉）；
② 不要用它写二进制内容；③ 不要用它把未变化的内容原样写回（无意义的写会污染文件时间戳与备份）。
副作用：**会修改磁盘文件**；写入前原文件自动备份到 .myagent/backups/，可用 /undo 撤销。"""


class WriteTool(BaseTool):
    """全量写入文件（写前备份）。

    Args:
        backups: 备份管理器（§14.5）；``.myagent/`` 内部文件不会被备份。
    """

    spec = ToolSpec(
        name="Write",
        description=_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文件路径（绝对或相对 cwd）"},
                "content": {"type": "string", "description": "要写入的完整文件内容"},
            },
            "required": ["file_path", "content"],
        },
        category="write",
        is_read_only=False,
        is_destructive=True,
        requires_user_approval_hint=True,
    )

    def __init__(self, backups: BackupManager) -> None:
        self.backups = backups

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """写入文件。

        Args:
            args: ``file_path``、``content``。
            ctx: 执行上下文（相对路径基于 ``ctx.cwd``）。

        Returns:
            ``ok=True`` 且 ``content`` 为简短确认；参数非法或写入失败时 ``ok=False``。
        """
        raw_path = args.get("file_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return self.fail(ctx, "缺少必填参数 file_path（应为文件路径字符串）")
        content = args.get("content")
        if not isinstance(content, str):
            return self.fail(ctx, "缺少必填参数 content（应为字符串）；清空文件请传空字符串")

        path = resolve_path(raw_path, ctx)
        if path.is_dir():
            return self.fail(ctx, f"{display_path(path, ctx)} 是目录，无法写入文件")
        return await asyncio.to_thread(self._write, path, content, ctx)

    def _write(self, path: Path, content: str, ctx: ToolContext) -> ToolResult:
        """同步写入实现（由 :meth:`run` 放进线程执行）。

        Args:
            path: 已解析的绝对路径。
            content: 完整文件内容。
            ctx: 执行上下文。

        Returns:
            写入结果。

        Raises:
            无：所有 ``OSError`` 都转成 ``ok=False`` 的结果。
        """
        shown = display_path(path, ctx)
        existed = path.is_file()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            backup = self.backups.snapshot(path, tool=self.spec.name, call_id=ctx.call_id)
            size = atomic_write_text(path, content)
        except OSError as exc:
            return self.fail(ctx, f"写入失败：{shown}（{exc}）")
        self.backups.mark_written(path, call_id=ctx.call_id)

        lines = count_lines(content)
        if not existed:
            return self.ok(ctx, f"已新建并写入 {shown}（{lines} 行，{size} 字节）")
        if backup is None:
            return self.ok(
                ctx, f"已覆盖写入 {shown}（{lines} 行，{size} 字节；内部文件未做备份）"
            )
        return self.ok(
            ctx,
            f"已覆盖写入 {shown}（{lines} 行，{size} 字节）；原文件已备份，可用 /undo 撤销",
        )
