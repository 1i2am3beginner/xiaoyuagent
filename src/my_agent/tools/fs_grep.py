"""``Grep`` 工具：纯 Python 的 ripgrep 语义内容搜索（§14.2）。

- 不依赖 ``rg`` 可执行文件：用 :mod:`re` 逐行匹配（与 ripgrep 默认行为一致，按行匹配）。
- 支持 ``path``（目录或单个文件）与 ``include``（文件名 glob 过滤）。
- 输出 ``file:line: 文本``；命中数超过
  :data:`~my_agent.config.defaults.GREP_MAX_INLINE_MATCHES` 时报告总数 + 前 N 条 + 收窄建议
  （大结果落盘由 §7.4 的 ``context/preview.py`` 统一处理）。
- 二进制文件、被 ``.gitignore`` 忽略的文件、``.myagent/`` 内部文件一律跳过。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from my_agent.config.defaults import GREP_MAX_INLINE_MATCHES
from my_agent.models.tools import ToolContext, ToolResult, ToolSpec
from my_agent.tools.base import (
    BaseTool,
    decode_text,
    display_path,
    is_binary_data,
    resolve_path,
    to_posix,
)
from my_agent.tools.fs_glob import iter_project_files, matches_pattern

__all__ = ["GrepTool"]

_DESCRIPTION = """用正则表达式搜索文件内容，返回 `file:line: 文本`（纯 Python 实现，不需要 rg 可执行文件）。

何时用：按内容定位代码（函数名、报错文本、TODO、配置项等），再用 Read 精读命中行。
何时不用：① 不要用它按文件名找文件——用 Glob；② 不要用它做替换/重构——用 Edit；
③ 命中过多时不要反复重跑同样的模式——用 path/include 收窄范围或写更精确的正则。
副作用：无（只读，不修改磁盘文件）。命中超过 250 处时只返回总数与前 250 条。"""


class GrepTool(BaseTool):
    """正则内容搜索（ripgrep 语义，纯 Python 实现）。"""

    spec = ToolSpec(
        name="Grep",
        description=_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python 正则表达式（按行匹配）"},
                "path": {
                    "type": "string",
                    "description": "搜索起始目录或单个文件（默认 cwd）",
                },
                "include": {
                    "type": "string",
                    "description": "文件名 glob 过滤，如 *.py、test_*.py",
                },
            },
            "required": ["pattern"],
        },
        category="read",
        is_read_only=True,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """执行搜索。

        Args:
            args: ``pattern``、可选 ``path``、可选 ``include``。
            ctx: 执行上下文（相对路径基于 ``ctx.cwd``）。

        Returns:
            ``ok=True`` 且 ``content`` 为 ``file:line: 文本`` 列表；
            正则非法、路径不存在时 ``ok=False``。
        """
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            return self.fail(ctx, "缺少必填参数 pattern（应为正则表达式字符串）")
        raw_path = args.get("path")
        if raw_path is not None and not isinstance(raw_path, str):
            return self.fail(ctx, "参数 path 非法：应为目录或文件路径字符串")
        include = args.get("include")
        if include is not None and not isinstance(include, str):
            return self.fail(ctx, "参数 include 非法：应为文件名 glob 字符串，如 *.py")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return self.fail(ctx, f"正则表达式非法：{pattern}（{exc}）")

        base = resolve_path(raw_path, ctx) if raw_path else Path(ctx.cwd).resolve()
        if not base.exists():
            return self.fail(ctx, f"搜索路径不存在：{display_path(base, ctx)}")
        return await asyncio.to_thread(self._grep, base, regex, include, pattern, ctx)

    def _grep(
        self,
        base: Path,
        regex: re.Pattern[str],
        include: str | None,
        pattern: str,
        ctx: ToolContext,
    ) -> ToolResult:
        """同步搜索实现（由 :meth:`run` 放进线程执行）。

        Args:
            base: 起始目录或文件。
            regex: 已编译的正则。
            include: 文件名 glob（``None`` = 全部）。
            pattern: 原始正则（用于提示文案）。
            ctx: 执行上下文。

        Returns:
            搜索结果。
        """
        lines: list[str] = []
        total = 0
        scanned = 0
        truncated = False
        for candidate in iter_project_files(base):
            relative = to_posix(candidate.relative_to(base)) if base.is_dir() else candidate.name
            if include and not matches_pattern(relative, include):
                continue
            scanned += 1
            try:
                data = candidate.read_bytes()
            except OSError:
                continue
            if is_binary_data(data):
                continue
            try:
                text = decode_text(data)
            except UnicodeDecodeError:
                continue
            shown = display_path(candidate, ctx)
            for number, line in enumerate(text.splitlines(), start=1):
                if not regex.search(line):
                    continue
                total += 1
                if len(lines) < GREP_MAX_INLINE_MATCHES:
                    lines.append(f"{shown}:{number}: {line}")
                else:
                    truncated = True

        if total == 0:
            return self.ok(
                ctx,
                f"未找到匹配 {pattern} 的内容（搜索位置：{display_path(base, ctx)}，"
                f"扫描文件 {scanned} 个）",
            )
        header = f"共 {total} 处命中（扫描文件 {scanned} 个）"
        body = "\n".join(lines)
        if truncated:
            return self.ok(
                ctx,
                f"{header}，仅显示前 {GREP_MAX_INLINE_MATCHES} 处：\n{body}\n\n"
                f"（结果已截断：命中过多。请用 path 限定目录、用 include 过滤后缀，"
                f"或写更精确的正则以减少命中）",
                truncated=True,
            )
        return self.ok(ctx, f"{header}：\n{body}")
