"""``Glob`` 工具与全项目文件遍历（§14.2）。

- 只返回**文件**（永不返回目录），按 mtime 从新到旧排序。
- 命中 ``.gitignore``（``pathspec`` 的 ``gitwildmatch`` 语义，含子目录 ``.gitignore``）。
- 内置忽略目录：``.git`` / ``node_modules`` / ``__pycache__`` / ``.venv`` / ``dist`` / ``build``；
  另外**永不搜索** ``.myagent/``（§7.4：落盘目录必须被自动忽略）。

:func:`iter_project_files` 同时被 :mod:`my_agent.tools.fs_grep` 复用，保证 Glob/Grep
的「可见文件集合」完全一致。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any

from pathspec import PathSpec

from my_agent.models.tools import ToolContext, ToolResult, ToolSpec
from my_agent.tools.backup import MYAGENT_DIR_NAME
from my_agent.tools.base import BaseTool, display_path, resolve_path, to_posix

__all__ = ["IGNORED_DIR_NAMES", "GlobTool", "iter_project_files", "matches_pattern"]

#: 内置忽略目录名（§14.2；可被 .gitignore 之外的规则补充，但不可删除）。
IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {".git", "node_modules", "__pycache__", ".venv", "dist", "build", MYAGENT_DIR_NAME}
)

_DESCRIPTION = """按 glob 模式查找文件路径（只返回文件，不返回目录），按修改时间从新到旧排序。

何时用：知道文件名或后缀但不确定在哪个目录时（如 **/*.py、test_*.py、src/**/*.ts）。
何时不用：① 不要用它搜文件内容——用 Grep；② 不要用它读文件内容——用 Read；
③ 不要指望它返回目录、.git/node_modules/.myagent 等被忽略目录里的文件。
副作用：无（只读，不修改磁盘文件）。"""


def _load_spec(directory: Path) -> PathSpec | None:
    """读取某个目录下的 ``.gitignore`` 并编译为 pathspec。

    Args:
        directory: 目录。

    Returns:
        编译好的 :class:`pathspec.PathSpec`；没有 ``.gitignore`` 或读取失败时 ``None``。
    """
    gitignore = directory / ".gitignore"
    if not gitignore.is_file():
        return None
    try:
        lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return PathSpec.from_lines("gitwildmatch", lines)


def _ignored(path: Path, specs: list[tuple[Path, PathSpec]], *, is_dir: bool) -> bool:
    """判断路径是否被任一 ``.gitignore`` 规则忽略。

    Args:
        path: 候选路径。
        specs: ``(gitignore 所在目录, 编译结果)`` 列表（从外到内）。
        is_dir: 候选是否为目录（目录会带尾斜杠参与匹配）。

    Returns:
        被忽略时 ``True``。
    """
    for root, spec in specs:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if spec.match_file(f"{relative}/" if is_dir else relative):
            return True
    return False


def _walk(directory: Path, specs: list[tuple[Path, PathSpec]]) -> Iterator[Path]:
    """递归遍历目录，产出未被忽略的文件。

    Args:
        directory: 当前目录。
        specs: 继承下来的 gitignore 规则。

    Yields:
        文件路径（不跟随符号链接目录，避免环）。
    """
    local_specs = specs
    loaded = _load_spec(directory)
    if loaded is not None:
        local_specs = [*specs, (directory, loaded)]
    try:
        children = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError:
        return
    for child in children:
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        if is_dir:
            if child.name in IGNORED_DIR_NAMES or child.is_symlink():
                continue
            if _ignored(child, local_specs, is_dir=True):
                continue
            yield from _walk(child, local_specs)
            continue
        if not child.is_file():
            continue
        if _ignored(child, local_specs, is_dir=False):
            continue
        yield child


def iter_project_files(base: Path) -> Iterator[Path]:
    """遍历 ``base`` 下的所有「可见」文件（已应用内置忽略与 .gitignore）。

    同步生成器：调用方需放进 ``asyncio.to_thread`` 消费，避免阻塞事件循环。

    Args:
        base: 起始目录。

    Yields:
        文件路径（含 ``base`` 目录自身的 ``.gitignore`` 规则影响）。
    """
    if base.is_file():
        yield base
        return
    if not base.is_dir():
        return
    yield from _walk(base, [])


def matches_pattern(relative: str, pattern: str) -> bool:
    """判断相对路径是否匹配 glob 模式。

    Args:
        relative: 相对起始目录的 posix 路径。
        pattern: 模型给出的 glob（``\\`` 会被归一化为 ``/``）。

    Returns:
        是否匹配。模式含 ``/`` 时对整条相对路径做 fnmatch；否则只匹配文件名。
    """
    normalized = pattern.replace("\\", "/").strip()
    if not normalized:
        return False
    if "/" in normalized:
        return fnmatch(relative, normalized)
    return fnmatch(PurePosixPath(relative).name, normalized)


class GlobTool(BaseTool):
    """按 glob 模式列文件（仅文件，按 mtime 倒序）。"""

    spec = ToolSpec(
        name="Glob",
        description=_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "glob 模式，如 **/*.py、src/**/*.ts、test_*.py",
                },
                "path": {
                    "type": "string",
                    "description": "搜索起始目录（默认 cwd）",
                },
            },
            "required": ["pattern"],
        },
        category="read",
        is_read_only=True,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """执行查找。

        Args:
            args: ``pattern``、可选 ``path``。
            ctx: 执行上下文（相对路径基于 ``ctx.cwd``；``path`` 缺省时用它）。

        Returns:
            ``ok=True`` 且 ``content`` 为相对路径列表（每行一个）；
            参数非法或目录不存在时 ``ok=False``。
        """
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            return self.fail(ctx, "缺少必填参数 pattern（应为 glob 模式字符串）")
        raw_base = args.get("path")
        if raw_base is not None and not isinstance(raw_base, str):
            return self.fail(ctx, "参数 path 非法：应为目录路径字符串")
        base = resolve_path(raw_base, ctx) if raw_base else Path(ctx.cwd).resolve()
        if not base.is_dir():
            return self.fail(ctx, f"搜索目录不存在或不是目录：{display_path(base, ctx)}")
        return await asyncio.to_thread(self._glob, base, pattern, ctx)

    def _glob(self, base: Path, pattern: str, ctx: ToolContext) -> ToolResult:
        """同步查找实现（由 :meth:`run` 放进线程执行）。

        Args:
            base: 起始目录。
            pattern: glob 模式。
            ctx: 执行上下文。

        Returns:
            查找结果。
        """
        hits: list[tuple[float, str]] = []
        for candidate in iter_project_files(base):
            relative = to_posix(candidate.relative_to(base))
            if not matches_pattern(relative, pattern):
                continue
            try:
                mtime = candidate.stat().st_mtime
            except OSError:
                mtime = 0.0
            hits.append((mtime, display_path(candidate, ctx)))
        if not hits:
            return self.ok(
                ctx,
                f"未找到匹配 {pattern} 的文件（搜索目录：{display_path(base, ctx)}）",
            )
        hits.sort(key=lambda item: (-item[0], item[1]))
        listing = "\n".join(path for _mtime, path in hits)
        return self.ok(ctx, f"{listing}\n\n（共 {len(hits)} 个文件，按修改时间从新到旧）")
