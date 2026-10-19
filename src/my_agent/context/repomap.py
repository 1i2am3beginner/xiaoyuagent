"""工作目录摘要（§7.3）。

**触发时机**：``QueryEngine.start()`` 时若 ``cwd`` 与上次记录不一致（或摘要文件不存在/过期）
则生成；``/init`` 可强制刷新（``force=True``）。

**两级生成（成本可控）**：

- **L1 确定性扫描**（必做，不调模型）：遍历目录（遵守 ``.gitignore`` + 内置忽略），
  限深度 ``REPOMAP_MAX_DEPTH``、限文件数 ``REPOMAP_MAX_FILES``，产出目录骨架与「包」
  （``pyproject.toml`` / ``package.json`` / ``go.mod`` / ``Cargo.toml`` 所在目录）；
- **L2 语义摘要**（调模型，角色 ``repomap``）：每个包取 README 前 1.5k + 入口文件前 3 个
  （各前 1k）+ 包元数据 → 每包**一行**说明；包数超过 ``REPOMAP_MAX_PACKAGES`` 的部分标「（未摘要）」。

**缓存**：``<project>/.myagent/repomap.md`` + ``repomap.meta.json``
（``{cwd, generated_at, tree_hash, file_count, llm_model}``）；
**失效**：``tree_hash`` 变化（目录结构级，sha1 of sorted dir names）或超过 ``REPOMAP_TTL_HOURS``。
命中缓存时**零模型调用**（C-11）。

所有文件 IO 经 :func:`asyncio.to_thread`，不阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import hashlib
import json
import os
import time
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from my_agent.config.defaults import (
    REPOMAP_BUDGET,
    REPOMAP_MAX_DEPTH,
    REPOMAP_MAX_FILES,
    REPOMAP_MAX_PACKAGES,
    REPOMAP_TTL_HOURS,
)
from my_agent.context.budget import truncate_to_tokens
from my_agent.errors import MyAgentError
from my_agent.llm.prompts import REPOMAP_PROMPT

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期导入
    from my_agent.config.schema import Config
    from my_agent.llm.client import LLMClient
    from my_agent.llm.tokenizer import TokenCounter

try:  # pragma: no cover - 依赖缺失时退化到「仅内置忽略」
    import pathspec

    _HAS_PATHSPEC = True
except ImportError:  # pragma: no cover
    pathspec = None  # type: ignore[assignment]
    _HAS_PATHSPEC = False

__all__ = [
    "IGNORED_DIR_NAMES",
    "MYAGENT_DIR_NAME",
    "PACKAGE_MANIFESTS",
    "REPOMAP_CACHE_NAME",
    "REPOMAP_META_NAME",
    "UNSUMMARIZED",
    "RepoMap",
    "RepoMapMeta",
    "TreeScan",
]

#: 项目内配置目录（§7.3 缓存位置）。
MYAGENT_DIR_NAME: str = ".myagent"
#: 人类可读缓存文件名（§7.3）。
REPOMAP_CACHE_NAME: str = "repomap.md"
#: 元数据文件名（§7.3）。
REPOMAP_META_NAME: str = "repomap.meta.json"

#: 包识别清单（§7.3 L1）。
PACKAGE_MANIFESTS: tuple[str, ...] = ("pyproject.toml", "package.json", "go.mod", "Cargo.toml")
#: 内置忽略目录（§7.3 + §7.4「``.myagent`` 自动加入忽略列表」）。
IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {".git", "node_modules", "__pycache__", ".venv", "dist", "build", MYAGENT_DIR_NAME}
)
#: 内置忽略文件模式（§7.3：``*.lock``）。
IGNORED_FILE_PATTERNS: tuple[str, ...] = ("*.lock",)
#: README 候选名（§7.3 L2）。
README_NAMES: tuple[str, ...] = ("README.md", "README.rst", "README.txt", "readme.md")
#: 入口文件候选名（§7.3 L2）。
ENTRY_FILE_CANDIDATES: tuple[str, ...] = (
    "__init__.py",
    "main.py",
    "cli.py",
    "app.py",
    "index.ts",
    "index.js",
    "main.go",
    "lib.rs",
)

#: README 摘录字符数（§7.3 L2）。
README_EXCERPT_CHARS: int = 1_500
#: 入口文件摘录字符数（§7.3 L2）。
ENTRY_EXCERPT_CHARS: int = 1_000
#: 每个包最多取几个入口文件（§7.3 L2）。
MAX_ENTRY_FILES: int = 3
#: 骨架文本最多渲染多少行（成本保护，非行为阈值）。
TREE_MAX_LINES: int = 600
#: 未做 L2 摘要的标注（§7.3 逐字）。
UNSUMMARIZED: str = "（未摘要）"
#: 每秒小时数（TTL 换算）。
SECONDS_PER_HOUR: int = 3_600
#: 原子写临时后缀。
_TMP_SUFFIX: str = ".tmp"


class RepoMapMeta(BaseModel):
    """``repomap.meta.json``（§7.3 缓存元数据）。"""

    cwd: str
    generated_at: float = Field(default_factory=time.time)
    tree_hash: str = ""
    file_count: int = 0
    llm_model: str = ""


class TreeScan(BaseModel):
    """L1 扫描结果（确定性、无模型）。"""

    dirs: list[str] = Field(default_factory=list, description="相对目录路径（posix，字典序）")
    files: list[str] = Field(default_factory=list, description="相对文件路径（posix）")
    manifests: dict[str, list[str]] = Field(
        default_factory=dict, description="包目录（相对路径，根为 ''）→ manifest 文件名"
    )
    file_count: int = 0
    truncated: bool = False
    ignored: list[str] = Field(default_factory=list, description="实际命中的内置忽略目录名")
    tree_hash: str = ""


class RepoMap:
    """工作目录摘要的生成与缓存（§7.3）。

    Args:
        config: 全局配置（当前版本无 repomap 专属配置项，保留以便后续扩展）。
        counter: token 计数器（用于骨架/包信息与预算截断）。
    """

    def __init__(self, *, config: Config, counter: TokenCounter | None = None) -> None:
        self._config = config
        self._counter = counter

    async def load_or_generate(
        self, *, cwd: Path, llm: LLMClient | None = None, force: bool = False
    ) -> str:
        """返回可注入上下文的工作目录摘要（命中缓存则零模型调用）。

        流程：L1 扫描 → 比 ``tree_hash`` / TTL / ``cwd`` → 命中则直接读
        ``repomap.md``；否则 L2 生成（``llm`` 为 ``None`` 或调用失败时退化为
        「（未摘要）」的 L1 文本）→ 原子写缓存 + 元数据。

        Args:
            cwd: 当前工作目录。
            llm: 模型客户端（角色 ``repomap``）；``None`` 表示只做 L1（不调模型）。
            force: ``True`` 时忽略缓存强制重新生成（``/init``）。

        Returns:
            §7.3 规定格式的 Markdown 摘要文本。
        """
        work_dir = Path(cwd)
        root = _project_root(work_dir)
        scan = await asyncio.to_thread(_scan_tree, root)
        cache_path = root / MYAGENT_DIR_NAME / REPOMAP_CACHE_NAME
        meta_path = root / MYAGENT_DIR_NAME / REPOMAP_META_NAME
        meta = await asyncio.to_thread(_read_meta, meta_path)
        if (
            not force
            and meta is not None
            and meta.cwd == str(work_dir)
            and meta.tree_hash == scan.tree_hash
            and _fresh(meta)
        ):
            cached = await asyncio.to_thread(_read_text, cache_path)
            if cached is not None:
                return cached
        text, model_label = await self._generate(
            root=root, work_dir=work_dir, scan=scan, llm=llm
        )
        await asyncio.to_thread(_atomic_write, cache_path, text)
        meta_out = RepoMapMeta(
            cwd=str(work_dir),
            generated_at=time.time(),
            tree_hash=scan.tree_hash,
            file_count=scan.file_count,
            llm_model=model_label,
        )
        await asyncio.to_thread(
            _atomic_write, meta_path, meta_out.model_dump_json(indent=2) + "\n"
        )
        return text

    # ------------------------------------------------------------------ #
    # 内部：生成
    # ------------------------------------------------------------------ #
    async def _generate(
        self,
        *,
        root: Path,
        work_dir: Path,
        scan: TreeScan,
        llm: LLMClient | None,
    ) -> tuple[str, str]:
        """L1 + L2 生成摘要文本，返回 ``(文本, 使用的模型标签)``。"""
        counter = self._resolve_counter()
        summaries: dict[str, str] = {}
        model_label = ""
        if llm is not None:
            packages = _package_dirs(scan)
            selected = packages[:REPOMAP_MAX_PACKAGES]
            packages_text = await asyncio.to_thread(
                _packages_excerpt, root, scan, selected, counter
            )
            tree_text = truncate_to_tokens(
                _tree_text(scan), max_tokens=REPOMAP_BUDGET // 2, counter=counter
            )
            try:
                response = await llm.complete_role(
                    "repomap",
                    user=_render_template(REPOMAP_PROMPT, tree=tree_text, packages=packages_text),
                )
            except (TimeoutError, MyAgentError):
                # §7.3 成本保护：L2 失败不阻塞启动，退化为 L1（全部标「（未摘要）」）。
                # LLMClient 已把 provider 异常统一映射为 MyAgentError（§13.1.5）。
                summaries = {}
            else:
                summaries = _parse_summary_lines(str(getattr(response, "text", "") or ""))
                model_label = _model_label(llm)
        return _render_summary(root, work_dir, scan, summaries), model_label

    def _resolve_counter(self) -> TokenCounter:
        """取 token 计数器（构造实例 > 默认实例）。"""
        if self._counter is not None:
            return self._counter
        from my_agent.llm.tokenizer import get_counter

        self._counter = get_counter(None)
        return self._counter


# --------------------------------------------------------------------------- #
# L1：确定性扫描
# --------------------------------------------------------------------------- #
def _project_root(cwd: Path) -> Path:
    """向上寻找项目根（``.git`` 或 ``.myagent``），否则返回 ``cwd``。"""
    current = Path(cwd)
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / MYAGENT_DIR_NAME).exists():
            return candidate
    return current


def _scan_tree(root: Path) -> TreeScan:
    """L1：遍历目录骨架（遵守 gitignore、深度/文件数上限，不调模型）。"""
    dirs: list[str] = []
    files: list[str] = []
    manifests: dict[str, list[str]] = {}
    ignored: set[str] = set()
    truncated = False
    stack: list[tuple[Path, int, list[Any]]] = [(Path(root), 0, [_load_gitignore(Path(root))])]
    while stack:
        if len(files) >= REPOMAP_MAX_FILES:
            truncated = True
            break
        directory, depth, specs = stack.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:  # pragma: no cover - 权限问题跳过该目录
            continue
        for entry in entries:
            if len(files) >= REPOMAP_MAX_FILES:
                truncated = True
                break
            relative = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name in IGNORED_DIR_NAMES or _ignored(f"{relative}/", specs):
                    ignored.add(entry.name)
                    continue
                if depth >= REPOMAP_MAX_DEPTH:
                    continue
                dirs.append(relative)
                stack.append((entry, depth + 1, [*specs, _load_gitignore(entry)]))
                continue
            if any(fnmatch.fnmatch(entry.name, pattern) for pattern in IGNORED_FILE_PATTERNS):
                continue
            if _ignored(relative, specs):
                continue
            files.append(relative)
            if entry.name in PACKAGE_MANIFESTS:
                parent = relative.rsplit("/", 1)[0] if "/" in relative else ""
                manifests.setdefault(parent, []).append(entry.name)
    for names in manifests.values():
        names.sort()
    return TreeScan(
        dirs=sorted(dirs),
        files=files,
        manifests=manifests,
        file_count=len(files),
        truncated=truncated,
        ignored=sorted(ignored),
        tree_hash=_tree_hash(dirs),
    )


def _tree_hash(dirs: Sequence[str]) -> str:
    """目录结构级哈希：``sha1`` of sorted dir names（§7.3 失效条件）。"""
    payload = "\n".join(sorted(dirs)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()  # noqa: S324 - 非安全用途，仅作缓存键


def _load_gitignore(directory: Path) -> Any:
    """加载某目录的 ``.gitignore``（无 pathspec 或读失败时返回 ``None``）。"""
    if not _HAS_PATHSPEC:
        return None
    path = directory / ".gitignore"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    except UnicodeDecodeError:  # pragma: no cover - 非 UTF-8 的 .gitignore
        return None
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _ignored(relative: str, specs: Iterable[Any]) -> bool:
    """相对路径是否被任一 ``.gitignore`` 忽略。"""
    return any(spec is not None and spec.match_file(relative) for spec in specs)


# --------------------------------------------------------------------------- #
# L2：包信息与模型摘要
# --------------------------------------------------------------------------- #
def _package_dirs(scan: TreeScan) -> list[str]:
    """包目录排序：根包（``""``）在前，其余按深度、字典序。"""
    return sorted(scan.manifests, key=lambda item: (item.count("/") if item else 0, item))


def _packages_excerpt(
    root: Path, scan: TreeScan, packages: Sequence[str], counter: TokenCounter
) -> str:
    """拼出附录 B.2 的「【包信息】」文本（README 1.5k + 入口文件 3×1k + 元数据）。"""
    chunks: list[str] = []
    for package in packages:
        chunks.append(_package_excerpt(root, scan, package))
    text = "\n\n".join(chunks) if chunks else "（未识别到包）"
    return truncate_to_tokens(text, max_tokens=REPOMAP_BUDGET // 2, counter=counter)


def _package_excerpt(root: Path, scan: TreeScan, package: str) -> str:
    """单个包的摘录文本。"""
    prefix = f"{package}/" if package else ""
    names = scan.manifests.get(package, [])
    parts = [f"## {prefix or './'}"]
    for name in names:
        excerpt = _read_excerpt(root / prefix / name, ENTRY_EXCERPT_CHARS)
        parts.append(f"manifest {name}: {_compact(excerpt)}")
    readme = next(
        (name for name in README_NAMES if (root / prefix / name).is_file()),
        None,
    )
    if readme is not None:
        excerpt = _read_excerpt(root / prefix / readme, README_EXCERPT_CHARS)
        parts.append(f"README {readme}: {_compact(excerpt)}")
    for entry in _entry_files(scan, package):
        excerpt = _read_excerpt(root / entry, ENTRY_EXCERPT_CHARS)
        parts.append(f"{entry}: {_compact(excerpt)}")
    return "\n".join(parts)


def _entry_files(scan: TreeScan, package: str) -> list[str]:
    """挑出包的入口文件（候选名优先，其次目录内文件按字典序）。"""
    prefix = f"{package}/" if package else ""
    candidates = [
        f"{prefix}{name}" for name in ENTRY_FILE_CANDIDATES if f"{prefix}{name}" in scan.files
    ]
    if len(candidates) < MAX_ENTRY_FILES:
        depth = prefix.count("/")
        for relative in scan.files:
            if relative in candidates or not relative.startswith(prefix):
                continue
            if relative.rsplit("/", 1)[0].count("/") != depth:
                continue
            candidates.append(relative)
            if len(candidates) >= MAX_ENTRY_FILES:
                break
    return candidates[:MAX_ENTRY_FILES]


def _read_excerpt(path: Path, limit: int) -> str:
    """读取文件前 ``limit`` 个字符（读失败返回空串）。"""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def _tree_text(scan: TreeScan) -> str:
    """把 L1 目录骨架渲染成缩进树（供附录 B.2 的【仓库骨架】）。"""
    lines: list[str] = ["./"]
    for relative in scan.dirs:
        depth = relative.count("/") + 1
        lines.append(f"{'  ' * depth}{relative.rsplit('/', 1)[-1]}/")
        if len(lines) >= TREE_MAX_LINES:
            lines.append("…（目录过多已截断）")
            break
    for relative in scan.files:
        if len(lines) >= TREE_MAX_LINES * 2:
            break
        depth = relative.count("/") + 1
        lines.append(f"{'  ' * depth}{relative.rsplit('/', 1)[-1]}")
    return "\n".join(lines)


def _parse_summary_lines(text: str) -> dict[str, str]:
    """解析模型输出的一行说明：``- <包路径> — <说明>`` → ``{规范化路径: 说明}``。"""
    summaries: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(("-", "*")):
            continue
        body = stripped.lstrip("-* ").strip()
        for sep in (" — ", " – ", " - ", "—", "："):
            if sep in body:
                raw_path, description = body.split(sep, 1)
                key = raw_path.strip().strip("`").strip().strip("/")
                if key in (".", "./"):
                    key = ""
                if key or raw_path.strip() in (".", "./"):
                    summaries[key] = description.strip()
                break
    return summaries


def _model_label(llm: LLMClient) -> str:
    """尽量取模型标签用于缓存元数据（取不到时给 ``unknown``）。"""
    for attribute in ("model_name", "model"):
        value = getattr(llm, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(llm).__name__


# --------------------------------------------------------------------------- #
# 渲染与缓存
# --------------------------------------------------------------------------- #
def _render_summary(
    root: Path, work_dir: Path, scan: TreeScan, summaries: dict[str, str]
) -> str:
    """渲染 §7.3 规定的注入文本（标题 + 仓库根 + 包/顶层目录 + 统计行）。"""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    lines = [f"## 工作目录摘要（{work_dir}，生成于 {stamp}）"]
    root_manifests = scan.manifests.get("", [])
    if root_manifests:
        lines.append(f"- 仓库根：{_manifest_label(root, root_manifests)}")
    listed: set[str] = set()
    for package in _package_dirs(scan):
        listed.add(package)
        description = summaries.get(package, UNSUMMARIZED)
        lines.append(f"- {package}/ — {description}" if package else f"- ./ — {description}")
    for relative in sorted(scan.dirs):
        if "/" in relative or relative in listed:
            continue
        lines.append(f"- {relative}/ — {summaries.get(relative, UNSUMMARIZED)}")
    top_levels = [relative for relative in scan.dirs if "/" not in relative]
    footer = f"（共 {len(top_levels)} 个顶层目录"
    if scan.ignored:
        footer += "，已忽略 " + " / ".join(scan.ignored)
    footer += "）"
    lines.append(footer)
    if scan.truncated:
        lines.append(f"（文件数已达上限 {REPOMAP_MAX_FILES}，骨架已截断）")
    return "\n".join(lines) + "\n"


def _manifest_label(root: Path, names: Sequence[str]) -> str:
    """仓库根一行：``pyproject.toml（Python 包 `my-agent`）``（§7.3 样例格式）。"""
    primary = names[0]
    detail = ""
    if primary == "pyproject.toml":
        name = _toml_name(_read_full(root / primary))
        detail = f"Python 包 `{name}`" if name else "Python 包"
    elif primary == "package.json":
        detail = "Node 包"
    elif primary == "go.mod":
        detail = "Go module"
    elif primary == "Cargo.toml":
        detail = "Rust crate"
    others = "、".join(names[1:])
    if others:
        return f"{primary}（{detail}、另有 {others}）" if detail else f"{primary}（另有 {others}）"
    return f"{primary}（{detail}）" if detail else primary


def _compact(text: str) -> str:
    """把多行文本压成单行（包信息段落用）。"""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _render_template(template: str, /, **values: Any) -> str:
    """渲染附录 B 模板（占位符 ``{tree}`` / ``{packages}``）。"""
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        return template


def _read_meta(path: Path) -> RepoMapMeta | None:
    """读取缓存元数据（缺失/损坏返回 ``None`` = 视为失效）。"""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    try:
        return RepoMapMeta.model_validate(raw)
    except ValueError:
        return None


def _read_text(path: Path) -> str | None:
    """读取缓存摘要文本（缺失/不可读返回 ``None``）。"""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return text if text.strip() else None


def _fresh(meta: RepoMapMeta) -> bool:
    """缓存是否仍在 TTL 内（§7.3 ``REPOMAP_TTL_HOURS``）。"""
    age = time.time() - meta.generated_at
    return age < REPOMAP_TTL_HOURS * SECONDS_PER_HOUR


def _atomic_write(path: Path, text: str) -> Path:
    """原子写：``<path>.tmp`` → :func:`os.replace`。

    Args:
        path: 目标文件。
        text: 内容。

    Returns:
        目标文件路径。

    Raises:
        OSError: 目录创建/写入/替换失败。
    """
    tmp = path.with_name(path.name + _TMP_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    return path


def _toml_name(text: str) -> str:
    """从 ``pyproject.toml`` 文本里取项目名（解析失败返回空串）。"""
    try:
        data = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError):
        return ""
    project = data.get("project")
    if isinstance(project, dict) and isinstance(project.get("name"), str):
        return str(project["name"])
    return ""


def _read_full(path: Path) -> str:
    """读取整个文本文件（读失败返回空串）。"""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
