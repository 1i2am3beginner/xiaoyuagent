"""归档与回读（§7.7.2 归档 / §7.7.3 回读工具）。

目录：``<project>/.myagent/archive/<session_id>/``

- ``turns-0001-0040.jsonl``：每行一条 :class:`~my_agent.models.messages.Message` 的 JSON
  （字段与 §4.1 一致，含 ``id`` / ``created_at`` / ``tokens``，``ensure_ascii=False``）；
- ``index.json``：段索引 ``[{file, turn_start, turn_end, tokens, summary_id}]``，**最后写**
  （保证索引不指向半截文件）；
- 原子性：先写 ``.tmp`` 再 :func:`os.replace`。

:class:`ArchiveStore.search` 即 ``ArchiveSearch`` 工具的行为实现（§7.7.3）：
至少一个检索条件、消息级检索、紧凑命中视图 ``[turn 12 | assistant] …``、
总返回 ≤ 8k token、附带回读提示、无归档时返回「本会话暂无归档历史」。

本模块的方法是**同步**的（§7 契约冻结）；异步调用方必须用 :func:`asyncio.to_thread`
包装，避免阻塞事件循环。
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
import os
import re
import time
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from my_agent.errors import ArchiveCorruptedError, ToolExecutionError
from my_agent.models.messages import Message

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期导入
    from my_agent.llm.tokenizer import TokenCounter

__all__ = [
    "ARCHIVE_DIR_NAME",
    "ARCHIVE_SEARCH_MAX_TOKENS",
    "EMPTY_SESSION_TEXT",
    "INDEX_FILE_NAME",
    "ArchiveIndex",
    "ArchiveIndexEntry",
    "ArchiveStore",
]

#: 归档根目录名（§7.7.2）。
ARCHIVE_DIR_NAME: str = "archive"
#: 段索引文件名（§7.7.2）。
INDEX_FILE_NAME: str = "index.json"
#: 归档分片文件名前缀（§7.7.2：``turns-0001-0040.jsonl``）。
SHARD_PREFIX: str = "turns-"
#: 归档分片扩展名。
SHARD_SUFFIX: str = ".jsonl"
#: 原子写临时后缀。
_TMP_SUFFIX: str = ".tmp"

#: §7.7.3：``ArchiveSearch`` 单次返回上限（8k token）。
ARCHIVE_SEARCH_MAX_TOKENS: int = 8_000
#: 单条命中片段的最大字符数（展示细节，非行为阈值）。
HIT_SNIPPET_MAX_CHARS: int = 400
#: 无归档时的固定返回文案（§7.7.3，逐字）。
EMPTY_SESSION_TEXT: str = "本会话暂无归档历史"

#: 从正文/参数里提取「像路径的 token」。
_PATH_RE = re.compile(r"[\w.\-]+(?:[/\\][\w.\-]+)+")
#: 命中行折叠成单行时的连接符。
_FLATTEN_SEP: str = " / "


class ArchiveIndexEntry(BaseModel):
    """``index.json`` 中的一条段索引（§7.7.2）。"""

    file: str = Field(description="归档分片文件名（相对会话归档目录）")
    turn_start: int
    turn_end: int
    tokens: int = 0
    summary_id: str | None = None
    messages: int = Field(default=0, description="该分片内的消息条数")
    created_at: float = Field(default_factory=time.time)


class _Hit(BaseModel):
    """一次检索命中（内部结构）。"""

    path: str
    line: int
    turn: int
    role: str
    score: float
    snippet: str


class ArchiveStore:
    """会话归档库（§7.7.2）与检索入口（§7.7.3）。

    Args:
        archive_root: ``<project>/.myagent/archive``。
        counter: token 计数器；用于检索返回的 8k 预算与分片 token 统计。
    """

    def __init__(self, archive_root: Path, *, counter: TokenCounter | None = None) -> None:
        self._root = Path(archive_root)
        self._counter = counter

    @property
    def root(self) -> Path:
        """归档根目录。

        Returns:
            ``<project>/.myagent/archive``。
        """
        return self._root

    def session_dir(self, session_id: str) -> Path:
        """某个会话的归档目录。

        Args:
            session_id: 会话 id。

        Returns:
            ``<archive_root>/<session_id>``。
        """
        return self._root / _safe_name(session_id)

    def archive_messages(
        self,
        session_id: str,
        messages: list[Message],
        *,
        turn_start: int,
        turn_end: int,
        summary_id: str | None = None,
    ) -> Path:
        """把一批消息写成归档分片并更新索引（先分片、后索引；均为原子写）。

        Args:
            session_id: 会话 id。
            messages: 要归档的消息（通常为压缩前的**全量** history 快照）。
            turn_start: 该分片起始轮次（含）。
            turn_end: 该分片结束轮次（含）。
            summary_id: 对应摘要消息 id（可空）。

        Returns:
            写入的分片路径。

        Raises:
            OSError: 磁盘/权限问题（调用方决定是否降级）。
        """
        shard_messages = list(messages)
        target_dir = self.session_dir(session_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = _unique_path(
            target_dir / f"{SHARD_PREFIX}{turn_start:04d}-{turn_end:04d}{SHARD_SUFFIX}"
        )
        payload = "".join(message.model_dump_json() + "\n" for message in shard_messages)
        _atomic_write_text(target, payload)
        entry = ArchiveIndexEntry(
            file=target.name,
            turn_start=turn_start,
            turn_end=turn_end,
            tokens=self._count_tokens(shard_messages),
            summary_id=summary_id,
            messages=len(shard_messages),
        )
        self._append_index(session_id, entry)
        return target

    def index(self, session_id: str) -> list[ArchiveIndexEntry]:
        """读取会话段索引。

        Args:
            session_id: 会话 id。

        Returns:
            段索引列表（无归档时为空列表）。

        Raises:
            ArchiveCorruptedError: ``index.json`` 不是合法 JSON 或结构非法。
        """
        return self._read_index(self.session_dir(session_id) / INDEX_FILE_NAME)

    def shards(self, session_id: str) -> list[Path]:
        """列出会话的所有归档分片（按文件名排序）。

        Args:
            session_id: 会话 id。

        Returns:
            分片路径列表。
        """
        directory = self.session_dir(session_id)
        if not directory.is_dir():
            return []
        return sorted(directory.glob(f"{SHARD_PREFIX}*{SHARD_SUFFIX}"))

    def path_for(self, session_id: str, entry: ArchiveIndexEntry) -> Path:
        """把索引项还原为分片路径。

        Args:
            session_id: 会话 id。
            entry: 索引项。

        Returns:
            分片绝对/相对路径。
        """
        return self.session_dir(session_id) / entry.file

    def latest_shard(self, session_id: str) -> Path | None:
        """返回最近写入的分片（``/context`` 报告用）。

        Args:
            session_id: 会话 id。

        Returns:
            最近的分片路径；无归档时 ``None``。
        """
        shards = self.shards(session_id)
        return shards[-1] if shards else None

    def search(
        self,
        session_id: str,
        *,
        query: str | None = None,
        turn_start: int | None = None,
        turn_end: int | None = None,
        file_glob: str | None = None,
        limit: int = 20,
        context_lines: int = 2,
    ) -> str:
        """按关键词 / 轮次区间 / 文件 glob 检索归档（§7.7.3）。

        Args:
            session_id: 会话 id。
            query: 关键词（不区分大小写），匹配 ``role`` + ``content`` +
                ``tool_calls`` 的 ``name`` / ``arguments``。
            turn_start: 轮次区间起点（含）。
            turn_end: 轮次区间终点（含）。
            file_glob: 只保留涉及该路径模式的消息，如 ``**/auth.py``。
            limit: 最多返回多少条命中（默认 20，契约上限 100）。
            context_lines: 命中行上下各取几行作为片段。

        Returns:
            命中消息的紧凑视图 + 回读提示；无归档时返回 :data:`EMPTY_SESSION_TEXT`；
            无命中时返回「未找到匹配」说明。

        Raises:
            ToolExecutionError: 一个检索条件都没给（§7.7.3「否则报参数错误」）。
            ArchiveCorruptedError: 归档分片/索引损坏。
        """
        if query is None and file_glob is None and turn_start is None and turn_end is None:
            raise ToolExecutionError(
                "至少需要提供一个检索条件：query / turn_start+turn_end / file_glob",
                name="ArchiveSearch",
            )
        shards = self.shards(session_id)
        if not shards:
            return EMPTY_SESSION_TEXT

        scanned = 0
        hits: list[_Hit] = []
        for shard in shards:
            for line_number, message in self._iter_messages(shard):
                scanned += 1
                if not _in_range(message.turn_index, turn_start, turn_end):
                    continue
                score, snippet = _match(
                    message, query=query, file_glob=file_glob, context_lines=context_lines
                )
                if score <= 0:
                    continue
                hits.append(
                    _Hit(
                        path=str(shard),
                        line=line_number,
                        turn=message.turn_index,
                        role=message.role,
                        score=score,
                        snippet=snippet,
                    )
                )
        if not hits:
            return (
                f"未找到匹配的归档记录（已检索 {scanned} 条消息 / {len(shards)} 个归档文件）；"
                f"可放宽轮次范围或更换关键词，也可用 Read 直接读取 {shards[-1]}"
            )
        return self._render(hits, limit=limit)

    def _render(self, hits: list[_Hit], *, limit: int) -> str:
        """按相关性排序 + 8k token 预算渲染命中视图。"""
        counter = self._counter
        if counter is None:
            from my_agent.llm.tokenizer import get_counter

            counter = get_counter(None)
        ordered = sorted(hits, key=lambda hit: (-hit.score, hit.path, hit.line))
        candidates = ordered[: max(1, limit)]
        header = f"归档检索命中 {len(candidates)} 条（共匹配 {len(hits)} 条，按相关性排序）："
        lines = [header]
        used = counter.count_text(header)
        shown: list[_Hit] = []
        truncated = False
        for hit in candidates:
            rendered = f"[turn {hit.turn} | {hit.role}] {hit.snippet}"
            cost = counter.count_text(rendered)
            if used + cost > ARCHIVE_SEARCH_MAX_TOKENS:
                truncated = True
                break
            lines.append(rendered)
            used += cost
            shown.append(hit)
        if truncated:
            lines.append(
                f"（返回已达 {ARCHIVE_SEARCH_MAX_TOKENS} token 上限，"
                f"已按相关性截断；请缩小检索范围或提高 query 精度）"
            )
        lines.append("如需完整原文，可用 Read 工具读取以下归档文件的对应行：")
        for path, line_numbers in _group_lines(shown):
            joined = ", ".join(str(number) for number in line_numbers)
            lines.append(f"- {path}（命中行：{joined}）")
        return "\n".join(lines)

    def _iter_messages(self, shard: Path) -> Iterator[tuple[int, Message]]:
        """逐行读取归档分片，产出 ``(行号, Message)``。

        Raises:
            ArchiveCorruptedError: 某行不是合法 JSON 或不满足 §4.1 消息契约。
        """
        with shard.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    payload: Any = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ArchiveCorruptedError(
                        detail=f"第 {line_number} 行不是合法 JSON（{exc.msg}）", path=str(shard)
                    ) from exc
                try:
                    yield line_number, Message.model_validate(payload)
                except ValidationError as exc:
                    raise ArchiveCorruptedError(
                        detail=f"第 {line_number} 行不满足消息契约（{exc.error_count()} 处错误）",
                        path=str(shard),
                    ) from exc

    def _append_index(self, session_id: str, entry: ArchiveIndexEntry) -> None:
        """读取 → 追加/替换 → 原子写回 ``index.json``（最后写，§7.7.2）。"""
        index_path = self.session_dir(session_id) / INDEX_FILE_NAME
        entries = [item for item in self._read_index(index_path) if item.file != entry.file]
        entries.append(entry)
        entries.sort(key=lambda item: (item.turn_start, item.turn_end, item.file))
        payload = json.dumps(
            [item.model_dump() for item in entries], ensure_ascii=False, indent=2
        )
        _atomic_write_text(index_path, payload + "\n")

    def _read_index(self, index_path: Path) -> list[ArchiveIndexEntry]:
        """读取 ``index.json``（缺失 = 空列表）。"""
        if not index_path.exists():
            return []
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8") or "[]")
        except (OSError, json.JSONDecodeError) as exc:
            raise ArchiveCorruptedError(
                detail=f"索引不可读（{exc}）", path=str(index_path)
            ) from exc
        if not isinstance(raw, list):
            raise ArchiveCorruptedError(detail="索引结构应为数组", path=str(index_path))
        entries: list[ArchiveIndexEntry] = []
        for item in raw:
            try:
                entries.append(ArchiveIndexEntry.model_validate(item))
            except ValidationError as exc:
                raise ArchiveCorruptedError(
                    detail=f"索引项字段非法（{exc.error_count()} 处错误）", path=str(index_path)
                ) from exc
        return entries

    def _count_tokens(self, messages: Sequence[Message]) -> int:
        """统计分片 token：优先用消息自带计数，全为 0 时用 counter 现算。"""
        total = sum(message.tokens for message in messages)
        if total or self._counter is None:
            return total
        return self._counter.count_messages(list(messages))


class ArchiveIndex:
    """绑定到某个会话的归档检索入口（``ArchiveSearchTool`` 的依赖，CONTRACTS §5）。

    Args:
        store: 归档库。
        session_id: 当前会话 id。
    """

    def __init__(self, store: ArchiveStore, session_id: str) -> None:
        self._store = store
        self._session_id = session_id

    @property
    def store(self) -> ArchiveStore:
        """底层归档库。

        Returns:
            :class:`ArchiveStore`。
        """
        return self._store

    @property
    def session_id(self) -> str:
        """当前会话 id。

        Returns:
            会话 id。
        """
        return self._session_id

    def index(self) -> list[ArchiveIndexEntry]:
        """当前会话的段索引。

        Returns:
            段索引列表。
        """
        return self._store.index(self._session_id)

    def paths(self) -> list[str]:
        """当前会话的全部分片路径。

        Returns:
            分片路径字符串列表。
        """
        return [str(path) for path in self._store.shards(self._session_id)]

    def search(
        self,
        *,
        query: str | None = None,
        turn_start: int | None = None,
        turn_end: int | None = None,
        file_glob: str | None = None,
        limit: int = 20,
        context_lines: int = 2,
    ) -> str:
        """检索当前会话的归档（语义同 :meth:`ArchiveStore.search`）。

        Args:
            query: 关键词（不区分大小写）。
            turn_start: 轮次区间起点（含）。
            turn_end: 轮次区间终点（含）。
            file_glob: 路径模式过滤，如 ``**/auth.py``。
            limit: 最多返回条数。
            context_lines: 命中行上下各取几行。

        Returns:
            紧凑命中视图或说明文案。

        Raises:
            ToolExecutionError: 未提供任何检索条件。
            ArchiveCorruptedError: 归档损坏。
        """
        return self._store.search(
            self._session_id,
            query=query,
            turn_start=turn_start,
            turn_end=turn_end,
            file_glob=file_glob,
            limit=limit,
            context_lines=context_lines,
        )


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _safe_name(value: str) -> str:
    """把 session id 净化为安全目录名。"""
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(value))
    return cleaned.strip("._") or "unknown"


def _unique_path(path: Path) -> Path:
    """避免覆盖同名分片：``turns-0001-0040.jsonl`` → ``turns-0001-0040-2.jsonl``。"""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise ArchiveCorruptedError(detail=f"同名分片过多：{path.name}", path=str(path))


def _atomic_write_text(path: Path, text: str) -> Path:
    """原子写：``<path>.tmp`` → :func:`os.replace`（§7.7.2）。

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


def _in_range(turn: int, turn_start: int | None, turn_end: int | None) -> bool:
    """消息轮次是否落在检索区间内。"""
    if turn_start is not None and turn < turn_start:
        return False
    return not (turn_end is not None and turn > turn_end)


def _searchable_text(message: Message) -> str:
    """拼出消息的「可检索文本」：role + content + tool_calls(name/arguments)。"""
    parts: list[str] = [message.role, message.content]
    for call in message.tool_calls:
        parts.append(call.name)
        parts.append(json.dumps(call.arguments, ensure_ascii=False, default=str))
        parts.append(call.raw_arguments)
    return "\n".join(part for part in parts if part)


def _match(
    message: Message,
    *,
    query: str | None,
    file_glob: str | None,
    context_lines: int,
) -> tuple[float, str]:
    """判定一条消息是否命中，并返回 ``(相关性分数, 片段)``（0 分 = 未命中）。"""
    text = _searchable_text(message)
    lowered = text.lower()
    score = 1.0
    if query:
        occurrences = lowered.count(query.lower())
        if occurrences <= 0:
            return 0.0, ""
        score += occurrences
    if file_glob:
        tokens = _PATH_RE.findall(text)
        if not any(_path_matches(token, file_glob) for token in tokens):
            return 0.0, ""
        score += 1.0
    needle = query.lower() if query else ""
    return score, _snippet(message.content or text, needle, context_lines)


def _snippet(content: str, needle: str, context_lines: int) -> str:
    """取命中行上下 ``context_lines`` 行，压平为单行并截断。"""
    lines = content.splitlines() or [content]
    if needle:
        matched = [index for index, line in enumerate(lines) if needle in line.lower()]
        if matched:
            low = max(0, min(matched) - max(0, context_lines))
            high = min(len(lines), max(matched) + max(0, context_lines) + 1)
            body = "\n".join(lines[low:high])
            prefix = "…" if low > 0 else ""
            suffix = "…" if high < len(lines) else ""
            return _clip(prefix + _flatten(body) + suffix)
    return _clip(_flatten(content))


def _flatten(text: str) -> str:
    """把多行片段压成单行（保持命中视图「一行一条」）。"""
    return _FLATTEN_SEP.join(part.strip() for part in text.splitlines() if part.strip())


def _clip(text: str, max_chars: int = HIT_SNIPPET_MAX_CHARS) -> str:
    """截断片段到 ``max_chars``。"""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _path_matches(token: str, pattern: str) -> bool:
    """路径 token 是否匹配 glob（兼容 ``**/auth.py`` 这类后缀模式与 Windows 反斜杠）。"""
    normalized = token.replace("\\", "/").lstrip("./")
    glob = pattern.replace("\\", "/")
    if fnmatch.fnmatch(normalized, glob) or fnmatch.fnmatch(normalized.rsplit("/", 1)[-1], glob):
        return True
    try:
        return PurePosixPath(normalized).match(glob)
    except ValueError:  # pragma: no cover - 非法 glob
        return False


def _group_lines(hits: Sequence[_Hit]) -> list[tuple[str, list[int]]]:
    """按分片聚合命中行号（保持首次出现顺序）。"""
    grouped: dict[str, list[int]] = {}
    for hit in hits:
        grouped.setdefault(hit.path, []).append(hit.line)
    return list(grouped.items())
