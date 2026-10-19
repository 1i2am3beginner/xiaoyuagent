"""记忆文件存储：Markdown 契约解析、外科式写入与并发控制（§8.1、§8.3、§8.4 步骤 5）。

设计要点（逐条对应规格）：

- **存储形态**：每个作用域一个 Markdown 文件（§8.1），人类可读、可手改、可 git 管理。
- **只动自己的行**（§8.3）：写入时按「章节跨度」拼接，未知章节、注释、用户手写内容
  **原样保留**；只有本模块管理的条目行（``- [id] (date, conf=x) text``）会被增删改。
- **容忍无 id 的旧条目**：解析时自动补 ``m-`` + 6 位 base36 的 id，并在回写前备份 ``.bak``。
- **原子写**：同目录临时文件 + ``os.replace``；任何改动前先写 ``.bak``。
- **并发**：进程内 ``asyncio.Lock``（:meth:`MemoryStore.locked`）+ 跨进程 ``O_EXCL`` 锁文件
  （超时 :data:`~my_agent.config.defaults.MEMORY_LOCK_TIMEOUT_S` 抛
  :class:`~my_agent.errors.MemoryLockError`）。

规格：``my_agent-开发文档.md`` §8.1、§8.3、§8.4 步骤 5；``CONTRACTS.md`` §8、§17.1。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import secrets
import tempfile
import time
from collections.abc import AsyncIterator, Collection, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from my_agent.config.defaults import MEMORY_LOCK_TIMEOUT_S, MEMORY_TEXT_MAX_CHARS
from my_agent.errors import MemoryLockError, MyAgentError
from my_agent.memory.sections import Scope, label_for, scope_of_section, section_from_label
from my_agent.models.permission import Confidence

__all__ = [
    "MANAGED_COMMENT",
    "MemoryEntry",
    "MemoryStore",
    "file_title",
    "managed_entry_line",
    "sanitize_text",
]

#: 托管标记注释：写在文件头部，提示用户「可手工编辑，但 id 请勿修改」（§8.3）。
MANAGED_COMMENT = "<!-- managed-by: myagent; 可手工编辑；条目 id 请勿修改 -->"

#: id 字母表与长度（§8.3：``m-`` + 6 位 base36）。
_ID_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_ID_LENGTH = 6

#: 锁文件轮询粒度（实现细节：等待锁文件释放的忙等间隔，不是行为阈值）。
_LOCK_POLL_S = 0.05

#: 锁文件后缀。
_LOCK_SUFFIX = ".lock"

#: 备份文件后缀（§8.3、§8.4 步骤 5：改动前 ``.bak``）。
_BACKUP_SUFFIX = ".bak"

_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.*?)\s*$")

_ENTRY_RE = re.compile(
    r"^-\s+\[(?P<id>m-[0-9a-z]+)\]\s+"
    r"\((?P<date>\d{4}-\d{2}-\d{2}),\s*conf=(?P<conf>high|medium|low)\)\s*(?P<text>.*)$"
)

#: 旧格式（无 id）条目：``- (2026-09-14, conf=high) 文本``。
_LEGACY_ENTRY_RE = re.compile(
    r"^-\s+\((?P<date>\d{4}-\d{2}-\d{2}),\s*conf=(?P<conf>high|medium|low)\)\s*(?P<text>.*)$"
)

_WHITESPACE_RE = re.compile(r"\s+")

#: 文件一级标题（§8.1）。
_FILE_TITLES: dict[Scope, str] = {
    "global": "# 全局记忆（myagent）",
    "project": "# 项目记忆（myagent）",
}


def file_title(scope: Scope) -> str:
    """返回某作用域记忆文件的一级标题。

    Args:
        scope: ``"global"`` 或 ``"project"``。

    Returns:
        形如 ``# 全局记忆（myagent）`` 的标题行。
    """
    return _FILE_TITLES[scope]


def sanitize_text(text: str) -> str:
    """把一条记忆文本规范成「单行、≤300 字符」（§8.3）。

    Args:
        text: 原始文本（可能含换行、多余空白）。

    Returns:
        折叠空白后的单行文本，最长 :data:`~my_agent.config.defaults.MEMORY_TEXT_MAX_CHARS` 字符。
    """
    collapsed = _WHITESPACE_RE.sub(" ", str(text)).strip()
    if len(collapsed) <= MEMORY_TEXT_MAX_CHARS:
        return collapsed
    return collapsed[:MEMORY_TEXT_MAX_CHARS].rstrip()


def managed_entry_line(entry: MemoryEntry) -> str:
    """返回一条记忆的托管行文本（§8.3 条目行语法）。

    Args:
        entry: 记忆条目。

    Returns:
        ``- [<id>] (<date>, conf=<conf>) <text>``。
    """
    return entry.to_line()


class MemoryEntry(BaseModel):
    """一条记忆（§8.3 条目行 + §8.2 章节归属）。

    Attributes:
        id: ``m-`` + 6 位 base36，文件内唯一。
        text: 单行文本，≤300 字符（:func:`sanitize_text` 保证）。
        date: 首次写入日期；更新时改为最新更新日期（ISO ``YYYY-MM-DD``）。
        confidence: 置信度（high/medium/low）。
        section: §8.2 的章节键。
        scope: 作用域。
    """

    id: str
    text: str
    date: str
    confidence: Confidence = "medium"
    section: str
    scope: Scope

    model_config = {"extra": "ignore"}

    def to_line(self) -> str:
        """渲染成文件中的条目行。

        Returns:
            ``- [<id>] (<date>, conf=<confidence>) <text>``。
        """
        return f"- [{self.id}] ({self.date}, conf={self.confidence}) {self.text}"


def _new_id(used: Collection[str]) -> str:
    """生成一个不与 ``used`` 冲突的条目 id。

    Args:
        used: 已占用的 id 集合。

    Returns:
        ``m-`` + 6 位 base36 的 id。

    Raises:
        MyAgentError: 连续尝试仍无法得到唯一 id（base36 空间耗尽，实际不可能发生）。
    """
    for _ in range(64):
        candidate = "m-" + "".join(secrets.choice(_ID_ALPHABET) for _ in range(_ID_LENGTH))
        if candidate not in used:
            return candidate
    raise MyAgentError("无法生成唯一的记忆条目 id", detail="id 空间冲突")


@dataclass(slots=True)
class _Span:
    """一个二级标题所辖的行区间（``[heading, end)``）。"""

    heading: int
    end: int
    key: str | None
    """章节键；``None`` = 未知章节或重复章节（原样保留，不解析、不改写）。"""


@dataclass(slots=True)
class _Parsed:
    """一次解析的结果。"""

    entries: list[MemoryEntry]
    spans: list[_Span]
    needs_repair: bool
    """是否存在无 id 的旧条目（需要补 id 并回写）。"""


class MemoryStore:
    """一个作用域的 Markdown 记忆文件（§8.3）。

    Args:
        path: 记忆文件路径（如 ``~/.myagent/memory.md``）。
        scope: 该文件所属作用域。

    Attributes:
        path: 记忆文件路径（``/memory`` 等调用点直接引用）。
        scope: 该文件所属作用域。
    """

    def __init__(self, path: Path, *, scope: Scope) -> None:
        self.path: Path = Path(path)
        self.scope: Scope = scope
        self._lock = asyncio.Lock()
        self._lock_depth = 0

    # ----------------------------------------------------------------------------------
    # 读
    # ----------------------------------------------------------------------------------

    def read(self) -> list[MemoryEntry]:
        """读取全部条目（§8.3）。

        容忍无 id 的旧条目：自动补 id，并在回写前备份 ``.bak``。

        Returns:
            按文件顺序排列的条目列表；文件不存在时返回空列表。

        Raises:
            MemoryLockError: 需要回写补 id 时锁文件被占用超时。
            MyAgentError: 文件不是合法 UTF-8 文本。
        """
        lines, newline = self._read_lines()
        parsed = _parse(lines, self.scope)
        if not parsed.needs_repair:
            return parsed.entries
        with self._file_lock():
            lines, newline = self._read_lines()  # 等锁期间可能已被改写，重新读
            parsed = _parse(lines, self.scope)
            if parsed.needs_repair:
                self._write_lines(lines, parsed.spans, _group(parsed.entries), newline)
        return parsed.entries

    def sections(self) -> dict[str, list[MemoryEntry]]:
        """按章节分组返回条目（文件顺序，仅含非空章节）。

        Returns:
            章节键 → 条目列表。
        """
        grouped: dict[str, list[MemoryEntry]] = {}
        for entry in self.read():
            grouped.setdefault(entry.section, []).append(entry)
        return grouped

    def get(self, entry_id: str) -> MemoryEntry | None:
        """按 id 查一条记忆。

        Args:
            entry_id: 条目 id。

        Returns:
            命中的条目；不存在时返回 ``None``。
        """
        return next((entry for entry in self.read() if entry.id == entry_id), None)

    # ----------------------------------------------------------------------------------
    # 写（只增删改自管条目行）
    # ----------------------------------------------------------------------------------

    def write_entries(self, entries: Sequence[MemoryEntry]) -> None:
        """把给定条目写回文件（**只影响这些条目所在的章节**）。

        未出现在 ``entries`` 中的章节（含用户手写的未知章节、注释）原样保留。

        Args:
            entries: 目标状态条目；每个条目的 ``section`` 必须是 §8.2 中的合法键。

        Raises:
            MyAgentError: 章节键非法。
            MemoryLockError: 锁文件被占用超时。
        """
        normalized = [self._normalize(entry) for entry in entries]
        with self._file_lock():
            lines, newline = self._read_lines()
            spans = _parse(lines, self.scope).spans
            self._write_lines(lines, spans, _group(normalized), newline)

    def add(self, text: str, section: str, confidence: Confidence = "medium") -> MemoryEntry:
        """新增一条记忆。

        Args:
            text: 记忆文本（自动规范为单行 ≤300 字符）。
            section: 章节键（§8.2）。
            confidence: 置信度。

        Returns:
            新建的条目（含分配好的 id）。

        Raises:
            MyAgentError: 章节键非法。
            MemoryLockError: 锁文件被占用超时。
        """
        key = self._require_section(section)
        with self._file_lock():
            lines, newline = self._read_lines()
            parsed = _parse(lines, self.scope)
            entry = MemoryEntry(
                id=_new_id({item.id for item in parsed.entries}),
                text=sanitize_text(text),
                date=date.today().isoformat(),
                confidence=confidence,
                section=key,
                scope=self.scope,
            )
            merged = [*parsed.entries, entry]
            self._write_lines(lines, parsed.spans, _group(merged), newline)
        return entry

    def update(
        self, entry_id: str, text: str, section: str, confidence: Confidence = "medium"
    ) -> MemoryEntry:
        """改写一条记忆（可换章节）。

        Args:
            entry_id: 目标条目 id。
            text: 新文本（自动规范为单行 ≤300 字符）。
            section: 新章节键（§8.2）。
            confidence: 新置信度。

        Returns:
            更新后的条目（``date`` 改为今天，§8.3）。

        Raises:
            MyAgentError: 章节键非法，或条目不存在。
            MemoryLockError: 锁文件被占用超时。
        """
        key = self._require_section(section)
        with self._file_lock():
            lines, newline = self._read_lines()
            parsed = _parse(lines, self.scope)
            target = next((item for item in parsed.entries if item.id == entry_id), None)
            if target is None:
                raise MyAgentError(f"记忆条目不存在：{entry_id}", detail=entry_id)
            updated = MemoryEntry(
                id=target.id,
                text=sanitize_text(text),
                date=date.today().isoformat(),
                confidence=confidence,
                section=key,
                scope=self.scope,
            )
            merged = [updated if item.id == entry_id else item for item in parsed.entries]
            self._write_lines(
                lines,
                parsed.spans,
                _group(merged),
                newline,
                known_keys=_all_keys(parsed, extra=key),
            )
        return updated

    def delete(self, entry_id: str) -> MemoryEntry | None:
        """删除一条记忆（删除前已写 ``.bak``，可人工恢复）。

        Args:
            entry_id: 目标条目 id。

        Returns:
            被删除的条目；不存在时返回 ``None``。

        Raises:
            MemoryLockError: 锁文件被占用超时。
        """
        with self._file_lock():
            lines, newline = self._read_lines()
            parsed = _parse(lines, self.scope)
            target = next((item for item in parsed.entries if item.id == entry_id), None)
            if target is None:
                return None
            remaining = [item for item in parsed.entries if item.id != entry_id]
            self._write_lines(
                lines,
                parsed.spans,
                _group(remaining),
                newline,
                known_keys=_all_keys(parsed, extra=target.section),
            )
        return target

    # ----------------------------------------------------------------------------------
    # 并发
    # ----------------------------------------------------------------------------------

    @asynccontextmanager
    async def locked(self) -> AsyncIterator[None]:
        """进程内互斥（``asyncio.Lock``）：保护「读 → 判定 → 写」整段序列。

        跨进程互斥由每次写入时的 ``O_EXCL`` 锁文件负责；本锁解决的是**同一进程内**
        多个协程（如并发回合）交错读改写导致丢更新的问题。

        Yields:
            持有进程内锁的上下文。
        """
        async with self._lock:
            yield

    # ----------------------------------------------------------------------------------
    # 内部实现
    # ----------------------------------------------------------------------------------

    def _normalize(self, entry: MemoryEntry) -> MemoryEntry:
        """校验章节并把作用域对齐到本文件。

        Args:
            entry: 待写入条目。

        Returns:
            规范化后的条目副本。

        Raises:
            MyAgentError: 章节键非法。
        """
        return entry.model_copy(
            update={
                "section": self._require_section(entry.section),
                "scope": self.scope,
                "text": sanitize_text(entry.text),
                "confidence": entry.confidence,
            }
        )

    def _require_section(self, section: str) -> str:
        """校验章节键属于本文件的作用域。

        Args:
            section: 章节键。

        Returns:
            原样返回合法键。

        Raises:
            MyAgentError: 键不在对应作用域的 §8.2 表中。
        """
        if scope_of_section(section) != self.scope:
            raise MyAgentError(
                f"未知记忆章节：{section}（{self.scope} 作用域）", detail=section
            )
        return section

    def _read_lines(self) -> tuple[list[str], str]:
        """读取文件行（保留原换行风格）。

        Returns:
            ``(行列表, 换行符)``；文件不存在时返回 ``([], "\\n")``。

        Raises:
            MyAgentError: 文件不是合法 UTF-8 文本（避免静默损坏用户内容）。
        """
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return [], "\n"
        except UnicodeDecodeError as exc:
            raise MyAgentError(
                f"记忆文件不是 UTF-8 文本，已跳过：{self.path}", detail=str(self.path)
            ) from exc
        newline = "\r\n" if "\r\n" in text else "\n"
        return text.splitlines(), newline

    def _write_lines(
        self,
        lines: Sequence[str],
        spans: Sequence[_Span],
        writes: Mapping[str, list[MemoryEntry]],
        newline: str,
        *,
        known_keys: Collection[str] | None = None,
    ) -> None:
        """拼接并原子落盘（改动前先写 ``.bak``）。

        Args:
            lines: 文件原有行。
            spans: 原有章节跨度。
            writes: 章节键 → 该章节的目标条目。
            newline: 落盘换行符。
            known_keys: **本次落盘涉及的章节键**（含被删空的章节）。
                缺省退化为「只重建 ``writes`` 里出现的章节」——注意此时
                「把某章节删空」不会生效，因为该键已从 ``writes`` 消失。
                ``delete``/``update`` 这类整体重建的调用必须显式传入。

        Raises:
            MemoryLockError: 锁文件被占用超时（由调用方持锁时不会触发）。
            OSError: 落盘失败。
        """
        rendered = _render(
            lines,
            spans,
            writes,
            scope=self.scope,
            newline=newline,
            known_keys=known_keys,
        )
        self._backup()
        _atomic_write_text(self.path, rendered)

    def _backup(self) -> None:
        """把当前文件复制为 ``<name>.bak``（文件不存在时什么都不做）。"""
        if not self.path.is_file():
            return
        backup = self.path.with_name(self.path.name + _BACKUP_SUFFIX)
        backup.write_bytes(self.path.read_bytes())

    def _lock_path(self) -> Path:
        """返回锁文件路径。

        Returns:
            ``<memory 文件>.lock``。
        """
        return self.path.with_name(self.path.name + _LOCK_SUFFIX)

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        """获取跨进程锁文件（``O_EXCL`` 语义，可重入）。

        Yields:
            持锁上下文。

        Raises:
            MemoryLockError: 等待超过 :data:`~my_agent.config.defaults.MEMORY_LOCK_TIMEOUT_S`
                秒仍未拿到锁，或锁文件无法创建。
        """
        if self._lock_depth > 0:  # 可重入：同一调用栈内的嵌套写入复用同一把锁
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_path()
        deadline = time.monotonic() + MEMORY_LOCK_TIMEOUT_S
        handle = -1
        while True:
            try:
                handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise MemoryLockError(
                        detail=f"{lock_path} 被其它进程持有，等待超过 {MEMORY_LOCK_TIMEOUT_S}s"
                    ) from None
                time.sleep(_LOCK_POLL_S)
            except OSError as exc:
                raise MemoryLockError(detail=f"无法创建锁文件 {lock_path}：{exc}") from exc

        try:
            with contextlib.suppress(OSError):
                os.write(handle, str(os.getpid()).encode("ascii"))
            os.close(handle)
            handle = -1
            self._lock_depth = 1
            try:
                yield
            finally:
                self._lock_depth = 0
        finally:
            if handle >= 0:
                with contextlib.suppress(OSError):
                    os.close(handle)
            with contextlib.suppress(OSError):
                lock_path.unlink()


def _parse(lines: Sequence[str], scope: Scope) -> _Parsed:
    """解析文件行（§8.3）。

    Args:
        lines: 文件行。
        scope: 作用域（决定哪些标题是「已知章节」）。

    Returns:
        解析结果：条目、章节跨度、是否需要补 id 回写。
    """
    spans: list[_Span] = []
    headings: list[tuple[int, str | None]] = []
    claimed: set[str] = set()
    for index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match is None or len(match.group("hashes")) != 2:
            continue
        key = section_from_label(scope, match.group("title"))
        if key is None or key in claimed:
            key = None  # 未知章节 / 重复章节：不解析、不改写（§8.3）
        else:
            claimed.add(key)
        headings.append((index, key))
    for position, (index, key) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        spans.append(_Span(heading=index, end=end, key=key))

    entries: list[MemoryEntry] = []
    used: set[str] = set()
    needs_repair = False
    for span in spans:
        if span.key is None:
            continue
        for line in lines[span.heading + 1 : span.end]:
            parsed = _parse_entry_line(line, span.key, scope, used)
            if parsed is None:
                continue
            entry, legacy = parsed
            used.add(entry.id)
            needs_repair = needs_repair or legacy
            entries.append(entry)
    return _Parsed(entries=entries, spans=spans, needs_repair=needs_repair)


def _parse_entry_line(
    line: str, section: str, scope: Scope, used: Collection[str]
) -> tuple[MemoryEntry, bool] | None:
    """解析单行条目。

    Args:
        line: 文件行。
        section: 所属章节键。
        scope: 作用域。
        used: 已占用的 id（为旧条目生成新 id 时避免冲突）。

    Returns:
        ``(条目, 是否为无 id 旧条目)``；该行不是条目行时返回 ``None``。
    """
    legacy = False
    match = _ENTRY_RE.match(line)
    if match is None:
        match = _LEGACY_ENTRY_RE.match(line)
        legacy = match is not None
    if match is None:
        return None
    entry_id = _new_id(used) if legacy else match.group("id")
    return (
        MemoryEntry(
            id=entry_id,
            text=match.group("text").strip(),
            date=match.group("date"),
            confidence=match.group("conf"),
            section=section,
            scope=scope,
        ),
        legacy,
    )


def _group(entries: Sequence[MemoryEntry]) -> dict[str, list[MemoryEntry]]:
    """按章节分组（保持给定顺序）。

    注意：**不含空章节**——条目被删空的章节不会出现在结果里。因此「整体重建」
    （``update``/``delete``）必须同时用 :func:`_all_keys` 把涉及的章节键显式告诉
    :meth:`MemoryStore._write_lines`，否则删空不会生效。

    Args:
        entries: 条目序列。

    Returns:
        章节键 → 条目列表。
    """
    grouped: dict[str, list[MemoryEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.section, []).append(entry)
    return grouped


def _all_keys(parsed: _Parsed, *, extra: str | None = None) -> set[str]:
    """返回本次落盘应当重建的章节键集合。

    Args:
        parsed: 解析结果（提供文件中已识别的章节跨度）。
        extra: 额外要纳入的章节键（如 ``update`` 的目标章节）。

    Returns:
        章节键集合（含已删空的章节）。
    """
    keys = {span.key for span in parsed.spans if span.key is not None}
    if extra:
        keys.add(extra)
    return keys


def _rebuild_body(body: Sequence[str], entries: Sequence[MemoryEntry]) -> list[str]:
    """重建一个章节的正文：只替换托管条目行，其余行原位保留。

    Args:
        body: 原章节正文行（不含标题行）。
        entries: 该章节的目标条目。

    Returns:
        新正文行。
    """
    managed = [
        index
        for index, line in enumerate(body)
        if _ENTRY_RE.match(line) is not None or _LEGACY_ENTRY_RE.match(line) is not None
    ]
    managed_set = set(managed)
    kept = [line for index, line in enumerate(body) if index not in managed_set]
    insert_at = managed[0] if managed else len(kept)
    new_lines = [entry.to_line() for entry in entries]
    return [*kept[:insert_at], *new_lines, *kept[insert_at:]]


def _render(
    lines: Sequence[str],
    spans: Sequence[_Span],
    writes: Mapping[str, list[MemoryEntry]],
    *,
    scope: Scope,
    newline: str,
    known_keys: Collection[str] | None = None,
) -> str:
    """把「文件原有行 + 目标条目」拼成新文件文本（未知内容原样保留）。

    关键点：``writes`` 用 ``_group`` 生成时会**丢掉空章节**（条目被删空后该键消失）。
    若只按 ``key in writes`` 判断，删空章节的旧条目行会被当作「非托管内容」原样保留，
    导致删除/更新不生效。因此这里额外接收 ``known_keys``——**本次落盘涉及的全部章节键**
    （含被删空的），对这些章节一律按 ``writes``（缺省为空列表）重建正文。

    Args:
        lines: 文件原有行。
        spans: 章节跨度。
        writes: 章节键 → 目标条目。
        scope: 作用域（决定新建文件时的标题）。
        newline: 落盘换行符。
        known_keys: 本次涉及的全部章节键；``None`` = 退化为「只重建 ``writes`` 中的章节」。

    Returns:
        新文件文本（以换行符结尾）。
    """
    out: list[str] = []
    if not any(line.strip() for line in lines):
        out.extend([file_title(scope), MANAGED_COMMENT])
    touched = set(writes) if known_keys is None else set(known_keys)
    cursor = 0
    for span in spans:
        out.extend(lines[cursor : span.heading])
        if span.key is not None and span.key in touched:
            out.append(lines[span.heading])
            out.extend(_rebuild_body(lines[span.heading + 1 : span.end], writes.get(span.key, [])))
        else:
            out.extend(lines[span.heading : span.end])
        cursor = span.end
    out.extend(lines[cursor:])
    existing_keys = {span.key for span in spans}
    for key, entries in writes.items():
        if not entries or key in existing_keys:
            continue
        out.extend(["", f"## {label_for(key)}", *(entry.to_line() for entry in entries)])

    normalized: list[str] = []
    for line in out:
        if line.startswith("## ") and normalized and normalized[-1].strip():
            normalized.append("")
        normalized.append(line)
    text = "\n".join(normalized).rstrip("\n") + "\n"
    return text.replace("\n", newline) if newline != "\n" else text


def _atomic_write_text(path: Path, text: str) -> None:
    """原子写入文本（同目录临时文件 + ``os.replace``）。

    Args:
        path: 目标文件。
        text: 文本内容（已含目标换行符）。

    Raises:
        OSError: 写入或替换失败（临时文件会被清理）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(text)
            handle.flush()
        Path(tmp_name).replace(path)
    except OSError:
        if tmp_name is not None:
            with contextlib.suppress(OSError):
                Path(tmp_name).unlink()
        raise
