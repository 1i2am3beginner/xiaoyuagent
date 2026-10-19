"""审计日志与打码（§6.8）。

规格：``my_agent-开发文档.md`` §6.8。

* 路径：``<project>/.myagent/audit/audit-YYYY-MM-DD.jsonl``，一行一条，
  ``ensure_ascii=False``。
* 写入**异步且不阻塞**判定（队列 + :func:`asyncio.create_task`），
  退出时 :meth:`AuditWriter.aclose` 必须被 ``await``。
* :func:`redact` 同时供 UI 复用（§16.3：参数展示必须复用本函数）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import Counter, deque
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from my_agent.models.permission import AuditRecord

__all__ = [
    "FILE_CONTENT_KEYS",
    "REDACTED",
    "AuditWriter",
    "audit_file_name",
    "redact",
]

REDACTED = "***REDACTED***"
"""被判定为敏感值的替换文案（§6.8）。"""

REDACT_KEY_PATTERN = (
    r"(?i)(key|token|secret|password|passwd|pwd|credential|authorization|api[-_]?key)"
)
"""§6.8 键名打码正则。"""

_SECRET_KEY_RE = re.compile(REDACT_KEY_PATTERN)
_SECRET_VALUE_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|ghp_[A-Za-z0-9]{8,})")
_BASE64_LIKE_RE = re.compile(r"^[A-Za-z0-9+/=_\-]{40,}$")

BASE64_MIN_LENGTH = 40
"""§6.8：长 base64 判定的最小长度（≥40 字符）。"""

BASE64_MIN_ENTROPY_BITS = 4.0
"""§6.8：高熵判定的最小 Shannon 熵（bits/char）。"""

FILE_CONTENT_KEYS: frozenset[str] = frozenset({"content"})
"""文件内容类参数键名（§6.8：只记 sha1 与前 200 字符）。"""

FILE_CONTENT_PREVIEW_CHARS = 200
"""§6.8：文件内容预览长度。"""

AUDIT_DIR_NAME = "audit"
"""审计子目录名（§6.8）。"""


def audit_file_name(day: date | None = None) -> str:
    """给出某一天的审计文件名（§6.8）。

    Args:
        day: 日期；缺省为今天。

    Returns:
        形如 ``audit-2026-05-10.jsonl`` 的文件名。
    """
    return f"audit-{(day or date.today()):%Y-%m-%d}.jsonl"


def _shannon_entropy(text: str) -> float:
    """计算字符串的 Shannon 熵（bits/char）。

    Args:
        text: 待计算字符串。

    Returns:
        熵值；空串返回 ``0.0``。
    """
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


def _is_high_entropy_secret(value: str) -> bool:
    """判断值是否形如「长 base64 且高熵」（§6.8）。

    Args:
        value: 待判断字符串。

    Returns:
        命中返回 ``True``。
    """
    if len(value) < BASE64_MIN_LENGTH or not _BASE64_LIKE_RE.match(value):
        return False
    return _shannon_entropy(value) >= BASE64_MIN_ENTROPY_BITS


def _content_digest(value: str) -> dict[str, Any]:
    """把文件内容参数压成「sha1 + 前 200 字符」（§6.8）。

    Args:
        value: 文件内容。

    Returns:
        含 ``sha1``/``preview``/``chars``/``truncated`` 的字典。
    """
    return {
        "sha1": hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest(),
        "preview": value[:FILE_CONTENT_PREVIEW_CHARS],
        "chars": len(value),
        "truncated": len(value) > FILE_CONTENT_PREVIEW_CHARS,
    }


def _redact_value(key: str, value: Any) -> Any:
    """对单个键值对做打码（递归下钻 dict/list）。

    Args:
        key: 键名（用于键名规则与文件内容规则）。
        value: 值。

    Returns:
        打码后的值。
    """
    if _SECRET_KEY_RE.search(key):
        return REDACTED
    if isinstance(value, str):
        if key.lower() in FILE_CONTENT_KEYS:
            return _content_digest(value)
        if _SECRET_VALUE_RE.search(value) or _is_high_entropy_secret(value):
            return REDACTED
        return value
    if isinstance(value, Mapping):
        return {str(item_key): _redact_value(str(item_key), item) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(key, item) for item in value]
    return value


def redact(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """按 §6.8 对工具参数打码（UI 展示复用，§16.3）。

    三条规则：键名命中 :data:`REDACT_KEY_PATTERN` → 整体替换；值形如
    ``sk-...``/``ghp_...``/长高熵 base64 → 替换；``content`` 类参数 → 只留
    sha1 与前 :data:`FILE_CONTENT_PREVIEW_CHARS` 字符。

    Args:
        arguments: 原始工具参数。

    Returns:
        打码后的新字典（不修改入参）。
    """
    return {str(key): _redact_value(str(key), value) for key, value in arguments.items()}


class AuditWriter:
    """异步审计写入器（§6.8）。

    Args:
        audit_dir: 审计目录（通常 ``<project>/.myagent/audit``）。
        enabled: 是否启用（``config.safety.audit_enabled``）；关闭时 :meth:`submit`
            为无副作用空操作。
    """

    def __init__(self, audit_dir: Path, *, enabled: bool = True) -> None:
        self._audit_dir = Path(audit_dir)
        self._enabled = bool(enabled)
        self._pending: deque[str] = deque()
        self._task: asyncio.Task[None] | None = None
        self._write_error: OSError | None = None
        self._closed = False
        self._written = 0

    @property
    def enabled(self) -> bool:
        """审计是否启用。"""
        return self._enabled

    @property
    def audit_dir(self) -> Path:
        """审计目录。"""
        return self._audit_dir

    @property
    def written(self) -> int:
        """已落盘条数。"""
        return self._written

    @property
    def pending(self) -> int:
        """尚未落盘的队列长度。"""
        return len(self._pending)

    def submit(self, rec: AuditRecord) -> None:
        """非阻塞入队一条审计记录（§6.8）。

        无运行中的事件循环（同步上下文）或已 :meth:`aclose` 时改为同步落盘，
        保证审计记录**永不静默丢失**。

        Args:
            rec: 审计记录。
        """
        if not self._enabled:
            return
        line = json.dumps(rec.model_dump(mode="json"), ensure_ascii=False)
        if self._closed:
            self._write_lines([line])
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._write_lines([line])
            return
        self._pending.append(line)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        """把队列内容批量落盘（后台任务）。"""
        while self._pending:
            batch: list[str] = []
            while self._pending:
                batch.append(self._pending.popleft())
            try:
                self._write_lines(batch)
            except OSError as exc:
                self._write_error = exc
                return

    async def flush(self) -> None:
        """等待队列落盘（进程退出前必须 ``await``）。

        Raises:
            OSError: 落盘失败（审计不可用时必须让调用方感知）。
        """
        task = self._task
        if task is not None and not task.done():
            await task
        if self._pending:
            batch: list[str] = []
            while self._pending:
                batch.append(self._pending.popleft())
            self._write_lines(batch)
        if self._write_error is not None:
            raise self._write_error

    async def aclose(self) -> None:
        """flush 并关闭（之后再 :meth:`submit` 会同步落盘）。

        Raises:
            OSError: 落盘失败。
        """
        await self.flush()
        self._closed = True

    def _write_lines(self, lines: list[str]) -> None:
        """把若干行按日期分组追加写盘。

        Args:
            lines: 已序列化的 JSONL 行（不含换行符）。

        Raises:
            OSError: 目录创建或写文件失败。
        """
        if not lines:
            return
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        path = self._audit_dir / audit_file_name()
        with path.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(f"{line}\n")
        self._written += len(lines)
