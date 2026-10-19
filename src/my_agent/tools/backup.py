"""写前备份与 ``/undo`` 支撑（§14.5）。

没有沙盒，所以「写前备份 + 可撤销」是硬要求（【待确认假设 U17】）：

- 备份目录：``<project>/.myagent/backups/<session_id>/<n>-<sanitized_path>``。
- 清单：同目录下 ``manifest.json``，每条记录 ``{path, backup, ts, tool, call_id}``（**字段冻结**），
  另附 ``sha1``（备份时原文件指纹）与 ``sha1_after``（写入完成后登记的指纹）用于 §14.5 冲突检测。
- 淘汰：超过 :data:`~my_agent.config.defaults.BACKUP_MAX_MB` 或
  :data:`~my_agent.config.defaults.BACKUP_MAX_FILES` 时按最旧淘汰。
- 拒绝备份 ``.myagent/`` 内部文件与落盘 artifacts（§14.5「不备份」）。

规格：``my_agent-开发文档.md`` §14.5；验收：J-3、J-6。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from my_agent.config.defaults import BACKUP_MAX_FILES, BACKUP_MAX_MB
from my_agent.tools.base import atomic_write_bytes, atomic_write_text, to_posix

__all__ = ["BackupEntry", "BackupManager", "UndoResult"]

#: 内部状态目录名（备份、会话、归档、todos 等都在其下；不备份、不被 Glob/Grep 搜索）。
MYAGENT_DIR_NAME = ".myagent"

#: 清单文件名（冻结）。
MANIFEST_NAME = "manifest.json"

#: 备份文件名中路径片段的最大长度（超过则截断，规避 Windows 路径长度限制）。
MAX_SANITIZED_LEN = 120


class BackupEntry(BaseModel):
    """一条备份记录（``manifest.json`` 的元素）。

    Attributes:
        path: 原始文件绝对路径。
        backup: 备份文件绝对路径。
        ts: 备份时间（epoch 秒）。
        tool: 触发备份的工具名（``Write`` / ``Edit`` / ``MultiEdit``）。
        call_id: 触发备份的工具调用 id。
        sha1: 备份时**原文件**的 sha1（空串 = 当时文件不存在）。
        sha1_after: 写入完成后登记的**新文件** sha1（``None`` = 未登记）。
    """

    path: str
    backup: str
    ts: float
    tool: str
    call_id: str
    sha1: str = ""
    sha1_after: str | None = None

    model_config = {"extra": "ignore"}


class UndoResult(BaseModel):
    """``/undo`` 的结果（§14.5）。

    Attributes:
        ok: 是否完成恢复。
        conflict: 是否检测到「写入后文件又被修改」的冲突。
        path: 被恢复的文件路径（无记录时为 ``None``）。
        backup: 使用的备份文件路径（无记录时为 ``None``）。
        message: 给用户看的中文说明。
    """

    ok: bool
    conflict: bool = False
    path: str | None = None
    backup: str | None = None
    message: str = ""

    model_config = {"extra": "ignore"}


def sha1_of(path: Path) -> str:
    """计算文件 sha1（§14.5 冲突检测用）。

    Args:
        path: 目标文件。

    Returns:
        十六进制 sha1；文件不存在时返回空串。

    Raises:
        OSError: 读取失败（权限、是目录等）。
    """
    if not path.is_file():
        return ""
    digest = hashlib.sha1()  # 仅用于内容比对（§14.5 冲突检测），非安全用途
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BackupManager:
    """写类工具的备份管理器（§14.5）。

    方法均为同步阻塞实现（调用方负责放进 ``asyncio.to_thread``）；内部用一把可重入锁
    保护 manifest 的读改写，避免并发写工具把清单写坏。

    Args:
        project_root: 项目根目录（备份落在其 ``.myagent/backups/`` 下）。
        session_id: 会话 id（备份按会话分目录）。
    """

    def __init__(self, project_root: Path, session_id: str) -> None:
        self.project_root = Path(project_root).resolve()
        self.session_id = session_id
        self.backups_root = self.project_root / MYAGENT_DIR_NAME / "backups"
        self.session_dir = self.backups_root / session_id
        self.manifest_path = self.session_dir / MANIFEST_NAME
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def is_internal(self, path: Path) -> bool:
        """判断路径是否属于 ``.myagent/`` 内部（§14.5：不备份、不搜索）。

        Args:
            path: 待判断路径。

        Returns:
            路径任一层级含 ``.myagent`` 目录时返回 ``True``。
        """
        return MYAGENT_DIR_NAME in Path(path).resolve().parts

    def list_entries(self) -> list[BackupEntry]:
        """读取本会话的备份清单（按时间从旧到新）。

        Returns:
            备份记录列表；清单不存在或损坏时返回空列表。
        """
        with self._lock:
            return self._read_manifest()

    def latest(self) -> BackupEntry | None:
        """返回最后一条备份记录（``/undo`` 的输入）。

        Returns:
            最后一条记录；无记录时 ``None``。
        """
        entries = self.list_entries()
        return entries[-1] if entries else None

    # ------------------------------------------------------------------ #
    # 写入路径
    # ------------------------------------------------------------------ #
    def snapshot(self, path: Path, *, tool: str, call_id: str) -> Path | None:
        """在写入前备份原文件（§14.5）。

        Args:
            path: 即将被写入的文件（可不存在——新建文件无需备份）。
            tool: 触发工具名。
            call_id: 工具调用 id。

        Returns:
            备份文件路径；以下情况返回 ``None``：文件不存在、不是普通文件、
            路径位于 ``.myagent/`` 内部（**拒绝备份内部状态文件**）。

        Raises:
            OSError: 备份复制或清单写入失败（此时调用方必须放弃写入）。
        """
        target = Path(path)
        if self.is_internal(target) or not target.is_file():
            return None

        with self._lock:
            entries = self._read_manifest()
            index = len(entries) + 1
            self.session_dir.mkdir(parents=True, exist_ok=True)
            suffix = self._sanitize(target)
            backup_path = self.session_dir / f"{index}-{suffix}"
            while backup_path.exists():  # 极端情况（清单被外部删改）下保证不覆盖旧备份
                index += 1
                backup_path = self.session_dir / f"{index}-{suffix}"
            backup_path.write_bytes(target.read_bytes())
            entry = BackupEntry(
                path=str(target.resolve()),
                backup=str(backup_path.resolve()),
                ts=time.time(),
                tool=tool,
                call_id=call_id,
                sha1=sha1_of(target),
            )
            entries.append(entry)
            self._prune(entries)
            self._write_manifest(entries)
            return backup_path

    def mark_written(self, path: Path, *, call_id: str = "") -> BackupEntry | None:
        """登记「写入完成后的文件指纹」，供 §14.5 冲突检测。

        写类工具在落盘成功后调用；``/undo`` 时若当前文件 sha1 与该指纹不一致，
        说明写入之后文件又被别的进程/工具改过 → 报冲突而不是直接覆盖。

        Args:
            path: 刚写完的文件。
            call_id: 可选，用于在并发场景下定位对应的那条记录。

        Returns:
            被更新的记录；无匹配记录时返回 ``None``。

        Raises:
            OSError: 读取文件或写清单失败。
        """
        target = Path(path).resolve()
        with self._lock:
            entries = self._read_manifest()
            for entry in reversed(entries):
                if Path(entry.path) != target:
                    continue
                if call_id and entry.call_id and entry.call_id != call_id:
                    continue
                if entry.sha1_after is not None:
                    continue
                entry.sha1_after = sha1_of(target)
                self._write_manifest(entries)
                return entry
            return None

    def restore(self, entry: BackupEntry) -> None:
        """用备份内容覆盖目标文件（原子写）。

        Args:
            entry: 备份记录。

        Raises:
            FileNotFoundError: 备份文件已丢失（可能已被淘汰）。
            OSError: 写入失败。
        """
        backup_path = Path(entry.backup)
        if not backup_path.is_file():
            raise FileNotFoundError(f"备份文件已不存在：{backup_path}")
        target = Path(entry.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(target, backup_path.read_bytes())

    def undo_last(self, *, force: bool = False) -> UndoResult:
        """撤销最近一次写类工具造成的文件修改（§14.5，``/undo``）。

        Args:
            force: ``True`` 时忽略冲突直接覆盖（对应「确认覆盖？」里用户选是）。

        Returns:
            撤销结果。冲突且未 ``force`` 时 ``ok=False, conflict=True``，**不修改任何文件**。
        """
        entry = self.latest()
        if entry is None:
            return UndoResult(ok=False, message="没有可撤销的记录（本会话还没有写类工具备份）")

        target = Path(entry.path)
        current = sha1_of(target)
        if not force and entry.sha1_after is not None and current != entry.sha1_after:
            return UndoResult(
                ok=False,
                conflict=True,
                path=entry.path,
                backup=entry.backup,
                message=f"该文件在写入后又被修改，确认覆盖？{entry.path}",
            )
        try:
            self.restore(entry)
        except OSError as exc:
            return UndoResult(
                ok=False,
                path=entry.path,
                backup=entry.backup,
                message=f"恢复失败：{exc}",
            )
        with self._lock:
            entries = self._read_manifest()
            for index in range(len(entries) - 1, -1, -1):
                if entries[index].backup == entry.backup:
                    entries.pop(index)
                    break
            self._write_manifest(entries)
        return UndoResult(
            ok=True,
            path=entry.path,
            backup=entry.backup,
            message=f"已恢复 {entry.path}（来自 {entry.backup}）",
        )

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    def _sanitize(self, path: Path) -> str:
        """把绝对路径压成安全的备份文件名片段。

        Args:
            path: 原始文件路径。

        Returns:
            形如 ``src__my_agent__tools__base.py`` 的片段（含盘符时以 ``C__`` 开头）。
        """
        try:
            relative = path.resolve().relative_to(self.project_root)
        except ValueError:
            relative = Path(str(path.resolve()).replace(":", "").lstrip("\\/"))
        raw = to_posix(relative)
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "__" for ch in raw)
        if len(safe) > MAX_SANITIZED_LEN:
            safe = safe[-MAX_SANITIZED_LEN:]
        return safe or "file"

    def _read_manifest(self) -> list[BackupEntry]:
        """读取清单（容忍单条记录损坏）。

        Returns:
            记录列表；文件不存在时返回空列表。

        Raises:
            json.JSONDecodeError: 清单整体不是合法 JSON（交由调用方决定是否重建）。
        """
        if not self.manifest_path.is_file():
            return []
        raw = self.manifest_path.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        records = payload.get("entries", [])
        if not isinstance(records, list):
            return []
        entries: list[BackupEntry] = []
        for record in records:
            try:
                entries.append(BackupEntry.model_validate(record))
            except ValueError:
                continue
        return entries

    def _write_manifest(self, entries: list[BackupEntry]) -> None:
        """原子写入清单（临时文件 + ``os.replace``）。

        Args:
            entries: 全量记录。

        Raises:
            OSError: 写入失败。
        """
        self.session_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "updated_at": time.time(),
            "entries": [entry.model_dump() for entry in entries],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        atomic_write_text(self.manifest_path, text)

    def _prune(self, entries: list[BackupEntry]) -> None:
        """按 ``BACKUP_MAX_FILES`` / ``BACKUP_MAX_MB`` 淘汰最旧备份（§14.5，就地修改 ``entries``）。

        Args:
            entries: 全量记录（会被就地裁剪）。
        """
        max_bytes = BACKUP_MAX_MB * 1024 * 1024
        # 先按文件数淘汰
        while len(entries) > BACKUP_MAX_FILES:
            self._drop_oldest(entries)
        # 再按总字节数淘汰
        while entries and self._total_bytes(entries) > max_bytes:
            self._drop_oldest(entries)

    def _drop_oldest(self, entries: list[BackupEntry]) -> None:
        """删除最旧的一条备份并移出清单。

        Args:
            entries: 全量记录（会被就地修改）。
        """
        if not entries:
            return
        oldest = min(entries, key=lambda item: (item.ts, item.backup))
        entries.remove(oldest)
        try:
            Path(oldest.backup).unlink(missing_ok=True)
        except OSError:
            # 删除失败不应阻碍当前写入；下次 prune 会再次尝试（备份目录可能被杀软/同步盘占用）
            return

    @staticmethod
    def _total_bytes(entries: list[BackupEntry]) -> int:
        """统计备份总字节数。

        Args:
            entries: 全量记录。

        Returns:
            所有备份文件大小之和（缺失文件按 0 计）。
        """
        total = 0
        for entry in entries:
            try:
                total += Path(entry.backup).stat().st_size
            except OSError:
                continue
        return total
