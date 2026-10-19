"""会话持久化（§15.6 ⇒ §15.4）。

布局：``<project>/.myagent/sessions/<session_id>/``

- ``state.json``：``SessionState``（原子写）
- ``messages.jsonl``：每条 ``Message`` 一行，**只追加不重写**（恢复时的真相源）
- ``meta.json``：标题、创建/更新时间、模型、模式、token 总计

会话文件中**不存** API Key（K-5）。
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from my_agent.config.defaults import SESSION_KEEP_RECENT
from my_agent.errors import ConfigError
from my_agent.models.messages import Message
from my_agent.models.permission import SafetyMode
from my_agent.models.session import SessionState

__all__ = ["SessionMeta", "SessionStore"]

_LOG = logging.getLogger(__name__)

SESSION_ID_PREFIX = "sess-"


class SessionMeta(BaseModel):
    """会话元信息（``meta.json``）。"""

    session_id: str
    title: str = ""
    cwd: str = ""
    mode: SafetyMode = "default"
    model: str = ""
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    turn_index: int = 0
    usage_totals: dict[str, int] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class SessionStore:
    """会话的创建、加载、追加与导出。"""

    def __init__(self, project_root: Path) -> None:
        """初始化。

        Args:
            project_root: 项目根目录（会话落在其下的 ``.myagent/sessions``）。
        """
        self.project_root = Path(project_root)
        self.root = self.project_root / ".myagent" / "sessions"

    # ------------------------------------------------------------------ 路径
    def session_dir(self, session_id: str) -> Path:
        """返回会话目录。

        Args:
            session_id: 会话 id。

        Returns:
            会话目录路径。
        """
        return self.root / session_id

    def _state_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "state.json"

    def _messages_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "messages.jsonl"

    def _meta_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "meta.json"

    # ------------------------------------------------------------------ 创建
    def new_session_id(self) -> str:
        """生成新的会话 id。

        Returns:
            形如 ``sess-1a2b3c4d`` 的 id。
        """
        return f"{SESSION_ID_PREFIX}{uuid.uuid4().hex[:8]}"

    def new_session(self, *, cwd: str, mode: SafetyMode = "default", project_root: str = "") -> SessionState:
        """新建一个会话状态。

        Args:
            cwd: 工作目录。
            mode: 初始安全模式。
            project_root: 项目根（缺省用 store 的 project_root）。

        Returns:
            新的 :class:`~my_agent.models.session.SessionState`。
        """
        now = time.time()
        session = SessionState(
            session_id=self.new_session_id(),
            cwd=cwd,
            mode=mode,
            created_at=now,
            updated_at=now,
            project_root=project_root or str(self.project_root),
        )
        self.persist(session)
        return session

    # ------------------------------------------------------------------ 读取
    def latest_session_id(self) -> str | None:
        """返回最近更新的会话 id（``--continue``）。

        Returns:
            会话 id；没有任何会话时返回 ``None``。
        """
        metas = self.list_sessions()
        return metas[0].session_id if metas else None

    def list_sessions(self) -> list[SessionMeta]:
        """列出全部会话元信息，按更新时间倒序。

        Returns:
            :class:`SessionMeta` 列表。
        """
        if not self.root.is_dir():
            return []
        metas: list[SessionMeta] = []
        for entry in self.root.iterdir():
            if not entry.is_dir():
                continue
            meta = self._read_meta(entry.name)
            if meta is not None:
                metas.append(meta)
        metas.sort(key=lambda item: item.updated_at, reverse=True)
        return metas

    def _read_meta(self, session_id: str) -> SessionMeta | None:
        path = self._meta_path(session_id)
        if not path.is_file():
            state = self._state_path(session_id)
            if not state.is_file():
                return None
            try:
                data = json.loads(state.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                _LOG.warning("会话状态损坏：%s", state)
                return None
            return SessionMeta(
                session_id=session_id,
                cwd=str(data.get("cwd", "")),
                mode=data.get("mode", "default"),
                created_at=float(data.get("created_at", 0.0)),
                updated_at=float(data.get("updated_at", 0.0)),
                turn_index=int(data.get("turn_index", 0)),
            )
        try:
            return SessionMeta.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _LOG.warning("会话元信息损坏：%s", path)
            return None

    def load(self, session_id: str) -> SessionState:
        """加载会话状态；``messages.jsonl`` 为历史真相源（§15.4）。

        Args:
            session_id: 会话 id。

        Returns:
            :class:`~my_agent.models.session.SessionState`。

        Raises:
            ConfigError: 会话不存在或状态文件损坏。
        """
        state_path = self._state_path(session_id)
        if not state_path.is_file():
            raise ConfigError(
                f"会话 {session_id} 不存在", detail=f"找不到 {state_path}"
            )
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(
                f"会话 {session_id} 状态文件损坏：{exc}", detail=str(exc)
            ) from exc

        raw["history"] = []
        session = SessionState.model_validate(raw)

        messages = self.read_messages(session_id)
        if messages:
            session.history = messages
        return session

    def read_messages(self, session_id: str) -> list[Message]:
        """从 ``messages.jsonl`` 重建历史（跳过损坏行）。

        Args:
            session_id: 会话 id。

        Returns:
            :class:`~my_agent.models.messages.Message` 列表。
        """
        path = self._messages_path(session_id)
        if not path.is_file():
            return []
        out: list[Message] = []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            _LOG.warning("无法读取会话消息：%s（%s）", path, exc)
            return []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                out.append(Message.model_validate_json(line))
            except ValueError:
                _LOG.warning("跳过损坏的消息行 %s:%d", path, lineno)
        return out

    # ------------------------------------------------------------------ 写入
    def persist(self, session: SessionState) -> None:
        """原子写入 ``state.json`` 与 ``meta.json``。

        Args:
            session: 会话状态。

        Raises:
            ConfigError: 写盘失败。
        """
        session.updated_at = time.time()
        directory = self.session_dir(session.session_id)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self._atomic_write(
                self._state_path(session.session_id),
                session.model_dump_json(),
            )
            meta = SessionMeta(
                session_id=session.session_id,
                title=session.title or session.session_id,
                cwd=session.cwd,
                mode=session.mode,
                model=str(session.usage_totals.get("model", "")),
                created_at=session.created_at,
                updated_at=session.updated_at,
                turn_index=session.turn_index,
                usage_totals=dict(session.usage_totals),
            )
            self._atomic_write(self._meta_path(session.session_id), meta.model_dump_json())
        except OSError as exc:
            raise ConfigError(
                f"会话写盘失败：{exc}", detail=f"{directory}: {exc}"
            ) from exc

    def append_messages(self, session_id: str, msgs: list[Message]) -> None:
        """把一个批次的消息**追加**到 ``messages.jsonl``（只追加不重写）。

        Args:
            session_id: 会话 id。
            msgs: 待追加消息。

        Raises:
            ConfigError: 写盘失败。
        """
        if not msgs:
            return
        path = self._messages_path(session_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for msg in msgs:
                    handle.write(msg.model_dump_json())
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ConfigError(
                f"会话消息追加失败：{exc}", detail=f"{path}: {exc}"
            ) from exc

    def _atomic_write(self, path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    # ------------------------------------------------------------------ 导出
    def export(self, session_id: str, dest: Path | None = None) -> Path:
        """把会话目录打包为 zip（§5.4 ``/session export``）。

        Args:
            session_id: 会话 id。
            dest: 目标 zip 路径；缺省为 ``<project>/.myagent/exports/<id>.zip``。

        Returns:
            生成的 zip 路径。

        Raises:
            ConfigError: 会话目录不存在或打包失败。
        """
        directory = self.session_dir(session_id)
        if not directory.is_dir():
            raise ConfigError(f"会话 {session_id} 不存在", detail=str(directory))
        target = dest or (self.project_root / ".myagent" / "exports" / f"{session_id}.zip")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(directory.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(directory.parent))
        except OSError as exc:
            raise ConfigError(f"会话导出失败：{exc}", detail=str(exc)) from exc
        return target

    # ------------------------------------------------------------------ 清理
    def prune(self, keep: int = SESSION_KEEP_RECENT) -> list[str]:
        """按更新时间保留最近 ``keep`` 个会话，删除更旧的（§15.4 清理）。

        Args:
            keep: 保留数量。

        Returns:
            被删除的会话 id 列表。
        """
        metas = self.list_sessions()
        removed: list[str] = []
        for meta in metas[keep:]:
            directory = self.session_dir(meta.session_id)
            try:
                for path in sorted(directory.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    else:
                        path.rmdir()
                directory.rmdir()
                removed.append(meta.session_id)
            except OSError as exc:
                _LOG.warning("清理会话 %s 失败：%s", meta.session_id, exc)
        return removed

    def dump_meta(self, session: SessionState) -> dict[str, Any]:
        """返回 ``meta.json`` 的字典形式（测试与调试用）。

        Args:
            session: 会话状态。

        Returns:
            元信息字典。
        """
        path = self._meta_path(session.session_id)
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
