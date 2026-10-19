"""权限判定缓存（§6.7）。

规格：``my_agent-开发文档.md`` §6.7。

* **缓存键**：``sha1(tool_name + "\\x00" + canonical_json(arguments))``；
  命令类工具（``category == "execute"``）先把命令字符串归一化（去多余空白）再哈希。
* **缓存粒度**：精确键匹配，不做前缀/通配（避免误放行）。
* **禁止缓存**：阶段 0 任一命中项（``permanent_eligible=False`` → 抛
  :class:`~my_agent.errors.PermanentGrantNotAllowedError`）、含 ``*``/``?``
  通配参数的命令、``risk == "critical"`` 的项。
* **失效**：``/mode`` 切换与 ``/clear`` 都调用 :meth:`ApprovalCache.clear`。
* **持久化**：默认关闭；开启时写 ``<project>/.myagent/approvals.json``。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from my_agent.config.defaults import APPROVAL_CACHE_WILDCARD_CHARS
from my_agent.errors import PermanentGrantNotAllowedError
from my_agent.security.phase0_hard_block import COMMAND_ARG_KEYS

__all__ = ["ApprovalCache", "ApprovalEntry"]

MYAGENT_DIR_NAME = ".myagent"
"""项目级配置/审计目录名（§6.7、§6.8）。"""

APPROVALS_FILENAME = "approvals.json"
"""持久化审批文件名（§6.7）。"""

_STORE_VERSION = 1


class ApprovalEntry(BaseModel):
    """一条永久放行记录（只存摘要，不存原始参数）。

    Attributes:
        key: 缓存键（sha1 十六进制）。
        tool: 工具名。
        created_at: 写入时间戳。
        permanent_eligible: 恒为 ``True``（不可永久放行的项根本不允许入缓存）。
    """

    model_config = ConfigDict(extra="ignore")

    key: str
    tool: str
    created_at: float = Field(default_factory=time.time)
    permanent_eligible: bool = True


class ApprovalCache:
    """会话级精确键审批缓存（§6.7）。

    Args:
        project_root: 项目根目录（持久化用）；``None`` 时禁用持久化。
        persist: 是否启用 ``.myagent/approvals.json`` 持久化
            （对应 ``config.safety.persist_approvals``，默认 ``False``）。
    """

    def __init__(self, *, project_root: Path | None = None, persist: bool = False) -> None:
        self._project_root = project_root
        self._persist_enabled = bool(persist) and project_root is not None
        self._entries: dict[str, ApprovalEntry] = {}
        if self._persist_enabled:
            self.load()

    @property
    def path(self) -> Path | None:
        """持久化文件路径；未启用持久化时为 ``None``。"""
        if self._project_root is None:
            return None
        return self._project_root / MYAGENT_DIR_NAME / APPROVALS_FILENAME

    def __len__(self) -> int:
        """当前缓存条数。"""
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        """缓存中是否存在该键。"""
        return key in self._entries

    def entries(self) -> list[ApprovalEntry]:
        """返回缓存内容快照（按写入时间排序）。

        Returns:
            记录列表的副本。
        """
        return sorted(self._entries.values(), key=lambda item: item.created_at)

    def key(self, tool: str, arguments: Mapping[str, Any], *, category: str | None = None) -> str:
        """计算缓存键（§6.7）。

        Args:
            tool: 工具名。
            arguments: 工具参数。
            category: 工具类别；``"execute"`` 时对命令字符串做空白归一化后再哈希。

        Returns:
            sha1 十六进制摘要。
        """
        normalized = _normalize_arguments(arguments, is_command=category == "execute")
        payload = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha1(f"{tool}\x00{payload}".encode()).hexdigest()

    def is_cacheable(
        self, tool: str, arguments: Mapping[str, Any], *, permanent_eligible: bool
    ) -> bool:
        """判断本次调用是否**允许**写入缓存（不抛异常，供流水线预检）。

        Args:
            tool: 工具名。
            arguments: 工具参数。
            permanent_eligible: 该调用是否允许永久放行（阶段 0 命中项为 ``False``）。

        Returns:
            ``True`` 表示可缓存。
        """
        if not permanent_eligible:
            return False
        return not _has_wildcard(arguments)

    def get(
        self, tool: str, arguments: Mapping[str, Any], *, category: str | None = None
    ) -> ApprovalEntry | None:
        """按精确键查询缓存。

        Args:
            tool: 工具名。
            arguments: 工具参数。
            category: 工具类别（命令类工具需与写入时一致）。

        Returns:
            命中则返回记录，否则 ``None``。
        """
        return self._entries.get(self.key(tool, arguments, category=category))

    def put(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        *,
        permanent_eligible: bool,
        category: str | None = None,
    ) -> ApprovalEntry:
        """写入一条永久放行记录（§6.7、B-5）。

        Args:
            tool: 工具名。
            arguments: 工具参数。
            permanent_eligible: 是否允许永久放行；``False`` 时**拒绝**写入。
            category: 工具类别。

        Returns:
            写入的记录。

        Raises:
            PermanentGrantNotAllowedError: ``permanent_eligible=False``（阶段 0 命中项），
                或参数含通配符（``*``/``?``）——两者都不得永久放行。
        """
        if not permanent_eligible:
            raise PermanentGrantNotAllowedError(
                detail="该操作在任何模式下都需要逐次确认，无法永久放行",
                tool=tool,
            )
        if _has_wildcard(arguments):
            raise PermanentGrantNotAllowedError(
                detail="参数含通配符（* 或 ?），命中范围不可精确判定，不予永久放行",
                tool=tool,
            )
        entry = ApprovalEntry(
            key=self.key(tool, arguments, category=category),
            tool=tool,
        )
        self._entries[entry.key] = entry
        if self._persist_enabled:
            self.persist()
        return entry

    def clear(self) -> None:
        """清空会话缓存（``/mode`` 切换、``/clear`` 调用；§6.7）。

        持久化开启时同时删除落盘文件（更保守：不让旧模式的永久放行跨模式生效）。
        """
        self._entries.clear()
        path = self.path
        if self._persist_enabled and path is not None and path.exists():
            try:
                path.unlink()
            except OSError:
                return

    def persist(self) -> None:
        """把缓存写入 ``<project>/.myagent/approvals.json``（未启用时为空操作）。

        Raises:
            OSError: 目录/文件不可写（由调用方决定是否降级）。
        """
        path = self.path
        if not self._persist_enabled or path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _STORE_VERSION,
            "entries": [entry.model_dump() for entry in self.entries()],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        """从磁盘载入缓存（未启用持久化或文件不存在时为空操作）。

        文件损坏时**丢弃全部记录**（fail-safe：少放行＝更保守），不抛异常。
        """
        path = self.path
        if not self._persist_enabled or path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_entries = payload["entries"]
            self._entries = {
                entry.key: entry
                for entry in (ApprovalEntry.model_validate(item) for item in raw_entries)
            }
        except (OSError, KeyError, TypeError, json.JSONDecodeError, ValidationError):
            self._entries = {}


def _normalize_arguments(arguments: Mapping[str, Any], *, is_command: bool) -> dict[str, Any]:
    """生成缓存键用的规范化参数副本。

    Args:
        arguments: 原始参数。
        is_command: 是否为命令类工具（命令字符串去多余空白，§6.7）。

    Returns:
        新的参数字典（命令类工具仅归一化 :data:`COMMAND_ARG_KEYS` 中的字符串值）。
    """
    if not is_command:
        return dict(arguments)
    normalized: dict[str, Any] = dict(arguments)
    for key in COMMAND_ARG_KEYS:
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = " ".join(value.split())
    return normalized


def _has_wildcard(value: Any) -> bool:
    """递归判断参数中是否含通配符（§6.7：``*``/``?`` 的参数不缓存）。

    Args:
        value: 任意参数值。

    Returns:
        命中任一通配符字符返回 ``True``。
    """
    if isinstance(value, str):
        return any(char in value for char in APPROVAL_CACHE_WILDCARD_CHARS)
    if isinstance(value, Mapping):
        return any(_has_wildcard(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_wildcard(item) for item in value)
    return False
