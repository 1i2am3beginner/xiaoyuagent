"""阶段 0：硬拦截（§6.3）。

规格：``my_agent-开发文档.md`` §6.3。三条子规则按顺序求值，命中即返回，
不再进入后续阶段（§6.2）：

1. ``0.1`` 全局 deny/ask 名单（§6.3.1）：匹配范围为「所有 ``category == "execute"``
   工具的命令字符串」以及「文件写类工具的目标路径」。
2. ``0.2`` 敏感路径（§6.3.2，**仅写操作**）：分段精确匹配，``.gitignore`` 不命中。
3. ``0.3`` 不可逆操作（§6.3.3）：仅对 ``execute`` 类工具的命令字符串做大小写不敏感匹配。

本阶段的 ``ask`` 一律带 ``permanent_eligible=False``：即使用户选「永久放行」，
:meth:`my_agent.security.cache.ApprovalCache.put` 也会抛
:class:`~my_agent.errors.PermanentGrantNotAllowedError`（§6.3.3、B-5）。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from my_agent.config.defaults import SENSITIVE_DIR_SEGMENTS, WRITE_CATEGORIES
from my_agent.config.schema import Config, Rule
from my_agent.errors import ConfigError
from my_agent.models.permission import Decision, Risk
from my_agent.models.tools import ToolInvocation
from my_agent.security.patterns import (
    GLOBAL_ASK,
    GLOBAL_DENY,
    IRREVERSIBLE_PATTERNS,
    merge_dirs,
    merge_rules,
)
from my_agent.security.phase1_precheck import extract_target_path

__all__ = [
    "COMMAND_ARG_KEYS",
    "Phase0Outcome",
    "hard_block",
    "hits_global_list",
    "hits_irreversible",
    "hits_sensitive_path",
]

COMMAND_ARG_KEYS: tuple[str, ...] = ("command", "cmd", "script", "shell")
"""命令字符串可能出现的参数键名（§14.2 Bash 工具用 ``command``；其余为兼容别名）。"""


class Phase0Outcome(BaseModel):
    """阶段 0 命中结果。

    Attributes:
        kind: 命中的子规则类别。
        decision: 该子规则给出的终态裁决。
        risk: 风险级别（不可逆/全局拒绝 = ``critical``，敏感路径/全局询问 = ``high``）。
        detail: 中文理由（进 ``phase0.detail``、审计 ``reason`` 与询问卡片）。
        pattern: 命中的正则（名单类规则才有）。
        path: 命中的目标路径（敏感路径类规则才有）。
        permanent_eligible: 恒为 ``False``（阶段 0 命中一律禁止永久放行）。
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    kind: Literal["global_deny", "global_ask", "sensitive_path", "irreversible"]
    decision: Decision
    risk: Risk
    detail: str
    pattern: str | None = None
    path: str | None = None
    permanent_eligible: bool = Field(default=False)

    def as_trace(self) -> dict[str, Any]:
        """转成 :attr:`my_agent.models.permission.PermissionDecision.phase0` 的痕迹字典。

        Returns:
            含 ``hit``/``detail``/``risk``/``permanent_eligible`` 等键的字典。
        """
        return {
            "hit": self.kind,
            "detail": self.detail,
            "risk": self.risk,
            "pattern": self.pattern,
            "path": self.path,
            "permanent_eligible": self.permanent_eligible,
        }


@lru_cache(maxsize=128)
def _compile(
    rules: tuple[tuple[str, str], ...], ignore_case: bool
) -> tuple[tuple[re.Pattern[str], Rule], ...]:
    """编译规则表（带缓存）。

    Args:
        rules: ``(pattern, reason)`` 元组序列。
        ignore_case: 是否大小写不敏感（§6.3.3 不可逆清单为 ``True``）。

    Returns:
        编译后的 ``(正则, Rule)`` 元组。

    Raises:
        ConfigError: 规则中的正则非法（用户配置写错时必须显式报错，不静默忽略）。
    """
    flags = re.IGNORECASE if ignore_case else 0
    compiled: list[tuple[re.Pattern[str], Rule]] = []
    for pattern, reason in rules:
        try:
            compiled.append((re.compile(pattern, flags), Rule(pattern=pattern, reason=reason)))
        except re.error as exc:
            raise ConfigError(detail=f"安全规则正则非法：{pattern!r}（{exc}）") from exc
    return tuple(compiled)


def _key(rules: Sequence[Rule]) -> tuple[tuple[str, str], ...]:
    """把规则序列压成可哈希的缓存键。

    Args:
        rules: 规则序列。

    Returns:
        ``(pattern, reason)`` 元组。
    """
    return tuple((rule.pattern, rule.reason) for rule in rules)


def hits_global_list(text: str, rules: Sequence[Rule]) -> Rule | None:
    """在文本中查找命中的全局名单规则（§6.3.1，大小写敏感）。

    Args:
        text: 待匹配文本（命令行或目标路径）。
        rules: 名单规则（应已用 :func:`~my_agent.security.patterns.merge_rules` 合并内置项）。

    Returns:
        第一条命中的规则；无命中返回 ``None``。

    Raises:
        ConfigError: 规则中的正则非法。
    """
    if not text:
        return None
    for pattern, rule in _compile(_key(rules), False):
        if pattern.search(text):
            return rule
    return None


def hits_irreversible(command: str, extra: Sequence[Rule]) -> Rule | None:
    """在命令字符串中查找命中的不可逆操作（§6.3.3，大小写不敏感）。

    Args:
        command: 命令行字符串。
        extra: 配置追加的不可逆规则（内置项始终参与匹配，不可被删除）。

    Returns:
        第一条命中的规则（内置项优先）；无命中返回 ``None``。

    Raises:
        ConfigError: 规则中的正则非法。
    """
    if not command:
        return None
    rules = merge_rules(IRREVERSIBLE_PATTERNS, extra) if extra else IRREVERSIBLE_PATTERNS
    for pattern, rule in _compile(_key(rules), True):
        if pattern.search(command):
            return rule
    return None


def hits_sensitive_path(
    path: str, cwd: str, *, extra_dirs: Sequence[str] | None = None
) -> str | None:
    """判断路径是否命中敏感目录（§6.3.2，**路径分段**匹配）。

    关键点（Windows 兼容）：先把反斜杠统一为 ``/``，再分段做**精确**比较，
    因此 ``.git/config``、``.\\.git\\config`` 命中，而 ``.gitignore`` **不**命中。
    除原始路径外，还会把相对路径按 ``cwd`` 解析成绝对路径后再匹配一次（更保守：
    符号链接或相对路径绕不出敏感目录）。

    Args:
        path: 目标路径（绝对或相对）。
        cwd: 工作目录，用于解析相对路径。
        extra_dirs: 配置追加的敏感目录段（内置 :data:`SENSITIVE_DIR_SEGMENTS` 始终生效）。

    Returns:
        命中的目录段名；未命中返回 ``None``。
    """
    segments = merge_dirs(SENSITIVE_DIR_SEGMENTS, extra_dirs or ())
    hit = _match_segments(path, segments)
    if hit is not None:
        return hit
    resolved = _resolve_text(path, cwd)
    if resolved is None:
        return None
    return _match_segments(resolved, segments)


def _resolve_text(path: str, cwd: str) -> str | None:
    """把相对路径按 ``cwd`` 拼成绝对文本（纯字符串拼接，不访问文件系统）。

    Args:
        path: 目标路径。
        cwd: 工作目录。

    Returns:
        拼接后的绝对路径文本；``path`` 为空、已是绝对路径或 ``cwd`` 为空时返回 ``None``。
    """
    if not path or _is_absolute(path):
        return None
    base = cwd.rstrip("/\\") if cwd else ""
    if not base:
        return None
    return f"{base}/{path}"


def _is_absolute(path: str) -> bool:
    """粗略判断路径是否为绝对路径（同时认 Windows 盘符与 POSIX 根）。

    Args:
        path: 路径字符串。

    Returns:
        绝对路径返回 ``True``。
    """
    norm = path.replace("\\", "/")
    return norm.startswith("/") or (len(norm) > 1 and norm[1] == ":")


def _match_segments(path: str, segments: Sequence[str]) -> str | None:
    """按 ``/`` 分段做精确匹配（§6.3.2 的实现规格）。

    Args:
        path: 路径字符串。
        segments: 敏感目录段集合。

    Returns:
        命中的目录段名；未命中返回 ``None``。
    """
    parts = [part for part in path.replace("\\", "/").split("/") if part not in ("", ".")]
    for part in parts:
        if part in segments:
            return part
    return None


def _command_texts(arguments: Mapping[str, Any]) -> list[str]:
    """取出命令行候选文本（§6.3.1 的匹配范围）。

    优先取 :data:`COMMAND_ARG_KEYS` 中的字符串参数；一个都没有时退化为
    「拼接所有字符串参数」，宁可多匹配也不漏匹配。

    Args:
        arguments: 工具调用参数。

    Returns:
        待匹配的命令字符串列表（可能为空）。
    """
    texts: list[str] = []
    for key in COMMAND_ARG_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value)
    if texts:
        return texts
    return [value for value in arguments.values() if isinstance(value, str) and value.strip()]


def _path_texts(arguments: Mapping[str, Any]) -> list[str]:
    """取出写类工具的目标路径候选（§6.3.1/§6.3.2 的匹配范围）。

    Args:
        arguments: 工具调用参数。

    Returns:
        目标路径列表（无路径时为空）。
    """
    target = extract_target_path(arguments)
    return [target] if target else []


#: 从命令行里提取「看起来像路径」的片段（§6.3.2：execute 视为潜在写操作）。
#: 只用于敏感路径的**分段匹配**，不追求 shell 语义完整——宁可多匹配也不漏匹配。
_COMMAND_TOKEN_RE = re.compile(r"[^\s\"'|&;<>()]+")

#: 触发扫描的敏感目录名（与 SENSITIVE_DIR_SEGMENTS 同源，用于快速筛选）。
def _paths_in_command(command: str) -> list[str]:
    """从命令行文本中提取可能的目标路径片段。

    ``echo hi > .git/config``、``Remove-Item .vscode/settings.json``、
    ``cat .bin/tool`` 都会给出对应片段。命中判断仍由
    :func:`hits_sensitive_path` 的分段精确匹配完成（因此 ``.gitignore`` 不会误命中）。

    Args:
        command: 命令行字符串。

    Returns:
        候选路径片段列表（可能为空）。
    """
    candidates: list[str] = []
    for token in _COMMAND_TOKEN_RE.findall(command):
        cleaned = token.strip().strip("\"'")
        if not cleaned or cleaned.startswith("-"):
            continue
        # 只保留含敏感段名或含路径分隔符的片段，避免把普通单词当路径
        normalized = cleaned.replace("\\", "/")
        segments = [part for part in normalized.split("/") if part not in ("", ".")]
        if not segments:
            continue
        if any(segment in SENSITIVE_DIR_SEGMENTS for segment in segments):
            candidates.append(normalized)
    return candidates


def hard_block(inv: ToolInvocation, *, config: Config, cwd: str) -> Phase0Outcome | None:
    """阶段 0 硬拦截（§6.3）。

    求值顺序：``0.1`` 全局 deny → ``0.1`` 全局 ask → ``0.2`` 敏感路径（仅写操作）
    → ``0.3`` 不可逆操作（仅 execute 命令）。

    Args:
        inv: 待判定调用。
        config: 全局配置（读取 ``safety.global_deny`` / ``global_ask`` /
            ``irreversible_patterns`` / ``sensitive_dirs``，均为**追加**语义）。
        cwd: 工作目录（相对路径的解析基准）。

    Returns:
        命中时的 :class:`Phase0Outcome`；未命中返回 ``None``。
    """
    safety = config.safety
    deny_rules = merge_rules(GLOBAL_DENY, safety.global_deny)
    ask_rules = merge_rules(GLOBAL_ASK, safety.global_ask)
    extra_dirs = [d for d in safety.sensitive_dirs if d not in SENSITIVE_DIR_SEGMENTS]

    category = inv.spec.category
    is_write_class = category in WRITE_CATEGORIES
    is_read_only = bool(inv.spec.is_read_only)

    command_texts = _command_texts(inv.arguments) if category == "execute" else []
    path_texts = _path_texts(inv.arguments) if is_write_class else []
    # §6.3.2：execute 视为潜在写操作 → 命令行中的目标路径同样要过敏感路径检查
    command_path_texts: list[str] = []
    if category == "execute":
        for command in command_texts:
            command_path_texts.extend(_paths_in_command(command))

    for text in [*command_texts, *path_texts]:
        rule = hits_global_list(text, deny_rules)
        if rule is not None:
            return Phase0Outcome(
                kind="global_deny",
                decision="deny",
                risk="critical",
                detail=f"命中全局拒绝名单「{rule.reason}」（正则 /{rule.pattern}/），已直接拒绝",
                pattern=rule.pattern,
            )
    for text in [*command_texts, *path_texts]:
        rule = hits_global_list(text, ask_rules)
        if rule is not None:
            return Phase0Outcome(
                kind="global_ask",
                decision="ask",
                risk="high",
                detail=f"命中全局确认名单「{rule.reason}」（正则 /{rule.pattern}/），需逐次确认",
                pattern=rule.pattern,
            )

    if is_write_class and not is_read_only:
        for path in [*path_texts, *command_path_texts]:
            segment = hits_sensitive_path(path, cwd, extra_dirs=extra_dirs)
            if segment is not None:
                return Phase0Outcome(
                    kind="sensitive_path",
                    decision="ask",
                    risk="high",
                    detail=(
                        f"命中敏感路径段「{segment}」：该目录下的写操作在任何模式下都需逐次确认，"
                        "无法永久放行"
                    ),
                    path=path,
                )

    for command in command_texts:
        rule = hits_irreversible(command, safety.irreversible_patterns)
        if rule is not None:
            return Phase0Outcome(
                kind="irreversible",
                decision="ask",
                risk="critical",
                detail=f"命中不可逆操作「{rule.reason}」（正则 /{rule.pattern}/），在任何模式下都需逐次确认",
                pattern=rule.pattern,
            )

    return None
