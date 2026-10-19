"""配置加载与五层合并（§15.1、§15.3）。

优先级（高 → 低）：CLI 参数 → 环境变量 ``MYAGENT_*`` → 项目配置
``<project>/.myagent/config.toml`` → 全局配置 ``~/.myagent/config.toml`` → 内置默认。

列表类（``global_ask`` / ``global_deny`` / ``irreversible_patterns`` / ``sensitive_dirs``）
**追加**而非覆盖，且内置项**不可被删除**（K-2）。
"""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from my_agent.config.schema import Config
from my_agent.errors import ConfigError

__all__ = [
    "deep_merge",
    "load_config",
    "myagent_home",
    "project_root_for",
    "read_toml",
]

_LOG = logging.getLogger(__name__)

#: 需要「追加而非覆盖」的列表键（叶子键名）。
_APPEND_LIST_KEYS = frozenset(
    {"global_deny", "global_ask", "irreversible_patterns", "sensitive_dirs"}
)

#: 项目运行时目录名。
PROJECT_DIR_NAME = ".myagent"


def myagent_home(env: Mapping[str, str] | None = None) -> Path:
    """返回 myagent 的全局目录。

    Args:
        env: 环境变量映射（默认取 ``os.environ``）。

    Returns:
        ``$MYAGENT_HOME`` 或 ``~/.myagent``。
    """
    source = os.environ if env is None else env
    raw = source.get("MYAGENT_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / PROJECT_DIR_NAME


def project_root_for(cwd: Path) -> Path:
    """从 ``cwd`` 向上查找项目根（§8.1）。

    第一个含 ``.git`` 或 ``.myagent`` 的目录即项目根；找不到则 ``cwd`` 自身。

    Args:
        cwd: 起始目录。

    Returns:
        项目根目录（绝对路径）。
    """
    start = Path(cwd).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() or (candidate / PROJECT_DIR_NAME).exists():
            return candidate
    return start


def read_toml(path: Path) -> dict[str, Any]:
    """读取 TOML 文件。

    Args:
        path: 文件路径。

    Returns:
        解析后的字典；文件不存在时返回空字典。

    Raises:
        ConfigError: 文件存在但无法解析。
    """
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"{path} 解析失败：{exc}", detail=f"{path} 解析失败：{exc}"
        ) from exc
    except OSError as exc:
        raise ConfigError(f"无法读取 {path}：{exc}", detail=str(exc)) from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} 顶层必须是表（table）", detail=str(path))
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并两个配置字典。

    规则（§15.1）：
        - 字典递归合并；
        - ``_APPEND_LIST_KEYS`` 中的列表**追加**（base 在前）；
        - 其它值由 ``override`` 整体覆盖 ``base``。

    Args:
        base: 低优先级字典（不会被修改）。
        override: 高优先级字典。

    Returns:
        合并后的新字典。
    """
    out = dict(base)
    for key, value in override.items():
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            out[key] = deep_merge(current, value)
        elif key in _APPEND_LIST_KEYS and isinstance(current, list) and isinstance(value, list):
            out[key] = [*current, *value]
        else:
            out[key] = value
    return out


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """把 ``MYAGENT_*`` 环境变量翻译成配置覆盖字典。

    支持的形式：``MYAGENT_MODE``、``MYAGENT_SKILLS__ROUTER_MODE``（``__`` 表示层级）、
    ``MYAGENT_LLM__ENDPOINTS__MAIN__BASE_URL``。

    Args:
        env: 环境变量映射。

    Returns:
        嵌套字典形式的覆盖项。
    """
    out: dict[str, Any] = {}
    for key, raw in env.items():
        if not key.startswith("MYAGENT_") or key in {"MYAGENT_HOME"}:
            continue
        path = key[len("MYAGENT_") :].lower().split("__")
        if not path or not path[0]:
            continue
        cursor = out
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = _coerce(raw)
    return out


def _coerce(raw: str) -> Any:
    """把环境变量字符串转换成合适的标量类型。

    Args:
        raw: 原始字符串。

    Returns:
        bool / int / float / 字符串。
    """
    lowered = raw.strip().lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _cli_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """把 CLI 参数翻译成嵌套配置覆盖字典。

    支持扁平键（``mode`` → ``safety.mode``、``model`` → 端点 model）与嵌套字典。

    Args:
        overrides: CLI 覆盖项。

    Returns:
        嵌套字典形式的覆盖项。
    """
    if not overrides:
        return {}
    flat_aliases = {
        "mode": ("safety", "mode"),
        "debug": ("ui", "debug"),
    }
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        if key in flat_aliases:
            section, leaf = flat_aliases[key]
            out.setdefault(section, {})[leaf] = value
        elif isinstance(value, dict):
            out = deep_merge(out, {key: dict(value)})
        else:
            out[key] = value
    return out


def _builtin_rules() -> dict[str, Any]:
    """构造内置安全规则的配置骨架（§6.3.1、§6.3.3）。

    ``security/patterns.py`` 是规则文本的权威定义；这里把它注入配置的
    ``builtin=True`` 标记位，保证项目配置**无法删除**内置项。

    Returns:
        含 ``safety.global_deny`` / ``safety.global_ask`` / ``safety.irreversible_patterns``
        的字典。
    """
    from my_agent.security.patterns import (  # 延迟导入避免循环依赖
        GLOBAL_ASK,
        GLOBAL_DENY,
        IRREVERSIBLE_PATTERNS,
    )

    def dump(rules: Sequence[Any]) -> list[dict[str, Any]]:
        return [
            {"pattern": rule.pattern, "reason": rule.reason, "builtin": True} for rule in rules
        ]

    return {
        "safety": {
            "global_deny": dump(GLOBAL_DENY),
            "global_ask": dump(GLOBAL_ASK),
            "irreversible_patterns": dump(IRREVERSIBLE_PATTERNS),
        }
    }


def _dedupe_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 ``pattern`` 去重（保留首次出现，即内置项优先）。

    Args:
        rules: 规则字典列表。

    Returns:
        去重后的列表。
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for rule in rules:
        pattern = str(rule.get("pattern", ""))
        if not pattern or pattern in seen:
            continue
        seen.add(pattern)
        out.append(rule)
    return out


def load_config(
    *,
    cwd: Path,
    cli_overrides: Mapping[str, Any] | None = None,
    config_path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    """按五层优先级合并并校验配置。

    Args:
        cwd: 工作目录。
        cli_overrides: 最高优先级覆盖项（扁平的 ``{"mode": ...}`` 或嵌套字典）。
        config_path: 显式指定的配置文件（``--config``）；给出时**只**读它作为文件层。
        env: 环境变量映射（默认 ``os.environ``）。

    Returns:
        校验通过的 :class:`~my_agent.config.schema.Config`。

    Raises:
        ConfigError: 配置解析失败或字段非法。
    """
    source = os.environ if env is None else env
    home = myagent_home(source)
    project_root = project_root_for(cwd)

    try:
        builtin = _builtin_rules()
    except ImportError:  # 骨架阶段 security/ 尚未就绪时降级为空规则集
        _LOG.debug("security.patterns 不可用，内置安全规则暂为空")
        builtin = {"safety": {}}

    merged: dict[str, Any] = dict(builtin)

    if config_path is not None:
        merged = deep_merge(merged, read_toml(Path(config_path)))
    else:
        merged = deep_merge(merged, read_toml(home / "config.toml"))
        merged = deep_merge(merged, read_toml(project_root / PROJECT_DIR_NAME / "config.toml"))

    merged = deep_merge(merged, _env_overrides(source))
    merged = deep_merge(merged, _cli_overrides(cli_overrides))

    safety = merged.get("safety")
    if isinstance(safety, dict):
        for key in ("global_deny", "global_ask", "irreversible_patterns"):
            rules = safety.get(key)
            if isinstance(rules, list):
                safety[key] = _dedupe_rules([r for r in rules if isinstance(r, dict)])

    try:
        return Config.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(
            f"配置校验失败：{_format_validation_error(exc)}",
            detail=_format_validation_error(exc),
        ) from exc


def _format_validation_error(exc: ValidationError) -> str:
    """把 pydantic 校验错误整理成一行可读文本。

    Args:
        exc: pydantic 异常。

    Returns:
        形如 ``safety.mode: Input should be ...`` 的多行文本。
    """
    lines: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        lines.append(f"{loc}: {error.get('msg', 'invalid')}")
    return "; ".join(lines) or str(exc)
