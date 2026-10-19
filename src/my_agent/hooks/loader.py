"""``hooks.yaml`` 加载与环境变量展开（§11.2）。

读取顺序（**project first**，符合「本地优先」直觉与 §15.1 的项目优先原则）：

1. ``<project>/.myagent/hooks.yaml``（默认文件名，可由 ``config.hooks.project_file`` 覆盖）
2. ``<home>/hooks.yaml``（默认文件名，可由 ``config.hooks.global_file`` 覆盖）
3. ``config.hooks.hooks``（若调用方已在内存中装配了 hook —— 例如测试）

所有非法条目（事件名未知、YAML 解析失败、缺 ``command``）一律 **警告 + 跳过**，
绝不抛异常导致启动崩溃（§11.2 未定义处按「宽容加载」实现）。
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from my_agent.config.schema import Config, HookSpec
from my_agent.hooks.events import HookEvent

__all__ = [
    "PROJECT_DIR_NAME",
    "expand_env",
    "expand_mapping",
    "load_hooks",
    "read_hooks_file",
]

_LOG = logging.getLogger(__name__)

#: 默认运行时目录名（与 ``config.loader.PROJECT_DIR_NAME`` 一致；此处独立定义避免循环依赖）。
PROJECT_DIR_NAME = ".myagent"

#: ``${VAR}`` 形式的占位符（§11.2 示例 ``env: {PROJECT: "${PWD}"}``）。
_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: str, *, extra: Mapping[str, str] | None = None) -> str:
    """展开字符串中的 ``${VAR}`` 占位符（§11.2）。

    未定义的变量**原样保留**（不报错），保证 ``hooks.yaml`` 在任何环境下都能加载。

    Args:
        value: 待展开的字符串。
        extra: 额外变量（优先级高于 ``os.environ``），例如 ``{"PWD": str(cwd)}``。

    Returns:
        展开后的字符串。
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if extra is not None and name in extra:
            return extra[name]
        return os.environ.get(name, match.group(0))

    return _VAR_PATTERN.sub(_replace, value)


def _identifier(value: Any) -> str:
    """把未知输入安全地转成日志用字符串。

    Args:
        value: 任意输入。

    Returns:
        ``str(value)``（``env`` 值按 YAML 约定都是标量，不会抛异常）。
    """
    return str(value)


def expand_mapping(
    mapping: Mapping[str, Any], *, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """展开 ``env`` 映射中的每个值。

    Args:
        mapping: 原始 ``env`` 映射（值可为非字符串，会被 ``str`` 化）。
        extra: 额外变量（如 ``PWD``）。

    Returns:
        展开后的 ``dict[str, str]``。
    """
    return {str(key): expand_env(_identifier(val), extra=extra) for key, val in mapping.items()}


def _parse_event(raw: Any) -> HookEvent | None:
    """把 YAML 中的事件名解析为 :class:`HookEvent`。

    Args:
        raw: 事件名（字符串）。

    Returns:
        解析成功时返回枚举成员；未知事件名返回 ``None``（调用方警告并跳过）。
    """
    if isinstance(raw, HookEvent):
        return raw
    try:
        return HookEvent(str(raw))
    except ValueError:
        return None


def _build_spec(entry: Any, *, source: Path, index: int) -> HookSpec | None:
    """把一条 YAML 条目转成 :class:`HookSpec`（非法 → ``None`` + 警告）。

    Args:
        entry: YAML 条目（应为 ``dict``，或已是 :class:`HookSpec`）。
        source: 来源文件（仅用于日志）。
        index: 条目序号（仅用于日志）。

    Returns:
        合法条目返回模型；否则 ``None``。
    """
    if isinstance(entry, HookSpec):
        return entry
    if not isinstance(entry, dict):
        _LOG.warning("hooks: %s 第 %d 条不是映射，已跳过", source, index)
        return None

    event = _parse_event(entry.get("event"))
    if event is None:
        _LOG.warning(
            "hooks: %s 第 %d 条事件名 %r 未知，已跳过（合法值：%s）",
            source,
            index,
            entry.get("event"),
            ", ".join(member.value for member in HookEvent),
        )
        return None

    try:
        spec = HookSpec.model_validate({**entry, "event": event.value})
    except ValidationError as exc:
        _LOG.warning("hooks: %s 第 %d 条校验失败（%s），已跳过", source, index, exc.error_count())
        return None

    if spec.timeout_s <= 0:
        _LOG.warning("hooks: %s 第 %d 条 timeout_s=%s 非正数，已跳过", source, index, spec.timeout_s)
        return None
    if spec.type == "command" and not spec.command:
        _LOG.warning("hooks: %s 第 %d 条缺少 command，已跳过", source, index)
        return None
    if spec.type == "python" and not spec.module:
        _LOG.warning("hooks: %s 第 %d 条缺少 module，已跳过", source, index)
        return None

    extra = {"PWD": str(source.parent.parent)}
    return spec.model_copy(update={"env": expand_mapping(spec.env, extra=extra)})


def read_hooks_file(path: Path, *, extra: Mapping[str, str] | None = None) -> list[HookSpec]:
    """读取单个 ``hooks.yaml`` 并预展开 ``env``。

    支持两种顶层结构：``{"hooks": [...]}``（§11.2）或裸列表 ``[...]``。
    条目级 ``env`` 的 ``${VAR}``/``${PWD}`` 已在此展开（:func:`load_hooks` 会再展开一次，
    覆盖 ``PWD`` 的最终取值）。

    Args:
        path: 文件路径；不存在时返回空列表。
        extra: 展开 ``env`` 时额外的变量映射（缺省只用 ``os.environ``）。

    Returns:
        合法条目对应的 :class:`HookSpec` 列表（非法条目被跳过并记录警告）。
    """
    if not path.is_file():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
        _LOG.warning("hooks: 无法读取 %s（%s），已跳过该文件", path, exc)
        return []

    if raw is None:
        return []
    if isinstance(raw, dict):
        entries = raw.get("hooks", [])
    else:
        entries = raw
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        _LOG.warning("hooks: %s 的 hooks 段不是列表，已跳过", path)
        return []

    context = dict(extra or {})
    context.setdefault("PWD", str(path.parent.parent))
    specs: list[HookSpec] = []
    for index, entry in enumerate(entries):
        spec = _build_spec(entry, source=path, index=index)
        if spec is None:
            continue
        specs.append(spec.model_copy(update={"env": expand_mapping(spec.env, extra=context)}))
    return specs


def _config_hooks(config: Config | None) -> list[HookSpec]:
    """取出 ``config.hooks.hooks`` 中已装配的 hook。

    Args:
        config: 全局配置；``None`` 时返回空列表。

    Returns:
        配置内的 :class:`HookSpec` 列表。
    """
    if config is None:
        return []
    return [spec for spec in config.hooks.hooks if isinstance(spec, HookSpec)]


def load_hooks(project_root: Path, home: Path, config: Config) -> list[HookSpec]:
    """加载并合并全部 hook 声明（§11.2、CONTRACTS §17.3）。

    合并顺序：项目文件 → 全局文件 → ``config.hooks.hooks``；完全相同（事件 + matcher +
    type + 命令/模块）的重复条目只保留首次出现。

    Args:
        project_root: 项目根目录（``<project>/.myagent/hooks.yaml``）。
        home: 全局目录（``~/.myagent/hooks.yaml``）。
        config: 全局配置（提供文件名与内联 hook）。

    Returns:
        按执行顺序排列的 :class:`HookSpec` 列表；任何解析失败都只记录警告（返回
        已成功解析的部分，绝不抛异常）。
    """
    project_path = Path(project_root) / config.hooks.project_file
    home_path = Path(home) / config.hooks.global_file
    context = {"PWD": str(project_root)}

    merged: list[HookSpec] = []
    seen: set[tuple[str, str, str, str]] = set()
    for spec in (
        *read_hooks_file(project_path, extra=context),
        *read_hooks_file(home_path, extra=context),
        *_config_hooks(config),
    ):
        key = (spec.event, spec.matcher or "", spec.type, spec.command or spec.module or "")
        if key in seen:
            _LOG.debug("hooks: 重复条目 %s 已忽略", key)
            continue
        seen.add(key)
        merged.append(spec)
    return merged
