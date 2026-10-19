"""记忆注入：把记忆块拼进 system 消息（§8.6）。

策略（§8.6「启动全量注入 + 容量保护」）：

- 预算 :data:`~my_agent.config.defaults.MEMORY_INJECT_MAX_TOKENS`（默认 6k），全局 : 项目
  = :data:`~my_agent.config.defaults.MEMORY_GLOBAL_BUDGET_RATIO` :
  :data:`~my_agent.config.defaults.MEMORY_PROJECT_BUDGET_RATIO`（40% : 60%）；
- 超预算裁剪顺序：1) 低置信度 → 2) 章节优先级（**红线类最后丢**）→ 3) ``date`` 由旧到新；
- 注入去噪：**不注入 id / date / conf 元数据**，但保留章节标题（模型据此理解分类）；
- 外部只读记忆（``CLAUDE.md`` / ``MYAGENT.md``，§8.8）计入记忆预算，优先级低于本项目记忆。

规格：``my_agent-开发文档.md`` §8.6、§8.8。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from my_agent.config.defaults import (
    MEMORY_GLOBAL_BUDGET_RATIO,
    MEMORY_INJECT_MAX_TOKENS,
)
from my_agent.memory.sections import Scope, is_redline, label_for, priority_index
from my_agent.memory.store import MemoryEntry
from my_agent.models.permission import Confidence

__all__ = ["build_block"]

#: 置信度丢弃顺序（越小越先丢，§8.6 裁剪顺序第 1 档）。
_DROP_RANK: dict[Confidence, int] = {"low": 0, "medium": 1, "high": 2}


class _CounterLike(Protocol):
    """token 计数器协议（与 ``llm.tokenizer.TokenCounter`` 结构一致）。"""

    def count_text(self, text: str) -> int:
        """统计一段文本的 token 数。

        Args:
            text: 文本。

        Returns:
            token 数。
        """
        ...


class _FallbackCounter:
    """``counter`` 缺省时的兜底计数器（CJK 1 token/字，其余 4 字符/token）。"""

    def count_text(self, text: str) -> int:
        """统计文本 token 数。

        Args:
            text: 文本。

        Returns:
            估算的 token 数。
        """
        cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        return cjk + (len(text) - cjk) // 4


def build_block(
    global_entries: Sequence[MemoryEntry],
    project_entries: Sequence[MemoryEntry],
    max_tokens: int | None = None,
    counter: _CounterLike | None = None,
    external: str = "",
    project_name: str = "",
) -> str:
    """组装记忆注入块（§8.6）。

    Args:
        global_entries: 全局记忆条目。
        project_entries: 项目记忆条目。
        max_tokens: 注入预算；缺省用
            :data:`~my_agent.config.defaults.MEMORY_INJECT_MAX_TOKENS`。
        counter: token 计数器（``count_text``）；缺省用启发式兜底。
        external: 外部只读记忆文本（§8.8），最低优先级、超预算先丢。
        project_name: 项目名（渲染 ``## 记忆（项目：<name>）``）；缺省只写「项目」。

    Returns:
        注入到 system 消息末尾的 Markdown 文本；无任何记忆时返回空串。
    """
    budget = MEMORY_INJECT_MAX_TOKENS if max_tokens is None else max_tokens
    count: Callable[[str], int] = (counter or _FallbackCounter()).count_text

    global_budget = int(budget * MEMORY_GLOBAL_BUDGET_RATIO)
    project_budget = budget - global_budget

    kept_global = _trim(global_entries, global_budget, count, "global", project_name)
    kept_project = _trim(project_entries, project_budget, count, "project", project_name)
    block = _join(_render_scope(kept_global, "global", project_name),
                  _render_scope(kept_project, "project", project_name))

    # 兜底：章节标题等固定开销仍可能把总量顶出预算，按同一丢弃顺序继续裁
    victims = sorted([*kept_global, *kept_project], key=_drop_key)
    while block and victims and count(block) > budget:
        victim = victims.pop(0)
        if victim in kept_global:
            kept_global = [item for item in kept_global if item is not victim]
        else:
            kept_project = [item for item in kept_project if item is not victim]
        block = _join(_render_scope(kept_global, "global", project_name),
                      _render_scope(kept_project, "project", project_name))

    if external.strip():
        external_block = f"## 记忆（外部文件）\n{external.strip()}"
        candidate = _join(block, external_block)
        if count(candidate) <= budget:
            return candidate
    return block


def _drop_key(entry: MemoryEntry) -> tuple[int, int, int, str]:
    """裁剪排序键（小的先丢）。

    Args:
        entry: 记忆条目。

    Returns:
        ``(是否红线, 置信度档, -章节优先级, date)``：
        非红线先丢、低置信度先丢、低优先级章节先丢、旧日期先丢；红线类**最后**丢。
    """
    return (
        1 if is_redline(entry.section) else 0,
        _DROP_RANK.get(entry.confidence, 1),
        -priority_index(entry.scope, entry.section),
        entry.date,
    )


def _trim(
    entries: Sequence[MemoryEntry],
    budget: int,
    count: Callable[[str], int],
    scope: Scope,
    project_name: str,
) -> list[MemoryEntry]:
    """按丢弃顺序裁剪到预算内。

    Args:
        entries: 候选条目。
        budget: 该作用域的 token 预算。
        count: token 计数函数。
        scope: 作用域。
        project_name: 项目名（渲染用）。

    Returns:
        裁剪后的条目。
    """
    kept = list(entries)
    if count(_render_scope(kept, scope, project_name)) <= budget:
        return kept
    for victim in sorted(kept, key=_drop_key):
        kept = [item for item in kept if item is not victim]
        if count(_render_scope(kept, scope, project_name)) <= budget:
            break
    return kept


def _render_scope(entries: Sequence[MemoryEntry], scope: Scope, project_name: str) -> str:
    """渲染一个作用域的记忆块（§8.6 注入格式）。

    Args:
        entries: 条目（按文件顺序）。
        scope: 作用域。
        project_name: 项目名。

    Returns:
        形如 ``## 记忆（全局）\\n### 用户偏好\\n- ...`` 的文本；无条目时为空串。
    """
    if not entries:
        return ""
    lines = [_scope_header(scope, project_name)]
    for section, items in _group_in_order(entries):
        lines.append(f"### {label_for(section)}")
        lines.extend(f"- {item.text}" for item in items)
    return "\n".join(lines)


def _scope_header(scope: Scope, project_name: str) -> str:
    """渲染作用域标题行（§8.6）。

    Args:
        scope: 作用域。
        project_name: 项目名。

    Returns:
        ``## 记忆（全局）`` 或 ``## 记忆（项目：<name>）``。
    """
    if scope == "global":
        return "## 记忆（全局）"
    if project_name:
        return f"## 记忆（项目：{project_name}）"
    return "## 记忆（项目）"


def _group_in_order(entries: Sequence[MemoryEntry]) -> list[tuple[str, list[MemoryEntry]]]:
    """按章节首次出现顺序分组（保持文件顺序，与 §8.6 示例一致）。

    Args:
        entries: 条目序列。

    Returns:
        ``[(章节键, 条目列表), ...]``。
    """
    grouped: dict[str, list[MemoryEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.section, []).append(entry)
    return list(grouped.items())


def _join(*blocks: str) -> str:
    """用空行拼接非空块。

    Args:
        *blocks: 待拼接文本。

    Returns:
        拼接结果（空块被忽略）。
    """
    return "\n\n".join(block for block in blocks if block)
