"""上下文预算计算（§7.2）与超预算降级（§7.6）。

规格：``my_agent-开发文档.md`` §7.1 / §7.2 / §7.6；契约：``CONTRACTS.md`` §7。

本模块是「有效上下文 / 水位 / 五件套预算」的**唯一**计算入口：

- :func:`effective_context` / :func:`watermark` 是 §7.2 的冻结公式；
- :class:`PromptBudget` 是 §7.6 要求的预算报告（供 ``/context`` 展示）；
- :func:`fit_to_budget` 实现 §7.6 的超预算丢弃优先级
  （MCP/skills 明细 → 低优先级记忆 → repomap 包明细 → 工具描述 → 系统提示词**永不裁剪**）。

所有阈值一律取自 :mod:`my_agent.config.defaults`（CONTRACTS §0.1）；
模块内只保留「展示用」的少量命名常量（见各常量注释）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from my_agent.config.defaults import (
    COMPRESS_TRIGGER_RATIO,
    MEMORY_BUDGET,
    PROMPT_BUDGET_TOKENS,
    REPOMAP_BUDGET,
    RESERVED_FIXED_TOKENS,
    RESERVED_OUTPUT_TOKENS,
    SKILLS_MCP_BUDGET,
    SYSTEM_PROMPT_BUDGET,
    TOOLS_PROMPT_BUDGET,
)

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期导入，避免运行期硬耦合
    from my_agent.config.schema import Config
    from my_agent.llm.tokenizer import TokenCounter

__all__ = [
    "BLOCK_BUDGETS",
    "BLOCK_LABELS",
    "BLOCK_MEMORY",
    "BLOCK_REPOMAP",
    "BLOCK_SKILLS_MCP",
    "BLOCK_SYSTEM",
    "BLOCK_TOOLS",
    "DROP_ORDER",
    "FittedPrompt",
    "PromptBudget",
    "context_value",
    "count_tools_schema",
    "degrade_memory_block",
    "degrade_repomap_block",
    "degrade_skills_block",
    "degrade_tools_schema",
    "effective_context",
    "fit_to_budget",
    "needs_compaction",
    "truncate_to_tokens",
    "watermark",
]

# --------------------------------------------------------------------------- #
# 块标识（§7.1 五件套；tools 块不是文本，单独用 schema 计数）
# --------------------------------------------------------------------------- #
BLOCK_SYSTEM: str = "system"
BLOCK_REPOMAP: str = "repomap"
BLOCK_TOOLS: str = "tools"
BLOCK_MEMORY: str = "memory"
BLOCK_SKILLS_MCP: str = "skills_mcp"

#: 各块的中文标签（``/context`` 报告用，与 §7.7.5 示例一致）。
BLOCK_LABELS: dict[str, str] = {
    BLOCK_SYSTEM: "系统提示词",
    BLOCK_REPOMAP: "工作目录摘要",
    BLOCK_TOOLS: "tools 提示词",
    BLOCK_MEMORY: "记忆",
    BLOCK_SKILLS_MCP: "skills/mcp",
}

#: §7.1 各块预算（合计 30k = 上游「模型上下文 − 30k（系统提示词，工具，记忆）」）。
BLOCK_BUDGETS: dict[str, int] = {
    BLOCK_SYSTEM: SYSTEM_PROMPT_BUDGET,
    BLOCK_REPOMAP: REPOMAP_BUDGET,
    BLOCK_TOOLS: TOOLS_PROMPT_BUDGET,
    BLOCK_MEMORY: MEMORY_BUDGET,
    BLOCK_SKILLS_MCP: SKILLS_MCP_BUDGET,
}

#: §7.6 丢弃优先级（**从低到高**）。系统提示词永不裁剪，故不在其中。
DROP_ORDER: tuple[str, ...] = (BLOCK_SKILLS_MCP, BLOCK_MEMORY, BLOCK_REPOMAP, BLOCK_TOOLS)

#: 工具描述降级为「一行」时的截断长度（展示细节，非行为阈值）。
TOOL_ONELINE_MAX_CHARS: int = 120
#: 工具一行描述能压到的下限（再压就失去「何时用」的信息）。
TOOL_ONELINE_MIN_CHARS: int = 24
#: :func:`truncate_to_tokens` 每轮收缩比例（数值拟合步长，非行为阈值）。
_TRUNCATE_SHRINK: float = 0.8
#: :func:`truncate_to_tokens` 最多精修轮数。
_TRUNCATE_FIT_PASSES: int = 3

#: §8.6 记忆章节优先级（值越小越先保留）。**顺序即优先级**，用子串匹配标题。
#: 权威定义在 ``memory/sections.py``；此处只复刻「降级时的保留顺序」，不复制章节内容。
MEMORY_SECTION_PRIORITY: dict[str, int] = {
    "记忆（": -1,  # 容器标题（## 记忆（全局）/（项目：x）），必须保留结构
    "长期规则与红线": 0,
    "rules_redlines": 0,
    "项目级红线": 0,
    "project_redlines": 0,
    "行为纠正": 1,
    "behavior_correction": 1,
    "关键契约": 1,
    "contracts": 1,
    "用户偏好": 2,
    "user_preference": 2,
    "任务状态": 2,
    "task_status": 2,
    "工作流偏好": 3,
    "workflow_pref": 3,
    "技术栈与命令": 3,
    "stack_commands": 3,
    "编码风格偏好": 4,
    "coding_style": 4,
    "架构与目录地图": 4,
    "architecture": 4,
    "工具与环境": 5,
    "tooling_env": 5,
    "代码规范与模式": 5,
    "code_conventions": 5,
    "跨项目知识": 6,
    "cross_project_knowledge": 6,
    "项目概览": 6,
    "overview": 6,
    "工程环境": 7,
    "engineering_env": 7,
    "决策与历史": 8,
    "decisions_history": 8,
}
#: 未在优先级表中出现的章节（最后丢）。
_MEMORY_UNKNOWN_RANK: int = 9


# --------------------------------------------------------------------------- #
# §7.2 预算计算
# --------------------------------------------------------------------------- #
def context_value(config: Config | None, name: str, default: Any) -> Any:
    """读取 ``config.context.<name>``，缺失或为 ``None`` 时回退到 defaults 常量。

    Args:
        config: 全局配置（§15.2）；``None`` 或没有 ``context`` 段时直接回退。
        name: :class:`~my_agent.config.schema.ContextConfig` 字段名（§15.3）。
        default: 回退值，一律来自 :mod:`my_agent.config.defaults`。

    Returns:
        配置值；未配置时返回 ``default``。
    """
    if config is None:
        return default
    section = getattr(config, "context", None)
    if section is None:
        return default
    value = getattr(section, name, None)
    return default if value is None else value


def effective_context(model_window: int, config: Config) -> int:
    """有效上下文 = 模型上下文 − 预留固定 − 输出预留（§7.2，C-1）。

    Args:
        model_window: 模型上下文窗口（token），取自配置 ``llm.context_window``。
        config: 全局配置；``context.reserved_*`` 可覆盖 defaults 常量。

    Returns:
        有效上下文 token 数（不做下限截断；``<= 0`` 时 :func:`watermark` 返回 ``1.0``）。
    """
    reserved_fixed = int(context_value(config, "reserved_fixed_tokens", RESERVED_FIXED_TOKENS))
    reserved_output = int(context_value(config, "reserved_output_tokens", RESERVED_OUTPUT_TOKENS))
    return model_window - reserved_fixed - reserved_output


def watermark(history_tokens: int, effective: int) -> float:
    """计算上下文水位（§7.2）。

    Args:
        history_tokens: 对话历史（含当前用户消息）的 token 估算值。
        effective: :func:`effective_context` 的结果。

    Returns:
        ``history_tokens / effective``；``effective <= 0`` 时返回 ``1.0``（永远触发压缩）。
    """
    return history_tokens / effective if effective > 0 else 1.0


def needs_compaction(value: float, *, ratio: float = COMPRESS_TRIGGER_RATIO) -> bool:
    """水位是否达到自动压缩触发点（§7.7：``watermark >= 0.80``）。

    Args:
        value: :func:`watermark` 的结果。
        ratio: 触发比例；默认 ``COMPRESS_TRIGGER_RATIO``。

    Returns:
        ``True`` 表示需要触发全文压缩。
    """
    return value >= ratio


# --------------------------------------------------------------------------- #
# 预算报告
# --------------------------------------------------------------------------- #
class PromptBudget(BaseModel):
    """五件套预算报告（CONTRACTS §7，供 ``/context`` 展示）。"""

    system: int = 0
    repomap: int = 0
    tools: int = 0
    memory: int = 0
    skills_mcp: int = 0
    total: int = 0
    dropped_blocks: list[str] = Field(
        default_factory=list, description="被降级/裁剪的块名（按触发顺序，已去重）"
    )
    limit: int = PROMPT_BUDGET_TOKENS
    warning: str = Field(default="", description="降级到底仍超预算时的告警文案")

    def over(self) -> bool:
        """是否超出五件套总预算。

        Returns:
            ``total > limit``。
        """
        return self.total > self.limit


class FittedPrompt(BaseModel):
    """:func:`fit_to_budget` 的结果：降级后的块文本 + 工具 schema + 预算报告。"""

    blocks: dict[str, str] = Field(default_factory=dict)
    tools_schema: list[dict[str, Any]] = Field(default_factory=list)
    budget: PromptBudget = Field(default_factory=PromptBudget)


# --------------------------------------------------------------------------- #
# 块降级（§7.6 优先级表）
# --------------------------------------------------------------------------- #
def degrade_skills_block(
    text: str, *, counter: TokenCounter, max_tokens: int = SKILLS_MCP_BUDGET
) -> str:
    """把 skills/MCP 块降级为「名字列表」（§7.6 优先级 1）。

    Args:
        text: 原始块文本（``- name: description`` 列表）。
        counter: token 计数器。
        max_tokens: 降级后允许的最大 token。

    Returns:
        只保留标题行与条目名的块文本；超出 ``max_tokens`` 时按 token 截断。
    """
    if not text.strip() or max_tokens <= 0:
        return ""
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            kept.append(line)
            continue
        if not stripped.startswith(("-", "*")):
            continue
        body = stripped.lstrip("-* ").strip()
        for sep in ("：", ": ", " — ", " - ", "—"):
            if sep in body:
                body = body.split(sep, 1)[0].strip()
                break
        if body:
            kept.append(f"- {body}")
    return truncate_to_tokens("\n".join(kept), max_tokens=max_tokens, counter=counter)


def degrade_memory_block(
    text: str, *, counter: TokenCounter, max_tokens: int = MEMORY_BUDGET
) -> str:
    """按 §8.6 顺序截断记忆块：先丢低优先级章节，章节内从尾部丢条目（§7.6 优先级 2）。

    Args:
        text: 注入格式的记忆块（``## 记忆（全局）`` / ``### 用户偏好`` / ``- 条目``）。
        counter: token 计数器。
        max_tokens: 允许的最大 token。

    Returns:
        截断后的记忆块；若放不下任何内容则返回空串。
    """
    if not text.strip() or max_tokens <= 0:
        return ""
    sections = _split_sections(text)
    order = sorted(range(len(sections)), key=lambda i: (_section_rank(sections[i][0]), i))
    kept: dict[int, list[str]] = {}
    used = 0
    omitted = 0
    for idx in order:
        heading, body = sections[idx]
        heading_cost = counter.count_text(heading + "\n") if heading else 0
        room = max_tokens - used - heading_cost
        if room <= 0:
            omitted += len(body)
            continue
        chosen, dropped = _trim_lines(body, counter=counter, max_tokens=room)
        omitted += dropped
        if chosen or heading:
            kept[idx] = chosen
            used += heading_cost + counter.count_text("\n".join(chosen))
    parts: list[str] = []
    for idx, (heading, _body) in enumerate(sections):
        if idx not in kept:
            continue
        chunk = "\n".join(kept[idx])
        parts.append(f"{heading}\n{chunk}" if heading else chunk)
    out = "\n".join(part for part in parts if part.strip())
    if omitted:
        out += f"\n（…已省略 {omitted} 条低优先级记忆条目）"
    return out


def degrade_repomap_block(
    text: str, *, counter: TokenCounter, max_tokens: int = REPOMAP_BUDGET
) -> str:
    """工作目录摘要只留顶层目录（§7.6 优先级 3）。

    Args:
        text: repomap 块文本（``## 工作目录摘要…`` + ``- <path>/ — <说明>``）。
        counter: token 计数器。
        max_tokens: 允许的最大 token。

    Returns:
        只含标题、根信息、顶层条目与统计行的块文本。
    """
    if not text.strip() or max_tokens <= 0:
        return ""
    kept: list[str] = []
    for index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if index == 0 or not stripped.startswith("-") or stripped.startswith("（"):
            kept.append(line)
            continue
        path = _line_path(stripped)
        if path is None or _path_depth(path) <= 1:
            kept.append(line)
    return truncate_to_tokens("\n".join(kept), max_tokens=max_tokens, counter=counter)


def degrade_tools_schema(
    tools_schema: Sequence[Mapping[str, Any]],
    *,
    counter: TokenCounter,
    max_tokens: int = TOOLS_PROMPT_BUDGET,
) -> tuple[list[dict[str, Any]], int]:
    """工具描述降级为 ``name + 一行描述``（§7.6 优先级 4）。

    **永不删除工具**（否则模型会调用不存在的工具），也**不删参数 schema**
    （删了模型就无法正确传参，与「永不整体删除」同一意图）：只压缩 ``description``。

    Args:
        tools_schema: 发往 Provider 的 ``{"type": "function", "function": {...}}`` 列表。
        counter: token 计数器。
        max_tokens: 允许的最大 token。

    Returns:
        ``(降级后的 schema 列表, 降级后的 token 数)``。
    """
    limit = TOOL_ONELINE_MAX_CHARS
    out = _oneline_tools(tools_schema, limit)
    tokens = count_tools_schema(out, counter)
    while tokens > max_tokens and limit > TOOL_ONELINE_MIN_CHARS:
        limit = max(TOOL_ONELINE_MIN_CHARS, limit // 2)
        out = _oneline_tools(tools_schema, limit)
        tokens = count_tools_schema(out, counter)
    return out, tokens


def count_tools_schema(tools_schema: Sequence[Mapping[str, Any]], counter: TokenCounter) -> int:
    """统计 tools 提示词的 token 数（§7.1 第 3 件套）。

    Args:
        tools_schema: function schema 列表。
        counter: token 计数器。

    Returns:
        所有 schema JSON 文本的 token 之和。
    """
    return sum(counter.count_text(_dumps(item)) for item in tools_schema)


def fit_to_budget(
    blocks: Mapping[str, str],
    *,
    counter: TokenCounter,
    tools_schema: Sequence[Mapping[str, Any]] | None = None,
    limit: int = PROMPT_BUDGET_TOKENS,
    block_budgets: Mapping[str, int] | None = None,
) -> FittedPrompt:
    """把五件套压进预算（§7.1 分块预算 + §7.6 总量优先级）。

    两遍处理：

    1. **分块预算**：某块超过 §7.1 的自身预算（2/8/12/6/2k）就降级该块；
    2. **总量预算**：五件套合计超过 ``PROMPT_BUDGET_TOKENS`` 时，按 §7.6 的
       :data:`DROP_ORDER`（低 → 高）继续降级，直到进入预算。

    系统提示词块**永不**被降级或丢弃（C-10）；若最终仍超预算，只在
    :attr:`PromptBudget.warning` 记告警，不牺牲系统提示词。

    Args:
        blocks: 块名 → 文本；缺省块视为空串（键取 :data:`BLOCK_LABELS`）。
        counter: token 计数器（必须与预算判断处共用同一实例，§7.2）。
        tools_schema: 第 3 件套（tools 提示词）的 function schema 列表。
        limit: 五件套总量上限，默认 30k。
        block_budgets: 覆盖分块预算（测试/调参用）。

    Returns:
        :class:`FittedPrompt`（降级后的块、schema 与预算报告）。
    """
    budgets = dict(BLOCK_BUDGETS)
    if block_budgets:
        budgets.update(block_budgets)
    work = {name: str(blocks.get(name, "") or "") for name in BLOCK_LABELS}
    tools = [dict(item) for item in (tools_schema or [])]
    degraded: list[str] = []

    for name in DROP_ORDER:
        if name == BLOCK_TOOLS:
            continue
        cap = int(budgets.get(name, 0))
        if cap <= 0 or counter.count_text(work[name]) <= cap:
            continue
        candidate = _degrade_block(name, work[name], counter=counter, max_tokens=cap)
        if candidate != work[name]:
            work[name] = candidate
            degraded.append(name)

    tools_tokens = count_tools_schema(tools, counter)
    tools_cap = int(budgets.get(BLOCK_TOOLS, 0))
    if tools and tools_cap > 0 and tools_tokens > tools_cap:
        tools, tools_tokens = degrade_tools_schema(tools, counter=counter, max_tokens=tools_cap)
        degraded.append(BLOCK_TOOLS)

    def total() -> int:
        return (
            sum(
                counter.count_text(work[name])
                for name in BLOCK_LABELS
                if name != BLOCK_TOOLS
            )
            + tools_tokens
        )

    for name in DROP_ORDER:
        if total() <= limit:
            break
        if name == BLOCK_TOOLS:
            if BLOCK_TOOLS in degraded or not tools:
                continue
            tools, tools_tokens = degrade_tools_schema(tools, counter=counter, max_tokens=tools_cap)
            degraded.append(BLOCK_TOOLS)
            continue
        cap = int(budgets.get(name, 0))
        tightened = work[name]
        for factor in (2, 4, 8):
            candidate = _degrade_block(
                name, work[name], counter=counter, max_tokens=max(1, cap // factor)
            )
            if candidate != work[name]:
                tightened = candidate
                break
        if tightened != work[name]:
            work[name] = tightened
            if name not in degraded:
                degraded.append(name)

    budget = PromptBudget(
        system=counter.count_text(work[BLOCK_SYSTEM]),
        repomap=counter.count_text(work[BLOCK_REPOMAP]),
        tools=tools_tokens,
        memory=counter.count_text(work[BLOCK_MEMORY]),
        skills_mcp=counter.count_text(work[BLOCK_SKILLS_MCP]),
        dropped_blocks=list(dict.fromkeys(degraded)),
        limit=limit,
    )
    budget.total = sum(
        (budget.system, budget.repomap, budget.tools, budget.memory, budget.skills_mcp)
    )
    if budget.over():
        budget.warning = (
            f"⚠ 五件套已按 §7.6 全量降级，仍超出预算 "
            f"{budget.total - limit} tokens（系统提示词永不裁剪）"
        )
    return FittedPrompt(blocks=work, tools_schema=tools, budget=budget)


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _degrade_block(name: str, text: str, *, counter: TokenCounter, max_tokens: int) -> str:
    """按块名分派降级策略；系统提示词原样返回（永不裁剪）。"""
    if name == BLOCK_SKILLS_MCP:
        return degrade_skills_block(text, counter=counter, max_tokens=max_tokens)
    if name == BLOCK_MEMORY:
        return degrade_memory_block(text, counter=counter, max_tokens=max_tokens)
    if name == BLOCK_REPOMAP:
        return degrade_repomap_block(text, counter=counter, max_tokens=max_tokens)
    return text


def _dumps(item: Mapping[str, Any]) -> str:
    """把 schema 项序列化为 JSON 文本（不可序列化时退化为 ``str``）。"""
    try:
        return json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):  # pragma: no cover - default=str 已兜底
        return str(item)


def _oneline_tools(tools_schema: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    """返回工具 schema 的深拷贝，``description`` 压成一行且不超过 ``limit`` 字符。"""
    out: list[dict[str, Any]] = []
    for item in tools_schema:
        copied: dict[str, Any] = json.loads(_dumps(item))
        function = copied.get("function")
        if isinstance(function, dict):
            function["description"] = _first_line(str(function.get("description", "")), limit)
        out.append(copied)
    return out


def _first_line(text: str, limit: int) -> str:
    """取文本首个非空行并截断到 ``limit`` 字符。"""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped if len(stripped) <= limit else stripped[: limit - 1].rstrip() + "…"
    return ""


def _split_sections(text: str) -> list[tuple[str, list[str]]]:
    """按 Markdown 标题切分文本为 ``(标题行, 正文行)``；无标题前导段标题为空串。"""
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if heading or body:
                sections.append((heading, body))
            heading = line
            body = []
        else:
            body.append(line)
    if heading or body:
        sections.append((heading, body))
    return sections


def _section_rank(heading: str) -> int:
    """章节保留优先级：值越小越先保留（未登记的章节最后丢）。"""
    key = heading.lstrip("#").strip()
    for label, rank in MEMORY_SECTION_PRIORITY.items():
        if label in key:
            return rank
    return _MEMORY_UNKNOWN_RANK


def _trim_lines(
    lines: Sequence[str], *, counter: TokenCounter, max_tokens: int
) -> tuple[list[str], int]:
    """在 token 预算内保留正文行（顺序不变），返回 ``(保留行, 丢弃行数)``。"""
    kept: list[str] = []
    used = 0
    dropped = 0
    for line in lines:
        if not line.strip():
            kept.append(line)
            continue
        cost = counter.count_text(line + "\n")
        if used + cost > max_tokens:
            dropped += 1
            continue
        kept.append(line)
        used += cost
    return kept, dropped


def _line_path(line: str) -> str | None:
    """从 ``- <path> — <说明>`` 行里取出 ``<path>``；取不到返回 ``None``。"""
    body = line.lstrip("-").strip()
    for sep in (" — ", " – ", " - "):
        if sep in body:
            candidate = body.split(sep, 1)[0].strip().strip("`").strip()
            return candidate if candidate and " " not in candidate else None
    return None


def _path_depth(path: str) -> int:
    """目录深度：顶层目录为 0（``src/`` → 0，``src/my_agent/`` → 1）。"""
    segments = [part for part in path.strip("/").split("/") if part]
    return max(0, len(segments) - 1)


def truncate_to_tokens(text: str, *, max_tokens: int, counter: TokenCounter) -> str:
    """按 token 预算截断文本（按字符比例估算 + 最多 3 轮精修）。

    Args:
        text: 原始文本。
        max_tokens: 允许的最大 token；``<= 0`` 时返回空串。
        counter: token 计数器。

    Returns:
        截断后的文本（未超预算时原样返回）；发生截断时追加一行省略说明。
    """
    if max_tokens <= 0:
        return ""
    tokens = counter.count_text(text)
    if tokens <= max_tokens:
        return text
    cut = max(1, int(len(text) * (max_tokens / tokens)))
    out = text[:cut]
    for _ in range(_TRUNCATE_FIT_PASSES):
        if counter.count_text(out) <= max_tokens:
            break
        cut = int(cut * _TRUNCATE_SHRINK)
        if cut <= 0:
            return ""
        out = text[:cut]
    return out.rstrip() + "\n（…已省略剩余内容以适配预算）"
