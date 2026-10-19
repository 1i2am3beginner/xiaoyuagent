"""工具调用参数容错解析链（§13.3 加固）。

背景
====

模型流式返回的 ``tool_calls[].function.arguments`` 常出现「几乎合法」的 JSON：
被 markdown 围栏包住、键没加引号、用了 Python 字面量、少一个右括号……旧实现直接
``json.loads`` 失败即放弃，只能等模型下一轮自纠。本模块把「解析 → 修复 → 校验 →
下发」做成一条**永不抛出**的链：

.. code-block:: text

    原始 LLM 输出
      ↓
    2.1 预处理：正则剥离 markdown（```json … ``` 围栏、``` 围栏、前导说明文字），
                提取平衡的 {} 块（:func:`strip_markdown` / :func:`extract_json_block`）
      ↓
    2.2 标准 json.loads
          ✅ 成功 → 直接返回（``ArgParseStatus.OK``）
          ❌ 失败 → 降级：容错修复解析（``json_repair`` → 内置启发式）
                ✅ 修复成功 → 校验 schema（Pydantic）→ 通过才下发工具调用
                      （``ArgParseStatus.REPAIRED``）
                ❌ 修复失败 → 停止解析，返回错误给 Agent：
                      「输出格式非法，请输出严格 JSON」（``ArgParseStatus.INVALID``）

设计约束
========

* ``json_repair`` **懒加载**：模块导入不依赖它；缺失时自动退到内置启发式
  （:func:`_heuristic_repair`），但**默认仍优先**用 ``json_repair``。
* 修复器会「过度自信」——对 ``"{not json"`` 这类输入硬凑出 ``["not json"]``。因此
  修复结果要过两道闸：① 原文必须存在**可识别的机械型破损信号**（尾逗号 / 单引号 /
  裸键 / Python 字面量 / 括号不平衡 / markdown 围栏）；② 顶层键必须是像样的标识符形状。
  过不了闸就判 ``INVALID``，宁可让模型下一轮自纠，也不把瞎编的参数下发给工具。
* 长度上限 ``TOOL_ARG_MAX_REPAIR_CHARS``、总开关 ``TOOL_ARG_REPAIR_ENABLED`` 均取自
  :mod:`my_agent.config.defaults`。
* 本模块只抛 :mod:`my_agent.errors` 里的异常。

JSON Schema → Pydantic 映射
===========================

============================  ==========================================
JSON Schema                   Pydantic / Python
============================  ==========================================
``string``                    ``str``
``integer``                   ``int``
``number``                    ``float``
``boolean``                   ``bool``
``null``                      ``None``
``array``                     ``list[<items 类型>]``（``items`` 缺失 → ``list[Any]``）
``object``                    嵌套 ``create_model`` 模型（递归）
``enum``（全 str/int）          ``Literal[...]``
``required`` 里的字段          必填
非 required 字段               默认 ``None``（即 Optional）
``default``                   作为字段默认值（优先于 ``None``）
``"type": [A, B]``            ``Union``
============================  ==========================================

**已知限制（故意保留，见用户规格）**：``oneOf`` / ``anyOf`` / ``allOf`` / ``$ref`` /
``not`` / ``if`` 等组合关键字**不做语义展开**，一律把该处放宽为 ``Any`` 并记一条说明
（``未支持的 schema 关键字，已放宽为 Any: oneOf``）。``additionalProperties`` /
``patternProperties`` 不展开为具体模型（但**未知参数键一律保留**，不静默丢弃）。
schema 缺失、不可用或根不是 object 时**不做任何校验**，直接放行并记说明。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from my_agent.config import defaults as D
from my_agent.errors import ToolCallParseError

__all__ = [
    "INVALID_ARGS_MESSAGE",
    "ArgParseResult",
    "ArgParseStatus",
    "ToolArgsSchemaError",
    "extract_json_block",
    "parse_tool_arguments",
    "repair_json",
    "schema_validator",
    "strip_markdown",
    "validate_against_spec",
]

_LOG = logging.getLogger("my_agent.llm")

#: 解析彻底失败时回给 Agent / 模型的**逐字**文案（用户规格冻结）。
INVALID_ARGS_MESSAGE = "输出格式非法，请输出严格 JSON"

#: 无法展开的 JSON Schema 组合关键字（放宽为 ``Any`` 并记说明）。
UNSUPPORTED_KEYWORDS: tuple[str, ...] = (
    "$ref",
    "allOf",
    "anyOf",
    "if",
    "not",
    "oneOf",
    "patternProperties",
    "then",
    "else",
)

#: 标量类型 → Python 类型。
_SCALAR_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "null": type(None),
}

#: schema 哈希 → 动态模型（进程内缓存，避免每次调用都建一遍模型）。
_SCHEMA_MODEL_CACHE: dict[str, type[BaseModel] | None] = {}

#: 开围栏：允许 `` ```json `` / `` ```JSON `` / 裸 ```` ``` ````。
_FENCE_OPEN_RE = re.compile(r"`{3,}[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n?")
#: 无换行的开围栏（`` ```json {"a": 1} ``` `` 单行写法）。
_FENCE_INLINE_RE = re.compile(r"`{3,}[ \t]*[A-Za-z0-9_+-]*[ \t]*")
#: 闭围栏。
_FENCE_CLOSE_RE = re.compile(r"`{3,}[ \t]*")
#: 行首裸 ``json`` 标记（``json\n{"a": 1}``）。
_BARE_JSON_TOKEN_RE = re.compile(r"^[ \t]*(json|JSON|Json)[ \t]*\r?\n")
#: 开括号 → 对应闭括号。
_CLOSERS: dict[str, str] = {"{": "}", "[": "]"}
#: 裸键：``{`` / ``,`` / ``[`` 之后紧跟的 ``identifier:``。
_BARE_KEY_RE = re.compile(r"(?P<prefix>[\{,\[]\s*)(?P<key>[A-Za-z_$][A-Za-z0-9_$.\-]*)(?P<colon>\s*:)")
#: 尾逗号：``,]`` / `` ,}``。
_TRAILING_COMMA_RE = re.compile(r",(?P<tail>\s*[\}\]])")
#: 像样的顶层键名（拒绝修复器凭空编出来的 ``"not json"`` 之类；单字母键如 ``a`` 合法）。
_PLAUSIBLE_KEY_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$.\-]*$")
#: 字符串里的 Python 字面量（只在字符串外替换）。
_PY_LITERALS: dict[str, str] = {"True": "true", "False": "false", "None": "null"}
_LITERAL_RE = re.compile(r"(?<![A-Za-z0-9_$])(" + "|".join(_PY_LITERALS) + r")(?![A-Za-z0-9_$])")


class ArgParseStatus(StrEnum):
    """参数解析链的终态。"""

    OK = "ok"
    """标准 ``json.loads`` 直接成功。"""

    REPAIRED = "repaired"
    """标准解析失败，但容错修复成功（并已通过 schema 校验）。"""

    INVALID = "invalid"
    """修复也失败（或 schema 校验不通过）→ 停止解析，回错误文案给 Agent。"""


class ToolArgsSchemaError(ToolCallParseError):
    """修复后的参数未能通过 JSON Schema → Pydantic 校验。

    继承 :class:`~my_agent.errors.ToolCallParseError`（附录 A 中语义最贴近的一条），
    因此既有的 ``except ToolCallParseError`` 捕获点仍然生效。
    """

    code = "tool_call_schema_error"
    _TEMPLATE = "工具 {name} 的参数不满足 schema：{detail}"


class ArgParseResult(BaseModel):
    """一次参数解析的完整结果（供 ``llm/client.py`` 决定下发 / 回错）。"""

    status: ArgParseStatus = Field(description="终态：OK / REPAIRED / INVALID")
    arguments: dict[str, Any] = Field(default_factory=dict, description="解析后的参数字典")
    raw: str = Field(default="", description="原始（未预处理）字符串，供审计与自纠")
    repaired: bool = Field(default=False, description="是否走过容错修复路径")
    notes: list[str] = Field(default_factory=list, description="解析过程说明（可进日志）")
    error: str | None = Field(default=None, description="失败原因；INVALID 时为固定文案")

    model_config = ConfigDict(extra="ignore")

    @property
    def ok(self) -> bool:
        """是否可直接下发工具调用。

        Returns:
            ``status != ArgParseStatus.INVALID``。
        """
        return self.status != ArgParseStatus.INVALID


# --------------------------------------------------------------------------------------
# 2.1 预处理
# --------------------------------------------------------------------------------------
def strip_markdown(text: str) -> str:
    """剥离 markdown 围栏与前导说明文字，返回最像 JSON 的一段。

    处理顺序：去掉 ```` ```json ```` / ```` ``` ```` 围栏（含单行写法）→ 去掉行首裸
    ``json`` 标记 → 掐掉第一个 ``{`` / ``[`` 之前的说明文字（并尽量在最后一个
    ``}`` / ``]`` 处截断尾部解释）。

    Args:
        text: 模型给出的原始参数串。

    Returns:
        预处理后的文本；没找到 JSON 起止符时返回去掉围栏后的整段文本。
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = _FENCE_OPEN_RE.sub("", cleaned)
    cleaned = _FENCE_CLOSE_RE.sub("", cleaned)
    cleaned = _FENCE_INLINE_RE.sub("", cleaned)
    cleaned = _BARE_JSON_TOKEN_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    if cleaned[:1] in "{[":
        return cleaned  # 开头就是 JSON：原样返回，绝不截取（截取会把真垃圾「修」成合法片段）
    start = _first_json_start(cleaned)
    if start < 0:
        return cleaned
    block = _balanced_from(cleaned, start)
    if block is not None:
        return block
    # 不平衡（模型漏了收尾括号）：保留从起点到末尾的原文，交给修复器补齐。
    return cleaned[start:].strip()


def _first_json_start(text: str) -> int:
    """取第一个 ``{`` / ``[`` 的下标。

    Args:
        text: 待扫描文本。

    Returns:
        下标；没有则返回 ``-1``。
    """
    positions = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
    return min(positions) if positions else -1


def _balanced_from(text: str, start: int) -> str | None:
    """从 ``start`` 处取一段括号平衡的文本（忽略字符串内的括号）。

    Args:
        text: 待扫描文本。
        start: 起点下标（应为 ``{`` 或 ``[``）。

    Returns:
        平衡块；在 ``text`` 结束前没配平则返回 ``None``。
    """
    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1].strip()
    return None


def extract_json_block(text: str) -> str | None:
    """提取第一个**平衡**的 ``{...}`` 块。

    括号深度扫描**忽略字符串字面量内的括号**，并遵守 ``\\`` 转义（``"a\\"}"`` 里的
    ``}`` 不参与计数）。

    Args:
        text: 待扫描文本。

    Returns:
        平衡块（含首尾括号）；找不到起点或不平衡时返回 ``None``。
    """
    start = (text or "").find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    return None


# --------------------------------------------------------------------------------------
# 2.2 容错修复
# --------------------------------------------------------------------------------------
def repair_json(text: str) -> tuple[Any, list[str]] | None:
    """容错修复并解析一段「几乎合法」的 JSON。

    路径：``json_repair.loads``（懒加载）→ 内置启发式 :func:`_heuristic_repair`。
    修复结果必须满足：是可当参数用的 ``dict``、且**原文**存在可识别的破损信号
    （否则视为修复器瞎编，直接放弃）。

    Args:
        text: 待修复文本（原始串或预处理后的候选串均可；内部会自行剥 markdown）。

    Returns:
        ``(parsed, notes)``；无法修复时返回 ``None``。
    """
    if not D.TOOL_ARG_REPAIR_ENABLED:
        return None
    if len(text) > D.TOOL_ARG_MAX_REPAIR_CHARS:
        _LOG.warning(
            "参数串超长（%d > %d），跳过容错修复", len(text), D.TOOL_ARG_MAX_REPAIR_CHARS
        )
        return None
    stripped = strip_markdown(text)
    signals = _damage_signals(text)
    if not signals:
        _LOG.debug("无可识别的破损信号，放弃修复：%.80s", text)
        return None
    notes = [f"检测到破损信号：{', '.join(signals)}"]
    if _has_unterminated_string(stripped):
        notes.append("末尾字符串未闭合（疑似截断），放弃修复以免凭空补出空串")
        return None
    parsed = _loads_via_json_repair(stripped, notes)
    if parsed is None:
        parsed = _loads_via_heuristics(stripped, notes)
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        notes.append("修复结果不是 JSON 对象，放弃")
        return None
    if parsed and not all(isinstance(key, str) and _PLAUSIBLE_KEY_RE.match(key) for key in parsed):
        notes.append("修复结果的键名不像标识符，判定为修复器臆造，已放弃")
        _LOG.debug("修复结果键名可疑，放弃：%.120s", parsed)
        return None
    return parsed, notes


def _loads_via_json_repair(text: str, notes: list[str]) -> dict[str, Any] | None:
    """优先用 ``json_repair`` 修复（懒加载；缺失即静默降级到内置启发式）。

    Args:
        text: 待修复文本。
        notes: 追加说明（原地修改）。

    Returns:
        修复出的参数字典；包缺失 / 抛错 / 未产出对象时返回 ``None``。
    """
    try:
        from json_repair import loads as json_repair_loads  # noqa: PLC0415 - 懒加载是硬要求
    except ImportError:
        notes.append("json_repair 未安装，改用内置启发式修复")
        _LOG.info("json_repair 缺失 → 走内置启发式修复")
        return None
    try:
        parsed = json_repair_loads(text)
    except (ValueError, TypeError, RecursionError, IndexError) as exc:
        notes.append(f"json_repair 修复失败：{type(exc).__name__}: {exc}")
        _LOG.debug("json_repair 失败：%s", exc)
        return None
    if not isinstance(parsed, dict) or not parsed:
        notes.append("json_repair 未产出可用的参数字典")
        return None
    notes.append("json_repair 修复成功")
    return parsed


def _loads_via_heuristics(text: str, notes: list[str]) -> Any | None:
    """内置启发式兜底：尾逗号 / 单引号 / 裸键 / Python 字面量 / 括号补齐。

    Args:
        text: 待修复文本。
        notes: 追加说明（原地修改）。

    Returns:
        修复并解析出的对象；仍不合法时返回 ``None``。
    """
    candidate, applied = _heuristic_repair(text)
    if candidate is None or not applied:
        return None  # 一项都没修 → 原文根本不像参数，别硬凑
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError) as exc:
        notes.append(f"内置启发式修复后仍非法 JSON：{exc.msg}")
        return None
    notes.append(f"内置启发式修复：{', '.join(applied)}")
    return parsed


def _heuristic_repair(text: str) -> tuple[str | None, list[str]]:
    """按固定顺序施加机械型修补，返回修补后的文本。

    Args:
        text: 待修补文本。

    Returns:
        ``(修补后的文本, 实际生效的修补项)``；输入为空时返回 ``(None, [])``。
    """
    if not text.strip():
        return None, []
    applied: list[str] = []
    current = text
    swapped = _single_to_double_quotes(current)
    if swapped != current:
        current = swapped
        applied.append("单引号→双引号")
    literals = _LITERAL_RE.sub(lambda match: _PY_LITERALS[match.group(1)], current)
    if literals != current:
        current = literals
        applied.append("Python 字面量→JSON 字面量")
    keyed = _BARE_KEY_RE.sub(lambda m: f'{m.group("prefix")}"{m.group("key")}"{m.group("colon")}', current)
    if keyed != current:
        current = keyed
        applied.append("裸键加引号")
    trimmed = _TRAILING_COMMA_RE.sub(lambda m: m.group("tail"), current)
    if trimmed != current:
        current = trimmed
        applied.append("去尾逗号")
    if not current.lstrip().startswith(("{", "[")):
        current = "{" + current + "}"
        applied.append("补上缺失的左花括号")
    balanced = _balance_brackets(current)
    if balanced != current:
        current = balanced
        applied.append("补齐缺失的括号")
    return current, applied


def _damage_signals(text: str) -> list[str]:
    """识别原文里可解释失败原因的**机械型**破损信号。

    这是防「修复器硬凑」的第一道闸：没有任何可识别破损 → 不做修复。

    Args:
        text: 原始参数串。

    Returns:
        破损信号说明列表（空列表表示无可解释的破损）。
    """
    signals: list[str] = []
    if not text.strip():
        return signals
    if "```" in text:
        signals.append("markdown 围栏")
    if _TRAILING_COMMA_RE.search(text):
        signals.append("尾逗号")
    if _single_quote_strings(text):
        signals.append("单引号字符串")
    if _BARE_KEY_RE.search(text):
        signals.append("裸键")
    if _LITERAL_RE.search(text):
        signals.append("Python 字面量")
    if _bracket_balance(text) > 0:
        signals.append("括号未闭合")
    return signals


def _bracket_balance(text: str) -> int:
    """统计字符串外的括号净深度（>0 表示缺闭合）。

    做括号类型配对：``{]`` 这类「类型错配」也计入未闭合（返回正数）。

    Args:
        text: 待统计文本。

    Returns:
        净深度；已正确闭合为 0，多余 / 错配的闭合为负数。
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    orphan = 0
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack and _CLOSERS[stack[-1]] == char:
                stack.pop()
            else:
                orphan -= 1
    return len(stack) - abs(orphan)


def _single_quote_strings(text: str) -> bool:
    """是否存在单引号字符串（``'a'``）；字符串内的撇号（``"it's"``）不计。

    Args:
        text: 待扫描文本。

    Returns:
        ``True`` 表示存在单引号字符串（JSON 非法，但属可机械修复的破损）。
    """
    in_double = False
    escaped = False
    for char in text:
        if in_double:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_double = False
            continue
        if char == '"':
            in_double = True
        elif char == "'":
            return True
    return False


def _has_unterminated_string(text: str) -> bool:
    """末尾是否停在未闭合的字符串里（截断的强信号）。

    与「引号总数不配对」不同：只有**待补的引号落在串尾**才算截断。``{"a": "b"``
    这类「只缺右括号」的输入不算，属可安全补齐的正常破损。

    Args:
        text: 待检查文本。

    Returns:
        ``True`` 表示不该硬修（硬修会把内容吞成空串或凭空补值）。
    """
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
    if not in_string:
        return False
    # 仍停在串内：末字符就是那个「开引号」→ 字符串被截断（如 {"file_path": "）。
    # 末字符不是引号则说明是缺右引号（如 {"a": "b），属可安全补齐。
    return text.endswith(('"', "'"))


def _single_to_double_quotes(text: str) -> str:
    """把单引号字符串改写成双引号字符串（转义内部的双引号）。

    双引号与单引号会互相嵌套（``{"a": 'he said "hi"'}``），因此用显式状态机而不是正则。

    Args:
        text: 待改写文本。

    Returns:
        改写后的文本。
    """
    out: list[str] = []
    open_quote = ""
    escaped = False
    for char in text:
        if open_quote == "'":
            if escaped:
                out.append(char)
                escaped = False
            elif char == "\\":
                out.append(char)
                escaped = True
            elif char == "'":
                out.append('"')
                open_quote = ""
            elif char == '"':
                out.append('\\"')
            else:
                out.append(char)
            continue
        if open_quote == '"':
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                open_quote = ""
            continue
        if char in "\"'":
            open_quote = char
            out.append('"')
            continue
        out.append(char)
    return "".join(out)


def _balance_brackets(text: str) -> str:
    """补齐缺失的右括号 / 右方括号（字符串内的括号不参与）。

    括号类型错配或出现「孤儿闭合符」时**原样返回**（该文本已非「缺个尾巴」，
    而是结构坏了，硬修只会凑出空对象之类的东西）。

    Args:
        text: 待补齐文本。

    Returns:
        补齐后的文本；结构错乱时原样返回。
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and (not stack or stack.pop() != char):
            return text
    if in_string:
        return text + '"' + "".join(reversed(stack))
    return text + "".join(reversed(stack))


# --------------------------------------------------------------------------------------
# schema → Pydantic 动态模型
# --------------------------------------------------------------------------------------
def _schema_hash(schema: Mapping[str, Any]) -> str:
    """算出 schema 的稳定哈希（缓存键）。

    Args:
        schema: JSON Schema 片段。

    Returns:
        sha1 十六进制摘要；不可序列化时退化为 ``repr`` 的摘要。
    """
    try:
        payload = json.dumps(schema, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - 防御性
        payload = repr(sorted((str(k), str(v)) for k, v in schema.items()))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def schema_validator(schema: dict[str, Any]) -> type[BaseModel] | None:
    """由 JSON Schema 动态构建（并缓存）Pydantic 模型。

    Args:
        schema: 工具参数的 JSON Schema（``ToolSpec.parameters``）。

    Returns:
        动态模型类；根不是 ``object`` 或 schema 不可用时返回 ``None``（表示「不校验」）。
    """
    if not isinstance(schema, Mapping) or not schema:
        return None
    try:
        key = _schema_hash(schema)
    except (TypeError, ValueError):  # pragma: no cover - 防御性
        return None
    if key in _SCHEMA_MODEL_CACHE:
        return _SCHEMA_MODEL_CACHE[key]
    model: type[BaseModel] | None
    try:
        notes: list[str] = []
        model = _build_model(dict(schema), notes, name="ToolArguments")
    except (TypeError, ValueError, KeyError, RecursionError, AttributeError) as exc:
        _LOG.info("schema 无法构建 Pydantic 模型（%s），本次不校验", exc)
        model = None
    _SCHEMA_MODEL_CACHE[key] = model
    return model


def _build_model(
    schema: Mapping[str, Any],
    notes: list[str],
    *,
    name: str,
    depth: int = 0,
) -> type[BaseModel] | None:
    """递归构建模型（``object`` 型 schema → ``create_model``）。

    Args:
        schema: JSON Schema 片段。
        notes: 追加说明（原地修改）。
        name: 模型类名（嵌套时由父字段名派生）。
        depth: 递归深度（防御性上限）。

    Returns:
        模型类；不是 ``object`` 型或过深时返回 ``None``。
    """
    if depth > 6:  # pragma: no cover - 防御性
        notes.append(f"schema 嵌套过深，已放宽为 Any: {name}")
        return None
    if _collect_unsupported(schema, notes):
        return None
    if not _is_object_schema(schema):
        return None
    fields: dict[str, Any] = {}
    required = {str(item) for item in _as_sequence(schema.get("required"))}
    for key, raw in (schema.get("properties") or {}).items():
        if not isinstance(raw, Mapping):
            continue
        annotation = _annotation_for(raw, notes, name=f"{name}_{key}", depth=depth + 1)
        if key in required:
            fields[str(key)] = (annotation, ...)
        else:
            fields[str(key)] = (annotation, _field_default(raw))
    return create_model(name, __config__=ConfigDict(extra="allow"), **fields)


def _annotation_for(
    schema: Mapping[str, Any],
    notes: list[str],
    *,
    name: str,
    depth: int,
) -> Any:
    """把单个属性的 schema 映射成 Python 类型注解。

    Args:
        schema: 属性 schema。
        notes: 追加说明（原地修改）。
        name: 用于嵌套模型命名。
        depth: 当前递归深度。

    Returns:
        可直接交给 ``create_model`` 的类型注解。
    """
    if not schema:
        return Any
    if _collect_unsupported(schema, notes):
        return Any
    enum_values = _as_sequence(schema.get("enum"))
    if enum_values and all(isinstance(item, (str, int)) and not isinstance(item, bool) for item in enum_values):
        return Literal[tuple(enum_values)]  # type: ignore[valid-type]
    declared = schema.get("type")
    if isinstance(declared, Sequence) and not isinstance(declared, str):
        members = tuple(
            _annotation_for({**schema, "type": item}, notes, name=name, depth=depth)
            for item in declared
        )
        unique = tuple(dict.fromkeys(members))
        if len(unique) == 1:
            return unique[0]
        return Union[unique]  # noqa: UP007 - 运行期按 schema 长度拼 Union，无法写 X | Y
    if declared == "array":
        items = schema.get("items")
        element = (
            _annotation_for(items, notes, name=f"{name}_item", depth=depth)
            if isinstance(items, Mapping)
            else Any
        )
        return list[element]  # type: ignore[valid-type]
    if declared == "object":
        nested = _build_model(schema, notes, name=name, depth=depth)
        return nested if nested is not None else dict[str, Any]
    return _SCALAR_TYPES.get(str(declared), Any)


def _collect_unsupported(schema: Mapping[str, Any], notes: list[str]) -> bool:
    """检查并记录无法展开的 schema 关键字（放宽为 ``Any``）。

    Args:
        schema: JSON Schema 片段。
        notes: 追加说明（原地修改）。

    Returns:
        ``True`` 表示该处必须放宽为 ``Any``。
    """
    hit = False
    for keyword in UNSUPPORTED_KEYWORDS:
        if keyword in schema:
            note = f"未支持的 schema 关键字，已放宽为 Any: {keyword}"
            if note not in notes:
                notes.append(note)
            hit = True
    return hit


def _is_object_schema(schema: Mapping[str, Any]) -> bool:
    """判断 schema 是否为 ``object`` 型（缺省 ``type`` 但有 ``properties`` 也算）。

    Args:
        schema: JSON Schema 片段。

    Returns:
        是否可按 object 建模。
    """
    declared = schema.get("type")
    if declared is None:
        return "properties" in schema or "additionalProperties" in schema
    if isinstance(declared, Sequence) and not isinstance(declared, str):
        return "object" in declared
    return declared == "object"


def _field_default(schema: Mapping[str, Any]) -> Any:
    """取非必填字段的默认值。

    Args:
        schema: 属性 schema。

    Returns:
        ``default`` 原值（可为 ``None``）；没有 ``default`` 时返回 ``None``（即 Optional）。
    """
    if "default" in schema:
        return schema["default"]
    return None


def _as_sequence(value: Any) -> list[Any]:
    """把可能是 ``None`` / 标量的值统一成列表。

    Args:
        value: 任意值。

    Returns:
        列表；``None`` → 空列表，标量 → 单元素列表。
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return [value]
    return list(value)


# --------------------------------------------------------------------------------------
# schema 校验
# --------------------------------------------------------------------------------------
def validate_against_spec(
    arguments: dict[str, Any],
    schema: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """用「JSON Schema 动态生成的 Pydantic 模型」校验参数。

    Args:
        arguments: 待校验参数字典。
        schema: 工具参数的 JSON Schema；``None`` / 不可用 → 不校验直接放行。

    Returns:
        ``(coerced_arguments, notes)``。``coerced_arguments`` 为 Pydantic 规范化后的
        dump（类型按 schema 收敛；**未知键原样保留**）。

    Raises:
        ToolArgsSchemaError: 参数不满足 schema（必填缺失 / 类型不符 / enum 越界等）。
    """
    notes: list[str] = []
    if not isinstance(schema, Mapping) or not schema:
        notes.append("未提供可用 schema，跳过参数校验")
        return dict(arguments), notes
    model = schema_validator(dict(schema))
    if model is None:
        notes.append("schema 无法生成校验模型（根非 object 或不可用），跳过参数校验")
        return dict(arguments), notes
    try:
        validated = model.model_validate(arguments)
    except ValidationError as exc:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in item.get('loc', ())) or '<root>'}: {item.get('msg')}"
            for item in exc.errors()
        )
        raise ToolArgsSchemaError(detail or str(exc), detail=detail or str(exc)) from exc
    dumped = validated.model_dump(exclude_none=True)
    extra = set(validated.model_extra or {})
    known = set(dumped)
    if extra:
        notes.append(f"参数含 schema 未声明的键（已保留）：{', '.join(sorted(extra - known)) or '-'}")
    _collect_relaxed(schema, notes)
    return dumped, notes


def _collect_relaxed(node: Any, notes: list[str], *, depth: int = 0) -> None:
    """递归收集 schema 树里的「放宽为 Any」说明（缓存命中也必须能拿到）。

    建模路径 :func:`_build_model` 只在**首次**构建时记录；命中 ``_SCHEMA_MODEL_CACHE``
    后不会再跑建模，所以这里独立扫一遍，保证 notes 与是否缓存无关。

    Args:
        node: schema 子树（可为任意嵌套结构）。
        notes: 追加说明（原地修改，自动去重）。
        depth: 递归深度（防御性上限）。
    """
    if depth > 12:  # pragma: no cover - 防御性
        return
    if isinstance(node, Mapping):
        _collect_unsupported(node, notes)
        for value in node.values():
            _collect_relaxed(value, notes, depth=depth + 1)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_relaxed(item, notes, depth=depth + 1)


# --------------------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------------------
def parse_tool_arguments(
    raw: str,
    *,
    schema: dict[str, Any] | None = None,
) -> ArgParseResult:
    """执行完整解析链：预处理 → ``json.loads`` → 容错修复 → schema 校验。

    本函数**永不抛出**：任何失败都收敛成 ``status=INVALID`` 的结果对象，交给上层
    回错误文案给 Agent（「输出格式非法，请输出严格 JSON」）。

    Args:
        raw: 模型给出的原始参数字符串（已按 ``index`` 拼接完毕）。
        schema: 该工具参数的 JSON Schema；``None`` → 不校验。

    Returns:
        :class:`ArgParseResult`。
    """
    text = raw or ""
    notes: list[str] = []
    if not text.strip():
        return ArgParseResult(
            status=ArgParseStatus.OK,
            arguments={},
            raw=raw,
            repaired=False,
            notes=["参数串为空，按无参调用处理"],
        )
    candidate = _preprocess(text, notes)
    parsed = _try_standard_loads(candidate)
    if parsed is not None:
        if not isinstance(parsed, dict):
            return ArgParseResult(
                status=ArgParseStatus.OK,
                arguments={},
                raw=raw,
                repaired=False,
                notes=[*notes, "顶层不是 JSON 对象，交由上层按 value 包装处理"],
            )
        return _finalize(parsed, raw=raw, notes=notes, repaired=False, schema=schema)
    repaired = repair_json(candidate)
    if repaired is None:
        notes.append(f"标准解析与容错修复均失败：{_json_error_hint(candidate)}")
        _LOG.warning("参数解析失败（INVALID）：%s；原文=%.200s", notes[-1], raw)
        return ArgParseResult(
            status=ArgParseStatus.INVALID,
            arguments={},
            raw=raw,
            repaired=False,
            notes=notes,
            error=INVALID_ARGS_MESSAGE,
        )
    parsed, repair_notes = repaired
    notes.extend(repair_notes)
    if not isinstance(parsed, dict):
        return ArgParseResult(
            status=ArgParseStatus.OK,
            arguments={},
            raw=raw,
            repaired=True,
            notes=[*notes, "修复结果的顶层不是对象，交由上层按 value 包装处理"],
        )
    return _finalize(parsed, raw=raw, notes=notes, repaired=True, schema=schema)


def _preprocess(raw: str, notes: list[str]) -> str:
    """2.1 预处理：剥 markdown + 提取平衡块。

    Args:
        raw: 原始参数串。
        notes: 追加说明（原地修改）。

    Returns:
        交给标准解析的候选文本；没提取到平衡块时原样返回 ``raw``，
        好让 :func:`repair_json` 仍在**原文**上识别破损信号。
    """
    stripped = strip_markdown(raw)
    if stripped != raw.strip():
        notes.append("已剥离 markdown 围栏/前导说明文字")
    block = extract_json_block(stripped)
    if block is None:
        return raw
    if block != stripped:
        notes.append("已提取首个平衡的 {} 块")
    return block


def _try_standard_loads(candidate: str) -> Any | None:
    """尝试标准 ``json.loads``。

    Args:
        candidate: 预处理后的文本。

    Returns:
        解析结果；失败返回 ``None``。
    """
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None


def _json_error_hint(candidate: str) -> str:
    """给出标准解析失败的原因（供日志 / notes）。

    Args:
        candidate: 预处理后的文本。

    Returns:
        ``json`` 错误消息；无错误时给一句兜底说明。
    """
    try:
        json.loads(candidate)
    except (json.JSONDecodeError, ValueError) as exc:
        return f"{exc.msg}（pos={exc.pos}）"
    return "未知原因"


def _finalize(
    parsed: dict[str, Any],
    *,
    raw: str,
    notes: list[str],
    repaired: bool,
    schema: dict[str, Any] | None,
) -> ArgParseResult:
    """收尾：schema 校验 + 组装结果。

    Args:
        parsed: 已解析出的参数字典。
        raw: 原始参数串。
        notes: 已有说明。
        repaired: 是否走过修复路径。
        schema: 可选 JSON Schema。

    Returns:
        :class:`ArgParseResult`；校验失败时收敛成 ``INVALID``（不抛）。
    """
    try:
        arguments, schema_notes = validate_against_spec(parsed, schema)
    except ToolArgsSchemaError as exc:
        notes.append(f"schema 校验未通过：{exc.detail}")
        _LOG.warning("schema 校验未通过：%s", exc.detail)
        return ArgParseResult(
            status=ArgParseStatus.INVALID,
            arguments={},
            raw=raw,
            repaired=repaired,
            notes=notes,
            error=INVALID_ARGS_MESSAGE,
        )
    if repaired:
        _LOG.info("工具参数经容错修复后通过校验：%s", "; ".join(notes))
    return ArgParseResult(
        status=ArgParseStatus.REPAIRED if repaired else ArgParseStatus.OK,
        arguments=arguments,
        raw=raw,
        repaired=repaired,
        notes=[*notes, *schema_notes],
    )
