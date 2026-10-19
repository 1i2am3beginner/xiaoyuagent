"""``llm/tool_args.py`` 容错解析链回归测试。

覆盖：

* 预处理：``\u0060\u0060\u0060json`` 围栏 / 裸围栏 / 前导说明 + 围栏 / 行首裸 ``json``；
* 平衡块提取：嵌套对象、字符串内的花括号、转义引号；不平衡 → ``None``；
* 标准路径：干净 JSON → ``OK`` 且 ``repaired is False``；
* 修复路径（经典 LLM 手滑）：尾逗号 / 单引号 / 裸键 / ``True``/``False``/``None`` /
  缺右括号 → ``REPAIRED``；
* 真垃圾（``not json at all``）→ ``INVALID`` 且 ``error == INVALID_ARGS_MESSAGE``；
* schema 校验：必填缺失 / 类型收敛 / enum 越界 / 未知关键字 ``oneOf`` 放宽为 Any /
  未知参数键保留 / 未提供 schema 时不校验；
* **无 json_repair 的优雅降级**：伪造 ``ImportError`` 后内置启发式仍能修复；
* ``json_repair`` 在场时确实被调用（未安装则 skip）。
"""

from __future__ import annotations

import builtins
import json
import sys
from typing import Any

import pytest

from my_agent.config import defaults as D
from my_agent.errors import MyAgentError, ToolCallParseError
from my_agent.llm.tool_args import (
    INVALID_ARGS_MESSAGE,
    ArgParseResult,
    ArgParseStatus,
    ToolArgsSchemaError,
    extract_json_block,
    parse_tool_arguments,
    repair_json,
    schema_validator,
    strip_markdown,
    validate_against_spec,
)

#: 一个「真实工具」形态的 JSON Schema（用于校验类用例）。
_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string"},
        "limit": {"type": "integer", "default": 10},
        "mode": {"type": "string", "enum": ["read", "write"]},
    },
    "required": ["file_path"],
}


class _NoJsonRepair:
    """把 ``json_repair`` 伪装成「未安装」的上下文管理器。

    同时拦截 ``import json_repair`` 与 ``from json_repair import loads``：
    ＊ 从 ``sys.modules`` 摘掉已导入的模块；
    ＊ 在 ``builtins.__import__`` 上对 ``json_repair`` 抛 ``ImportError``。
    """

    def __init__(self) -> None:
        """保存原 import 钩子。"""
        self._original_import = builtins.__import__
        self._saved: Any = None
        self._had_module = False

    def __enter__(self) -> _NoJsonRepair:
        """安装 ImportError 钩子并摘掉缓存模块。

        Returns:
            自身，便于 ``with ... as`` 使用。
        """
        self._had_module = "json_repair" in sys.modules
        self._saved = sys.modules.pop("json_repair", None)

        def _fake_import(
            name: str,
            globals_: Any = None,
            locals_: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> Any:
            if name == "json_repair" or name.startswith("json_repair."):
                raise ImportError("No module named 'json_repair'（测试伪造）")
            return self._original_import(name, globals_, locals_, fromlist, level)

        builtins.__import__ = _fake_import
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """恢复 import 钩子与模块缓存。

        Args:
            *exc_info: 异常三元组（忽略）。
        """
        builtins.__import__ = self._original_import
        if self._had_module and self._saved is not None:
            sys.modules["json_repair"] = self._saved


# --------------------------------------------------------------------------------------
# 2.1 预处理：markdown 剥离
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('```json\n{"file_path": "a.py"}\n```', '{"file_path": "a.py"}'),
        ('```\n{"file_path": "a.py"}\n```', '{"file_path": "a.py"}'),
        ('Here you go:\n```json\n{"file_path": "a.py"}\n```\nDone.', '{"file_path": "a.py"}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('json\n{"a": 1}', '{"a": 1}'),
        ('{"a": 1}', '{"a": 1}'),
        ("", ""),
    ],
)
def test_strip_markdown_removes_fences_and_prose(raw: str, expected: str) -> None:
    """围栏（带语言标记 / 裸围栏）与前导说明文字都要剥掉。"""
    assert strip_markdown(raw) == expected


def test_strip_markdown_single_line_fence() -> None:
    """单行写法 ```` ```json {"a": 1} ``` ```` 同样要能剥。"""
    assert strip_markdown('```json {"a": 1} ```') == '{"a": 1}'


def test_strip_markdown_without_json_keeps_text() -> None:
    """没有 JSON 起止符时不臆造，原样（去空白后）返回。"""
    assert strip_markdown("  完全不是 JSON  ") == "完全不是 JSON"


# --------------------------------------------------------------------------------------
# 2.1 预处理：平衡块提取
# --------------------------------------------------------------------------------------
def test_extract_json_block_nested_objects() -> None:
    """嵌套对象要取到最外层平衡块。"""
    text = 'prefix {"a": {"b": {"c": 1}}, "d": [1, 2]} suffix'
    assert extract_json_block(text) == '{"a": {"b": {"c": 1}}, "d": [1, 2]}'


def test_extract_json_block_ignores_braces_in_strings() -> None:
    """字符串里的 ``{`` / ``}`` 不参与深度计数。"""
    text = '{"pattern": "a{b}c}", "other": "}"}'
    assert extract_json_block(text) == text


def test_extract_json_block_honours_backslash_escape() -> None:
    """``\\"`` 是转义引号，不能当成字符串结束。"""
    text = '{"q": "say \\"hi\\" }", "n": 1}'
    assert extract_json_block(text) == text


def test_extract_json_block_unbalanced_returns_none() -> None:
    """不平衡 → ``None``（缺右括号、孤右括号都算）。"""
    assert extract_json_block('{"a": 1') is None
    assert extract_json_block('{"a": {"b": 1}') is None
    assert extract_json_block("no braces at all") is None


def test_extract_json_block_stops_at_first_balanced_block() -> None:
    """取第一个平衡块，后面的兄弟块不并入。"""
    assert extract_json_block('{"a": 1} {"b": 2}') == '{"a": 1}'


# --------------------------------------------------------------------------------------
# 2.2 标准路径
# --------------------------------------------------------------------------------------
def test_clean_json_is_ok_and_not_repaired() -> None:
    """干净 JSON → ``OK``、``repaired is False``、参数原样。"""
    result = parse_tool_arguments('{"file_path": "a.py", "limit": 3}')

    assert isinstance(result, ArgParseResult)
    assert result.status == ArgParseStatus.OK
    assert result.repaired is False
    assert result.arguments == {"file_path": "a.py", "limit": 3}
    assert result.error is None
    assert result.ok is True


def test_fenced_clean_json_is_ok() -> None:
    """围栏包住的干净 JSON 走标准解析（剥围栏后无需修复）。"""
    result = parse_tool_arguments('```json\n{"file_path": "a.py"}\n```')

    assert result.status == ArgParseStatus.OK
    assert result.repaired is False
    assert result.arguments == {"file_path": "a.py"}
    assert any("markdown" in note for note in result.notes)


def test_empty_arguments_is_ok() -> None:
    """空串按「无参调用」处理，不算错误。"""
    result = parse_tool_arguments("")

    assert result.status == ArgParseStatus.OK
    assert result.arguments == {}
    assert result.raw == ""


# --------------------------------------------------------------------------------------
# 2.2 修复路径（经典 LLM 手滑）
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"file_path": "a.py",}', {"file_path": "a.py"}),
        ('{"a": [1, 2, 3,],}', {"a": [1, 2, 3]}),
        ("{'file_path': 'a.py'}", {"file_path": "a.py"}),
        ('{file_path: "a.py"}', {"file_path": "a.py"}),
        ('{"ok": True, "no": False, "nothing": None}', {"ok": True, "no": False, "nothing": None}),
        ('{"file_path": "a.py"', {"file_path": "a.py"}),
        ('{"a": [1, 2', {"a": [1, 2]}),
        ("```json\n{'path': 'a.py',}\n```", {"path": "a.py"}),
    ],
)
def test_repairable_input_yields_repaired(raw: str, expected: dict[str, Any]) -> None:
    """尾逗号 / 单引号 / 裸键 / Python 字面量 / 缺括号 → ``REPAIRED`` 且参数正确。"""
    result = parse_tool_arguments(raw)

    assert result.status == ArgParseStatus.REPAIRED
    assert result.repaired is True
    assert result.arguments == expected
    assert result.error is None
    assert result.notes, "修复路径必须留下可审计的 notes"


def test_trailing_comma_note_mentions_signal() -> None:
    """尾逗号这条经典手滑要能被点出来（便于日志定位模型习惯）。"""
    result = parse_tool_arguments('{"file_path": "a.py",}')

    assert any("尾逗号" in note for note in result.notes)
    assert result.raw == '{"file_path": "a.py",}'


def test_repair_json_returns_none_without_damage_signal() -> None:
    """无可识别破损信号 → 不修复（避免修复器把散文硬凑成参数）。"""
    assert repair_json("not json at all") is None
    assert repair_json("") is None


def test_unterminated_string_is_not_repaired() -> None:
    """末尾字符串未闭合属截断，不硬修（否则会凭空补出空串）。"""
    result = parse_tool_arguments('{"file_path": "')

    assert result.status == ArgParseStatus.INVALID
    assert result.arguments == {}


@pytest.mark.parametrize("raw", ['{"a": "', '{"file_path": "', "你好，请调用 Read", "hello world"])
def test_truncated_payload_is_invalid(raw: str) -> None:
    """值一个字符都没写出来（或纯散文）→ 不猜内容，判 ``INVALID`` 让模型重发。"""
    result = parse_tool_arguments(raw)

    assert result.status == ArgParseStatus.INVALID
    assert result.error == INVALID_ARGS_MESSAGE


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": "b', {"a": "b"}),
        ("{'a': 1", {"a": 1}),
        ("```json\n{'root': 1", {"root": 1}),
    ],
)
def test_truncated_tail_is_repaired(raw: str, expected: dict[str, Any]) -> None:
    """只缺收尾的引号/括号时，已写出的内容是确定的 → 放心补齐。"""
    result = parse_tool_arguments(raw)

    assert result.status == ArgParseStatus.REPAIRED
    assert result.arguments == expected


# --------------------------------------------------------------------------------------
# 2.2 失败路径
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["not json at all", "{not json", "}{", "hello world"])
def test_garbage_is_invalid_with_exact_message(raw: str) -> None:
    """真垃圾 → ``INVALID``、``error`` 逐字等于用户规格文案、``arguments`` 为空。"""
    result = parse_tool_arguments(raw)

    assert result.status == ArgParseStatus.INVALID
    assert result.error == INVALID_ARGS_MESSAGE
    assert result.error == "输出格式非法，请输出严格 JSON"
    assert result.arguments == {}
    assert result.raw == raw
    assert result.ok is False


def test_invalid_args_message_is_frozen() -> None:
    """用户可见文案冻结（改一个字都算破坏契约）。"""
    assert INVALID_ARGS_MESSAGE == "输出格式非法，请输出严格 JSON"


def test_result_model_carries_raw_for_audit() -> None:
    """``raw`` 始终保留原文，供审计与模型下一轮自纠。"""
    raw = "{'file_path': 'a.py'}"
    result = parse_tool_arguments(raw)

    assert result.raw == raw


# --------------------------------------------------------------------------------------
# schema 校验（JSON Schema → Pydantic 动态模型）
# --------------------------------------------------------------------------------------
def test_schema_validator_builds_model_and_caches() -> None:
    """``schema_validator`` 建模型并按 schema 哈希缓存。"""
    first = schema_validator(_FILE_SCHEMA)
    second = schema_validator(json.loads(json.dumps(_FILE_SCHEMA)))

    assert first is not None
    assert first is second


def test_schema_validator_returns_none_for_non_object_schema() -> None:
    """根不是 object → 返回 ``None``（表示「不校验」）。"""
    assert schema_validator({"type": "string"}) is None
    assert schema_validator({}) is None


def test_required_field_missing_is_rejected() -> None:
    """必填字段缺失 → 抛 :class:`ToolArgsSchemaError`（是 ``ToolCallParseError`` 子类）。"""
    with pytest.raises(ToolArgsSchemaError) as excinfo:
        validate_against_spec({"limit": 3}, _FILE_SCHEMA)

    assert "file_path" in excinfo.value.detail
    assert isinstance(excinfo.value, ToolCallParseError)
    assert isinstance(excinfo.value, MyAgentError)


def test_required_field_missing_makes_parse_invalid() -> None:
    """必填缺失在解析链里收敛成 ``INVALID``（不抛、不下发）。"""
    result = parse_tool_arguments('{"limit": 3}', schema=_FILE_SCHEMA)

    assert result.status == ArgParseStatus.INVALID
    assert result.error == INVALID_ARGS_MESSAGE
    assert result.arguments == {}
    assert any("schema" in note for note in result.notes)


def test_type_coercion_and_defaults() -> None:
    """``"3"`` → ``3``（pydantic 宽松模式）；未给的非必填字段吃 ``default``。"""
    result = parse_tool_arguments('{"file_path": "a.py", "limit": "3"}', schema=_FILE_SCHEMA)

    assert result.status == ArgParseStatus.OK
    assert result.arguments == {"file_path": "a.py", "limit": 3}


def test_wrong_type_is_rejected() -> None:
    """类型无法收敛（对象给了字符串）→ 校验失败 → ``INVALID``。"""
    result = parse_tool_arguments('{"file_path": {"a": 1}}', schema=_FILE_SCHEMA)

    assert result.status == ArgParseStatus.INVALID
    assert result.error == INVALID_ARGS_MESSAGE


def test_enum_violation_is_rejected() -> None:
    """``enum`` → ``Literal[...]``，越界值必须被拒。"""
    with pytest.raises(ToolArgsSchemaError):
        validate_against_spec({"file_path": "a.py", "mode": "delete"}, _FILE_SCHEMA)

    result = parse_tool_arguments(
        '{"file_path": "a.py", "mode": "delete"}', schema=_FILE_SCHEMA
    )
    assert result.status == ArgParseStatus.INVALID

    ok = parse_tool_arguments('{"file_path": "a.py", "mode": "write"}', schema=_FILE_SCHEMA)
    assert ok.status == ArgParseStatus.OK
    assert ok.arguments["mode"] == "write"


def test_boolean_and_number_types() -> None:
    """``boolean`` / ``number`` / ``array`` / 嵌套 ``object`` 都要能建出来的模型认。"""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "flag": {"type": "boolean"},
            "ratio": {"type": "number"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "nested": {
                "type": "object",
                "properties": {"inner": {"type": "integer"}},
                "required": ["inner"],
            },
        },
        "required": ["flag", "ratio", "tags", "nested"],
    }
    result = parse_tool_arguments(
        '{"flag": true, "ratio": 0.5, "tags": ["a", "b"], "nested": {"inner": 7}}',
        schema=schema,
    )

    assert result.status == ArgParseStatus.OK
    assert result.arguments == {
        "flag": True,
        "ratio": 0.5,
        "tags": ["a", "b"],
        "nested": {"inner": 7},
    }


def test_nested_object_violation_is_rejected() -> None:
    """嵌套 object 内部违约也要被拒。"""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {"inner": {"type": "integer"}},
                "required": ["inner"],
            }
        },
    }
    result = parse_tool_arguments('{"nested": {}}', schema=schema)

    assert result.status == ArgParseStatus.INVALID


@pytest.mark.parametrize("keyword", ["oneOf", "anyOf", "allOf", "$ref"])
def test_unsupported_schema_keywords_are_relaxed_to_any(keyword: str) -> None:
    """``oneOf`` / ``anyOf`` / ``allOf`` / ``$ref`` 不得崩：放宽为 ``Any`` + 记 note。"""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "weird": {keyword: [{"type": "string"}, {"type": "integer"}]},
        },
        "required": ["file_path"],
    }
    result = parse_tool_arguments('{"file_path": "a.py", "weird": {"any": "thing"}}', schema=schema)

    assert result.status == ArgParseStatus.OK
    assert result.arguments["weird"] == {"any": "thing"}
    assert any(f"放宽为 Any: {keyword}" in note for note in result.notes)


def test_unknown_argument_keys_are_kept_with_note() -> None:
    """schema 未声明的参数键保留（不静默丢弃）并记一条 note。"""
    result = parse_tool_arguments('{"file_path": "a.py", "extra": 1}', schema=_FILE_SCHEMA)

    assert result.status == ArgParseStatus.OK
    assert result.arguments["extra"] == 1
    assert any("未声明" in note for note in result.notes)


def test_missing_schema_validates_nothing() -> None:
    """没给 schema → 不校验，原样返回 + note。"""
    arguments, notes = validate_against_spec({"anything": 1}, None)

    assert arguments == {"anything": 1}
    assert any("跳过" in note for note in notes)

    result = parse_tool_arguments('{"anything": 1}')
    assert result.status == ArgParseStatus.OK
    assert result.arguments == {"anything": 1}


def test_unusable_schema_validates_nothing() -> None:
    """schema 不可用（根是 string）→ 不校验也不崩。"""
    arguments, notes = validate_against_spec({"anything": 1}, {"type": "string"})

    assert arguments == {"anything": 1}
    assert any("跳过" in note for note in notes)


def test_repaired_arguments_still_go_through_schema() -> None:
    """修复成功后仍要过 schema：修出来的参数不合规一样不下发。"""
    ok = parse_tool_arguments('{"file_path": "a.py",}', schema=_FILE_SCHEMA)

    assert ok.status == ArgParseStatus.REPAIRED
    assert ok.arguments == {"file_path": "a.py", "limit": 10}

    bad = parse_tool_arguments('{"limit": 1,}', schema=_FILE_SCHEMA)
    assert bad.status == ArgParseStatus.INVALID
    assert bad.repaired is True  # 修过，但校验没过 → 仍然不下发


# --------------------------------------------------------------------------------------
# 优雅降级：json_repair 缺席
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"file_path": "a.py",}', {"file_path": "a.py"}),
        ("{'file_path': 'a.py'}", {"file_path": "a.py"}),
        ('{file_path: "a.py"}', {"file_path": "a.py"}),
        ('{"ok": True, "no": False, "nothing": None}', {"ok": True, "no": False, "nothing": None}),
        ('{"file_path": "a.py"', {"file_path": "a.py"}),
        ('{"a": [1, 2', {"a": [1, 2]}),
    ],
)
def test_builtin_heuristic_repair_works_without_json_repair(
    raw: str, expected: dict[str, Any]
) -> None:
    """**无 json_repair 时不崩**：内置启发式仍能修好常见手滑（优雅降级核心用例）。"""
    with _NoJsonRepair():
        result = parse_tool_arguments(raw)

    assert result.status == ArgParseStatus.REPAIRED
    assert result.arguments == expected
    assert any("json_repair" in note for note in result.notes)


def test_without_json_repair_garbage_still_invalid() -> None:
    """缺 json_repair 时真垃圾依旧 ``INVALID``（不能因为降级就放行）。"""
    with _NoJsonRepair():
        result = parse_tool_arguments("not json at all")

    assert result.status == ArgParseStatus.INVALID
    assert result.error == INVALID_ARGS_MESSAGE


def test_without_json_repair_clean_json_still_ok() -> None:
    """缺 json_repair 时标准解析路径完全不受影响。"""
    with _NoJsonRepair():
        result = parse_tool_arguments('{"file_path": "a.py"}')

    assert result.status == ArgParseStatus.OK
    assert result.repaired is False


def test_import_error_is_simulated_and_restored() -> None:
    """确认测试用的 ``ImportError`` 伪装本身有效（否则上面的用例是假绿）。"""
    with _NoJsonRepair():
        with pytest.raises(ImportError):
            __import__("json_repair")

    assert __import__("json_repair") is not None


def test_module_import_does_not_require_json_repair() -> None:
    """模块级不 import json_repair（懒加载）：伪造 ImportError 后仍可重载本模块。"""
    import importlib

    from my_agent.llm import tool_args as module

    with _NoJsonRepair():
        reloaded = importlib.reload(module)

    assert reloaded.INVALID_ARGS_MESSAGE == INVALID_ARGS_MESSAGE


# --------------------------------------------------------------------------------------
# json_repair 在场时确实被调用
# --------------------------------------------------------------------------------------
def test_json_repair_is_used_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """``json_repair`` 在场时必须真的被调用（打桩计数，不依赖其内部实现）。"""
    json_repair = pytest.importorskip("json_repair")
    calls: list[str] = []
    original = json_repair.loads

    def _spy(text: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(text)
        return original(text, *args, **kwargs)

    monkeypatch.setattr(json_repair, "loads", _spy)
    result = parse_tool_arguments('{"file_path": "a.py",}')

    assert result.status == ArgParseStatus.REPAIRED
    assert calls, "json_repair.loads 未被调用"
    assert any("json_repair 修复成功" in note for note in result.notes)


def test_repair_disabled_by_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """``TOOL_ARG_REPAIR_ENABLED=False`` 时只走标准解析（灰度开关生效）。"""
    monkeypatch.setattr(D, "TOOL_ARG_REPAIR_ENABLED", False)

    result = parse_tool_arguments('{"file_path": "a.py",}')

    assert result.status == ArgParseStatus.INVALID
    assert result.repaired is False


def test_repair_skipped_when_over_char_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """超长参数串直接跳过修复器（``TOOL_ARG_MAX_REPAIR_CHARS`` 生效）。"""
    monkeypatch.setattr(D, "TOOL_ARG_MAX_REPAIR_CHARS", 8)

    assert repair_json('{"file_path": "a.py",}') is None
