"""``my_agent.hooks.events`` 单测（§11.1 事件表、§11.3 协议字段、§11.4 能力矩阵）。

这些断言把「事件名冻结」「``continue_`` → JSON 字段 ``continue``」「能力矩阵」
三件事钉死；引擎（``engine/query.py``、``engine/query_engine.py``、
``security/pipeline.py``）与 ``hooks.yaml`` 都依赖它们。
"""

from __future__ import annotations

import pytest

from my_agent.hooks.events import (
    EDITABLE_EVENTS,
    OUTPUT_EVENTS,
    SPEC_EVENTS,
    TOOL_EVENTS,
    HookEvent,
    HookOutcome,
    HookPayload,
    mutable_fields,
)
from my_agent.models.messages import Message
from my_agent.models.tools import ToolResult

#: §11.1 表中逐字冻结的 7 个事件名（U11：实现 7 个，不是 6 个）。
SPEC_EVENT_VALUES: tuple[str, ...] = (
    "user_input_received",
    "before_model_call",
    "after_model_call",
    "before_tool_call",
    "after_tool_call",
    "before_permission_check",
    "after_permission_check",
)


class TestHookEvent:
    """§11.1 事件名 + CONTRACTS §17.3 生命周期事件。"""

    def test_enum_has_the_seven_spec_events(self) -> None:
        """§11.1 的 7 个事件齐备且取值逐字一致。"""
        values = [member.value for member in HookEvent]
        for name in SPEC_EVENT_VALUES:
            assert name in values
        assert len(SPEC_EVENTS) == 7

    def test_lifecycle_events_present_for_engine(self) -> None:
        """CONTRACTS §17.3：引擎另外调用 session_start / session_end。"""
        assert HookEvent.SESSION_START.value == "session_start"
        assert HookEvent.SESSION_END.value == "session_end"
        assert len(list(HookEvent)) == 9

    def test_event_is_str_enum(self) -> None:
        """StrEnum 语义：可直接与字符串比较并写入 JSON。"""
        assert HookEvent.BEFORE_TOOL_CALL == "before_tool_call"
        assert f"{HookEvent.AFTER_TOOL_CALL}" == "after_tool_call"

    def test_unknown_event_value_raises(self) -> None:
        """未知事件名必须报错（加载器据此警告并跳过）。"""
        with pytest.raises(ValueError):
            HookEvent("before_everything")

    def test_tool_events_are_the_matcher_scoped_ones(self) -> None:
        """只有工具相关事件受 ``matcher`` 约束（G-6 的适用范围）。"""
        assert {
            HookEvent.BEFORE_TOOL_CALL,
            HookEvent.AFTER_TOOL_CALL,
            HookEvent.BEFORE_PERMISSION_CHECK,
            HookEvent.AFTER_PERMISSION_CHECK,
        } == TOOL_EVENTS


class TestCapabilityMatrix:
    """§11.4 能力矩阵：可改写入参 / 可改输出。"""

    def test_before_events_allow_argument_edits(self) -> None:
        """``before_tool_call`` / ``before_permission_check`` 可改写入参。"""
        assert mutable_fields(HookEvent.BEFORE_TOOL_CALL) == {"arguments"}
        assert mutable_fields(HookEvent.BEFORE_PERMISSION_CHECK) == {"arguments"}

    def test_user_input_event_can_rewrite_text(self) -> None:
        """``user_input_received`` 可改写用户文本（§11.4）。"""
        assert mutable_fields(HookEvent.USER_INPUT_RECEIVED) == {"text"}

    def test_after_model_call_rewrites_output_text_only(self) -> None:
        """§11.4「可改模型输出」：``after_model_call`` 只能改 ``text``。"""
        assert mutable_fields(HookEvent.AFTER_MODEL_CALL) == {"text"}
        assert HookEvent.AFTER_MODEL_CALL in OUTPUT_EVENTS

    def test_after_tool_call_rewrites_result_only(self) -> None:
        """§11.4「改写结果内容」：``after_tool_call`` 只能改 ``result``。"""
        assert mutable_fields(HookEvent.AFTER_TOOL_CALL) == {"result"}
        assert HookEvent.AFTER_TOOL_CALL in OUTPUT_EVENTS

    def test_after_permission_check_is_immutable(self) -> None:
        """§11.4：``after_permission_check`` 不可改写入参（裁决已定）。"""
        assert mutable_fields(HookEvent.AFTER_PERMISSION_CHECK) == frozenset()
        assert HookEvent.AFTER_PERMISSION_CHECK in EDITABLE_EVENTS  # 空集但仍在矩阵内

    def test_editable_events_cover_the_seven_spec_events(self) -> None:
        """能力矩阵必须覆盖 §11.1 全部 7 个事件（生命周期事件不进矩阵）。"""
        assert set(SPEC_EVENTS) == EDITABLE_EVENTS
        assert HookEvent.SESSION_START not in EDITABLE_EVENTS


class TestHookPayload:
    """§11.3 的 ``payload`` 对象：浅合并值域 + JSON 序列化。"""

    def test_defaults_are_all_none(self) -> None:
        """空负载（``session_start`` 场景）所有字段为 None。"""
        payload = HookPayload()
        assert payload.text is None
        assert payload.messages is None
        assert payload.tool is None
        assert payload.arguments is None
        assert payload.reason is None
        assert payload.result is None
        assert payload.decision is None
        assert payload.subagent_id is None

    def test_accepts_every_documented_field(self) -> None:
        """§11.3 示例 + §17.3：八个字段全部可构造。"""
        result = ToolResult(call_id="c1", tool="Read", ok=True, content="ok")
        payload = HookPayload(
            text="hi",
            messages=[Message(role="user", content="hi")],
            tool="Read",
            arguments={"file_path": "a.py"},
            reason="查看文件",
            result=result,
            decision="allow",
            subagent_id="sub-1",
        )
        assert payload.result is not None
        assert payload.result.content == "ok"
        assert payload.messages is not None
        assert payload.messages[0].role == "user"

    def test_to_json_dict_drops_none_fields(self) -> None:
        """stdin JSON 只包含有值字段（§11.3 示例结构）。"""
        payload = HookPayload(tool="Write", arguments={"file_path": "src/a.py"})
        assert payload.to_json_dict() == {
            "tool": "Write",
            "arguments": {"file_path": "src/a.py"},
        }

    def test_overlay_merges_only_allowed_fields(self) -> None:
        """浅合并：只有白名单内的字段被覆盖。"""
        base = HookPayload(tool="Write", arguments={"file_path": "a.py"}, reason="原原因")
        changes = HookPayload(
            arguments={"file_path": "b.py"}, reason="越权改写", text="越权文本"
        )
        merged = base.overlay(changes, frozenset({"arguments"}))
        assert merged.arguments == {"file_path": "b.py"}
        assert merged.reason == "原原因"
        assert merged.text is None
        assert merged.tool == "Write"

    def test_overlay_ignores_none_values(self) -> None:
        """hook 未提交的字段（None）不得清空原值。"""
        base = HookPayload(text="原文", arguments={"a": 1})
        merged = base.overlay(HookPayload(), frozenset({"text", "arguments"}))
        assert merged.text == "原文"
        assert merged.arguments == {"a": 1}

    def test_overlay_does_not_mutate_original(self) -> None:
        """``overlay`` 纯函数语义：原负载保持不变（G-7 链式可回溯）。"""
        base = HookPayload(arguments={"file_path": "a.py"})
        base.overlay(HookPayload(arguments={"file_path": "b.py"}), frozenset({"arguments"}))
        assert base.arguments == {"file_path": "a.py"}

    def test_tool_result_round_trips_through_json(self) -> None:
        """``result`` 字段可 JSON 往返（跨进程协议要求）。"""
        payload = HookPayload(
            tool="Read",
            result=ToolResult(call_id="c1", tool="Read", ok=False, error="boom"),
        )
        dumped = payload.to_json_dict()
        restored = HookPayload.model_validate(dumped)
        assert restored.result is not None
        assert restored.result.error == "boom"


class TestHookOutcome:
    """§11.3 的输出契约：``decision`` / ``modified_payload`` / ``message`` / ``continue``。"""

    def test_continue_alias_is_serialized_as_continue(self) -> None:
        """``continue_`` 的 JSON 字段名必须是 ``"continue"``（CONTRACTS §17.3）。"""
        outcome = HookOutcome(continue_=False)
        assert outcome.model_dump(by_alias=True)["continue"] is False
        assert "continue_" not in outcome.model_dump(by_alias=True)

    def test_continue_defaults_to_true(self) -> None:
        """默认继续回合。"""
        assert HookOutcome().continue_ is True
        assert HookOutcome().aborted is False

    def test_construct_by_field_name(self) -> None:
        """按字段名构造（引擎与测试都这么写）。"""
        assert HookOutcome(continue_=False).continue_ is False
        # aborted 是 continue_ 的派生属性（见 events.HookOutcome.aborted），必须同步为 True
        assert HookOutcome(continue_=False).aborted is True

    def test_construct_by_json_alias(self) -> None:
        """按 JSON 别名构造（子进程输出 → 模型）。"""
        outcome = HookOutcome.model_validate({"continue": False})
        assert outcome.continue_ is False
        assert outcome.continue_ is not True

    def test_decision_values_are_frozen(self) -> None:
        """``decision`` 只能是 allow / deny / ask 或省略。"""
        for value in ("allow", "deny", "ask"):
            assert HookOutcome(decision=value).decision == value  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            HookOutcome.model_validate({"decision": "maybe"})

    def test_modified_payload_carries_chained_state(self) -> None:
        """``modified_payload`` 承载链式改写后的负载（G-2 / G-7）。"""
        outcome = HookOutcome(
            modified_payload=HookPayload(arguments={"file_path": "patched.py"})
        )
        assert outcome.modified_payload is not None
        assert outcome.modified_payload.arguments == {"file_path": "patched.py"}

    def test_message_defaults_to_empty_string(self) -> None:
        """``message`` 缺省为空串（避免引擎里到处判 None）。"""
        assert HookOutcome().message == ""
