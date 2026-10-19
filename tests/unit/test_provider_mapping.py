"""``llm/provider.py`` 协议层测试（§13.1.4 参数映射 + §13.1.5 异常翻译）。

覆盖点与验收项对照：

* ``I-9`` **C7 依赖隔离**：用 ``ast`` 扫描 ``src/`` 全部 ``.py``，断言 ``import openai``
  只出现在 ``llm/provider.py``；
* ``I-8`` SDK 异常映射表逐条正确，且 ``APITimeoutError`` 先于 ``APIConnectionError``
  被匹配（规格强调的「子类在前」易错点）；
* ``I-4`` API Key 缺失 → :class:`ConfigError`，消息含环境变量名；
* ``I-5`` 日志中不出现密钥材料（正则扫日志文件）；
* ``I-10`` ``close()`` 关闭底层 SDK client；
* §13.1.4 调用映射：``max_tokens`` / ``temperature`` / ``timeout``（按请求传）/
  ``tools``（缺省用 ``NOT_GIVEN``）/ ``stream_options.include_usage``。

HTTP 层用 ``httpx.MockTransport`` 注入（SDK 自带的 transport 取决于安装的 SDK 版本，
因此这里对 ``httpx2`` / ``httpx`` 都做兼容探测），不触网。
"""

from __future__ import annotations

import ast
import importlib
import json
import shutil
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import openai
import pytest

from my_agent.config.schema import EndpointConfig
from my_agent.errors import (
    ConfigError,
    LLMRequestError,
    LLMUnavailableError,
    ModelUnavailableError,
    MyAgentError,
)
from my_agent.llm.provider import (
    SDK_ERROR_MAP,
    OpenAISDKClient,
    StreamChunk,
    ToolCallDelta,
    map_sdk_error,
    read_api_key,
)
from my_agent.models.messages import Message, ToolCallRequest

#: 被测源码根（``src/``）与唯一允许 import openai 的文件（C7）。
_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
_OPENAI_ALLOWLIST = ("my_agent/llm/provider.py",)

#: ``map_sdk_error`` 期望的映射（§13.1.5 冻结表，**顺序即优先级**）。
_EXPECTED_MAPPING: tuple[tuple[str, type[MyAgentError]], ...] = (
    ("APITimeoutError", ModelUnavailableError),
    ("APIConnectionError", LLMUnavailableError),
    ("RateLimitError", LLMUnavailableError),
    ("InternalServerError", LLMUnavailableError),
    ("AuthenticationError", ConfigError),
    ("PermissionDeniedError", ConfigError),
    ("BadRequestError", LLMRequestError),
    ("NotFoundError", ConfigError),
    ("APIStatusError", LLMUnavailableError),
)

#: 需要构造 ``httpx.Response`` 的异常 → HTTP 状态码。
_STATUS_CODES: dict[str, int] = {
    "RateLimitError": 429,
    "InternalServerError": 500,
    "AuthenticationError": 401,
    "PermissionDeniedError": 403,
    "BadRequestError": 400,
    "NotFoundError": 404,
    "APIStatusError": 418,
}

#: 写你自己的 key：这是测试用占位值（非真实密钥），仅用于验证 I-5「日志不泄漏密钥」。
_FAKE_KEY = "YOUR_API_KEY_HERE"


@pytest.fixture
def logs_dir() -> Iterator[Path]:
    """在仓库内提供一次性日志目录。

    pytest 的 ``tmp_path`` 基目录落在系统临时区，某些受限环境下不可写；这里改用
    仓库内的一次性目录，测试结束后删除。

    Yields:
        已创建的目录路径。
    """
    base = Path(__file__).resolve().parents[2] / ".tmp-llm-tests" / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _http_module() -> Any:
    """返回 SDK 自身使用的 httpx 模块（openai 3.x 用 ``httpx2``）。

    Returns:
        具备 ``Request`` / ``Response`` / ``MockTransport`` / ``AsyncClient`` 的模块。

    Raises:
        pytest.skip.Exception: 环境里的 SDK 用的是另一套 httpx，跳过协议层 HTTP 测试。
    """
    for name in ("httpx2", "httpx"):
        try:
            module = importlib.import_module(name)
        except ImportError:  # pragma: no cover - 取决于 SDK 版本
            continue
        if all(hasattr(module, attr) for attr in ("Request", "Response", "MockTransport")):
            return module
    pytest.skip("未找到 SDK 使用的 httpx 实现（httpx2/httpx），跳过 HTTP 层测试")


def _build_sdk_error(name: str) -> BaseException:
    """构造**真实**的 SDK 异常实例（不是 Mock）。

    Args:
        name: ``openai`` 模块里的异常类名。

    Returns:
        该异常的实例；状态码类带真实的 ``httpx.Response``。

    Raises:
        ValueError: 名字不在冻结映射表内。
    """
    cls = getattr(openai, name, None)
    if cls is None:  # pragma: no cover - SDK 版本漂移
        raise ValueError(f"当前 SDK 没有 {name}")
    httpx_mod = _http_module()
    request = httpx_mod.Request("POST", "http://127.0.0.1:9/v1/chat/completions")
    if name == "APITimeoutError":
        return cls(request=request)
    if name == "APIConnectionError":
        return cls(request=request, message="connection refused")
    status = _STATUS_CODES[name]
    response = httpx_mod.Response(
        status, request=request, json={"error": {"message": f"{name} from fake server"}}
    )
    return cls(f"{name} from fake server", response=response, body=None)


def _iter_sources() -> Iterator[tuple[str, ast.Module]]:
    """遍历 ``src/`` 下全部 Python 文件的 AST。

    Yields:
        ``(相对 src 的 POSIX 路径, 模块 AST)``。
    """
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(_SRC_ROOT).as_posix()
        yield relative, ast.parse(path.read_text(encoding="utf-8"))


def _imports_openai(tree: ast.Module) -> bool:
    """判断模块是否直接依赖 openai（含 ``import openai.x`` 与 ``from openai import x``）。

    会走到 ``ast.walk`` 的每个节点，因此 ``if TYPE_CHECKING:`` 分支里的导入同样被算作依赖。

    Args:
        tree: 模块 AST。

    Returns:
        ``True`` = 该模块 import 了 SDK。
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "openai" or alias.name.startswith("openai.") for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "openai" or module.startswith("openai."):
                return True
    return False


# --------------------------------------------------------------------------------------
# I-9：C7 依赖隔离（ast 扫描 src/）
# --------------------------------------------------------------------------------------


def test_openai_is_imported_only_by_llm_provider() -> None:
    """I-9：``src/`` 下只有 ``llm/provider.py`` 可以 ``import openai``（C7）。"""
    offenders = sorted(
        relative for relative, tree in _iter_sources() if _imports_openai(tree)
    )

    assert offenders == list(_OPENAI_ALLOWLIST), (
        f"违反 C7 的模块：{offenders}；只允许 {list(_OPENAI_ALLOWLIST)}"
    )


def test_openai_allowlist_target_really_imports_the_sdk() -> None:
    """反向断言：``llm/provider.py`` 确实是那个 import openai 的文件（防止扫描失效）。"""
    tree = ast.parse((_SRC_ROOT / _OPENAI_ALLOWLIST[0]).read_text(encoding="utf-8"))

    assert _imports_openai(tree) is True


# --------------------------------------------------------------------------------------
# I-8：SDK 异常 → 内部异常映射表
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sdk_name", "expected"), _EXPECTED_MAPPING, ids=[name for name, _ in _EXPECTED_MAPPING]
)
def test_sdk_error_mapping_is_exact(sdk_name: str, expected: type[MyAgentError]) -> None:
    """I-8：9 种 SDK 异常逐一映射到 §13.1.5 指定的内部异常。"""
    mapped = map_sdk_error(_build_sdk_error(sdk_name))

    assert type(mapped) is expected, (
        f"{sdk_name} → {type(mapped).__name__}（期望 {expected.__name__}）"
    )
    assert mapped.detail
    assert mapped.user_message()


def test_api_timeout_is_matched_before_api_connection_error() -> None:
    """I-8 易错点：``APITimeoutError`` 是 ``APIConnectionError`` 的子类。

    父子关系决定了「子类必须先被判」——否则超时会被误判成普通连接错误，
    判定角色就拿不到 :class:`ModelUnavailableError` 降级信号（§6.5.4）。
    """
    assert issubclass(openai.APITimeoutError, openai.APIConnectionError), (
        "SDK 层级已变：请重新核对 §13.1.5 的 except 顺序说明"
    )
    timeout = map_sdk_error(_build_sdk_error("APITimeoutError"))
    connection = map_sdk_error(_build_sdk_error("APIConnectionError"))

    assert type(timeout) is ModelUnavailableError
    assert type(connection) is LLMUnavailableError
    assert type(timeout) is not type(connection)


def test_rate_limit_and_internal_server_error_precede_api_status_error() -> None:
    """I-8 易错点：``RateLimitError`` / ``InternalServerError`` 亦是 ``APIStatusError`` 子类。"""
    assert issubclass(openai.RateLimitError, openai.APIStatusError)
    assert issubclass(openai.InternalServerError, openai.APIStatusError)
    assert type(map_sdk_error(_build_sdk_error("RateLimitError"))) is LLMUnavailableError
    assert type(map_sdk_error(_build_sdk_error("InternalServerError"))) is LLMUnavailableError
    assert type(map_sdk_error(_build_sdk_error("APIStatusError"))) is LLMUnavailableError


def test_error_map_table_order_puts_subclasses_first() -> None:
    """冻结表本身必须「子类在前」：``map_sdk_error`` 按表顺序做 ``isinstance`` 匹配。"""
    order = [name for name, _ in _EXPECTED_MAPPING]
    positions = {name: index for index, name in enumerate(order)}

    assert len(order) == 9
    assert positions["APITimeoutError"] < positions["APIConnectionError"]
    for name in (
        "RateLimitError",
        "InternalServerError",
        "AuthenticationError",
        "PermissionDeniedError",
        "BadRequestError",
        "NotFoundError",
    ):
        assert positions[name] < positions["APIStatusError"], name


def test_sdk_error_map_source_literal_keeps_subclasses_first() -> None:
    """源码里的 :data:`SDK_ERROR_MAP` **字面顺序**也必须「子类在前」（防止有人重排）。"""
    tree = ast.parse((_SRC_ROOT / _OPENAI_ALLOWLIST[0]).read_text(encoding="utf-8"))
    names = _sdk_error_map_literal_names(tree)

    assert names, "未能在 provider.py 中找到 SDK_ERROR_MAP 字面量表"
    assert names[0] == "APITimeoutError"
    assert names.index("APITimeoutError") < names.index("APIConnectionError")
    assert names[-1] == "APIStatusError"
    for name in ("RateLimitError", "InternalServerError", "BadRequestError", "NotFoundError"):
        assert names.index(name) < names.index("APIStatusError"), name


def _sdk_error_map_literal_names(tree: ast.Module) -> list[str]:
    """从 AST 里取出 ``SDK_ERROR_MAP`` 字面量的键名顺序。

    Args:
        tree: ``provider.py`` 的模块 AST。

    Returns:
        映射表左列的属性名列表（如 ``["APITimeoutError", ...]``）。
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id != "SDK_ERROR_MAP" or not isinstance(node.value, ast.Tuple):
            continue
        names: list[str] = []
        for element in node.value.elts:
            if not isinstance(element, ast.Tuple) or not element.elts:
                continue
            first = element.elts[0]
            if isinstance(first, ast.Attribute):
                names.append(first.attr)
        return names
    return []


def test_error_map_constant_matches_the_frozen_table() -> None:
    """运行期常量与冻结表逐条一致（名字与内部异常类都对得上）。"""
    actual = [(cls.__name__, target) for cls, target in SDK_ERROR_MAP]

    assert actual == [(name, target) for name, target in _EXPECTED_MAPPING]


def test_already_internal_errors_pass_through_unchanged() -> None:
    """内部异常再映射一次必须原样返回（``raise map_sdk_error(exc)`` 不会层层包装）。"""
    original = LLMUnavailableError("已翻译", detail="已翻译")

    assert map_sdk_error(original) is original


def test_unknown_exception_falls_back_to_llm_unavailable() -> None:
    """非 SDK 的未知异常收敛为 :class:`LLMUnavailableError`（内部异常体系内可重试的兜底）。"""
    mapped = map_sdk_error(RuntimeError("weird"))

    assert isinstance(mapped, LLMUnavailableError)
    assert "RuntimeError" in mapped.detail


def test_map_sdk_error_rejects_non_exception_input() -> None:
    """传入非异常对象是编程错误，必须显式暴露（不静默吞掉）。"""
    with pytest.raises(TypeError):
        map_sdk_error("not an exception")  # type: ignore[arg-type]


def test_map_sdk_error_includes_http_status_in_detail() -> None:
    """状态码要进 ``detail``（便于审计与用户排查），且不含任何密钥材料。"""
    mapped = map_sdk_error(_build_sdk_error("BadRequestError"))

    assert "400" in mapped.detail


# --------------------------------------------------------------------------------------
# I-4：API Key 缺失
# --------------------------------------------------------------------------------------


def test_missing_api_key_raises_config_error_naming_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-4：密钥缺失 → ``ConfigError``，消息含环境变量名与配置指引。"""
    monkeypatch.delenv("MYAGENT_MISSING_KEY", raising=False)
    with pytest.raises(ConfigError) as excinfo:
        read_api_key("MYAGENT_MISSING_KEY")

    message = excinfo.value.user_message()
    assert "MYAGENT_MISSING_KEY" in message
    assert "api_key_env" in message
    assert excinfo.value.code == "config_error"


def test_blank_api_key_is_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """只有空白字符的密钥等价于未设置（避免把空串发给端点）。"""
    monkeypatch.setenv("MYAGENT_BLANK_KEY", "   ")
    with pytest.raises(ConfigError):
        read_api_key("MYAGENT_BLANK_KEY")


def test_read_api_key_returns_value_from_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式传入 env 映射时按映射取值（不读进程环境）。"""
    monkeypatch.delenv("MYAGENT_MAPPING_KEY", raising=False)

    assert read_api_key("MYAGENT_MAPPING_KEY", {"MYAGENT_MAPPING_KEY": "v"}) == "v"


async def test_provider_raises_config_error_on_first_use_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-4：构造 Provider 本身不报错，**首次调用**时因密钥缺失抛 ``ConfigError``。"""
    monkeypatch.delenv("MYAGENT_ABSENT_KEY", raising=False)
    provider = OpenAISDKClient(
        EndpointConfig(
            name="main",
            base_url="http://127.0.0.1:9/v1",
            api_key_env="MYAGENT_ABSENT_KEY",
            model="test-model",
        )
    )
    with pytest.raises(ConfigError) as excinfo:
        await provider.complete(
            system="s",
            messages=[Message(role="user", content="hi")],
            max_tokens=8,
            temperature=0.0,
            timeout_s=1.0,
        )

    assert "MYAGENT_ABSENT_KEY" in excinfo.value.user_message()


# --------------------------------------------------------------------------------------
# §13.1.4 调用映射（httpx MockTransport 注入；不触网）
# --------------------------------------------------------------------------------------


async def test_non_streaming_call_maps_parameters_and_parses_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非流式：参数映射正确，响应（文本 + 工具调用 + usage）可解析。"""
    seen: list[dict[str, Any]] = []

    def handler(request: Any) -> Any:
        seen.append(json.loads(request.content))
        return _http_module().Response(
            200,
            request=request,
            json={
                "id": "cmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "看一下",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Read",
                                        "arguments": '{"file_path": "a.py"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
            },
        )

    provider, http_client = _provider_with_transport(handler, monkeypatch)
    result = await provider.complete(
        system="SYSTEM-PROMPT",
        messages=[Message(role="user", content="读文件", turn_index=4)],
        tools=_tool_schema(),
        max_tokens=123,
        temperature=0.3,
        timeout_s=7.5,
    )

    assert result.text == "看一下"
    assert result.finish_reason == "tool_calls"
    assert result.model == "test-model"
    assert result.usage["total"] == 12
    assert result.latency_ms >= 0
    [call] = result.tool_calls
    assert (call.id, call.name) == ("call_1", "Read")
    assert call.arguments == {"file_path": "a.py"}
    assert call.raw_arguments == '{"file_path": "a.py"}'

    [payload] = seen
    assert payload["model"] == "test-model"
    assert payload["messages"][0] == {"role": "system", "content": "SYSTEM-PROMPT"}
    # 消息净化：内部元数据（turn_index）绝不进入请求体。
    assert payload["messages"][1] == {"role": "user", "content": "读文件"}
    assert payload["max_tokens"] == 123
    assert payload["temperature"] == 0.3
    assert payload["tools"][0]["function"]["name"] == "Read"
    assert "stream" not in payload

    await provider.close()
    assert http_client.is_closed is True  # I-10 的一环：SDK client 真的被关掉了


async def test_tools_are_omitted_when_not_given(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tools=None`` 时请求体里不应出现 ``tools`` 键（用 SDK 哨兵而非 ``None``）。"""
    seen: list[dict[str, Any]] = []

    def handler(request: Any) -> Any:
        seen.append(json.loads(request.content))
        return _http_module().Response(
            200,
            request=request,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    provider, _ = _provider_with_transport(handler, monkeypatch)
    await provider.complete(
        system="s",
        messages=[Message(role="user", content="hi")],
        tools=None,
        max_tokens=16,
        temperature=0.0,
        timeout_s=1.0,
    )

    assert "tools" not in seen[0]
    await provider.close()


async def test_non_streaming_invalid_json_arguments_are_kept_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非流式路径同样保留非法 JSON 的原始串（交上层记 ``ToolCallParseError``）。"""

    def handler(request: Any) -> Any:
        return _http_module().Response(
            200,
            request=request,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_bad",
                                    "type": "function",
                                    "function": {"name": "Read", "arguments": "{not json"},
                                }
                            ],
                        },
                    }
                ],
            },
        )

    provider, _ = _provider_with_transport(handler, monkeypatch)
    result = await provider.complete(
        system="s",
        messages=[Message(role="user", content="hi")],
        max_tokens=16,
        temperature=0.0,
        timeout_s=1.0,
    )

    [call] = result.tool_calls
    assert call.raw_arguments == "{not json"
    assert call.arguments == {}
    await provider.close()


async def test_streaming_call_sets_include_usage_and_maps_fragments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """流式：``stream=True`` + ``stream_options.include_usage``，分片按 index/文本映射。"""
    seen: list[dict[str, Any]] = []
    events = [
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "你"},
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse({"choices": [{"index": 0, "delta": {"content": "好"}, "finish_reason": None}]}),
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_9",
                                    "type": "function",
                                    "function": {"name": "Read", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '{"file_p'}}]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'ath": "a.py"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
            }
        ),
    ]

    def handler(request: Any) -> Any:
        seen.append(json.loads(request.content))
        body = b"".join(events) + b"data: [DONE]\n\n"
        return _http_module().Response(
            200, request=request, headers={"content-type": "text/event-stream"}, content=body
        )

    provider, _ = _provider_with_transport(handler, monkeypatch)
    chunks: list[StreamChunk] = [
        chunk
        async for chunk in provider.stream(
            system="s",
            messages=[Message(role="user", content="hi")],
            tools=None,
            max_tokens=64,
            temperature=0.0,
            timeout_s=3.0,
        )
    ]

    assert seen[0]["stream"] is True
    assert seen[0]["stream_options"] == {"include_usage": True}
    assert "".join(chunk.text_delta for chunk in chunks) == "你好"
    deltas: list[ToolCallDelta] = [
        chunk.tool_call_delta for chunk in chunks if chunk.tool_call_delta is not None
    ]
    assert (deltas[0].index, deltas[0].id, deltas[0].name) == (0, "call_9", "Read")
    assert "".join(delta.arguments_delta for delta in deltas) == '{"file_path": "a.py"}'
    assert chunks[-1].usage == {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 12,
        "total": 12,
    }
    assert any(chunk.finish_reason == "tool_calls" for chunk in chunks)
    await provider.close()


async def test_request_timeout_becomes_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-8 端到端（真实超时）：``timeout=`` 生效 → SDK 抛 ``APITimeoutError`` → 判定降级异常。

    这是「超时按请求传」的直接证据：handler 故意睡过 ``timeout_s``，请求必须失败。
    """
    seen: list[dict[str, Any]] = []

    def handler(request: Any) -> Any:
        seen.append(json.loads(request.content))
        time.sleep(0.8)  # 足够大：让请求先发出去，再在「等待响应」阶段超时
        return _http_module().Response(
            200,
            request=request,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "too late"},
                    }
                ],
            },
        )

    provider, _ = _provider_with_transport(handler, monkeypatch)
    with pytest.raises(ModelUnavailableError):
        await provider.complete(
            system="s",
            messages=[Message(role="user", content="hi")],
            max_tokens=8,
            temperature=0.0,
            timeout_s=0.3,
        )

    assert seen, "请求确实发出去了（超时发生在等待响应时）"
    await provider.close()


# APITimeoutError 与 APIConnectionError 都没有对应的 HTTP 状态码：
#  - 超时由 test_request_timeout_becomes_model_unavailable 覆盖（真实挂钟超时）；
#  - 连接失败由 test_transport_level_connection_error_is_translated 覆盖。
_WITHOUT_STATUS: frozenset[str] = frozenset({"APITimeoutError", "APIConnectionError"})


@pytest.mark.parametrize(
    ("sdk_name", "expected"),
    [entry for entry in _EXPECTED_MAPPING if entry[0] not in _WITHOUT_STATUS],
    ids=[name for name, _ in _EXPECTED_MAPPING if name not in _WITHOUT_STATUS],
)
async def test_http_error_responses_are_translated_end_to_end(
    sdk_name: str,
    expected: type[MyAgentError],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-8 端到端：真实 HTTP 错误码经 SDK 抛出后翻译成约定的内部异常。

    ``APITimeoutError`` 不在此列——它没有对应的 HTTP 状态码（见
    :func:`test_request_timeout_becomes_model_unavailable`）。
    """
    status = _STATUS_CODES[sdk_name]

    def handler(request: Any) -> Any:
        return _http_module().Response(
            status, request=request, json={"error": {"message": f"{sdk_name} e2e"}}
        )

    provider, _ = _provider_with_transport(handler, monkeypatch)
    with pytest.raises(MyAgentError) as excinfo:
        await provider.complete(
            system="s",
            messages=[Message(role="user", content="hi")],
            max_tokens=8,
            temperature=0.0,
            timeout_s=1.0,
        )

    assert isinstance(excinfo.value, expected), (
        f"HTTP {status} → {type(excinfo.value).__name__}（期望 {expected.__name__}）"
    )
    await provider.close()


async def test_transport_level_connection_error_is_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """传输层直接失败（无 HTTP 响应）→ ``LLMUnavailableError``，不泄漏密钥。"""

    def handler(_request: Any) -> Any:
        raise _http_module().ConnectError("simulated network failure")

    provider, _ = _provider_with_transport(handler, monkeypatch, api_key=_FAKE_KEY)
    with pytest.raises(LLMUnavailableError) as excinfo:
        await provider.complete(
            system="s",
            messages=[Message(role="user", content="hi")],
            max_tokens=8,
            temperature=0.0,
            timeout_s=1.0,
        )

    assert _FAKE_KEY not in str(excinfo.value)
    await provider.close()


# --------------------------------------------------------------------------------------
# I-5：日志不含密钥
# --------------------------------------------------------------------------------------


async def test_log_file_never_contains_key_material(
    monkeypatch: pytest.MonkeyPatch,
    logs_dir: Path,
    make_config: Any,
) -> None:
    """I-5：失败调用的日志里不出现密钥本体（正则扫日志文件）。"""

    def handler(_request: Any) -> Any:
        raise _http_module().ConnectError("simulated network failure")

    provider, _ = _provider_with_transport(handler, monkeypatch, api_key=_FAKE_KEY)
    config = make_config()
    config.llm.log_dir = str(logs_dir)

    import my_agent.llm.client as client_module

    client = client_module.LLMClient(provider, config, client_module.UsageTracker())
    with pytest.raises(MyAgentError):
        await client.complete_role("judge", user="审计我的参数")
    await provider.close()

    logs = sorted(logs_dir.glob("llm-*.log"))
    assert logs, "未生成 LLM 日志文件"
    text = "\n".join(path.read_text(encoding="utf-8") for path in logs)
    assert _FAKE_KEY not in text
    assert "role=judge" in text  # 摘要确实写进去了，不是「什么都没写」


async def test_successful_call_log_summary_has_no_key(
    monkeypatch: pytest.MonkeyPatch,
    logs_dir: Path,
    make_config: Any,
) -> None:
    """I-5（成功路径）：日志只记摘要字段，密钥与 Authorization 头都不出现。"""

    def handler(request: Any) -> Any:
        return _http_module().Response(
            200,
            request=request,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider, _ = _provider_with_transport(handler, monkeypatch, api_key=_FAKE_KEY)
    config = make_config()
    config.llm.log_dir = str(logs_dir)

    import my_agent.llm.client as client_module

    client = client_module.LLMClient(provider, config, client_module.UsageTracker())
    await client.complete_role("judge", user="hi")
    await provider.close()

    [log] = sorted(logs_dir.glob("llm-*.log"))
    text = log.read_text(encoding="utf-8")
    assert _FAKE_KEY not in text
    assert "Authorization" not in text
    assert "role=judge mode=complete" in text
    assert "usage=" in text


# --------------------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------------------


def _provider_with_transport(
    handler: Callable[[Any], Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_key: str = "test-key-not-a-secret",
) -> tuple[OpenAISDKClient, Any]:
    """构造注入了 ``MockTransport`` 的 Provider。

    Args:
        handler: ``MockTransport`` 处理函数（``Request -> Response``）。
        monkeypatch: pytest 的 monkeypatch（用于设置环境变量）。
        api_key: 写入环境变量的密钥。

    Returns:
        ``(provider, http_client)``；用完请 ``await provider.close()``。
    """
    httpx_mod = _http_module()
    monkeypatch.setenv("MYAGENT_TEST_KEY", api_key)
    http_client = httpx_mod.AsyncClient(transport=httpx_mod.MockTransport(handler))
    provider = OpenAISDKClient(
        EndpointConfig(
            name="main",
            base_url="http://127.0.0.1:9/v1",
            api_key_env="MYAGENT_TEST_KEY",
            model="test-model",
        ),
        http_client=http_client,
    )
    return provider, http_client


def _tool_schema() -> list[dict[str, Any]]:
    """返回一个最小的 function schema。

    Returns:
        ``[{"type": "function", "function": {...}}]``。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "读文件",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            },
        }
    ]


def _sse(payload: dict[str, Any]) -> bytes:
    """把一条 chunk 编码为 SSE 帧。

    Args:
        payload: chunk 字典（与默认字段合并）。

    Returns:
        ``b"data: {...}\\n\\n"``。
    """
    base = {"id": "chunk", "object": "chat.completion.chunk", "created": 0, "model": "test-model"}
    return f"data: {json.dumps(base | payload)}\n\n".encode()


def _sample_tool_call() -> ToolCallRequest:
    """构造一个内部工具调用（供断言与脚本复用）。

    Returns:
        :class:`~my_agent.models.messages.ToolCallRequest`。
    """
    return ToolCallRequest(id="c1", name="Read", arguments={"file_path": "a.py"})
