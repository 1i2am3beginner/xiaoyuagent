"""``tools/web.py`` 重试行为单测：WebFetch / WebSearch 的工具层网络退避。

被测纪律（§14 + 工具层重试补齐）：

* 超时 / 连接失败 / 408 / 429 / 5xx → 按 ``TOOL_RETRY_BACKOFF_S`` 退避重试，
  次数上限 ``TOOL_RETRY_MAX_ATTEMPTS``；
* **4xx（如 404）立即失败**——只发起 1 次请求（重试无用）；
* 重试耗尽后**仍返回** ``ToolResult(ok=False)``，错误文案可操作，
  **绝不让异常逃出工具层**（§14.4 失败隔离）；
* ``WEB_FETCH_TIMEOUT_S`` / 大小上限行为不变。

实现手法：把 ``httpx.AsyncClient`` 换成带 ``httpx.MockTransport`` 的实例（不触网），
并把模块级 ``_retry_sleep`` 注入为记录器（**测试绝不真实等待退避**）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from my_agent.config import defaults as D
from my_agent.config.schema import Config
from my_agent.models.tools import ToolContext
from my_agent.tools import ensure_tool_context_ready
from my_agent.tools import web as web_module
from my_agent.tools.web import WebFetchTool, WebSearchTool

# ``models/tools.py`` 只在 TYPE_CHECKING 下引用 ``Config``，构造 ToolContext 前需显式重建
# （工具层提供的公开补救入口；不改冻结的 models/）。
assert ensure_tool_context_ready(), "ToolContext 前向引用未能解析，测试无法构造上下文"

FETCH_URL = "http://fetch.test/doc"
SEARCH_URL = "http://search.test/api"

#: 剧本元素：表示「按默认的成功响应」（区别于剧本用尽）。
DEFAULT = object()

#: 「一直失败」剧本用的长度 = 工具层总尝试次数（显式、可读）。
ATTEMPTS = D.TOOL_RETRY_MAX_ATTEMPTS


class SleepRecorder:
    """记录退避调用但**不真实等待**的注入式 ``sleep``。

    Attributes:
        delays: 依次收到的退避秒数。
    """

    def __init__(self) -> None:
        """初始化空记录。"""
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        """记录一次退避（立即返回）。

        Args:
            seconds: 退避秒数。
        """
        self.delays.append(seconds)


class FakeTransport:
    """按剧本回放响应/异常的 ``httpx.MockTransport`` 处理器（统计请求次数）。

    Attributes:
        calls: 已处理的请求数。
        requests: 收到的请求列表。
    """

    def __init__(self, script: list[Any]) -> None:
        """初始化处理器。

        Args:
            script: 每次请求的剧本元素；异常实例 = 抛出，``httpx.Response`` = 返回，
                :data:`DEFAULT` 或剧本用尽 = 返回默认的 200 HTML 响应。
        """
        self._script = list(script)
        self.calls = 0
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """处理一次请求。

        Args:
            request: 收到的请求。

        Returns:
            剧本中的响应。

        Raises:
            BaseException: 剧本元素是异常实例时原样抛出（模拟连接/超时故障）。
        """
        self.requests.append(request)
        index = self.calls
        self.calls += 1
        item = self._script[index] if index < len(self._script) else DEFAULT
        if item is DEFAULT:
            return text_response("ok body")
        if isinstance(item, BaseException):
            raise item
        return item


def json_response(payload: dict[str, Any], status_code: int = 200) -> httpx.Response:
    """构造一个 JSON 响应。

    Args:
        payload: JSON 体。
        status_code: 状态码。

    Returns:
        :class:`httpx.Response`（``content-type: application/json``）。
    """
    return httpx.Response(
        status_code,
        headers={"content-type": "application/json"},
        json=payload,
        request=httpx.Request("GET", FETCH_URL),
    )


def text_response(text: str, status_code: int = 200) -> httpx.Response:
    """构造一个 HTML 响应。

    Args:
        text: 响应体。
        status_code: 状态码。

    Returns:
        :class:`httpx.Response`（``content-type: text/html``）。
    """
    return httpx.Response(
        status_code,
        headers={"content-type": "text/html; charset=utf-8"},
        text=text,
        request=httpx.Request("GET", FETCH_URL),
    )


SEARCH_PAYLOAD: dict[str, Any] = {
    "results": [
        {"title": "文档", "url": "http://docs.test/a", "snippet": "摘要"},
    ]
}


@pytest.fixture
def ctx() -> ToolContext:
    """构造一个最小可用的工具上下文。

    Returns:
        指向工作区的 :class:`~my_agent.models.tools.ToolContext`。
    """
    return ToolContext(
        session_id="s-retry",
        turn_index=0,
        cwd=".",
        project_root=".",
        config=Config(),
        artifact_dir=".",
        call_id="call-1",
        permission_checked=True,
    )


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> SleepRecorder:
    """把 ``tools/web.py`` 的退避睡眠注入为记录器。

    Args:
        monkeypatch: pytest 的 monkeypatch。

    Returns:
        记录器（断言精确退避序列）。
    """
    recorder = SleepRecorder()
    monkeypatch.setattr(web_module, "_retry_sleep", recorder)
    return recorder


def install_transport(monkeypatch: pytest.MonkeyPatch, handler: FakeTransport) -> None:
    """把 ``tools/web.py`` 里的 ``httpx.AsyncClient`` 换成 MockTransport 版本。

    Args:
        monkeypatch: pytest 的 monkeypatch。
        handler: 剧本处理器。
    """
    real_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        """构造一个强制走 MockTransport 的客户端（忽略被测代码传入的 timeout 等）。

        Args:
            *args: 位置参数（忽略）。
            **kwargs: 关键字参数（忽略）。

        Returns:
            :class:`httpx.AsyncClient`。
        """
        del args, kwargs
        return real_client(transport=httpx.MockTransport(handler), follow_redirects=True)

    monkeypatch.setattr(web_module.httpx, "AsyncClient", factory)


def search_config() -> SimpleNamespace:
    """构造 WebSearch 的搜索后端配置（flat 键，走宽容查找）。

    Returns:
        带 ``web_search_*`` 属性的配置对象。
    """
    return SimpleNamespace(
        web_search_backend="custom",
        web_search_url=SEARCH_URL,
        web_search_method="POST",
        web_search_results_path="results",
    )


# --------------------------------------------------------------------------------------
# WebFetch
# --------------------------------------------------------------------------------------


async def test_web_fetch_retries_503_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    ctx: ToolContext,
    sleeps: SleepRecorder,
) -> None:
    """瞬时 503 两次后 200 → 成功，且退避序列为 ``[0.5, 1.0]``。"""
    handler = FakeTransport(
        [
            text_response("down", 503),
            text_response("down", 503),
            text_response("<html><title>标题</title><body>正文</body></html>"),
        ]
    )
    install_transport(monkeypatch, handler)

    result = await WebFetchTool().run({"url": FETCH_URL}, ctx)

    assert result.ok is True
    assert "正文" in result.content
    assert handler.calls == D.TOOL_RETRY_MAX_ATTEMPTS == 3
    assert sleeps.delays == [0.5, 1.0]


async def test_web_fetch_retries_connect_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    ctx: ToolContext,
    sleeps: SleepRecorder,
) -> None:
    """``httpx.ConnectError`` 两次后成功 → 成功，且只重试 2 次。"""
    handler = FakeTransport(
        [
            httpx.ConnectError("connection refused"),
            httpx.ConnectError("connection refused"),
            text_response("ok body"),
        ]
    )
    install_transport(monkeypatch, handler)

    result = await WebFetchTool().run({"url": FETCH_URL}, ctx)

    assert result.ok is True
    assert "ok body" in result.content
    assert handler.calls == 3
    assert sleeps.delays == [0.5, 1.0]


async def test_web_fetch_404_fails_after_exactly_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
    ctx: ToolContext,
    sleeps: SleepRecorder,
) -> None:
    """404 是客户端错误 → **只尝试 1 次**，不重试，且失败文案带状态码。"""
    handler = FakeTransport([text_response("not found", 404)] * ATTEMPTS)
    install_transport(monkeypatch, handler)

    result = await WebFetchTool().run({"url": FETCH_URL}, ctx)

    assert result.ok is False
    assert handler.calls == 1
    assert sleeps.delays == []
    assert result.error is not None
    assert "404" in result.error


async def test_web_fetch_exhausted_retries_returns_failed_result_not_exception(
    monkeypatch: pytest.MonkeyPatch,
    ctx: ToolContext,
    sleeps: SleepRecorder,
) -> None:
    """重试耗尽（一直 503）→ ``ToolResult(ok=False)``，异常**不外泄**。"""
    handler = FakeTransport([text_response("down", 503)] * ATTEMPTS)
    install_transport(monkeypatch, handler)

    result = await WebFetchTool().run({"url": FETCH_URL}, ctx)

    assert result.ok is False
    assert result.tool == "WebFetch"
    assert handler.calls == ATTEMPTS == 3
    assert sleeps.delays == [0.5, 1.0]
    assert result.error is not None
    assert "503" in result.error
    assert "已重试 2 次" in result.error


async def test_web_fetch_exhausted_connect_errors_returns_failed_result(
    monkeypatch: pytest.MonkeyPatch,
    ctx: ToolContext,
    sleeps: SleepRecorder,
) -> None:
    """连接一直失败 → ``ToolResult(ok=False)``（不是抛 ``ConnectError``）。"""
    handler = FakeTransport([httpx.ConnectError("boom")] * ATTEMPTS)
    install_transport(monkeypatch, handler)

    result = await WebFetchTool().run({"url": FETCH_URL}, ctx)

    assert result.ok is False
    assert handler.calls == ATTEMPTS
    assert sleeps.delays == [0.5, 1.0]
    assert result.error is not None
    assert "抓取失败" in result.error


async def test_web_fetch_timeout_is_retried_then_reported(
    monkeypatch: pytest.MonkeyPatch,
    ctx: ToolContext,
    sleeps: SleepRecorder,
) -> None:
    """``ReadTimeout``（超时属可重试网络异常）→ 重试耗尽后给出超时文案。"""
    handler = FakeTransport([httpx.ReadTimeout("too slow")] * ATTEMPTS)
    install_transport(monkeypatch, handler)

    result = await WebFetchTool().run({"url": FETCH_URL}, ctx)

    assert result.ok is False
    assert handler.calls == ATTEMPTS
    assert sleeps.delays == [0.5, 1.0]
    assert result.error is not None
    assert "超时" in result.error
    assert f"{D.WEB_FETCH_TIMEOUT_S:g}s" in result.error


async def test_web_fetch_success_on_first_attempt_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
    ctx: ToolContext,
    sleeps: SleepRecorder,
) -> None:
    """一次成功 → 恰好 1 次请求、0 次退避（重试不改变成功路径）。"""
    handler = FakeTransport([text_response("hello")])
    install_transport(monkeypatch, handler)

    result = await WebFetchTool().run({"url": FETCH_URL}, ctx)

    assert result.ok is True
    assert handler.calls == 1
    assert sleeps.delays == []


async def test_web_fetch_invalid_url_still_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
    ctx: ToolContext,
    sleeps: SleepRecorder,
) -> None:
    """非法 URL 在发起请求前就失败（重试逻辑不影响参数校验）。"""
    handler = FakeTransport([text_response("never")])
    install_transport(monkeypatch, handler)

    result = await WebFetchTool().run({"url": "ftp://bad.test/x"}, ctx)

    assert result.ok is False
    assert handler.calls == 0
    assert sleeps.delays == []


# --------------------------------------------------------------------------------------
# WebSearch
# --------------------------------------------------------------------------------------


async def test_web_search_retries_503_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    ctx: ToolContext,
    sleeps: SleepRecorder,
) -> None:
    """搜索后端瞬时 503 两次后成功 → 返回结果条目，退避 ``[0.5, 1.0]``。"""
    handler = FakeTransport(
        [
            json_response({"detail": "unavailable"}, 503),
            json_response({"detail": "unavailable"}, 503),
            json_response(SEARCH_PAYLOAD),
        ]
    )
    install_transport(monkeypatch, handler)

    result = await WebSearchTool(search_config()).run({"queries": ["httpx retry"]}, ctx)

    assert result.ok is True
    assert "文档" in result.content
    assert handler.calls == 3
    assert sleeps.delays == [0.5, 1.0]


async def test_web_search_exhausted_retries_returns_failed_result(
    monkeypatch: pytest.MonkeyPatch,
    ctx: ToolContext,
    sleeps: SleepRecorder,
) -> None:
    """搜索后端一直 429 → 重试 3 次后 ``ToolResult(ok=False)``，文案带状态码。"""
    handler = FakeTransport([json_response({"detail": "slow down"}, 429)] * ATTEMPTS)
    install_transport(monkeypatch, handler)

    result = await WebSearchTool(search_config()).run({"queries": ["x"]}, ctx)

    assert result.ok is False
    assert handler.calls == ATTEMPTS
    assert sleeps.delays == [0.5, 1.0]
    assert result.error is not None
    assert "429" in result.error
    assert "已重试 2 次" in result.error


async def test_web_search_404_fails_after_exactly_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
    ctx: ToolContext,
    sleeps: SleepRecorder,
) -> None:
    """搜索后端 404 → **只尝试 1 次**（客户端错误不重试）。"""
    handler = FakeTransport([json_response({"detail": "not found"}, 404)] * ATTEMPTS)
    install_transport(monkeypatch, handler)

    result = await WebSearchTool(search_config()).run({"queries": ["x"]}, ctx)

    assert result.ok is False
    assert handler.calls == 1
    assert sleeps.delays == []
    assert result.error is not None
    assert "404" in result.error


async def test_web_search_connect_error_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    ctx: ToolContext,
    sleeps: SleepRecorder,
) -> None:
    """搜索后端连接失败 1 次后成功 → 成功（整批查询重发一次）。"""
    handler = FakeTransport([httpx.ConnectError("boom"), json_response(SEARCH_PAYLOAD)])
    install_transport(monkeypatch, handler)

    result = await WebSearchTool(search_config()).run({"queries": ["x"]}, ctx)

    assert result.ok is True
    assert handler.calls == 2
    assert sleeps.delays == [0.5]


async def test_web_search_unconfigured_backend_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    ctx: ToolContext,
    sleeps: SleepRecorder,
) -> None:
    """未配置后端仍返回明确错误（U28），且不发起任何请求。"""
    handler = FakeTransport([json_response(SEARCH_PAYLOAD)])
    install_transport(monkeypatch, handler)

    result = await WebSearchTool().run({"queries": ["x"]}, ctx)

    assert result.ok is False
    assert handler.calls == 0
    assert sleeps.delays == []
    assert result.error is not None
    assert "未配置搜索后端" in result.error
