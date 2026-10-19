"""端到端集成冒烟：真实引擎 + 真实工具/权限流水线 + 假模型。

这是**契约对齐测试**：按真实装配路径构造 ``QueryEngine``，验证
``config → models → llm → tools → security → context → engine`` 全链路能跑通，
而不是各模块自测。模型侧用 ``tests/fakes/provider.py`` 的 ``FakeProvider``
按脚本顺序消费（第 1 条 = 第一轮，第 2 条 = 第二轮 …）。

临时目录使用 ``workdir`` fixture（见 ``tests/conftest.py``），不用 ``tmp_path``：
受管环境下系统临时目录与 pytest basetemp 的复用都不可靠。

覆盖：
1. 普通输入触发完整 ReAct 循环（A-2：1 次工具调用 + 1 次终答，工具结果回填）；
2. ``/help`` 不触发任何模型调用（A-1）；
3. ``plan`` 模式下写类工具被 deny，理由回填（B-10 / E2E-4）；
4. ``acceptEdits`` 下 cwd 内 ``Write`` 走快速通道、不询问用户（B-7 / E2E-2）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from my_agent.config.schema import Config, EndpointConfig, LLMConfig
from my_agent.models.protocols import PermissionRequest, UserAnswer
from tests.fakes.provider import FakeProvider


class _ScriptedConfirmer:
    """记录询问并固定回答的确认器。"""

    def __init__(self, choice: str = "once") -> None:
        self.choice = choice
        self.requests: list[PermissionRequest] = []

    async def ask(self, request: PermissionRequest) -> UserAnswer:
        """记录并返回预设回答。

        Args:
            request: 询问卡片。

        Returns:
            预设回答。
        """
        self.requests.append(request)
        return UserAnswer(choice=self.choice)  # type: ignore[arg-type]


def _config(project: Path, mode: str = "default") -> Config:
    """构造一份指向本地假端点的配置。

    Args:
        project: 项目根。
        mode: 安全模式。

    Returns:
        :class:`~my_agent.config.schema.Config`。
    """
    endpoint = EndpointConfig(
        name="main",
        base_url="http://127.0.0.1:9/v1",
        api_key_env="MYAGENT_TEST_API_KEY",
        model="test-model",
    )
    config = Config(llm=LLMConfig(endpoint="main", endpoints={"main": endpoint}))
    config.safety.mode = mode  # type: ignore[assignment]
    return config


async def _build_engine(
    workdir: Path,
    provider: FakeProvider,
    *,
    mode: str = "default",
    confirmer: Any = None,
) -> Any:
    """按真实装配路径构造并启动引擎（仅把模型 provider 换成假实现）。

    Args:
        workdir: 隔离的工作目录。
        provider: 假模型提供方。
        mode: 安全模式。
        confirmer: 权限确认器。

    Returns:
        已 ``start()`` 的 :class:`~my_agent.engine.query_engine.QueryEngine`。
    """
    from my_agent.engine.query_engine import QueryEngine
    from my_agent.llm.client import LLMClient, UsageTracker

    project = workdir / "proj"
    project.mkdir(parents=True, exist_ok=True)
    (project / ".git").mkdir(exist_ok=True)

    config = _config(project, mode)
    engine = QueryEngine(config, confirmer or _ScriptedConfirmer(), cwd=project)
    await engine.start()
    # start() 里已构造真实 SDK 客户端；这里整体替换为假 provider（不触网）
    engine.llm = LLMClient(provider, config, UsageTracker())
    return engine


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """注入测试用密钥环境变量。"""
    monkeypatch.setenv("MYAGENT_TEST_API_KEY", "test-key-not-a-secret")


async def test_plain_input_runs_full_react_loop(workdir: Path, api_key: None) -> None:
    """普通输入 → 1 次工具调用 + 1 次终答，工具结果回填进 history（A-2）。"""
    target = workdir / "proj" / "hello.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello world\n", encoding="utf-8")

    provider = FakeProvider(
        [
            FakeProvider.tool_call("Read", {"file_path": "hello.txt"}),
            FakeProvider.text("已完成：文件内容已读取。"),
        ]
    )
    engine = await _build_engine(workdir, provider, mode="bypassPermissions")
    try:
        result = await engine.handle_input("读一下 hello.txt")
        tool_msgs = [m for m in engine.session.history if m.role == "tool"]
    finally:
        await engine.shutdown("test")

    assert result.kind == "assistant_text", result.text
    assert "已完成" in result.text
    assert tool_msgs, "必须有一条 role='tool' 的结果回填（B2）"
    assert "hello world" in tool_msgs[0].content


async def test_slash_help_makes_no_model_call(workdir: Path, api_key: None) -> None:
    """``/help`` 不触发任何模型调用（A-1）。"""
    provider = FakeProvider([FakeProvider.text("不应被调用")])
    engine = await _build_engine(workdir, provider)
    try:
        result = await engine.handle_input("/help")
    finally:
        await engine.shutdown("test")

    assert result.kind == "slash_output"
    assert "/mode" in result.text
    assert provider.call_count == 0, "``/help`` 不得调用模型"


async def test_plan_mode_denies_write(workdir: Path, api_key: None) -> None:
    """``plan`` 模式下写类工具被 deny，理由回填（B-10 / E2E-4）。"""
    provider = FakeProvider(
        [
            FakeProvider.tool_call("Write", {"file_path": "new.txt", "content": "x"}),
            FakeProvider.text("明白，我只给出计划，不写文件。"),
        ]
    )
    engine = await _build_engine(workdir, provider, mode="plan")
    try:
        result = await engine.handle_input("创建 new.txt")
        denied = [
            m for m in engine.session.history if m.role == "tool" and "权限被拒绝" in m.content
        ]
    finally:
        await engine.shutdown("test")

    assert result.kind == "assistant_text"
    assert denied, "被 deny 的工具必须回填带理由的结果（B3）"
    assert not (workdir / "proj" / "new.txt").exists(), "plan 模式绝不允许落盘"


async def test_accept_edits_fast_path_allows_in_cwd_write(workdir: Path, api_key: None) -> None:
    """``acceptEdits`` 下 cwd 内 ``Write`` 走快速通道且不询问用户（B-7 / E2E-2）。"""
    provider = FakeProvider(
        [
            FakeProvider.tool_call("Write", {"file_path": "inside.txt", "content": "ok\n"}),
            FakeProvider.text("文件已写入。"),
        ]
    )
    confirmer = _ScriptedConfirmer()
    engine = await _build_engine(workdir, provider, mode="acceptEdits", confirmer=confirmer)
    try:
        result = await engine.handle_input("写一个 inside.txt")
    finally:
        await engine.shutdown("test")

    assert result.kind == "assistant_text", result.text
    written = workdir / "proj" / "inside.txt"
    assert written.is_file(), "acceptEdits 快速通道应直接放行 cwd 内写入"
    assert written.read_text(encoding="utf-8") == "ok\n"
    assert confirmer.requests == [], "cwd 内编辑不应触发权限询问（快速通道不调模型、不问用户）"
