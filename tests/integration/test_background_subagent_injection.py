"""集成：后台子 Agent 的结论真的会回到主上下文（§12.2）。

此前 ``SubagentRunner.drain_background()`` 只有定义、全仓没有调用点——后台子 Agent
跑完后结论被丢弃，主 Agent 永远看不到。本文件固定修复后的行为：

1. :func:`test_background_subagent_conclusion_reaches_main_context`
   —— 真实引擎 + 真实 ``SubagentRunner``：后台子 Agent 完成后，**下一轮**主对话的
   模型输入里必须出现 ``[后台子 Agent 完成]`` 结论块，且它以 ``role="user"`` 落入 history；
2. :func:`test_drained_block_is_not_injected_twice`
   —— ``drain_background`` 的「取走即清空」语义：同一结论只注入一次；
3. :func:`test_no_runner_means_no_drain`
   —— 子 Agent 内部构造的 ``Query`` 不传 runner，注入逻辑必须是无害空操作（深度隔离）；
4. :func:`test_runner_is_wired_only_when_agents_enabled`
   —— ``agents.enabled=false`` 时引擎不装配 runner，也不注册 ``Task`` 工具。

装配方式与 ``tests/integration/test_loop_detection_engine.py`` 一致：真实
``config → engine → tools → security`` 全链路，只把模型 provider 换成
``tests.fakes.provider.FakeProvider``（不触网）。临时目录用 ``workdir`` fixture，不用
``tmp_path``（受管环境下不可靠）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from my_agent.config.schema import Config, EndpointConfig, LLMConfig
from my_agent.models.protocols import PermissionRequest, UserAnswer
from tests.fakes.provider import FakeProvider

BACKGROUND_MARKER = "[后台子 Agent 完成]"
CONCLUSION = "子Agent结论：仓库里共有 3 处 TODO"


def _content_of(message: Any) -> str:
    """取出消息内容（Provider 侧记录的是净化后的 dict，也可能是 Message 对象）。

    Args:
        message: ``dict`` 或 :class:`~my_agent.models.messages.Message`。

    Returns:
        消息正文；取不到时返回空串。
    """
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


class _ScriptedConfirmer:
    """记录询问并固定回答的确认器（本文件用 bypassPermissions，正常不会被调用）。"""

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


def _config(project: Path, mode: str = "bypassPermissions", *, agents: bool = True) -> Config:
    """构造指向本地假端点、且关闭回合记忆写入的配置。

    Args:
        project: 项目根（未直接使用，保持与既有集成测试同签名）。
        mode: 安全模式。
        agents: 是否启用多 Agent。

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
    # 关掉回合结束的记忆抽取：否则它会额外调一次模型，污染「模型调用次数」断言。
    config.memory.write_per_turn = False
    config.agents.enabled = agents
    return config


async def _build_engine(
    workdir: Path,
    provider: FakeProvider,
    *,
    mode: str = "bypassPermissions",
    agents: bool = True,
) -> Any:
    """按真实装配路径构造并启动引擎（仅把模型 provider 换成假实现）。

    Args:
        workdir: 隔离的工作目录。
        provider: 假模型提供方。
        mode: 安全模式。
        agents: 是否启用多 Agent。

    Returns:
        已 ``start()`` 的 :class:`~my_agent.engine.query_engine.QueryEngine`。
    """
    from my_agent.engine.query_engine import QueryEngine
    from my_agent.llm.client import LLMClient, UsageTracker

    project = workdir / "proj"
    project.mkdir(parents=True, exist_ok=True)
    (project / ".git").mkdir(exist_ok=True)

    config = _config(project, mode, agents=agents)
    engine = QueryEngine(config, _ScriptedConfirmer(), cwd=project)
    await engine.start()
    # start() 里已构造真实 SDK 客户端；这里整体替换为假 provider（不触网）
    engine.llm = LLMClient(provider, config, UsageTracker())
    # SubagentRunner 在 start() 期间捕获了 llm 引用，替换 provider 后必须同步重指，
    # 否则子 Agent 会走真实 SDK（生产环境不会替换，仅测试需要）。
    if engine.subagent_runner is not None:
        engine.subagent_runner.llm = engine.llm
    return engine


async def _wait_for_background(runner: Any, *, timeout_s: float = 5.0) -> None:
    """等待后台子 Agent 全部完成。

    Args:
        runner: :class:`~my_agent.agents.subagent.SubagentRunner`。
        timeout_s: 最长等待秒数。
    """
    deadline = asyncio.get_running_loop().time() + timeout_s
    while runner.background_ids():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"后台子 Agent 未在 {timeout_s}s 内完成")
        await asyncio.sleep(0.01)


async def test_background_subagent_conclusion_reaches_main_context(workdir: Path) -> None:
    """后台子 Agent 的结论必须注入主上下文，并出现在下一轮的模型输入里（§12.2）。"""
    # 第 1 条脚本：子 Agent 自己的模型调用；第 2 条：主对话下一轮。
    provider = FakeProvider(
        [
            FakeProvider.text(CONCLUSION),
            FakeProvider.text("收到后台结论，继续。"),
        ]
    )
    engine = await _build_engine(workdir, provider)
    try:
        runner = engine.subagent_runner
        assert runner is not None, "agents.enabled=true 时引擎必须装配 SubagentRunner"

        receipt = await runner.spawn(
            description="数 TODO",
            prompt="数一下仓库里有几处 TODO，只回结论",
            agent_type="explore",
            run_in_background=True,
            parent=engine.session,
        )
        assert receipt.ok is True
        assert BACKGROUND_MARKER not in receipt.text  # 启动回执不是结论
        await _wait_for_background(runner)

        await engine.handle_input("继续")

        injected = [
            m for m in engine.session.history if m.role == "user" and BACKGROUND_MARKER in m.content
        ]
        assert injected, "后台结论必须以 role='user' 注入 history（§12.2）"
        assert CONCLUSION in injected[0].content

        # 真正重要的断言：结论进入了**发给模型的** messages，而不只是躺在 history 里
        last_call = provider.last_call
        assert last_call is not None
        rendered = "\n".join(_content_of(m) for m in last_call.messages)
        assert BACKGROUND_MARKER in rendered, "结论必须出现在模型输入中"
        assert CONCLUSION in rendered
    finally:
        await engine.shutdown("test")


async def test_drained_block_is_not_injected_twice(workdir: Path) -> None:
    """``drain_background`` 取走即清空：同一结论只注入一次。"""
    provider = FakeProvider(
        [
            FakeProvider.text(CONCLUSION),
            FakeProvider.text("第一轮"),
            FakeProvider.text("第二轮"),
        ]
    )
    engine = await _build_engine(workdir, provider)
    try:
        runner = engine.subagent_runner
        assert runner is not None
        await runner.spawn(
            description="数 TODO",
            prompt="数 TODO",
            agent_type="explore",
            run_in_background=True,
            parent=engine.session,
        )
        await _wait_for_background(runner)

        await engine.handle_input("第一轮")
        await engine.handle_input("第二轮")

        injected = [m for m in engine.session.history if BACKGROUND_MARKER in m.content]
        assert len(injected) == 1, f"结论应只注入一次，实际 {len(injected)} 次"
    finally:
        await engine.shutdown("test")


async def test_no_runner_means_no_drain(workdir: Path) -> None:
    """不传 runner 的 Query（子 Agent 内部）注入逻辑是无害空操作。"""
    provider = FakeProvider([FakeProvider.text("ok")])
    engine = await _build_engine(workdir, provider)
    try:
        query = engine._new_query()
        query.subagent_runner = None
        before = len(engine.session.history)
        query._drain_background(1)
        assert len(engine.session.history) == before, "无 runner 时不得改动 history"
    finally:
        await engine.shutdown("test")


async def test_runner_is_wired_only_when_agents_enabled(workdir: Path) -> None:
    """``agents.enabled=false`` 时不装配 runner，也不注册 Task 工具。"""
    provider = FakeProvider([FakeProvider.text("ok")])
    engine = await _build_engine(workdir, provider, agents=False)
    try:
        assert engine.subagent_runner is None
        assert engine.tools is not None
        assert "Task" not in engine.tools.names(), "关闭多 Agent 时不得注册 Task"
    finally:
        await engine.shutdown("test")
