"""集成：真实 ``Query`` + 真实工具/权限流水线，死循环检测在引擎层真的生效（U33）。

覆盖用户口径的两个条件：

1. :func:`test_identical_repeats_terminate_turn` —— 同一工具同一参数连调 3 次且结果相同
   → 判定死循环、本回合终止、会话历史里回填一条说明原因的 tool 消息、模型**不再**被第 4 次调用；
2. :func:`test_changing_results_do_not_terminate_turn` —— 同一工具同一参数连调 3 次但结果在变
   → **不**判死循环，模型继续被调用并给出终答（条件 2「内容相似度」的守门测试）。

装配方式与 ``tests/integration/test_engine_end_to_end.py`` 一致：真实
``config → engine → tools → security`` 全链路，只把模型 provider 换成
``tests.fakes.provider.FakeProvider``（不触网）。

临时目录使用 ``workdir`` fixture（见 ``tests/conftest.py``），不用 ``tmp_path``：
受管环境下系统临时目录与 pytest basetemp 的复用都不可靠。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from my_agent.config.schema import Config, EndpointConfig, LLMConfig
from my_agent.llm.provider import CompletionResult, StreamChunk
from my_agent.models.protocols import PermissionRequest, UserAnswer
from tests.fakes.provider import FakeProvider, ScriptedResult


class _ScriptedConfirmer:
    """记录询问并固定回答的确认器（本文件全部用 bypassPermissions，正常不会被调用）。"""

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


class _FileRewritingProvider(FakeProvider):
    """每次模型调用前改写目标文件的假 provider（模拟「结果在变」的工具）。

    写文件的时机是**模型调用前**：``模型 → 工具`` 交替执行，因此第 N 次 Read 看到的是
    第 N 版内容——参数完全不变，结果逐次不同。

    Args:
        script: 同 :class:`FakeProvider` 的脚本。
        target: 每次模型调用前要改写的文件。
        revisions: 依次写入的内容（用完最后一版后保持不再变化）。
    """

    def __init__(
        self,
        script: Sequence[ScriptedResult | BaseException | str],
        *,
        target: Path,
        revisions: Sequence[str],
    ) -> None:
        super().__init__(script)
        self._target = target
        self._revisions = tuple(revisions)
        self._revision = 0

    def _advance(self) -> None:
        """把文件改写成下一版内容（越界后保持最后一版）。"""
        if not self._revisions:
            return
        text = self._revisions[min(self._revision, len(self._revisions) - 1)]
        self._revision += 1
        self._target.write_text(text, encoding="utf-8")

    async def complete(self, **kwargs: Any) -> CompletionResult:
        """改写文件后走父类的非流式脚本消费。

        Args:
            **kwargs: 与 :meth:`FakeProvider.complete` 相同的实参。

        Returns:
            :class:`~my_agent.llm.provider.CompletionResult`。
        """
        self._advance()
        return await super().complete(**kwargs)

    async def stream(self, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """改写文件后走父类的流式脚本下发。

        Args:
            **kwargs: 与 :meth:`FakeProvider.stream` 相同的实参。

        Yields:
            :class:`~my_agent.llm.provider.StreamChunk` 帧。
        """
        self._advance()
        async for chunk in super().stream(**kwargs):
            yield chunk


def _config(project: Path, mode: str = "bypassPermissions") -> Config:
    """构造一份指向本地假端点、且关闭回合记忆写入的配置。

    Args:
        project: 项目根（未直接使用，保持与既有集成测试同签名）。
        mode: 安全模式（死循环测试用 ``bypassPermissions``，不触发权限询问）。

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
    return config


async def _build_engine(workdir: Path, provider: FakeProvider, *, mode: str = "bypassPermissions") -> Any:
    """按真实装配路径构造并启动引擎（仅把模型 provider 换成假实现）。

    Args:
        workdir: 隔离的工作目录。
        provider: 假模型提供方。
        mode: 安全模式。

    Returns:
        已 ``start()`` 的 :class:`~my_agent.engine.query_engine.QueryEngine`。
    """
    from my_agent.engine.query_engine import QueryEngine
    from my_agent.llm.client import LLMClient, UsageTracker

    project = workdir / "proj"
    project.mkdir(parents=True, exist_ok=True)
    (project / ".git").mkdir(exist_ok=True)

    config = _config(project, mode)
    engine = QueryEngine(config, _ScriptedConfirmer(), cwd=project)
    await engine.start()
    # start() 里已构造真实 SDK 客户端；这里整体替换为假 provider（不触网）
    engine.llm = LLMClient(provider, config, UsageTracker())
    return engine


def _tool_messages(engine: Any) -> list[Any]:
    """取会话历史里的全部 ``role="tool"`` 消息。

    Args:
        engine: 已启动的引擎。

    Returns:
        工具消息列表（按时间顺序）。
    """
    return [msg for msg in engine.session.history if msg.role == "tool"]


async def test_identical_repeats_terminate_turn(workdir: Path, isolated_env: dict[str, str]) -> None:
    """同工具同参数 3 次 + 结果相同 → 判死循环、终止回合、回填原因、不再调模型。"""
    target = workdir / "proj" / "loop.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("stable content\n", encoding="utf-8")

    provider = FakeProvider(
        [
            FakeProvider.tool_call("Read", {"file_path": "loop.txt"}, call_id="call_1"),
            FakeProvider.tool_call("Read", {"file_path": "loop.txt"}, call_id="call_2"),
            FakeProvider.tool_call("Read", {"file_path": "loop.txt"}, call_id="call_3"),
            FakeProvider.text("不应被调用：死循环已在第 3 次调用后终止本回合。"),
        ]
    )
    engine = await _build_engine(workdir, provider)
    try:
        result = await engine.handle_input("反复读 loop.txt，直到我说停")
        history = list(engine.session.history)
        turns = list(engine.session.turns)
    finally:
        await engine.shutdown("test")

    # (a) 死循环被检测到
    assert result.kind == "assistant_text", result.text
    assert "死循环" in result.text
    assert "Read" in result.text
    assert "3 次" in result.text

    # (b) 本回合已结束并被正常记账（kind 仍是 assistant_text，不新增 kind）
    assert turns, "回合必须被 _finish_turn 记录"
    assert turns[-1].assistant_final == result.text

    # (c) 会话历史里有说明死循环的 tool 消息（模型能看到原因）
    loop_msgs = [msg for msg in history if msg.role == "tool" and "死循环" in msg.content]
    assert loop_msgs, "判定后必须回填一条说明原因的 tool 消息"
    assert loop_msgs[0].name == "Read"
    tool_msgs = [msg for msg in history if msg.role == "tool"]
    assert len(tool_msgs) == 4, "3 条真实结果 + 1 条死循环说明"
    call_ids = {
        call.id for msg in history if msg.role == "assistant" for call in msg.tool_calls
    }
    assert all(msg.tool_call_id in call_ids for msg in tool_msgs), "§4.1：tool 消息必须挂在 tool_calls 上"

    # (d) 模型没有被第 4 次调用（脚本第 4 条原封未动）
    assert provider.call_count == 3, "检测到死循环后不得再请求模型"
    assert provider.remaining == 1


async def test_changing_results_do_not_terminate_turn(
    workdir: Path, isolated_env: dict[str, str]
) -> None:
    """**条件 2 的守门测试**：同工具同参数 3 次但结果在变 → 不判死循环，模型继续工作。

    去掉结果相似度这一步，本测试立刻失败：第 3 次会被判死循环而拿不到终答，
    ``provider.call_count`` 会停在 3 而不是 4。
    """
    target = workdir / "proj" / "run.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    provider = _FileRewritingProvider(
        [
            FakeProvider.tool_call("Read", {"file_path": "run.txt"}, call_id="call_1"),
            FakeProvider.tool_call("Read", {"file_path": "run.txt"}, call_id="call_2"),
            FakeProvider.tool_call("Read", {"file_path": "run.txt"}, call_id="call_3"),
            FakeProvider.text("三次读取内容都在变，任务已完成。"),
        ],
        target=target,
        revisions=(
            "alpha\nbeta\ngamma\n",
            "lorem ipsum dolor sit amet " * 30 + "\n",
            "zeta\n",
        ),
    )
    engine = await _build_engine(workdir, provider)
    try:
        result = await engine.handle_input("反复读 run.txt 看它变没变")
        tool_msgs = _tool_messages(engine)
    finally:
        await engine.shutdown("test")

    assert result.kind == "assistant_text", result.text
    assert "已完成" in result.text, "结果在变 → 必须继续循环并给出终答"
    assert provider.call_count == 4, "第 4 次模型调用必须发生（未被误判死循环拦下）"
    assert provider.remaining == 0
    assert len(tool_msgs) == 3
    assert not any("死循环" in msg.content for msg in tool_msgs)
    contents = [msg.content for msg in tool_msgs]
    assert len(set(contents)) == 3, "三次结果确实各不相同（测试前提）"


@pytest.mark.parametrize("tool", ["Read", "Glob"])
async def test_loop_message_is_never_a_new_engine_result_kind(
    workdir: Path, isolated_env: dict[str, str], tool: str
) -> None:
    """终止路径只产出既有 ``kind``（``assistant_text``），不新增 ``EngineResult.kind``。"""
    target = workdir / "proj" / "same.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("same\n", encoding="utf-8")

    args = {"file_path": "same.txt"} if tool == "Read" else {"pattern": "*.txt"}
    provider = FakeProvider(
        [FakeProvider.tool_call(tool, args, call_id=f"c{index}") for index in range(3)]
    )
    engine = await _build_engine(workdir, provider)
    try:
        result = await engine.handle_input("重复调用同一个工具")
    finally:
        await engine.shutdown("test")

    assert result.kind == "assistant_text"
    assert "死循环" in result.text
    assert provider.call_count == 3
