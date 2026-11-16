"""集成：任务分发审判在真实引擎里生效（U35）。

验证四件事（都用真实 ``Query`` + 真实工具/权限流水线，仅模型换成 ``FakeProvider``）：

1. :func:`test_reasonable_lets_batch_proceed` —— 判「合理」→ 原批次照常执行；
2. :func:`test_too_fine_rejects_batch_and_model_redispatches` —— 判「过细」→
   该 Task 调用被打回、模型看到补丁后**重新分发**（同一回合不再重复审判）；
3. :func:`test_judge_down_is_fail_open_and_notifies_user` —— 审判模型故障 →
   指数退避重试耗尽 → **放行**（Task 仍然执行）且**显式告警用户**；
4. :func:`test_judge_absent_when_agents_disabled` —— 关闭多 Agent 时不构造审判器。

调用顺序很关键：``Query._step_act`` 在**执行 Task 之前**调用审判，所以脚本里
第 2 条一定是审判的输出（而非子 Agent 的结论）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from my_agent.config.schema import Config, EndpointConfig, LLMConfig
from my_agent.errors import ConfigError
from my_agent.models.protocols import PermissionRequest, UserAnswer
from tests.fakes.provider import FakeProvider

FINAL_TEXT = "全部子任务已完成。"


class _ScriptedConfirmer:
    """记录询问并固定回答的确认器（用 bypassPermissions，正常不会被调用）。"""

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


class _RecordingConsole:
    """同时实现 DisplayPort 与 ConsolePort 的记录器（供引擎注入）。"""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def stream_text(self, delta: str) -> None:
        """忽略流式文本。"""

    def info(self, msg: str) -> None:
        """记录 info。"""
        self.infos.append(msg)

    def warn(self, msg: str) -> None:
        """记录 warning。"""
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        """忽略错误。"""

    def tool_start(self, name: str, summary: str) -> None:
        """忽略工具开始事件。"""

    def tool_result(self, name: str, ok: bool, summary: str) -> None:
        """忽略工具结束事件。"""

    def render_context_report(self, report: Any) -> None:
        """忽略。"""

    def render_cost(self, text: str) -> None:
        """忽略。"""

    def render_permission_card(self, request: PermissionRequest) -> None:
        """忽略。"""


def _config(project: Path, *, agents: bool = True) -> Config:
    """构造指向本地假端点、关闭回合记忆写入的配置。

    Args:
        project: 项目根（保持与既有集成测试同签名）。
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
    config.safety.mode = "bypassPermissions"  # type: ignore[assignment]
    config.memory.write_per_turn = False
    config.agents.enabled = agents
    return config


async def _build_engine(
    workdir: Path, provider: FakeProvider, *, agents: bool = True
) -> Any:
    """按真实装配路径构造并启动引擎。

    Args:
        workdir: 隔离工作目录。
        provider: 假模型提供方。
        agents: 是否启用多 Agent。

    Returns:
        ``(engine, console)``。
    """
    from my_agent.engine.query_engine import QueryEngine
    from my_agent.llm.client import LLMClient, UsageTracker

    project = workdir / "proj"
    project.mkdir(parents=True, exist_ok=True)
    (project / ".git").mkdir(exist_ok=True)

    console = _RecordingConsole()
    config = _config(project, agents=agents)
    engine = QueryEngine(config, _ScriptedConfirmer(), console, cwd=project)
    await engine.start()
    engine.llm = LLMClient(provider, config, UsageTracker())
    # 子 Agent 运行时与审判器都在 start() 期间捕获了 llm，替换 provider 后需同步重指
    if engine.subagent_runner is not None:
        engine.subagent_runner.llm = engine.llm
    if engine.dispatch_judge is not None:
        engine.dispatch_judge.llm = engine.llm

        async def _no_sleep(_delay: float) -> None:
            return None

        engine.dispatch_judge._sleep = _no_sleep  # 重试不真实等待
    return engine, console


def _judgment(**overrides: Any) -> str:
    """生成审判 JSON 文本。

    Args:
        **overrides: 覆盖字段。

    Returns:
        JSON 字符串。
    """
    payload: dict[str, Any] = {
        "verdict": "reasonable",
        "confidence": "high",
        "reason": "覆盖完整",
        "missing_tasks": [],
        "merged_into": None,
        "patch": [],
        "impact": [],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def _task_call(description: str, prompt: str, *, call_id: str = "call_task") -> Any:
    """构造一个 Task 工具调用脚本条目。

    Args:
        description: 子任务短标题。
        prompt: 子任务说明。
        call_id: 模型返回的 call id。

    Returns:
        :class:`tests.fakes.provider.ScriptedResult`。
    """
    return FakeProvider.tool_call(
        "Task",
        {"description": description, "prompt": prompt, "agent_type": "explore"},
        call_id=call_id,
    )


async def test_reasonable_lets_batch_proceed(workdir: Path) -> None:
    """判「合理」→ Task 照常执行，子 Agent 结论回填。"""
    provider = FakeProvider(
        [
            _task_call("数 TODO", "数一下仓库里的 TODO"),
            FakeProvider.text(_judgment()),  # 审判：合理
            FakeProvider.text("子Agent结论：有 3 处 TODO"),
            FakeProvider.text(FINAL_TEXT),
        ]
    )
    engine, console = await _build_engine(workdir, provider)
    try:
        result = await engine.handle_input("数一下仓库里的 TODO")
        assert result.kind == "assistant_text"
        joined = "\n".join(message.content for message in engine.session.history)
        assert "子Agent结论：有 3 处 TODO" in joined
        assert "过细" not in joined
        assert console.warnings == []
    finally:
        await engine.shutdown("test")


async def test_too_fine_rejects_batch_and_model_redispatches(workdir: Path) -> None:
    """判「过细」→ 该 Task 被打回，模型据补丁重新分发（不再重复审判）。"""
    provider = FakeProvider(
        [
            _task_call("数TODO并写文档", "一次性把数 TODO 和写文档都做了"),
            FakeProvider.text(
                _judgment(
                    verdict="too_fine",
                    reason="一个任务里混了两件事",
                    merged_into=2,
                    impact=["README.md"],
                    patch=[
                        {
                            "description": "数 TODO",
                            "prompt": "只数仓库里的 TODO 数量",
                            "agent_type": "explore",
                        }
                    ],
                )
            ),
            # 模型看到补丁后重新分发
            _task_call("数 TODO", "只数仓库里的 TODO 数量", call_id="call_task_2"),
            FakeProvider.text("子Agent结论：有 3 处 TODO"),
            FakeProvider.text(FINAL_TEXT),
        ]
    )
    engine, console = await _build_engine(workdir, provider)
    try:
        result = await engine.handle_input("数 TODO 并写文档")
        assert result.kind == "assistant_text"
        joined = "\n".join(message.content for message in engine.session.history)
        # 打回反馈必须进入模型可见的 tool 结果
        assert "过细" in joined
        assert "只数仓库里的 TODO 数量" in joined
        # 重新分发后的子 Agent 确实跑起来了
        assert "子Agent结论：有 3 处 TODO" in joined
        assert any("过细" in warning for warning in console.warnings)
    finally:
        await engine.shutdown("test")


async def test_judge_down_is_fail_open_and_notifies_user(workdir: Path) -> None:
    """审判模型故障 → 重试耗尽 → 放行（Task 仍执行）且显式告警用户。"""
    provider = FakeProvider(
        [
            _task_call("数 TODO", "数一下 TODO"),
            # 审判 4 次尝试全部失败（首试 + 3 次重试）
            FakeProvider.raise_(ConfigError("judge endpoint down")),
            FakeProvider.raise_(ConfigError("judge endpoint down")),
            FakeProvider.raise_(ConfigError("judge endpoint down")),
            FakeProvider.raise_(ConfigError("judge endpoint down")),
            FakeProvider.text("子Agent结论：有 3 处 TODO"),
            FakeProvider.text(FINAL_TEXT),
        ]
    )
    engine, console = await _build_engine(workdir, provider)
    try:
        result = await engine.handle_input("数一下 TODO")
        assert result.kind == "assistant_text"
        joined = "\n".join(message.content for message in engine.session.history)
        # fail-open：审判挂了但分发照常进行
        assert "子Agent结论：有 3 处 TODO" in joined
        # 但必须告警用户，不静默失效
        assert any("审判模型" in warning for warning in console.warnings), console.warnings
        assert engine.dispatch_judge.judge_degraded is True
    finally:
        await engine.shutdown("test")


async def test_judge_absent_when_agents_disabled(workdir: Path) -> None:
    """关闭多 Agent 时不构造审判器（不产生额外模型调用）。"""
    provider = FakeProvider([FakeProvider.text("ok")])
    engine, _console = await _build_engine(workdir, provider, agents=False)
    try:
        assert engine.dispatch_judge is None
        assert "Task" not in (engine.tools.names() if engine.tools else [])
    finally:
        await engine.shutdown("test")
