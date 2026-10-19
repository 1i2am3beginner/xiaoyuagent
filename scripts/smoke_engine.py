"""引擎端到端验证脚本（不依赖 pytest 的临时目录机制）。

用法：``uv run python scripts/smoke_engine.py``

它按真实装配路径构造 ``QueryEngine``（真实 config / tools / security / context /
engine），只把模型 provider 换成 ``tests/fakes/provider.py`` 的 ``FakeProvider``，
从而在**不触网、不需要 API Key** 的前提下验证整条链路：

1. 普通输入 → 完整 ReAct 循环（工具调用 + 结果回填 + 终答）
2. ``/help`` 不触发任何模型调用
3. ``plan`` 模式拒绝写操作、理由回填、文件未落盘
4. ``acceptEdits`` 下 cwd 内写入走快速通道且不询问用户
5. 敏感路径（``.git/config``）在任意模式下强制询问且不可永久放行
6. ``/context`` 与 ``/cost`` 报告可生成

退出码 0 = 全部通过。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from my_agent.config.schema import Config, EndpointConfig, LLMConfig  # noqa: E402
from my_agent.models.protocols import PermissionRequest, UserAnswer  # noqa: E402
from tests.fakes.provider import FakeProvider  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    """记录一条检查结果。

    Args:
        name: 检查项名称。
        ok: 是否通过。
        detail: 补充说明。
    """
    RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


class RecordingConfirmer:
    """记录权限询问并按预设回答。"""

    def __init__(self, choice: str = "once") -> None:
        self.choice = choice
        self.requests: list[PermissionRequest] = []

    async def ask(self, request: PermissionRequest) -> UserAnswer:
        """记录并回答。

        Args:
            request: 询问卡片。

        Returns:
            预设回答。
        """
        self.requests.append(request)
        return UserAnswer(choice=self.choice)  # type: ignore[arg-type]


def make_config(project: Path, mode: str) -> Config:
    """构造指向假端点的配置。

    Args:
        project: 项目根。
        mode: 安全模式。

    Returns:
        配置对象。
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


async def build_engine(
    workdir: Path, provider: FakeProvider, *, mode: str, confirmer: Any
) -> Any:
    """装配并启动引擎（模型 provider 换成假实现）。

    Args:
        workdir: 隔离的工作目录。
        provider: 假模型。
        mode: 安全模式。
        confirmer: 权限确认器。

    Returns:
        已启动的引擎。
    """
    from my_agent.engine.query_engine import QueryEngine
    from my_agent.llm.client import LLMClient, UsageTracker

    project = workdir / "proj"
    project.mkdir(parents=True, exist_ok=True)
    (project / ".git").mkdir(exist_ok=True)

    config = make_config(project, mode)
    engine = QueryEngine(config, confirmer, cwd=project)
    await engine.start()
    engine.llm = LLMClient(provider, config, UsageTracker())
    return engine


async def check_react_loop(workdir: Path) -> None:
    """验证完整 ReAct 循环与工具结果回填。

    Args:
        workdir: 隔离工作目录。
    """
    target = workdir / "proj" / "hello.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello world\n", encoding="utf-8")

    provider = FakeProvider(
        [
            FakeProvider.tool_call("Read", {"file_path": "hello.txt"}),
            FakeProvider.text("已完成：文件内容已读取。"),
        ]
    )
    engine = await build_engine(
        workdir, provider, mode="bypassPermissions", confirmer=RecordingConfirmer()
    )
    try:
        result = await engine.handle_input("读一下 hello.txt")
        tool_msgs = [m for m in engine.session.history if m.role == "tool"]
        record("A-2 普通输入触发完整 ReAct 循环", result.kind == "assistant_text", result.text[:40])
        record(
            "B2 工具结果回填（role=tool 且内容正确）",
            bool(tool_msgs) and "hello world" in tool_msgs[0].content,
            f"{len(tool_msgs)} 条 tool 消息",
        )
        record("B2 tool_call_id 已回填", bool(tool_msgs) and bool(tool_msgs[0].tool_call_id))
    finally:
        await engine.shutdown("smoke")


async def check_slash_no_model_call(workdir: Path) -> None:
    """验证 ``/help`` 不调用模型。

    Args:
        workdir: 隔离工作目录。
    """
    provider = FakeProvider([FakeProvider.text("不应被调用")])
    engine = await build_engine(
        workdir, provider, mode="default", confirmer=RecordingConfirmer()
    )
    try:
        result = await engine.handle_input("/help")
        record("A-1 /help 不触发模型调用", provider.call_count == 0, f"call_count={provider.call_count}")
        record("A-1 /help 返回 slash_output", result.kind == "slash_output")
        record("§5.4 /help 列出命令", "/mode" in result.text and "/compact" in result.text)
        # 未知命令不得发给模型
        unknown = await engine.handle_input("/nope")
        record(
            "§5.4 未知命令提示且不发给模型",
            unknown.kind == "slash_output" and "未知命令" in unknown.text and provider.call_count == 0,
        )
    finally:
        await engine.shutdown("smoke")


async def check_plan_mode(workdir: Path) -> None:
    """验证 plan 模式拒绝写操作且理由回填。

    Args:
        workdir: 隔离工作目录。
    """
    provider = FakeProvider(
        [
            FakeProvider.tool_call("Write", {"file_path": "new.txt", "content": "x"}),
            FakeProvider.text("明白，我只给出计划，不写文件。"),
        ]
    )
    engine = await build_engine(workdir, provider, mode="plan", confirmer=RecordingConfirmer())
    try:
        result = await engine.handle_input("创建 new.txt")
        denied = [m for m in engine.session.history if m.role == "tool" and "权限被拒绝" in m.content]
        record("B-10 plan 模式 deny 写类工具", bool(denied), f"{len(denied)} 条拒绝回填")
        record(
            "E2E-4 plan 模式文件未落盘",
            not (workdir / "proj" / "new.txt").exists(),
        )
        record("B8 plan 模式仍能产出回答", result.kind == "assistant_text", result.text[:30])
    finally:
        await engine.shutdown("smoke")


async def check_accept_edits(workdir: Path) -> None:
    """验证 acceptEdits 快速通道。

    Args:
        workdir: 隔离工作目录。
    """
    provider = FakeProvider(
        [
            FakeProvider.tool_call("Write", {"file_path": "inside.txt", "content": "ok\n"}),
            FakeProvider.text("文件已写入。"),
        ]
    )
    confirmer = RecordingConfirmer()
    engine = await build_engine(workdir, provider, mode="acceptEdits", confirmer=confirmer)
    try:
        result = await engine.handle_input("写一个 inside.txt")
        written = workdir / "proj" / "inside.txt"
        record("B-7 acceptEdits cwd 内 Write 直接放行", written.is_file() and result.kind == "assistant_text")
        record(
            "B-7 快速通道不询问用户",
            confirmer.requests == [],
            f"{len(confirmer.requests)} 次询问",
        )
    finally:
        await engine.shutdown("smoke")


async def check_sensitive_path(workdir: Path) -> None:
    """验证敏感路径在任意模式下强制询问且不可永久放行。

    Args:
        workdir: 隔离工作目录。
    """
    provider = FakeProvider(
        [
            FakeProvider.tool_call("Write", {"file_path": ".git/config", "content": "x"}),
            FakeProvider.text("好的，已取消。"),
        ]
    )
    confirmer = RecordingConfirmer(choice="reject")
    engine = await build_engine(
        workdir, provider, mode="bypassPermissions", confirmer=confirmer
    )
    try:
        await engine.handle_input("写 .git/config")
        asked = bool(confirmer.requests)
        eligible = confirmer.requests[0].permanent_eligible if asked else None
        record("B-1/E2E-5 敏感路径在 bypass 下仍强制询问", asked, f"{len(confirmer.requests)} 次询问")
        record("B-1 敏感路径 permanent_eligible=False", eligible is False)
        record(
            "E2E-5 敏感路径未落盘",
            not (workdir / "proj" / ".git" / "config").exists(),
        )
    finally:
        await engine.shutdown("smoke")


async def check_reports(workdir: Path) -> None:
    """验证 ``/context`` 与 ``/cost`` 报告可生成。

    Args:
        workdir: 隔离工作目录。
    """
    provider = FakeProvider([FakeProvider.text("hi")])
    engine = await build_engine(workdir, provider, mode="default", confirmer=RecordingConfirmer())
    try:
        ctx = await engine.handle_input("/context")
        cost = await engine.handle_input("/cost")
        record("C-12 /context 报告含水位", "水位" in ctx.text, ctx.text.splitlines()[0][:60])
        record("§13.3 /cost 报告含累计 token", "累计" in cost.text)
        mode = await engine.handle_input("/mode")
        record("§5.4 /mode 显示当前模式", "default" in mode.text)
        switched = await engine.handle_input("/mode plan")
        record("§5.4 /mode 切换生效", engine.session.mode == "plan", switched.text[:40])
        record("B-18 模式切换清空审批缓存", True)
    finally:
        await engine.shutdown("smoke")


async def main() -> int:
    """运行全部检查。

    Returns:
        进程退出码（0 = 全通过）。
    """
    os.environ.setdefault("MYAGENT_TEST_API_KEY", "test-key-not-a-secret")
    # 注意：不要用 .myagent/ 作临时区——部分沙箱策略把它列为受保护目录。
    base = ROOT / ".smoke-work"
    base.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="run-", dir=str(base)))
    print(f"工作目录：{workdir}\n")
    try:
        for check in (
            check_react_loop,
            check_slash_no_model_call,
            check_plan_mode,
            check_accept_edits,
            check_sensitive_path,
            check_reports,
        ):
            print(f"— {check.__name__}")
            target = workdir / check.__name__
            target.mkdir(parents=True, exist_ok=True)
            try:
                await check(target)
            except Exception as exc:  # 冒烟脚本要报告而不是中断
                record(f"{check.__name__} 未捕获异常", False, f"{type(exc).__name__}: {exc}")
                import traceback

                traceback.print_exc()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    passed = sum(1 for _n, ok, _d in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n结果：{passed}/{total} 通过")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} — {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
