"""Typer CLI 入口（§16.1）。

```
myagent                          # 交互式 REPL（默认）
myagent "帮我看下这个报错"        # 一次性提问（等价 --print）
myagent --print "..."            # 非交互：输出后退出，遇 ask 走 §6.9 规则
myagent --mode plan              # 指定安全模式
myagent --continue               # 继续最近会话
myagent --resume <session_id>    # 恢复指定会话
myagent --cwd <path>             # 指定工作目录
myagent --config <path>          # 指定配置文件
myagent --debug                  # DEBUG 日志
```
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from my_agent import __version__
from my_agent.config.defaults import assumption_report
from my_agent.config.loader import load_config, myagent_home, project_root_for
from my_agent.engine.query_engine import EngineResult, QueryEngine
from my_agent.errors import ConfigError, MyAgentError
from my_agent.models.permission import ALL_MODES
from my_agent.ui.console import RichConsole
from my_agent.ui.prompt import TerminalConfirmer, read_line

__all__ = ["app", "main"]

_LOG = logging.getLogger("my_agent")

EXIT_HINT = "输入 /help 查看命令，/exit 退出，Ctrl-D 退出。"
NO_SANDBOX_WARNING = (
    "⚠ 本工具在你本机以你的权限运行，没有沙盒隔离；写/执行类操作会被逐次询问。"
)

app = typer.Typer(
    name="myagent",
    help="本地运行的编程 CLI Agent（双层 Agent-Loop + 四阶段权限流水线）",
    add_completion=False,
    no_args_is_help=False,
)


def _setup_logging(debug: bool) -> None:
    """配置日志（审计与调试分离：日志走 stderr，用户输出走 rich）。

    Args:
        debug: 是否 DEBUG 级别。
    """
    home = myagent_home()
    log_dir = home / "logs"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "myagent.log", encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _resolve_resume(*, resume: str | None, continue_: bool, cwd: Path) -> str | None:
    """解析 ``--resume`` / ``--continue`` 得到目标会话 id。

    Args:
        resume: ``--resume`` 的值。
        continue_: 是否 ``--continue``。
        cwd: 工作目录。

    Returns:
        会话 id；都没有指定时返回 ``None``。
    """
    if resume:
        return resume
    if continue_:
        from my_agent.engine.session_store import SessionStore

        return SessionStore(project_root_for(cwd)).latest_session_id()
    return None


def _build_engine(
    *,
    cwd: Path,
    mode: str | None,
    resume: str | None,
    continue_: bool,
    config_path: Path | None,
    print_mode: bool,
    debug: bool,
    console: RichConsole | None = None,
) -> QueryEngine:
    """装配引擎（含配置校验与错误美化）。

    Args:
        cwd: 工作目录。
        mode: ``--mode``。
        resume: ``--resume``。
        continue_: ``--continue``。
        config_path: ``--config``。
        print_mode: 是否非交互模式。
        debug: ``--debug``。
        console: 已构造的输出端口（缺省新建）。

    Returns:
        :class:`~my_agent.engine.query_engine.QueryEngine`。

    Raises:
        ConfigError: 配置非法（含模式名非法）。
    """
    if mode is not None and mode not in ALL_MODES:
        raise ConfigError(
            f"未知模式：{mode}；可用：{' / '.join(ALL_MODES)}",
            detail=f"未知模式：{mode}",
        )
    workdir = Path(cwd).expanduser().resolve()
    config = load_config(
        cwd=workdir,
        cli_overrides={"mode": mode} if mode else None,
        config_path=config_path,
    )
    if mode:
        config.safety.mode = mode  # type: ignore[assignment]
    if debug:
        _LOG.setLevel(logging.DEBUG)

    output = console or RichConsole(
        no_color=config.ui.no_color or None,
        emoji=config.ui.emoji,
        stream=config.ui.stream,
        preview_lines=config.ui.tool_result_preview_lines,
    )
    confirmer = TerminalConfirmer(output, interactive=not print_mode)
    return QueryEngine(
        config,
        confirmer,
        output,
        cwd=workdir,
        resume=_resolve_resume(resume=resume, continue_=continue_, cwd=workdir),
        non_interactive=print_mode,
    )


def _print_result(result: EngineResult, console: RichConsole) -> int:
    """输出一次结果并返回进程退出码。

    Args:
        result: 引擎结果。
        console: 输出端口。

    Returns:
        进程退出码。
    """
    if result.kind == "error":
        console.error(result.text)
        return 1
    if result.text:
        console.info(result.text)
    return 0


async def _run_repl(engine: QueryEngine, console: RichConsole) -> int:
    """交互式 REPL 主循环。

    Args:
        engine: 已启动的引擎。
        console: 输出端口。

    Returns:
        进程退出码。
    """
    console.info(NO_SANDBOX_WARNING)
    console.info(EXIT_HINT)
    interrupted_once = False

    while True:
        console.render_status(engine.status_line())
        try:
            text = await read_line("› ")
        except KeyboardInterrupt:
            console.info("\n（Ctrl-D 或 /exit 退出）")
            continue

        if not text.strip():
            continue
        try:
            result = await engine.handle_input(text)
        except KeyboardInterrupt:
            if interrupted_once:
                console.warn("已中止回合")
                interrupted_once = False
                continue
            interrupted_once = True
            console.warn("已取消当前操作（再按一次 Ctrl-C 中止整个回合）")
            continue
        interrupted_once = False

        if result.kind == "aborted" and result.text == "__exit__":
            return 0
        if result.kind == "error":
            console.error(result.text)
        elif result.text:
            console.print_markdown(result.text)


async def _run_print(engine: QueryEngine, prompt: str, console: RichConsole) -> int:
    """非交互一次性执行。

    Args:
        engine: 未启动的引擎。
        prompt: 用户输入。
        console: 输出端口。

    Returns:
        进程退出码。
    """
    await engine.start()
    try:
        result = await engine.handle_input(prompt)
        return _print_result(result, console)
    finally:
        await engine.shutdown("print_mode")


def _async_main(coro: Any) -> int:
    """运行顶层协程并统一异常处理。

    Args:
        coro: 待运行协程。

    Returns:
        进程退出码。
    """
    try:
        return int(asyncio.run(coro))
    except KeyboardInterrupt:
        return 130
    except ConfigError as exc:
        sys.stderr.write(f"{exc.user_message()}\n")
        return 2
    except MyAgentError as exc:
        sys.stderr.write(f"{exc.user_message()}\n")
        return 1


@app.command()
def main(
    prompt: Annotated[str | None, typer.Argument(help="一次性提问（等价 --print）")] = None,
    print_mode: Annotated[
        bool, typer.Option("--print", "-p", help="非交互：输出后退出；遇 ask 一律拒绝")
    ] = False,
    mode: Annotated[
        str | None,
        typer.Option("--mode", "-m", help=f"安全模式：{' / '.join(ALL_MODES)}"),
    ] = None,
    continue_: Annotated[bool, typer.Option("--continue", "-c", help="继续最近会话")] = False,
    resume: Annotated[str | None, typer.Option("--resume", help="恢复指定会话 id")] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd", help="指定工作目录")] = None,
    config: Annotated[Path | None, typer.Option("--config", help="指定配置文件")] = None,
    debug: Annotated[bool, typer.Option("--debug", help="DEBUG 日志")] = False,
    assumptions: Annotated[
        bool, typer.Option("--assumptions", help="打印全部【待确认假设】的当前实现口径")
    ] = False,
    version: Annotated[bool, typer.Option("--version", "-V", help="显示版本")] = False,
) -> None:
    """myagent 主入口。

    Raises:
        typer.Exit: 以对应退出码结束进程。
    """
    if version:
        typer.echo(f"myagent {__version__}")
        raise typer.Exit(0)
    if assumptions:
        typer.echo(assumption_report())
        raise typer.Exit(0)

    _setup_logging(debug)
    workdir = Path(cwd) if cwd else Path.cwd()
    non_interactive = print_mode or prompt is not None

    try:
        engine = _build_engine(
            cwd=workdir,
            mode=mode,
            resume=resume,
            continue_=continue_,
            config_path=config,
            print_mode=non_interactive,
            debug=debug,
        )
    except ConfigError as exc:
        typer.secho(exc.user_message(), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc

    console: RichConsole = engine.console  # type: ignore[assignment]

    if prompt is not None:
        code = _async_main(_run_print(engine, prompt, console))
        raise typer.Exit(code)

    async def _session() -> int:
        await engine.start()
        try:
            return await _run_repl(engine, console)
        finally:
            await engine.shutdown("user_exit")

    raise typer.Exit(_async_main(_session()))


def _entry() -> None:  # pragma: no cover - console_scripts 入口
    """``myagent`` 控制台脚本入口。"""
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
