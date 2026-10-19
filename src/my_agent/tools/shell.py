"""``Bash`` 工具：执行 shell 命令（§14.2）。

- **平台 shell**：Windows 用 ``pwsh``（缺失则 ``powershell``，再退 ``cmd.exe``），
  其它平台用 ``/bin/bash``；命令**始终作为参数传给 shell 程序**，不使用 ``shell=True`` 拼接。
- **cwd**：固定项目根（``ctx.project_root``，缺失时退 ``ctx.cwd``）。
- **超时**：默认 :data:`~my_agent.config.defaults.BASH_DEFAULT_TIMEOUT_MS`；超时后
  **杀掉整棵进程树**（Windows ``taskkill /F /T /PID``，POSIX ``os.killpg``）并返回超时错误（J-4）。
- **输出**：stdout 与 stderr 合并捕获，超过 :data:`BASH_OUTPUT_LIMIT_BYTES` 即截断并终止进程。
- ``ctx.abort_event`` 被置位时同样杀掉进程树（Ctrl-C 语义，§B6）。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from my_agent.config.defaults import BASH_DEFAULT_TIMEOUT_MS
from my_agent.models.tools import ToolContext, ToolResult, ToolSpec
from my_agent.tools.base import BaseTool, as_positive_int, decode_text

__all__ = ["BashTool", "shell_argv", "shell_name"]

#: 输出字节上限。【待补】``config/defaults.py`` 目前没有对应常量（§0.1 未列）；
#: 建议新增 ``BASH_MAX_OUTPUT_BYTES = 1_000_000``，届时本常量删掉改为导入。
BASH_OUTPUT_LIMIT_BYTES = 1_000_000

#: 单次读取的块大小。
READ_CHUNK_BYTES = 65_536

#: 杀掉进程树后等待退出的宽限时间（秒）。
KILL_GRACE_S = 3.0

_DESCRIPTION = """在项目根目录执行一条 shell 命令（Windows: pwsh/powershell；其它平台: /bin/bash），返回合并后的输出与退出码。

何时用：运行测试、构建、git 状态、安装依赖等必须走命令行的操作。
何时不用：① 不要用它读写或查找文件——用 Read/Write/Edit/Glob/Grep（更安全、更可审计）；
② 不要用它执行需要交互输入或长时间挂起的命令（默认 120s 超时后会强杀整个进程树）；
③ 不要用它启动常驻服务（除非确实需要后台进程）。
副作用：**会修改磁盘文件**（命令可能改动工作区、安装依赖、发起网络请求）；执行前请说明意图。"""


def shell_name() -> str:
    """返回当前平台使用的 shell 显示名。

    Returns:
        ``"pwsh"`` / ``"powershell"`` / ``"cmd"`` / ``"bash"`` / ``"sh"`` 之一。
    """
    argv = shell_argv("")
    return Path(argv[0]).stem.lower()


def shell_argv(command: str) -> list[str]:
    """构造「shell 程序 + 参数」的命令行列表。

    Args:
        command: 要执行的命令文本（作为 shell 的 ``-Command`` / ``-c`` 参数，不做字符串拼接）。

    Returns:
        可直接交给 :func:`asyncio.create_subprocess_exec` 的参数列表。

    Raises:
        无：找不到任何 shell 时回退到平台默认程序名。
    """
    if os.name == "nt":
        for candidate in ("pwsh", "powershell"):
            found = shutil.which(candidate)
            if found:
                return [
                    found,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    command,
                ]
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return [comspec, "/d", "/s", "/c", command]
    for candidate in ("/bin/bash", "/usr/bin/bash"):
        if Path(candidate).is_file():
            return [candidate, "-lc", command]
    return ["/bin/sh", "-c", command]


class BashTool(BaseTool):
    """执行 shell 命令（超时杀进程树，输出限流）。"""

    spec = ToolSpec(
        name="Bash",
        description=_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的命令"},
                "timeout_ms": {
                    "type": "integer",
                    "description": f"超时毫秒，默认 {BASH_DEFAULT_TIMEOUT_MS}",
                },
                "description": {"type": "string", "description": "这条命令做什么（给用户看）"},
            },
            "required": ["command"],
        },
        category="execute",
        is_read_only=False,
        is_destructive=True,
        requires_user_approval_hint=True,
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """执行命令。

        Args:
            args: ``command``、可选 ``timeout_ms`` 与 ``description``。
            ctx: 执行上下文（cwd 取 ``ctx.project_root``）。

        Returns:
            ``ok=True`` 且 ``exit_code == 0`` 时为成功；非零退出、超时、输出超限或被中止时
            ``ok=False``（超时/中止时进程树已被杀掉）。
        """
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return self.fail(ctx, "缺少必填参数 command（应为要执行的命令字符串）")
        timeout_ms = as_positive_int(args.get("timeout_ms"), BASH_DEFAULT_TIMEOUT_MS)
        if timeout_ms is None:
            return self.fail(ctx, "参数 timeout_ms 非法：应为 ≥1 的整数（毫秒）")
        timeout_s = timeout_ms / 1000.0

        workdir = Path(ctx.project_root or ctx.cwd)
        if not workdir.is_dir():
            return self.fail(ctx, f"工作目录不存在：{workdir}")

        argv = shell_argv(command)
        started = time.monotonic()
        try:
            proc = await self._spawn(argv, workdir)
        except (OSError, ValueError) as exc:
            return self.fail(ctx, f"启动 shell 失败（{argv[0]}）：{exc}")

        data, status = await self._supervise(proc, timeout_s=timeout_s, abort=ctx.abort_event)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        try:
            output = decode_text(data)
        except UnicodeDecodeError:
            output = data.decode("utf-8", errors="replace")
        if status == "timeout":
            return self.fail(
                ctx,
                f"命令执行超时（{timeout_s:g}s，已终止整棵进程树，PID {proc.pid}）。"
                "请缩小任务范围、加上非交互参数（如 --yes/--no-input），或调大 timeout_ms 重试",
                content=output,
                exit_code=-1,
                duration_ms=elapsed_ms,
            )
        if status == "abort":
            return self.fail(
                ctx,
                "命令已被用户中止（Ctrl-C），进程树已终止",
                content=output,
                exit_code=-1,
                duration_ms=elapsed_ms,
            )

        code = proc.returncode if proc.returncode is not None else -1
        if status == "overflow":
            notice = (
                f"\n\n[输出超过 {BASH_OUTPUT_LIMIT_BYTES} 字节已截断，进程树已终止。"
                "请把输出重定向到文件后用 Grep 检索，或缩小命令范围]"
            )
            return self.fail(
                ctx,
                f"命令输出过大（>{BASH_OUTPUT_LIMIT_BYTES} 字节），已截断并终止进程树",
                content=output + notice,
                exit_code=code,
                duration_ms=elapsed_ms,
            )
        if code != 0:
            return self.fail(
                ctx,
                f"命令退出码 {code}",
                content=output,
                exit_code=code,
                duration_ms=elapsed_ms,
            )
        return self.ok(ctx, output or "（命令无输出）", exit_code=0, duration_ms=elapsed_ms)

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    async def _spawn(self, argv: list[str], workdir: Path) -> asyncio.subprocess.Process:
        """启动子进程（POSIX 建新会话，便于整组杀进程）。

        Args:
            argv: 命令行参数列表。
            workdir: 工作目录。

        Returns:
            子进程对象。

        Raises:
            OSError: 启动失败（可执行文件缺失、cwd 非法等）。
        """
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        return await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(workdir),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **kwargs,
        )

    async def _supervise(
        self,
        proc: asyncio.subprocess.Process,
        *,
        timeout_s: float,
        abort: Any,
    ) -> tuple[bytes, str]:
        """等待命令结束，超时/中止/输出超限时杀掉进程树。

        Args:
            proc: 子进程。
            timeout_s: 超时秒数。
            abort: ``asyncio.Event | None``，被置位表示用户中止。

        Returns:
            ``(输出字节, 状态)``；状态取值 ``"ok"`` / ``"overflow"`` / ``"timeout"`` / ``"abort"``。
        """
        collect = asyncio.create_task(self._collect(proc))
        waiters: set[asyncio.Task[Any]] = {collect}
        abort_task: asyncio.Task[Any] | None = None
        if abort is not None and callable(getattr(abort, "wait", None)):
            abort_task = asyncio.create_task(abort.wait())
            waiters.add(abort_task)

        done, _pending = await asyncio.wait(
            waiters, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
        )
        if collect in done:
            if abort_task is not None:
                abort_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await abort_task
            data, overflow = collect.result()
            return data, "overflow" if overflow else "ok"

        status = "abort" if abort_task is not None and abort_task in done else "timeout"
        collect.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await collect
        await self._kill_tree(proc)
        return await self._partial_output(proc), status

    async def _collect(self, proc: asyncio.subprocess.Process) -> tuple[bytes, bool]:
        """读取输出直到进程结束或超过字节上限。

        Args:
            proc: 子进程。

        Returns:
            ``(输出字节, 是否因超限被截断)``；超限时已杀掉进程树。
        """
        data, overflow = await self._drain(proc)
        if overflow:
            await self._kill_tree(proc)
        await proc.wait()
        return data, overflow

    async def _drain(self, proc: asyncio.subprocess.Process) -> tuple[bytes, bool]:
        """按块读取合并后的输出。

        Args:
            proc: 子进程。

        Returns:
            ``(输出字节, 是否超限)``。
        """
        stream = proc.stdout
        if stream is None:  # pragma: no cover - 我们总是开 PIPE
            return b"", False
        buffer = bytearray()
        while True:
            chunk = await stream.read(READ_CHUNK_BYTES)
            if not chunk:
                return bytes(buffer), False
            room = BASH_OUTPUT_LIMIT_BYTES - len(buffer)
            if len(chunk) >= room:
                buffer.extend(chunk[:room])
                return bytes(buffer), True
            buffer.extend(chunk)

    async def _partial_output(self, proc: asyncio.subprocess.Process) -> bytes:
        """在杀掉进程后尽量取回已产生的部分输出。

        Args:
            proc: 子进程。

        Returns:
            已读到的输出（拿不到时为 ``b""``）。
        """
        stream = proc.stdout
        if stream is None:  # pragma: no cover - 我们总是开 PIPE
            return b""
        try:
            return await asyncio.wait_for(stream.read(), timeout=KILL_GRACE_S)
        except (TimeoutError, OSError):
            return b""

    async def _kill_tree(self, proc: asyncio.subprocess.Process) -> None:
        """终止整棵进程树并等待回收。

        Args:
            proc: 子进程。

        Returns:
            无。
        """
        if proc.returncode is None:
            pid = proc.pid
            if os.name == "nt":
                await asyncio.to_thread(
                    subprocess.run,
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    check=False,
                )
            else:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=KILL_GRACE_S)
        except TimeoutError:
            return
