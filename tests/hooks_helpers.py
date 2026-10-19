"""Hook 测试公共工具：在 ``tmp_path`` 里生成**真子进程** hook 脚本。

集成测试的 hook 必须是独立进程（§11.3 的 stdin/stdout 协议只在子进程边界上成立）；
本模块把「写脚本 + 造 HookSpec + 跑 HookRunner + 读调用日志」收敛成几个小函数。

脚本模板自身不 import 项目代码（避免 PYTHONPATH 依赖），只读 stdin JSON、追加调用记录、
按注入的 ``decision`` / ``modified_payload`` 输出返回值。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from my_agent.config.schema import Config, HookSpec
from my_agent.hooks.events import HookEvent
from my_agent.models.protocols import DisplayPort

if TYPE_CHECKING:
    from my_agent.hooks.runner import HookRunner

__all__ = [
    "PROJECT_DIR",
    "RecordingDisplay",
    "collect_calls",
    "make_runner",
    "python_available",
    "spec_for",
    "subprocess_spawn_ok",
    "write_recording_hook",
]

#: 项目内运行时目录名。
PROJECT_DIR = ".myagent"

#: 记录型 hook 脚本模板。
#:
#: 占位符均为 **Python 字面量**（由 ``repr`` 注入）：``__LOG__``（调用记录路径）、
#: ``__DECISION__``（返回值 decision 或 None）、``__MODIFIED__``（``modified_payload``
#: 的 JSON 文本 —— 脚本内 ``json.loads`` 还原）、``__MESSAGE__``、``__CONTINUE__``
#: （``True``/``False``/``None``）、``__SLEEP__``（秒）、``__EXIT_CODE__``（整数）、
#: ``__STDERR__``（写往 stderr 的文本，退出码 2 时作为 deny 理由）。
_HOOK_TEMPLATE = '''\
import json
import sys
import time

LOG = {log!r}
DECISION = {decision!r}
MODIFIED_JSON = {modified!r}
MESSAGE = {message!r}
CONTINUE = {continue_!r}
SLEEP = {sleep!r}
EXIT_CODE = {exit_code!r}
STDERR = {stderr!r}

MODIFIED = json.loads(MODIFIED_JSON) if MODIFIED_JSON else None


def log(entry):
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\\n")


def main():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except Exception:
        data = {{"raw": raw}}
    log({{"seen": data, "ts": time.time()}})
    if SLEEP:
        time.sleep(SLEEP)
    if STDERR:
        sys.stderr.write(STDERR)
    if EXIT_CODE:
        sys.exit(EXIT_CODE)
    out = {{}}
    if DECISION is not None:
        out["decision"] = DECISION
    if MODIFIED is not None:
        out["modified_payload"] = MODIFIED
    if MESSAGE is not None:
        out["message"] = MESSAGE
    if CONTINUE is not None:
        out["continue"] = CONTINUE
    if out:
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.exit(0)


main()
'''


def subprocess_spawn_ok() -> bool:
    """检测当前平台能否创建子进程（不能则相关测试整体**显式跳过**）。

    注意：受限沙箱下 ``create_subprocess_exec`` 会抛 ``PermissionError [WinError 5]``；
    本函数把这种情况识别为「平台不可用」，由调用方 ``pytest.mark.skipif`` 显式报告。

    Returns:
        ``True`` 表示可以正常 ``spawn`` 一个 Python 解释器。
    """
    exe = shutil.which(sys.executable) or sys.executable
    if not exe:
        return False
    try:
        return asyncio.run(_spawn_probe(exe))
    except (OSError, RuntimeError, ValueError):
        return False


async def _spawn_probe(exe: str) -> bool:
    """真正 spawn 一次并回收（避免假阳性）。

    Args:
        exe: 解释器路径。

    Returns:
        退出码为 0 时 ``True``。
    """
    proc = await asyncio.create_subprocess_exec(
        exe,
        "-c",
        "print('ok')",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    return proc.returncode == 0


def python_available() -> bool:
    """测试用解释器是否可用。

    Returns:
        ``sys.executable`` 是否存在。
    """
    return bool(shutil.which(sys.executable) or Path(sys.executable).exists())


def write_recording_hook(
    path: Path,
    log_path: Path,
    *,
    decision: str | None = None,
    modified_payload: dict[str, Any] | None = None,
    message: str | None = None,
    continue_: bool | None = None,
    sleep: float = 0.0,
    exit_code: int = 0,
    stderr: str = "",
) -> Path:
    """把「记录 + 返回固定结果」的 hook 脚本写到 ``path``。

    Args:
        path: 脚本落盘路径（父目录需已存在）。
        log_path: 每次调用的记录文件（JSONL）。
        decision: 返回的 ``decision``；``None`` = 不返回该字段。
        modified_payload: 返回的 ``modified_payload``。
        message: 返回的 ``message``。
        continue_: 返回的 ``continue``。
        sleep: 读到 stdin 后先 sleep 的秒数（用于超时用例）。
        exit_code: 非 0 时直接按该退出码退出（用于 G-3/G-4）。
        stderr: 写往 stderr 的文本（退出码 2 时即 deny 理由）。

    Returns:
        写入的脚本路径。
    """
    source = _HOOK_TEMPLATE.format(
        log=str(log_path),
        decision=decision,
        modified=json.dumps(modified_payload, ensure_ascii=False)
        if modified_payload is not None
        else None,
        message=message,
        continue_=continue_,
        sleep=sleep,
        exit_code=exit_code,
        stderr=stderr,
    )
    path.write_text(source, encoding="utf-8")
    return path


def spec_for(
    event: HookEvent | str,
    script: Path,
    *,
    matcher: str | None = None,
    timeout_s: float = 5.0,
    on_error: str = "allow",
    blocking: bool = True,
) -> HookSpec:
    """构造一条指向记录型脚本的 :class:`HookSpec`。

    Args:
        event: 事件（枚举或字符串）。
        script: 脚本路径。
        matcher: 工具名正则。
        timeout_s: 超时秒数。
        on_error: ``allow`` / ``deny`` / ``abort``。
        blocking: 是否阻塞等待。

    Returns:
        ``type=command`` 的 hook 声明。
    """
    value = event.value if isinstance(event, HookEvent) else str(event)
    return HookSpec(
        event=value,
        matcher=matcher,
        type="command",
        command=f'"{sys.executable}" "{script}"',
        timeout_s=timeout_s,
        blocking=blocking,
        on_error=on_error,
    )


class RecordingDisplay:
    """把 warn/info/error 收集到列表的显示端口（断言告警用）。

    Attributes:
        warnings: ``warn()`` 收到的文案。
        infos: ``info()`` 收到的文案.
        errors: ``error()`` 收到的文案。
    """

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.errors: list[str] = []

    def stream_text(self, delta: str) -> None:
        """忽略流式文本。

        Args:
            delta: 文本增量。
        """

    def info(self, msg: str) -> None:
        """记录提示。

        Args:
            msg: 文案。
        """
        self.infos.append(msg)

    def warn(self, msg: str) -> None:
        """记录警告。

        Args:
            msg: 文案。
        """
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        """记录错误。

        Args:
            msg: 文案。
        """
        self.errors.append(msg)

    def tool_start(self, name: str, summary: str) -> None:
        """忽略工具开始。

        Args:
            name: 工具名。
            summary: 摘要。
        """

    def tool_result(self, name: str, ok: bool, summary: str) -> None:
        """忽略工具结束。

        Args:
            name: 工具名。
            ok: 是否成功。
            summary: 摘要。
        """


def make_runner(
    specs: list[HookSpec],
    *,
    cwd: Path,
    config: Config | None = None,
    display: DisplayPort | None = None,
) -> HookRunner:
    """构造 :class:`~my_agent.hooks.runner.HookRunner`。

    Args:
        specs: hook 声明。
        cwd: 工作目录。
        config: 全局配置（缺省为默认 :class:`Config`）。
        display: 显示端口（缺省为记录型）。

    Returns:
        可直接 ``await runner.run(...)`` 的执行器。
    """
    from my_agent.hooks.runner import HookRunner

    return HookRunner(
        specs,
        cwd=cwd,
        config=config if config is not None else Config(),
        display=display if display is not None else RecordingDisplay(),
    )


def collect_calls(log_path: Path) -> list[dict[str, Any]]:
    """读取记录型 hook 的调用日志。

    Args:
        log_path: 记录文件路径。

    Returns:
        调用记录列表（文件不存在时为空列表）。
    """
    if not log_path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries
