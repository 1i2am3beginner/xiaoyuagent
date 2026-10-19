"""Hook 执行器（§11.3 协议、§11.4 能力矩阵、§11.5 验收 G-1…G-8）。

职责：

* **串行链式**执行同一事件的全部 hook（配置顺序），后继 hook 看到前者的
  ``modified_payload``（§11.3）。
* ``deny`` **立即短路**，后续 hook 不再执行（G-8）。
* ``ask`` **强制**转人工确认（由调用方落实，见 ``security/pipeline.py``）。
* ``continue=false`` 中止整个回合。
* 退出码语义：``0`` 读 stdout JSON；``2`` → 视为 ``deny``（stderr 作理由）；
  其它非零 → 按 ``on_error``（``allow``/``deny``/``abort``）处理并记警告（§11.3、G-3/G-4）。
* 超时上限 = ``timeout_s + HOOK_TIMEOUT_GRACE_S``；崩溃/超时的 hook **绝不**阻塞更久。
* **安全不变量（G-5）**：``after_permission_check`` 只能收紧裁决，永远不能把
  ``deny`` 放宽为 ``allow``；该合并逻辑是可直接单测的纯函数
  :func:`merge_permission_decision`。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from my_agent.config.defaults import (
    HOOK_ALLOW_PERMISSION_OVERRIDE,
    HOOK_DEFAULT_TIMEOUT_S,
    HOOK_TIMEOUT_GRACE_S,
)
from my_agent.config.schema import Config, HookSpec
from my_agent.errors import HookError
from my_agent.hooks.events import (
    TOOL_EVENTS,
    HookEvent,
    HookOutcome,
    HookPayload,
    mutable_fields,
)
from my_agent.hooks.loader import load_hooks
from my_agent.hooks.python_hook import run_python_hook
from my_agent.models.permission import Decision
from my_agent.models.protocols import DisplayPort, NullDisplay

__all__ = [
    "HookDecisionMerge",
    "HookRunner",
    "merge_permission_decision",
    "split_command",
]

_LOG = logging.getLogger(__name__)

#: Windows 下不弹控制台窗口（§10.5 同款要求，hook 子进程同样适用）。
_CREATE_NO_WINDOW = 0x08000000

#: 裁决严格度：数值越大越严格（``deny`` > ``ask`` > ``allow``）。
_STRICTNESS: dict[str, int] = {"allow": 0, "ask": 1, "deny": 2}


class HookDecisionMerge(BaseModel):
    """:func:`merge_permission_decision` 的结果（G-5 的显式证据）。

    Attributes:
        decision: 合并后的生效裁决。
        warning: 需要展示/记录的告警；无告警时为空串。
        ignored_relaxation: 是否发生了「被拒绝的放宽尝试」（安全不变量触发）。
    """

    model_config = ConfigDict(extra="ignore")

    decision: Decision
    warning: str = ""
    ignored_relaxation: bool = False


def merge_permission_decision(
    current: Decision | None,
    proposed: Decision | None,
    *,
    allow_override: bool | None = None,
) -> HookDecisionMerge:
    """把 hook 提议的裁决合并进流水线裁决（**纯函数**，§11.4 安全不变量 G-5）。

    规则：hook 只能**收紧**（``allow → ask → deny``）。任何放宽方向的尝试
    （``deny → allow``、``deny → ask``、``ask → allow``）在
    ``HOOK_ALLOW_PERMISSION_OVERRIDE = False``（默认，U12）下一律**忽略并告警**。

    Args:
        current: 流水线已产出的裁决；``None`` 表示尚无裁决（hook 可自由决定）。
        proposed: hook 提议的裁决；``None`` 表示不改变。
        allow_override: 是否允许放宽；缺省读
            :data:`~my_agent.config.defaults.HOOK_ALLOW_PERMISSION_OVERRIDE`。

    Returns:
        :class:`HookDecisionMerge`：生效裁决 + 告警 + 是否发生过被拒绝的放宽。

    Raises:
        HookError: ``current`` 与 ``proposed`` 均为 ``None``（调用方不应这样使用）。
    """
    if current is None and proposed is None:
        raise HookError(
            "merge_permission_decision 需要至少一个非空裁决",
            detail="current 与 proposed 同时为 None",
        )
    if proposed is None:
        assert current is not None  # noqa: S101 - 上一分支已保证
        return HookDecisionMerge(decision=current)
    if current is None:
        return HookDecisionMerge(decision=proposed)

    override = HOOK_ALLOW_PERMISSION_OVERRIDE if allow_override is None else allow_override
    if _STRICTNESS[proposed] >= _STRICTNESS[current]:
        return HookDecisionMerge(decision=proposed)
    if override:
        _LOG.warning(
            "hook 放宽裁决 %s → %s（allow_permission_override=true，风险自负）",
            current,
            proposed,
        )
        return HookDecisionMerge(
            decision=proposed,
            warning=f"hook 放宽了裁决：{current} → {proposed}（allow_permission_override=true）",
        )
    return HookDecisionMerge(
        decision=current,
        warning=(
            f"已忽略 hook 的放宽尝试（{current} → {proposed}）："
            f"hook 不得把更严格的裁决放宽（§11.4 安全不变量）"
        ),
        ignored_relaxation=True,
    )


def split_command(command: str) -> list[str]:
    """把 ``command`` 字符串拆成 argv（不经过 shell，避免注入）。

    先按 POSIX 规则解析（支持引号）；失败时退化为「按空白切分 + 去掉成对引号」，
    以保证 Windows 反斜杠路径（``python C:\\x\\hook.py``）也能正确拆开。

    Args:
        command: 配置中的命令行。

    Returns:
        argv 列表（不含空串）。

    Raises:
        HookError: ``command`` 为空或拆分后无有效参数。
    """
    text = command.strip()
    if not text:
        raise HookError("hook 的 command 为空", detail="command 为空")
    try:
        argv = shlex.split(text, posix=True)
    except ValueError:
        argv = [token.strip("\"'") for token in re.split(r"\s+", text) if token.strip()]
    argv = [token for token in (part.strip() for part in argv) if token]
    if not argv:
        raise HookError(f"hook 的 command 无法拆分为可执行命令：{command!r}", detail=command)
    return argv


class _CallResult(BaseModel):
    """一次 hook 调用的原始结果（内部结构）。

    Attributes:
        outcome: 解析出的结果；``None`` = 不干预。
        timed_out: 是否超时。
        failed: 是否属于「失败」（非零退出 / 解析失败 / 调用异常），需按 ``on_error`` 处理。
        detail: 失败原因（用于日志与告警）。
        duration_ms: 调用耗时（毫秒）。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    outcome: HookOutcome | None = None
    timed_out: bool = False
    failed: bool = False
    detail: str = ""
    duration_ms: int = 0


class _Accumulator(BaseModel):
    """多 hook 链式累积状态（内部结构）。

    Attributes:
        payload: 当前生效负载（被前序 hook 改写后的版本）。
        decision: 已生效的裁决。
        messages: 依次拼接的 hook 文案。
        continue_: 是否继续回合。
        touched: 是否有 hook 提交过 ``modified_payload``。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    payload: HookPayload
    decision: Decision | None = None
    messages: list[str] = Field(default_factory=list)
    continue_: bool = True
    touched: bool = False


class HookRunner:
    """按事件调度 hook 的执行器（CONTRACTS §17.3）。

    Args:
        specs: hook 声明列表（``load_hooks`` 的产物）。
        cwd: 工作目录（写入 stdin JSON 的 ``cwd``，也是 ``${PWD}`` 的来源）。
        config: 全局配置（读取 ``hooks.*`` 开关与文件名）。
        display: 输出端口（默认 :class:`~my_agent.models.protocols.NullDisplay`）。
        project_root: 项目根；给出时 :meth:`reload` 能真正重新读取 ``hooks.yaml``。
        home: 全局目录；与 ``project_root`` 配合供 :meth:`reload` 使用。

    Attributes:
        events_for: ``matcher`` 正则过滤（G-6）。
    """

    def __init__(
        self,
        specs: list[HookSpec],
        *,
        cwd: Path,
        config: Config,
        display: DisplayPort | None = None,
        project_root: Path | None = None,
        home: Path | None = None,
    ) -> None:
        self.specs: list[HookSpec] = list(specs)
        self.cwd = Path(cwd)
        self.config = config
        self.display: DisplayPort = display if display is not None else NullDisplay()
        self.project_root = Path(project_root) if project_root is not None else None
        self.home = Path(home) if home is not None else None
        self._last_reload_error: str = ""

    # ------------------------------------------------------------------ 查询
    def events_for(self, tool: str, event: HookEvent | None = None) -> list[HookSpec]:
        """返回匹配给定工具名的 hook 列表（CONTRACTS §10，G-6）。

        Args:
            tool: 工具名（如 ``"Write"``）。
            event: 限定事件；``None`` = 全部工具相关事件
                （``before/after_tool_call``、``before/after_permission_check``）。

        Returns:
            按配置顺序排列的匹配项。非工具事件（``matcher`` 不适用）不在此列。
        """
        wanted = {event} if event is not None else set(TOOL_EVENTS)
        matched: list[HookSpec] = []
        for spec in self.specs:
            try:
                spec_event = HookEvent(spec.event)
            except ValueError:
                continue
            if spec_event not in wanted:
                continue
            if not self._matches(spec, tool):
                continue
            matched.append(spec)
        return matched

    def describe(self) -> str:
        """生成 ``/hook list`` 的输出文本（CONTRACTS §17.3）。

        Returns:
            多行文本；无 hook 时给出配置指引。
        """
        lines = [f"已加载 hook：{len(self.specs)} 个"]
        if self._last_reload_error:
            lines.append(f"（上次 reload 失败：{self._last_reload_error}）")
        if not self.specs:
            lines.append(
                f"（未配置 hook；写入 {self.config.hooks.project_file} 即可启用，"
                f'示例：hooks: [{{event: before_tool_call, matcher: "Bash|Write", '
                f'type: command, command: "python .myagent/hooks/check_write.py"}}]）'
            )
            return "\n".join(lines)

        lines.append("序号 | 事件 | 匹配 | 类型 | 目标 | 超时 | 失败策略")
        for index, spec in enumerate(self.specs, start=1):
            target = spec.command or spec.module or "(未配置)"
            lines.append(
                f"{index}. {spec.event} | {spec.matcher or '*'} | {spec.type} | "
                f"{target} | {spec.timeout_s:g}s | on_error={spec.on_error}"
            )
        return "\n".join(lines)

    def reload(self) -> None:
        """重新读取 ``hooks.yaml`` 并替换内存中的声明（``/hook reload``）。

        未提供 ``project_root`` 时只重置内部状态（并记录原因），不抛异常。

        Raises:
            HookError: 加载器本身抛错（正常情况下加载器只记警告，不会走到这里）。
        """
        if self.project_root is None or self.home is None:
            self._last_reload_error = "构造 HookRunner 时未提供 project_root/home，无法重新读盘"
            _LOG.warning("hooks.reload：%s", self._last_reload_error)
            return
        try:
            self.specs = load_hooks(self.project_root, self.home, self.config)
        except HookError as exc:
            self._last_reload_error = exc.user_message()
            _LOG.warning("hooks.reload 失败：%s", self._last_reload_error)
            raise
        self._last_reload_error = ""
        _LOG.info("hooks.reload 完成：%d 个 hook", len(self.specs))

    # ------------------------------------------------------------------ 执行
    async def run(
        self,
        event: HookEvent,
        *,
        session_id: str,
        turn_index: int,
        payload: HookPayload,
    ) -> HookOutcome:
        """按配置顺序串行执行该事件的全部 hook（§11.3）。

        Args:
            event: 事件。
            session_id: 会话 ID（写入 stdin JSON）。
            turn_index: 轮次号（写入 stdin JSON）。
            payload: 事件负载；会被前序 hook 的 ``modified_payload`` 链式改写。

        Returns:
            :class:`HookOutcome`：``decision``（``allow``/``deny``/``ask``/``None``）、
            ``modified_payload``（链式累积）、``message``、``continue_``。

        Raises:
            HookError: 负载构造或裁决合并出现不可恢复的内部错误。
        """
        state = _Accumulator(payload=payload)
        data: dict[str, Any] = {
            "event": event.value,
            "session_id": session_id,
            "turn_index": turn_index,
            "cwd": str(self.cwd),
            "payload": payload.to_json_dict(),
        }
        for spec in self._specs_for(event, payload):
            if (
                self.config.hooks.subagent_isolation
                and state.payload.subagent_id
                and event not in {HookEvent.SESSION_START, HookEvent.SESSION_END}
            ):
                _LOG.debug("hooks: subagent_isolation=true，跳过子 Agent 事件 %s", event)
                break
            timeout_s = self._timeout_for(spec)
            try:
                call = await self._invoke(spec, data=data, timeout_s=timeout_s)
            except HookError as exc:  # 仅 python hook 的加载/超时错误会走到这里
                call = _CallResult(failed=True, detail=exc.user_message())
            if call.failed:
                policy = self._apply_error_policy(spec, event=event, call=call, state=state)
                if policy is not None:
                    return policy
                if not state.continue_:
                    break
                continue
            if call.outcome is None:
                continue
            stop = self._merge_outcome(spec, event=event, outcome=call.outcome, state=state)
            if stop is not None:
                return stop
        return self._finish(state)

    # ------------------------------------------------------------ 内部：调度
    def _specs_for(self, event: HookEvent, payload: HookPayload) -> list[HookSpec]:
        """挑出该事件应当执行的 hook（应用 ``matcher``）。

        Args:
            event: 事件。
            payload: 事件负载（提供工具名）。

        Returns:
            按配置顺序排列的 hook 列表。
        """
        matched: list[HookSpec] = []
        for spec in self.specs:
            try:
                spec_event = HookEvent(spec.event)
            except ValueError:  # 加载器已过滤；这里再兜底一次
                _LOG.warning("hooks: 未知事件 %r，已跳过", spec.event)
                continue
            if spec_event is not event:
                continue
            if event in TOOL_EVENTS and not self._matches(spec, payload.tool or ""):
                continue
            matched.append(spec)
        return matched

    @staticmethod
    def _matches(spec: HookSpec, tool: str) -> bool:
        """判断 ``matcher`` 正则是否命中工具名（G-6）。

        Args:
            spec: hook 声明。
            tool: 工具名。

        Returns:
            ``True`` 表示应当执行；``matcher`` 为空表示全部匹配，非法正则按全部匹配处理。
        """
        if not spec.matcher:
            return True
        try:
            return re.search(spec.matcher, tool) is not None
        except re.error as exc:
            _LOG.warning("hooks: matcher %r 不是合法正则（%s），按全部匹配处理", spec.matcher, exc)
            return True

    def _timeout_for(self, spec: HookSpec) -> float:
        """计算单次 hook 的超时上限。

        Args:
            spec: hook 声明。

        Returns:
            ``spec.timeout_s``（非正数时回落到
            :data:`~my_agent.config.defaults.HOOK_DEFAULT_TIMEOUT_S`）。
        """
        timeout_s = spec.timeout_s if spec.timeout_s > 0 else HOOK_DEFAULT_TIMEOUT_S
        return float(timeout_s)

    # ------------------------------------------------------------ 内部：调用
    async def _invoke(self, spec: HookSpec, *, data: dict[str, Any], timeout_s: float) -> _CallResult:
        """调用一个 hook（子进程或本地函数）并解析其输出。

        Args:
            spec: hook 声明。
            data: 完整 stdin JSON 结构。
            timeout_s: 超时上限（秒）。

        Returns:
            :class:`_CallResult`。

        Raises:
            HookError: python hook 无法加载，或调用出现不可恢复错误。
        """
        started = time.monotonic()
        if spec.type == "python":
            return await self._invoke_python(spec, data=data, timeout_s=timeout_s, started=started)
        return await self._invoke_command(spec, data=data, timeout_s=timeout_s, started=started)

    async def _invoke_python(
        self,
        spec: HookSpec,
        *,
        data: dict[str, Any],
        timeout_s: float,
        started: float,
    ) -> _CallResult:
        """执行 ``type: python`` 的本地函数 hook（§11.2，仅内置测试用）。

        Args:
            spec: hook 声明（``module`` 必填）。
            data: 完整 stdin JSON 结构。
            timeout_s: 超时上限（秒）。
            started: ``time.monotonic()`` 起点。

        Returns:
            :class:`_CallResult`。

        Raises:
            HookError: ``module`` 缺失或无法解析。
        """
        target = spec.module or ""
        if not target:
            raise HookError("type=python 的 hook 缺少 module", detail="module 缺失")
        elapsed = lambda: int((time.monotonic() - started) * 1000)  # noqa: E731 - 局部小工具
        try:
            raw = await run_python_hook(
                target,
                data,
                cwd=self.cwd,
                project_root=self.project_root,
                timeout_s=timeout_s,
            )
        except TimeoutError:
            return _CallResult(
                timed_out=True,
                failed=True,
                detail=f"python hook {target} 超时（{timeout_s:g}s）",
                duration_ms=elapsed(),
            )
        except HookError:
            raise
        except Exception as exc:  # noqa: BLE001 - hook 自身异常必须被隔离（G-4）
            return _CallResult(
                failed=True,
                detail=f"python hook {target} 抛出 {type(exc).__name__}: {exc}",
                duration_ms=elapsed(),
            )
        outcome, error = self._parse_output(raw)
        if error:
            return _CallResult(failed=True, detail=error, duration_ms=elapsed())
        return _CallResult(outcome=outcome, duration_ms=elapsed())

    async def _invoke_command(
        self,
        spec: HookSpec,
        *,
        data: dict[str, Any],
        timeout_s: float,
        started: float,
    ) -> _CallResult:
        """以子进程方式执行 ``type: command`` 的 hook。

        超时上限 = ``timeout_s + HOOK_TIMEOUT_GRACE_S``；超时后 ``kill()`` 并立即返回，
        保证「崩溃/超时的 hook 绝不阻塞更久」（G-3）。

        Args:
            spec: hook 声明（``command`` 必填）。
            data: 完整 stdin JSON 结构。
            timeout_s: 超时上限（秒，不含宽限）。
            started: ``time.monotonic()`` 起点。

        Returns:
            :class:`_CallResult`。
        """
        budget = timeout_s + HOOK_TIMEOUT_GRACE_S
        deadline = started + budget
        try:
            argv = split_command(spec.command or "")
        except HookError as exc:
            return _CallResult(failed=True, detail=exc.user_message(), duration_ms=self._elapsed(started))
        stdin_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
        env = self._build_env(spec)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.cwd),
                env=env,
                creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except (OSError, ValueError) as exc:
            return _CallResult(
                failed=True,
                detail=f"无法启动 hook 进程 {argv[0]!r}：{exc}",
                duration_ms=self._elapsed(started),
            )

        timed_out = False
        stdout = b""
        stderr = b""
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_bytes), timeout=max(deadline - time.monotonic(), 0.01)
            )
        except TimeoutError:
            timed_out = True
            await self._terminate(proc, deadline=deadline)
        except (OSError, ValueError) as exc:
            await self._terminate(proc, deadline=deadline)
            return _CallResult(
                failed=True,
                detail=f"hook 进程 {argv[0]!r} 通信失败：{exc}",
                duration_ms=self._elapsed(started),
            )

        duration_ms = self._elapsed(started)
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        if timed_out:
            return _CallResult(
                timed_out=True,
                failed=True,
                detail=f"hook {argv[0]!r} 超时（{timeout_s:g}s + {HOOK_TIMEOUT_GRACE_S:g}s 宽限）",
                duration_ms=duration_ms,
            )

        code = proc.returncode if proc.returncode is not None else -1
        if code == 2:  # §11.3：退出码 2 = deny，stderr 作为理由
            reason = stderr_text or "hook 以退出码 2 拒绝（未提供 stderr 理由）"
            return _CallResult(
                outcome=HookOutcome(decision="deny", message=reason), duration_ms=duration_ms
            )
        if code != 0:
            return _CallResult(
                failed=True,
                detail=f"hook {argv[0]!r} 退出码 {code}：{stderr_text or '无 stderr 输出'}",
                duration_ms=duration_ms,
            )

        if not stdout.strip():  # 空输出 = 不干预
            return _CallResult(outcome=None, duration_ms=duration_ms)
        try:
            parsed = json.loads(self._last_json_line(stdout))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return _CallResult(
                failed=True,
                detail=f"hook 输出不是合法 JSON（{exc}）：{self._stdout_preview(stdout)}",
                duration_ms=duration_ms,
            )
        outcome, error = self._parse_output(parsed)
        if error:
            return _CallResult(failed=True, detail=error, duration_ms=duration_ms)
        return _CallResult(outcome=outcome, duration_ms=duration_ms)

    @staticmethod
    def _elapsed(started: float) -> int:
        """返回自 ``started`` 起的毫秒数。

        Args:
            started: ``time.monotonic()`` 起点。

        Returns:
            毫秒整数。
        """
        return int((time.monotonic() - started) * 1000)

    async def _terminate(self, proc: asyncio.subprocess.Process, *, deadline: float) -> None:
        """尽力终止超时的 hook 进程（``terminate`` → 等待 → ``kill``）。

        Args:
            proc: 子进程句柄。
            deadline: 必须返回的时间点（``time.monotonic()`` 标度）。
        """
        try:
            proc.terminate()
        except ProcessLookupError:  # pragma: no cover - 进程已自行退出
            return
        except OSError:  # pragma: no cover - 平台差异（Windows 权限等）
            proc.kill()
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=max(deadline - time.monotonic(), 0.01))
            return
        except TimeoutError:
            pass
        except ProcessLookupError:  # pragma: no cover - 已回收
            return
        try:
            proc.kill()
        except ProcessLookupError:  # pragma: no cover - 已回收
            return
        except OSError:  # pragma: no cover - 平台差异
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=max(deadline - time.monotonic(), 0.01))
        except (TimeoutError, ProcessLookupError):  # pragma: no cover - 交给事件循环回收
            _LOG.debug("hook 进程未在宽限期内回收，交由事件循环回收")

    def _build_env(self, spec: HookSpec) -> dict[str, str]:
        """构造 hook 子进程环境（继承当前环境 + 条目 ``env`` + ``PWD``）。

        Args:
            spec: hook 声明。

        Returns:
            环境变量字典。
        """
        env = dict(os.environ)
        env["PWD"] = str(self.cwd)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.update(spec.env)
        return env

    @staticmethod
    def _last_json_line(stdout: bytes) -> str:
        """取出 stdout 中最后一行非空文本（容忍 hook 先打印日志再输出 JSON）。

        Args:
            stdout: 原始 stdout 字节。

        Returns:
            最后一行非空文本。
        """
        text = stdout.decode("utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def _stdout_preview(stdout: bytes, limit: int = 200) -> str:
        """生成 stdout 预览（用于 JSON 解析失败的告警）。

        Args:
            stdout: 原始 stdout 字节。
            limit: 最大字符数。

        Returns:
            截断后的预览文本。
        """
        text = stdout.decode("utf-8", errors="replace").strip()
        return text[:limit] + ("…" if len(text) > limit else "")

    def _parse_output(self, raw: Any) -> tuple[HookOutcome | None, str]:
        """把 hook 的返回结构解析成 :class:`HookOutcome`。

        Args:
            raw: ``None``（不干预）、``dict``、``str``（JSON）或 :class:`HookOutcome`。

        Returns:
            ``(结果, 错误说明)``；``错误说明`` 非空表示输出非法（按 ``on_error`` 处理）。
        """
        if raw is None:
            return None, ""
        if isinstance(raw, HookOutcome):
            return raw, ""
        payload = raw
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return None, ""
            try:
                payload = json.loads(self._last_json_line(text.encode("utf-8")))
            except json.JSONDecodeError as exc:
                return None, f"hook 返回的字符串不是合法 JSON：{exc}"
        if not isinstance(payload, dict):
            return None, f"hook 返回值必须是对象，实际是 {type(payload).__name__}"

        unknown = sorted(set(payload) - {"decision", "modified_payload", "message", "continue"})
        if unknown:
            _LOG.debug("hooks: 忽略未知输出字段 %s", ", ".join(unknown))

        decision_raw = payload.get("decision")
        decision: Decision | None = None
        if decision_raw in ("allow", "deny", "ask"):
            decision = decision_raw
        elif decision_raw is not None:
            _LOG.warning("hooks: decision=%r 非法（允许 allow/deny/ask），已忽略", decision_raw)

        modified: HookPayload | None = None
        modified_raw = payload.get("modified_payload")
        if isinstance(modified_raw, dict):
            try:
                modified = HookPayload.model_validate(modified_raw)
            except ValueError as exc:
                return None, f"modified_payload 非法：{exc}"
        elif modified_raw is not None:
            return None, f"modified_payload 必须是对象，实际是 {type(modified_raw).__name__}"

        message_raw = payload.get("message", "")
        message = message_raw if isinstance(message_raw, str) else str(message_raw)
        continue_raw = payload.get("continue")
        if continue_raw is None:
            continue_ = True
        elif isinstance(continue_raw, bool):
            continue_ = continue_raw
        else:
            _LOG.warning("hooks: continue=%r 非法（应为布尔），按 true 处理", continue_raw)
            continue_ = True
        return (
            HookOutcome(
                decision=decision,
                modified_payload=modified,
                message=message,
                continue_=continue_,
            ),
            "",
        )

    # ------------------------------------------------------------ 内部：合并
    def _apply_error_policy(
        self,
        spec: HookSpec,
        *,
        event: HookEvent,
        call: _CallResult,
        state: _Accumulator,
    ) -> HookOutcome | None:
        """按 ``on_error`` 处理崩溃/超时的 hook（G-3、G-4）。

        Args:
            spec: 出错的 hook。
            event: 事件（用于文案）。
            call: 调用结果（含失败原因）。
            state: 累积状态（``deny``/``abort`` 会就地生效）。

        Returns:
            需要立即返回的结果（``deny`` / ``abort``）；``allow`` 策略返回 ``None`` 继续执行。
        """
        label = "超时" if call.timed_out else "失败"
        warning = f"hook {label}（{event.value} · {spec.command or spec.module}）：{call.detail}"
        self._warn(warning, spec=spec, event=event, duration_ms=call.duration_ms)
        if spec.on_error == "allow":
            return None
        state.continue_ = False
        state.decision = "deny"
        state.messages.append(f"{warning}；按 on_error={spec.on_error} 处理")
        return self._finish(state)

    def _merge_outcome(
        self,
        spec: HookSpec,
        *,
        event: HookEvent,
        outcome: HookOutcome,
        state: _Accumulator,
    ) -> HookOutcome | None:
        """把单个 hook 的结果并入累积状态（链式）。

        Args:
            spec: 产出该结果的 hook。
            event: 事件。
            outcome: hook 的结果。
            state: 累积状态（就地更新）。

        Returns:
            需要立即返回的结果（``deny`` / ``abort`` 短路）；否则 ``None``。
        """
        source = f"{event.value} · {spec.command or spec.module}"
        if outcome.message:
            state.messages.append(outcome.message)
            self.display.warn(f"[hook] {outcome.message}")

        if outcome.decision == "deny":
            self.display.warn(f"[hook] 拒绝（{event.value}）：{outcome.message or '未说明理由'}")
            state.decision = "deny"
            return self._finish(state)

        if outcome.modified_payload is not None:
            self._apply_modification(
                spec, event=event, modified=outcome.modified_payload, state=state
            )

        if event is HookEvent.AFTER_PERMISSION_CHECK and outcome.decision is not None:
            merged = merge_permission_decision(
                state.payload.decision,
                outcome.decision,
                allow_override=self.config.hooks.allow_permission_override,
            )
            if merged.warning:
                self._warn(merged.warning, spec=spec, event=event, duration_ms=0)
                state.messages.append(merged.warning)
            state.decision = merged.decision
        elif outcome.decision is not None:
            state.decision = outcome.decision

        if not outcome.continue_:
            state.continue_ = False
            state.messages.append(f"hook 要求中止回合（{source}）")
            return self._finish(state)
        return None

    def _apply_modification(
        self,
        spec: HookSpec,
        *,
        event: HookEvent,
        modified: HookPayload,
        state: _Accumulator,
    ) -> None:
        """把 ``modified_payload`` 链式并入当前负载（§11.3、G-2）。

        ``after_*`` 事件的**入参**修改被忽略并告警（§11.3）；只有模型/工具输出
        （``text`` / ``result``，§11.4）在 ``after_model_call`` / ``after_tool_call``
        上可被改写 —— 这两点由 :data:`~my_agent.hooks.events.mutable_fields` 统一表达。

        Args:
            spec: 产出改写的 hook。
            event: 事件。
            modified: hook 提交的 ``modified_payload``。
            state: 累积状态（``payload`` 就地更新）。
        """
        state.touched = True
        allowed = mutable_fields(event)
        if not allowed:
            self._warn(
                f"已忽略 {event.value} 的 modified_payload：该事件不允许修改负载（§11.3）",
                spec=spec,
                event=event,
                duration_ms=0,
            )
            return
        attempted = modified.model_fields_set
        ignored = sorted(attempted - allowed)
        if ignored:
            self._warn(
                f"已忽略 {event.value} 中不可改写的字段：{', '.join(ignored)}（§11.4 能力矩阵）",
                spec=spec,
                event=event,
                duration_ms=0,
            )
        state.payload = state.payload.overlay(modified, allowed)

    def _warn(self, message: str, *, spec: HookSpec, event: HookEvent, duration_ms: int) -> None:
        """同时写日志与显示端口（警告绝不静默）。

        Args:
            message: 告警文案。
            spec: 相关 hook。
            event: 事件。
            duration_ms: 耗时（毫秒）。
        """
        _LOG.warning("hooks: %s", message)
        suffix = f"（{duration_ms}ms）" if duration_ms else ""
        self.display.warn(f"[hook] {message}{suffix}")

    def _finish(self, state: _Accumulator) -> HookOutcome:
        """把累积状态收敛为 :class:`HookOutcome`。

        Args:
            state: 累积状态。

        Returns:
            对外结果；无任何改写时 ``modified_payload`` 为 ``None``。
        """
        return HookOutcome(
            decision=state.decision,
            modified_payload=state.payload if state.touched else None,
            message="\n".join(state.messages),
            continue_=state.continue_,
            aborted=not state.continue_,
        )
