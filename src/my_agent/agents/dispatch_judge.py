"""任务分发审判 Agent（§12 扩展 / 【待确认假设 U35】）。

**定位**：在 ``Task`` 批次真正执行之前，由独立的「审判角色」判断主 Agent 的分发计划是否合理。
它与权限流水线的 ``judge``（§6.5）是**两个不同的东西**：

============================ ==========================================
权限判定 judge                 分发审判 dispatch_judge
============================ ==========================================
判「这个调用**安不安全**」        判「这份分发计划**合不合理**」
输出 allow/deny/ask            输出 reasonable / too_coarse / too_fine
拒绝 → 不执行                  过粗 → 审判器**亲自补发**；过细 → 返回补丁
============================ ==========================================

## 输入：最小信息集（用户明确要求）

**不打包整个会话历史**，只给四样东西（:class:`JudgmentContext`）：

1. ``user_requirement`` —— 原始用户需求（本轮用户输入）
2. ``subtasks`` —— 拟分发的子任务描述
3. ``constraints`` —— 已知约束/验收标准（项目红线 + 本轮用户输入）
4. ``file_paths`` —— 相关文件/代码的**路径清单**

第 4 项**不含文件内容**：文件由审判器**自己读**（:func:`read_file_excerpts`），
读到的摘录只进审判提示词，**永不进入主 Agent 的上下文**。

> 实现说明：这里用**受上限约束的只读直读**（不经过工具注册表），原因是若走工具层，
> ``default`` 模式下每读一个文件都会弹权限询问，审判就无法无感进行；
> 直读同为只读、且有行数与字节上限（:data:`DISPATCH_JUDGE_MAX_FILES` /
> :data:`DISPATCH_JUDGE_FILE_EXCERPT_CHARS`），安全面等价。

## 两条派生路径（用户明确要求）

- **过粗**（遗漏必要工作）→ 审判器**亲自派生**补发的子任务（``_dispatch_missing``），
  结论以 ``[审判 Agent 补发的子任务结论]`` 形式注入主上下文。
- **过细**（拆得太碎/重复）→ 审判器**不派生**，只返回结论 / 补丁 / 影响分析 / 理由摘要，
  由主 Agent 简单验证后**重新分发**。

## 审判模型不可用时（用户明确要求的行为）

**先按指数退避重试**（:data:`DISPATCH_JUDGE_BACKOFF_S` = 1s → 2s → 4s，
最多 :data:`DISPATCH_JUDGE_MAX_RETRIES` 次）；仍失败则
**不做任何处理（放行该批分发）**，并且**必须显式告诉用户「审判模型出现故障」**
（:meth:`DispatchJudge.review` 调用 ``display.warn``）。即 fail-open，但**不静默失效**。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from my_agent.config.defaults import (
    DISPATCH_JUDGE_BACKOFF_S,
    DISPATCH_JUDGE_FILE_EXCERPT_CHARS,
    DISPATCH_JUDGE_MAX_FILES,
    DISPATCH_JUDGE_MAX_OUTPUT_CHARS,
    DISPATCH_JUDGE_MAX_RETRIES,
    DISPATCH_JUDGE_TIMEOUT_S,
)
from my_agent.errors import MyAgentError
from my_agent.llm.prompts import DISPATCH_JUDGE_USER_TEMPLATE, render
from my_agent.models.permission import Confidence
from my_agent.models.protocols import DisplayPort, NullDisplay

if TYPE_CHECKING:
    from my_agent.agents.subagent import SubagentRunner
    from my_agent.config.schema import Config
    from my_agent.llm.client import LLMClient
    from my_agent.models.session import SessionState

__all__ = [
    "DISPATCH_JUDGE_ROLE",
    "DispatchJudge",
    "DispatchJudgment",
    "DispatchVerdict",
    "JudgmentContext",
    "MissingDispatch",
    "SubtaskSpec",
    "read_file_excerpts",
]

_LOG = logging.getLogger(__name__)

#: 审判角色名（对应 ``llm/prompts.py::ROLE_PROMPTS["dispatch_judge"]``）。
DISPATCH_JUDGE_ROLE = "dispatch_judge"

#: 补发结论注入主上下文时的前缀（唯一产生点，便于测试断言）。
SUPPLEMENT_MARKER = "[审判 Agent 补发的子任务结论]"

#: 审判模型彻底不可用时的用户告警文案（必须显式告知，不静默失效）。
JUDGE_DOWN_WARNING = (
    "⚠ 任务分发审判模型出现故障（已按指数退避重试 {retries} 次），本次不做审判、放行分发。"
    "若反复出现请检查模型配置或网络。"
)

#: 从子任务描述里提取「像路径」的片段（只取路径，不取内容）。
#: 同时支持**裸文件名**（``auth.py``）与多级路径（``src/my_agent/auth.py``）；
#: 扩展名必须以字母开头，避免把 ``v1.2`` 之类版本号当成文件。
_PATH_TOKEN_RE = re.compile(
    r"(?:[A-Za-z]:)?[\\/]?(?:[\w.\-@]+[\\/])*[\w\-@]+\.[A-Za-z][A-Za-z0-9]{0,7}\b"
)


class DispatchVerdict(StrEnum):
    """分发审判的三种裁决。"""

    REASONABLE = "reasonable"
    """分发合理 → 原计划照常执行。"""

    TOO_COARSE = "too_coarse"
    """分发过粗（遗漏必要工作）→ 审判器亲自补发。"""

    TOO_FINE = "too_fine"
    """分发过细（拆得太碎/重复）→ 返回补丁，由主 Agent 重新分发。"""


class SubtaskSpec(BaseModel):
    """一个子任务的规格（与 ``Task`` 工具参数对齐，可直接用于派生）。"""

    description: str = Field(description="3-5 词的短标题")
    prompt: str = Field(description="完整、自包含的任务说明")
    agent_type: str = Field(default="general", description="general / explore / review")

    model_config = {"extra": "ignore"}


class DispatchJudgment(BaseModel):
    """审判输出（严格 JSON 契约）。"""

    verdict: DispatchVerdict
    confidence: Confidence = "medium"
    reason: str = ""
    missing_tasks: list[SubtaskSpec] = Field(
        default_factory=list, description="TOO_COARSE 时应补发的子任务"
    )
    merged_into: int | None = Field(default=None, description="TOO_FINE 时应合并为几个任务")
    patch: list[SubtaskSpec] = Field(
        default_factory=list, description="TOO_FINE 时修订后的任务列表（可直接执行）"
    )
    impact: list[str] = Field(default_factory=list, description="影响分析：涉及的文件/模块")
    judge_error: str | None = Field(
        default=None, description="非空表示审判模型不可用、本判决为放行占位"
    )

    model_config = {"extra": "ignore"}

    @property
    def is_fallback(self) -> bool:
        """是否为「审判不可用」的放行占位判决。

        Returns:
            ``judge_error`` 非空时 ``True``。
        """
        return self.judge_error is not None

    def feedback_text(self) -> str:
        """生成回填给主 Agent 的结构化反馈（TOO_FINE 场景）。

        Returns:
            含结论、补丁、影响分析与理由的文本；主 Agent 据此重新分发。
        """
        lines = [f"任务分发被判为**过细**：{self.reason or '（未给出理由）'}"]
        if self.merged_into is not None:
            lines.append(f"- 建议合并为：{self.merged_into} 个子任务")
        if self.impact:
            lines.append(f"- 影响范围：{', '.join(self.impact)}")
        if self.patch:
            lines.append("- 修订后的任务列表（请据此重新分发）：")
            for index, task in enumerate(self.patch, start=1):
                lines.append(
                    f"  {index}. [{task.agent_type}] {task.description}\n     {task.prompt}"
                )
        else:
            lines.append("- 请自行合并重复任务后重新分发。")
        lines.append("请用修订后的任务重新调用 Task（同一回合不再重复审判）。")
        return "\n".join(lines)


class JudgmentContext(BaseModel):
    """审判器的**最小信息集**（不含完整会话历史）。"""

    user_requirement: str = Field(description="原始用户需求")
    subtasks: list[SubtaskSpec] = Field(default_factory=list, description="拟分发的子任务")
    constraints: str = Field(default="", description="已知约束/验收标准")
    file_paths: list[str] = Field(default_factory=list, description="相关文件路径（不含内容）")
    file_excerpts: dict[str, str] = Field(
        default_factory=dict, description="审判器自行读取的文件摘录（永不进入主上下文）"
    )

    model_config = {"extra": "ignore"}


class MissingDispatch(BaseModel):
    """补发结果（审判器亲自派生的补发任务结论）。"""

    description: str
    ok: bool
    text: str = ""
    error: str | None = None

    model_config = {"extra": "ignore"}


def read_file_excerpts(
    file_paths: list[str],
    *,
    root: Path,
    max_files: int = DISPATCH_JUDGE_MAX_FILES,
    max_chars: int = DISPATCH_JUDGE_FILE_EXCERPT_CHARS,
) -> dict[str, str]:
    """由审判器**自己**读取相关文件（只读、限量、带截断）。

    读到的内容只交给审判器，**绝不写入主 Agent 的 history**。

    Args:
        file_paths: 相关文件路径清单（相对 ``root`` 或绝对路径）。
        root: 解析相对路径的基准目录（项目根）。
        max_files: 最多读取的文件数。
        max_chars: 单文件最大字符数（超出截断并标注）。

    Returns:
        路径 → 摘录；读取失败的文件不出现在结果里（审判器会看到「未读到」）。
    """
    excerpts: dict[str, str] = {}
    for raw in list(dict.fromkeys(file_paths))[:max_files]:
        candidate = Path(raw)
        path = candidate if candidate.is_absolute() else (root / candidate)
        try:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > max_chars:
            text = f"{text[:max_chars]}\n…[已截断，原文件更长]"
        excerpts[raw] = text
    return excerpts


def extract_paths(text: str, *, limit: int = DISPATCH_JUDGE_MAX_FILES) -> list[str]:
    """从子任务描述中提取「像路径」的片段（主 Agent 只传路径，不传内容）。

    Args:
        text: 子任务描述拼接文本。
        limit: 最多返回的路径数。

    Returns:
        去重后的路径列表。
    """
    found = [match.group(0) for match in _PATH_TOKEN_RE.finditer(text)]
    return list(dict.fromkeys(found))[:limit]


class DispatchJudge:
    """任务分发审判器。"""

    def __init__(
        self,
        *,
        llm: LLMClient,
        config: Config,
        display: DisplayPort | None = None,
        runner: SubagentRunner | None = None,
        project_root: Path | None = None,
        timeout_s: float = DISPATCH_JUDGE_TIMEOUT_S,
        max_retries: int = DISPATCH_JUDGE_MAX_RETRIES,
        schedule: tuple[float, ...] = DISPATCH_JUDGE_BACKOFF_S,
        sleep: Any = None,
    ) -> None:
        """初始化。

        Args:
            llm: 模型客户端（审判走独立角色 ``dispatch_judge``）。
            config: 生效配置。
            display: 输出端口（告警用户用）。
            runner: 子 Agent 运行时——**过粗时由审判器亲自补发**需要它；缺省则无法补发。
            project_root: 相对路径解析基准。
            timeout_s: 单次审判超时。
            max_retries: 失败重试次数。
            schedule: 指数退避秒数序列。
            sleep: 可注入的 sleep（测试不真实等待）。
        """
        self.llm = llm
        self.config = config
        self.display: DisplayPort = display or NullDisplay()
        self.runner = runner
        self.project_root = Path(project_root or Path.cwd())
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.schedule = tuple(schedule)
        self._sleep = sleep or asyncio.sleep
        self.judge_degraded = False
        """最近一次审判是否因模型不可用而降级（供测试与状态展示）。"""

    # ------------------------------------------------------------------ 主入口
    async def review(
        self,
        context: JudgmentContext,
        *,
        parent: SessionState | None = None,
    ) -> tuple[DispatchJudgment, list[MissingDispatch]]:
        """审判一份分发计划。

        Args:
            context: 最小信息集（见 :class:`JudgmentContext`）。
            parent: 父会话（判为过粗时，补发的子 Agent 需要继承其安全模式）。

        Returns:
            ``(判决, 补发结果列表)``。判决为 ``TOO_COARSE`` 且补发成功时，第二项非空；
            审判模型彻底不可用时返回 :meth:`_fallback` 的放行判决且第二项为空。
        """
        prompt = self._build_prompt(context)
        try:
            judgment = await self._call_with_backoff(prompt)
        except MyAgentError as exc:
            self.judge_degraded = True
            self.display.warn(
                JUDGE_DOWN_WARNING.format(retries=self.max_retries) + f"（原因：{exc.user_message()}）"
            )
            return self._fallback(str(exc)), []

        self.judge_degraded = False
        if judgment.verdict is not DispatchVerdict.TOO_COARSE:
            return judgment, []

        if self.runner is None:
            _LOG.warning("判为过粗但未注入 SubagentRunner，无法补发；按原计划执行")
            return judgment, []
        supplements = await self._dispatch_missing(judgment.missing_tasks, parent=parent)
        return judgment, supplements

    # ------------------------------------------------------------------ 提示词
    def _build_prompt(self, context: JudgmentContext) -> str:
        """按最小信息集渲染审判提示词。

        Args:
            context: 最小信息集。

        Returns:
            用户消息文本（system prompt 由角色 ``dispatch_judge`` 提供）。
        """
        subtasks = "\n".join(
            f"{index}. [{task.agent_type}] {task.description}\n   {task.prompt[:DISPATCH_JUDGE_MAX_OUTPUT_CHARS]}"
            for index, task in enumerate(context.subtasks, start=1)
        ) or "（无：主 Agent 尚未分发任何子任务）"
        paths = "\n".join(f"- {path}" for path in context.file_paths) or "（无）"
        excerpts = "\n\n".join(
            f"--- {path} ---\n{text}" for path, text in context.file_excerpts.items()
        ) or "（无）"
        return render(
            DISPATCH_JUDGE_USER_TEMPLATE,
            user_requirement=context.user_requirement or "（未提供）",
            subtask_count=len(context.subtasks),
            subtasks=subtasks,
            constraints=context.constraints or "（未提供）",
            file_paths=paths,
            file_excerpts=excerpts,
        )

    # ------------------------------------------------------------------ 调用
    async def _call_with_backoff(self, prompt: str) -> DispatchJudgment:
        """调用审判模型，失败按指数退避重试。

        Args:
            prompt: 审判用户消息。

        Returns:
            解析好的 :class:`DispatchJudgment`。

        Raises:
            MyAgentError: 重试耗尽仍失败（调用方据此走 fail-open 放行）。
        """
        last: MyAgentError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self.llm.complete_role(
                        DISPATCH_JUDGE_ROLE,
                        user=prompt,
                        timeout_s=self.timeout_s,
                    ),
                    timeout=self.timeout_s + 1.0,
                )
                return self._parse(getattr(result, "text", "") or "")
            except (MyAgentError, TimeoutError) as exc:
                last = exc if isinstance(exc, MyAgentError) else MyAgentError(str(exc))
                if attempt >= self.max_retries:
                    break
                delay = self.schedule[min(attempt, len(self.schedule) - 1)]
                _LOG.warning(
                    "审判调用第 %d 次失败（%.1fs 后重试）：%s", attempt + 1, delay, last
                )
                await self._sleep(delay)
        raise last or MyAgentError("审判调用失败", detail="unknown")

    def _parse(self, text: str) -> DispatchJudgment:
        """解析审判输出（严格 JSON，失败则抛错以触发重试）。

        Args:
            text: 模型输出。

        Returns:
            :class:`DispatchJudgment`。

        Raises:
            MyAgentError: 无法解析为合法判决（调用方会重试）。
        """
        payload = _extract_json_object(text)
        if payload is None:
            raise MyAgentError("审判输出不是合法 JSON", detail=text[:200])
        raw_verdict = str(payload.get("verdict") or "").strip().lower()
        try:
            verdict = DispatchVerdict(raw_verdict)
        except ValueError as exc:
            raise MyAgentError(
                f"审判 verdict 非法：{raw_verdict}", detail=raw_verdict
            ) from exc
        confidence = str(payload.get("confidence") or "medium").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        return DispatchJudgment(
            verdict=verdict,
            confidence=confidence,  # type: ignore[arg-type]
            reason=str(payload.get("reason") or "")[:500],
            missing_tasks=_specs(payload.get("missing_tasks")),
            merged_into=_as_int(payload.get("merged_into")),
            patch=_specs(payload.get("patch")),
            impact=[str(item) for item in (payload.get("impact") or []) if item],
        )

    # ------------------------------------------------------------------ 补发
    async def _dispatch_missing(
        self, tasks: list[SubtaskSpec], *, parent: SessionState | None = None
    ) -> list[MissingDispatch]:
        """**过粗**路径：审判器亲自派生补发的子任务（第二条派生路径）。

        Args:
            tasks: 待补发的子任务。
            parent: 父会话（继承安全模式）。

        Returns:
            每个补发任务的结论。
        """
        results: list[MissingDispatch] = []
        if self.runner is None or not tasks:
            return results
        for task in tasks:
            try:
                outcome = await self.runner.spawn(
                    description=task.description,
                    prompt=task.prompt,
                    agent_type=task.agent_type,
                    run_in_background=False,
                    parent=parent,
                    parent_call_id="dispatch-judge",
                )
            except MyAgentError as exc:
                results.append(
                    MissingDispatch(
                        description=task.description, ok=False, error=exc.user_message()
                    )
                )
                continue
            results.append(
                MissingDispatch(
                    description=task.description, ok=outcome.ok, text=outcome.text,
                    error=outcome.error,
                )
            )
        return results

    # ------------------------------------------------------------------ 降级
    def _fallback(self, error: str) -> DispatchJudgment:
        """审判不可用时的放行占位判决（fail-open，但已告警用户）。

        Args:
            error: 失败原因。

        Returns:
            带 ``judge_error`` 的放行判决。
        """
        return DispatchJudgment(
            verdict=DispatchVerdict.REASONABLE,
            confidence="low",
            reason="审判模型不可用，按放行处理",
            judge_error=error,
        )


def supplement_text(supplements: list[MissingDispatch]) -> str:
    """把补发结论渲染成注入主上下文的文本块。

    Args:
        supplements: 补发结果。

    Returns:
        以 :data:`SUPPLEMENT_MARKER` 开头的文本；无结果时返回空串。
    """
    if not supplements:
        return ""
    lines = [SUPPLEMENT_MARKER, "（审判 Agent 判定原分发过粗，已代为补发以下子任务）", ""]
    for item in supplements:
        status = "完成" if item.ok else f"失败：{item.error or '未知原因'}"
        lines.append(f"## {item.description} —— {status}")
        if item.text:
            lines.append(item.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def build_context(
    *,
    user_requirement: str,
    subtasks: list[SubtaskSpec],
    constraints: str = "",
    file_paths: list[str] | None = None,
    root: Path,
    read_files: bool = True,
) -> JudgmentContext:
    """构造审判用的最小信息集（主 Agent 侧只传路径清单）。

    Args:
        user_requirement: 原始用户需求。
        subtasks: 拟分发的子任务。
        constraints: 已知约束/验收标准。
        file_paths: 相关文件路径；``None`` = 从子任务描述里自动提取。
        root: 相对路径解析基准。
        read_files: 是否由审判器自行读取文件摘录。

    Returns:
        :class:`JudgmentContext`。
    """
    paths = (
        list(dict.fromkeys(file_paths))
        if file_paths is not None
        else extract_paths("\n".join(task.prompt for task in subtasks))
    )
    excerpts = read_file_excerpts(paths, root=root) if read_files else {}
    return JudgmentContext(
        user_requirement=user_requirement,
        subtasks=subtasks,
        constraints=constraints,
        file_paths=paths,
        file_excerpts=excerpts,
    )


def _specs(value: Any) -> list[SubtaskSpec]:
    """把 JSON 里的任务数组规整成 :class:`SubtaskSpec` 列表（跳过非法项）。

    Args:
        value: 原始值（通常为 list）。

    Returns:
        合法的子任务列表。
    """
    out: list[SubtaskSpec] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not description or not prompt:
            continue
        agent_type = str(item.get("agent_type") or "general").strip()
        if agent_type not in {"general", "explore", "review"}:
            agent_type = "general"
        out.append(SubtaskSpec(description=description, prompt=prompt, agent_type=agent_type))
    return out


def _as_int(value: Any) -> int | None:
    """把可选值转成整数或 ``None``。

    Args:
        value: 原始值。

    Returns:
        整数或 ``None``。
    """
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出中提取第一个平衡的 JSON 对象（容忍前后噪声）。

    Args:
        text: 模型输出。

    Returns:
        解析出的字典；失败返回 ``None``。
    """
    stripped = text.strip()
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
