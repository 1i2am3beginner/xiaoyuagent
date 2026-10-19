"""工具调用大结果落盘与预览（§7.4）。

规则（上游）：单条调用结果 > 50k → 写文件保存到本地，只留 2kb 预览。

本模块实现 §7.4 的全部「必须实现的点」：

- 落盘**原子**（写 ``.tmp`` 再 :func:`os.replace`），避免半截文件；
- 落盘失败（磁盘满/权限）→ **降级为强截断**（保头部 + 尾部）并在内容里说明
  「落盘失败，已截断」，**绝不**把全量内容塞回上下文（C-3）；
- 落盘文件登记到 ``SessionState.archive_files``，供 ``/session export`` 与清理；
- 落盘路径固定在 ``<project>/.myagent/artifacts/<session_id>/`` 内
  （**永不在用户源码目录产生文件**）。

所有同步文件 IO 一律经 :func:`asyncio.to_thread` 执行，不得阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from my_agent.config.defaults import (
    LARGE_RESULT_CHAR_THRESHOLD,
    LARGE_RESULT_TOKEN_THRESHOLD,
    PREVIEW_HEAD_TOKENS,
    PREVIEW_TAIL_TOKENS,
    PREVIEW_TOKENS,
)
from my_agent.models.tools import ToolResult

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期导入
    from my_agent.llm.tokenizer import TokenCounter
    from my_agent.models.session import SessionState

__all__ = [
    "ARTIFACTS_DIR_NAME",
    "DEFAULT_ARTIFACT_EXT",
    "MYAGENT_DIR_NAME",
    "PREVIEW_OVERHEAD_TOKENS",
    "TOOL_EXTENSIONS",
    "artifact_path",
    "head_tail_preview",
    "post_tool_result",
]

#: 落盘根目录名（§7.4：``<project>/.myagent/artifacts/<session_id>/``）。
MYAGENT_DIR_NAME: str = ".myagent"
ARTIFACTS_DIR_NAME: str = "artifacts"

#: 预览替换后 ``ToolResult.tokens`` 的固定开销（§7.4 伪码：``PREVIEW_TOKENS + 60``）。
PREVIEW_OVERHEAD_TOKENS: int = 60

#: 按工具选择落盘扩展名（§7.4 ``{ext}``；默认 ``txt``）。
TOOL_EXTENSIONS: dict[str, str] = {
    "Bash": "log",
    "Grep": "txt",
    "Read": "txt",
    "WebFetch": "md",
    "WebSearch": "md",
}
DEFAULT_ARTIFACT_EXT: str = "txt"

#: ``_atomic_write_text`` 的临时文件后缀（§7.4）。
_TMP_SUFFIX: str = ".tmp"

#: 预览精修的轮数上限（只影响拟合速度，不影响语义）。
_PREVIEW_FIT_PASSES: int = 3
#: 预览精修时的收缩比例（拟合步长，非行为阈值）。
_PREVIEW_SHRINK: float = 0.7


def artifact_path(
    artifact_dir: Path,
    *,
    session_id: str,
    turn: int,
    call_id: str,
    tool: str,
) -> Path:
    """构造落盘路径 ``{turn:03d}-{call_id}-{tool}.{ext}``（§7.4）。

    ``artifact_dir`` 若不在 ``.myagent/`` 内，会被视为项目根并补上
    ``.myagent/artifacts/``，保证「永不在用户源码目录产生文件」这一硬约束。

    Args:
        artifact_dir: 落盘目录（``<project>/.myagent/artifacts`` 或项目根）。
        session_id: 会话 id（作为子目录名）。
        turn: 轮次号（用于文件名前缀，3 位零填充）。
        call_id: 工具调用 id。
        tool: 工具名（决定扩展名）。

    Returns:
        目标文件的绝对/相对路径（父目录可能尚不存在）。
    """
    base = _ensure_myagent_dir(Path(artifact_dir))
    ext = TOOL_EXTENSIONS.get(tool, DEFAULT_ARTIFACT_EXT)
    name = f"{max(0, turn):03d}-{_safe_name(call_id)}-{_safe_name(tool)}.{ext}"
    return base / _safe_name(session_id) / name


def head_tail_preview(
    text: str,
    tokens: int = PREVIEW_TOKENS,
    *,
    counter: TokenCounter | None = None,
) -> str:
    """生成「头部 + 尾部」预览，中间插入 ``…[已省略 N 字符]…``（§7.4）。

    Args:
        text: 原始全文。
        tokens: 预览总 token 预算；默认 :data:`PREVIEW_TOKENS`（头 1.5k + 尾 0.5k）。
        counter: token 计数器；``None`` 时用 :func:`my_agent.llm.tokenizer.get_counter`
            取默认实例（保证与其他预算判断同口径）。

    Returns:
        形如 ``<head>\\n\\n…[已省略 N 字符]…\\n\\n<tail>`` 的预览文本。
    """
    if counter is None:
        from my_agent.llm.tokenizer import get_counter

        counter = get_counter(None)
    if tokens <= 0 or not text:
        return ""
    head_tokens = max(1, round(tokens * PREVIEW_HEAD_TOKENS / PREVIEW_TOKENS))
    tail_tokens = max(0, round(tokens * PREVIEW_TAIL_TOKENS / PREVIEW_TOKENS))
    total = max(1, counter.count_text(text))
    chars_per_token = len(text) / total
    head_chars = min(len(text), max(1, int(head_tokens * chars_per_token)))
    tail_chars = min(max(0, len(text) - head_chars), max(0, int(tail_tokens * chars_per_token)))
    head = text[:head_chars]
    tail = text[len(text) - tail_chars :] if tail_chars else ""
    for _ in range(_PREVIEW_FIT_PASSES):
        omitted = len(text) - len(head) - len(tail)
        candidate = _compose(head, tail, omitted)
        if counter.count_text(candidate) <= tokens or (len(head) <= 1 and not tail):
            return candidate
        shrink = _PREVIEW_SHRINK
        head_chars = max(1, int(len(head) * shrink))
        tail_chars = max(0, int(len(tail) * shrink))
        head = text[:head_chars]
        tail = text[len(text) - tail_chars :] if tail_chars else ""
    omitted = len(text) - len(head) - len(tail)
    return _compose(head, tail, omitted)


async def post_tool_result(
    result: ToolResult,
    session: SessionState,
    *,
    artifact_dir: Path,
    counter: TokenCounter,
) -> ToolResult:
    """大结果落盘 + 预览替换，返回**发回模型**的版本（§7.4）。

    小于阈值的结果原样返回（仅在 ``tokens`` 为 0 时补齐估算值）。
    大结果：

    1. 原子写入 ``<artifact_dir>/<session_id>/{turn:03d}-{call_id}-{tool}.{ext}``；
    2. 内容替换为 ``head_tail_preview`` + 「完整输出 … 已保存到 <path>」提示；
    3. 路径登记进 ``SessionState.archive_files``；
    4. 写盘失败 → 退化为强截断（``is_large_result=False``、``artifact_path=None``、
       ``truncated=True``），内容里显式说明「落盘失败」，并附上完整 token 数。

    本函数不修改传入的 ``result``（返回 ``model_copy``），只按规格要求登记
    ``SessionState.archive_files``。

    Args:
        result: 工具执行结果（``tokens`` 为 0 时用 ``counter`` 现算）。
        session: 当前会话状态（用于 session_id / turn_index / archive_files）。
        artifact_dir: 落盘目录（``<project>/.myagent/artifacts`` 或项目根）。
        counter: token 计数器。

    Returns:
        需要回填给模型的结果（大结果 = 预览版）。
    """
    tokens = result.tokens if result.tokens > 0 else counter.count_text(result.content)
    char_threshold = LARGE_RESULT_CHAR_THRESHOLD
    is_large = tokens > LARGE_RESULT_TOKEN_THRESHOLD or (
        char_threshold is not None and len(result.content) > char_threshold
    )
    if not is_large:
        if tokens == result.tokens:
            return result
        return result.model_copy(update={"tokens": tokens})

    preview = head_tail_preview(result.content, PREVIEW_TOKENS, counter=counter)
    target = artifact_path(
        artifact_dir,
        session_id=session.session_id,
        turn=session.turn_index,
        call_id=result.call_id,
        tool=result.tool,
    )
    try:
        await asyncio.to_thread(_atomic_write_text, target, result.content)
    except OSError as exc:
        return result.model_copy(
            update={
                "content": (
                    f"{preview}\n\n[⚠ 落盘失败（{exc.strerror or exc}），已截断："
                    f"完整内容 {tokens} tokens 未写入磁盘，且不会塞回上下文；"
                    f"请缩小调用范围（如加 --limit / 更精确的路径）后重试]"
                ),
                "tokens": PREVIEW_TOKENS + PREVIEW_OVERHEAD_TOKENS,
                "truncated": True,
                "artifact_path": None,
                "is_large_result": False,
            }
        )
    path_text = str(target)
    if path_text not in session.archive_files:
        session.archive_files.append(path_text)
    return result.model_copy(
        update={
            "content": (
                f"{preview}\n\n"
                f"[完整输出 {tokens} tokens 已保存到 {path_text}；"
                f"可用 Read 工具读取该文件，或用 Grep 在文件中检索]"
            ),
            "tokens": PREVIEW_TOKENS + PREVIEW_OVERHEAD_TOKENS,
            "truncated": True,
            "artifact_path": path_text,
            "is_large_result": True,
        }
    )


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _compose(head: str, tail: str, omitted: int) -> str:
    """拼装预览文本：头 + 省略标记 + 尾（§7.4 规定的中插格式）。"""
    marker = f"…[已省略 {max(0, omitted)} 字符]…"
    parts = [part for part in (head.rstrip(), marker, tail.lstrip()) if part]
    return "\n\n".join(parts)


def _ensure_myagent_dir(artifact_dir: Path) -> Path:
    """保证落盘目录位于 ``.myagent/`` 内（§7.4 硬约束）。"""
    if MYAGENT_DIR_NAME in artifact_dir.parts:
        return artifact_dir
    if artifact_dir.name == ARTIFACTS_DIR_NAME:
        return artifact_dir.parent / MYAGENT_DIR_NAME / ARTIFACTS_DIR_NAME
    return artifact_dir / MYAGENT_DIR_NAME / ARTIFACTS_DIR_NAME


def _safe_name(value: str) -> str:
    """把 session_id / call_id / 工具名净化为安全文件名片段。"""
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(value))
    return cleaned.strip("._") or "unknown"


def _atomic_write_text(path: Path, text: str) -> Path:
    """原子写文本：``<path>.tmp`` → :func:`os.replace`（§7.4）。

    Args:
        path: 目标文件路径。
        text: 内容。

    Returns:
        目标文件路径。

    Raises:
        OSError: 目录创建/写入/替换失败（调用方负责降级）。
    """
    tmp = path.with_name(path.name + _TMP_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    return path
