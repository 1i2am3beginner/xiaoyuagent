"""大工具结果裁剪（§7.5）。

规则（上游）：只保留最近 5 个，以前的大工具调用显示 ``Old tool result content cleared``。

精确化规格（§7.5）：

- **作用对象**：仅「大工具结果」（``is_large_result == True``，即曾触发 §7.4 落盘的结果）；
- **保留数**：``KEEP_RECENT_LARGE_RESULTS``（全局，跨工具、按时间序）；
- **替换文本**：``CLEARED_MARKER`` 原样 + 一行引用 ``[原始内容见 <artifact_path>，可用 Read/Grep 回读]``；
- **触发时机**：每次 ``ContextAssembler.build()`` 之前，幂等（已 cleared 的不重复处理）；
- **不删除**：``SessionState.history`` 原文保留（C4），只改「发给模型的视图」（C-5）。

本模块所有函数都是**纯函数**：返回新列表 + 替换项的新 :class:`~my_agent.models.messages.Message`
对象，绝不修改传入的 ``history`` 或其元素。
"""

from __future__ import annotations

import re

from my_agent.config.defaults import CLEARED_MARKER, KEEP_RECENT_LARGE_RESULTS
from my_agent.models.messages import Message

__all__ = ["CLEARED_REFERENCE_TEMPLATE", "prune_large_results"]

#: §7.5 规定的追加引用行模板（``artifact_path`` 来自 ``Message.archive_ref``）。
CLEARED_REFERENCE_TEMPLATE: str = "[原始内容见 {ref}，可用 Read/Grep 回读]"

#: 从已落盘结果文本里兜底提取 artifact 路径（``archive_ref`` 缺失时用）。
_ARTIFACT_RE = re.compile(r"\S*[/\\]\.myagent[/\\]artifacts[/\\]\S+")


def prune_large_results(
    history: list[Message],
    keep: int = KEEP_RECENT_LARGE_RESULTS,
) -> list[Message]:
    """返回**视图**（浅拷贝），把最旧的（超出 ``keep`` 的）大结果替换为清除标记。

    Args:
        history: 会话唯一真相源（**不会被修改**）。
        keep: 保留最近多少条大结果；默认 ``KEEP_RECENT_LARGE_RESULTS``。

    Returns:
        与 ``history`` 等长的新列表：未命中项与原对象同一实例，命中项为
        ``kind="cleared"``、``content`` 以 :data:`CLEARED_MARKER` 开头的新对象。
        对同一输入多次调用结果一致（幂等）。
    """
    view = list(history)
    candidates = [index for index, message in enumerate(history) if message.is_large_result]
    keep_recent = max(0, keep)
    to_clear = candidates[: len(candidates) - keep_recent] if keep_recent else candidates
    for index in to_clear:
        view[index] = _cleared_view(history[index])
    return view


def _cleared_view(message: Message) -> Message:
    """构造一条大结果的「已清除」视图（幂等：已清除的原样返回）。"""
    if message.kind == "cleared" and message.content.startswith(CLEARED_MARKER):
        return message
    reference = message.archive_ref or _extract_artifact_ref(message.content)
    content = CLEARED_MARKER
    if reference:
        content = f"{CLEARED_MARKER}\n{CLEARED_REFERENCE_TEMPLATE.format(ref=reference)}"
    return message.model_copy(
        update={
            "content": content,
            "kind": "cleared",
            "cleared_marker": CLEARED_MARKER,
            # 内容已变更 → 置 0 让 TokenCounter 重新计数（§7.2「内容变更时失效」）。
            "tokens": 0,
        }
    )


def _extract_artifact_ref(content: str) -> str | None:
    """从落盘提示文本中提取 artifact 路径（``archive_ref`` 缺失时的兜底）。"""
    match = _ARTIFACT_RE.search(content)
    if match is None:
        return None
    return match.group(0).strip("[]（）()；;，,。：:")
