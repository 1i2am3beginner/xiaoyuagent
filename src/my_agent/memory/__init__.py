"""模块 D：Memory（§8）。

写入管线（§8.4）：``extractor`` → ``scope_router`` → ``merger`` → ``store`` → ``writer``；
注入（§8.6）：``injector``；工具（§8.7）：``tools``；范围表唯一权威（§8.2）：``sections``。

本包顶层只定义 :class:`MemoryLLM` 协议与惰性再导出，**不在 import 期加载子模块**，
避免与 ``engine`` / ``tools`` 形成循环导入。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期
    from my_agent.llm.provider import CompletionResult

__all__ = [
    "GLOBAL_SECTIONS",
    "PROJECT_SECTIONS",
    "SECTION_LABELS",
    "MemoryEntry",
    "MemoryLLM",
    "MemoryStore",
    "MemoryWriter",
    "build_block",
    "choose",
    "extract",
    "merge",
]


class MemoryLLM(Protocol):
    """记忆管线需要的最小模型接口（``LLMClient`` 结构化满足）。

    角色（§13.2 冻结）：``memory_extract`` / ``memory_scope`` / ``memory_merge``。
    """

    async def complete_role(self, role: str, *, user: Any, **kwargs: Any) -> CompletionResult:
        """按角色完成一次非流式对话。

        Args:
            role: 角色名（决定 system prompt）。
            user: 用户消息文本。
            **kwargs: 透传给底层客户端的可选参数。

        Returns:
            含 ``text`` 字段的完成结果。
        """
        ...


#: 惰性再导出表：属性名 → (子模块, 属性名)。
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "GLOBAL_SECTIONS": ("my_agent.memory.sections", "GLOBAL_SECTIONS"),
    "PROJECT_SECTIONS": ("my_agent.memory.sections", "PROJECT_SECTIONS"),
    "SECTION_LABELS": ("my_agent.memory.sections", "SECTION_LABELS"),
    "MemoryEntry": ("my_agent.memory.store", "MemoryEntry"),
    "MemoryStore": ("my_agent.memory.store", "MemoryStore"),
    "MemoryWriter": ("my_agent.memory.writer", "MemoryWriter"),
    "build_block": ("my_agent.memory.injector", "build_block"),
    "choose": ("my_agent.memory.scope_router", "choose"),
    "extract": ("my_agent.memory.extractor", "extract"),
    "merge": ("my_agent.memory.merger", "merge"),
}


def __getattr__(name: str) -> Any:
    """按需导入子模块符号（PEP 562）。

    Args:
        name: 属性名。

    Returns:
        目标对象。

    Raises:
        AttributeError: 名称不在 :data:`_LAZY_EXPORTS` 中。
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """列出可用属性（含惰性导出）。

    Returns:
        排序后的属性名列表。
    """
    return sorted({*globals(), *_LAZY_EXPORTS})
