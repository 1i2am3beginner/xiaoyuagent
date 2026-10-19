"""模型层（模块 I，§13）。

对外只暴露三件事：**内部 Provider 协议 + 实现**（:mod:`my_agent.llm.provider`）、
**唯一的业务入口 `LLMClient`**（:mod:`my_agent.llm.client`）与**角色提示词**
（:mod:`my_agent.llm.prompts`，唯一区分手段，§13.2）。

C7 依赖隔离：``openai`` 只允许被 ``llm/provider.py`` import。因此本包的 ``__init__``
**不做任何 eager import**——``from my_agent.llm import LLMClient`` 走 PEP 562
``__getattr__`` 惰性导入，``from my_agent.llm.provider import OpenAISDKClient`` 亦不受影响。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查器与 IDE 使用
    from my_agent.llm.client import AssistantTurn, LLMClient, UsageTracker
    from my_agent.llm.provider import (
        CompletionResult,
        ModelProvider,
        OpenAISDKClient,
        StreamChunk,
        ToolCallDelta,
    )
    from my_agent.llm.tokenizer import (
        HeuristicCounter,
        TiktokenCounter,
        TokenCounter,
        get_counter,
    )

__all__ = [
    "AssistantTurn",
    "CompletionResult",
    "HeuristicCounter",
    "LLMClient",
    "ModelProvider",
    "OpenAISDKClient",
    "StreamChunk",
    "TiktokenCounter",
    "TokenCounter",
    "ToolCallDelta",
    "UsageTracker",
    "get_counter",
]

#: 惰性导出名 → 定义它的子模块。
_LAZY_EXPORTS: dict[str, str] = {
    "AssistantTurn": "my_agent.llm.client",
    "LLMClient": "my_agent.llm.client",
    "UsageTracker": "my_agent.llm.client",
    "CompletionResult": "my_agent.llm.provider",
    "ModelProvider": "my_agent.llm.provider",
    "OpenAISDKClient": "my_agent.llm.provider",
    "StreamChunk": "my_agent.llm.provider",
    "ToolCallDelta": "my_agent.llm.provider",
    "HeuristicCounter": "my_agent.llm.tokenizer",
    "TiktokenCounter": "my_agent.llm.tokenizer",
    "TokenCounter": "my_agent.llm.tokenizer",
    "get_counter": "my_agent.llm.tokenizer",
}


def __getattr__(name: str) -> Any:
    """按需导入子模块符号（PEP 562）。

    Args:
        name: 请求的属性名。

    Returns:
        对应子模块中的对象。

    Raises:
        AttributeError: ``name`` 不是本包导出的符号。
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """列出本包的公开符号（含惰性导出）。

    Returns:
        排序后的属性名列表。
    """
    return sorted({*globals(), *_LAZY_EXPORTS})
