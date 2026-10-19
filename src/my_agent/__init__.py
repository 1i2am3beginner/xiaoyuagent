"""my_agent：本地运行的编程 CLI Agent。

双层 Agent-Loop（外层 QueryEngine + 内层 ReAct Query）配合一条四阶段权限流水线
（硬拦截 → 前置修正 → 大模型判定 → 后置修正），保证每一次工具调用可控、可审计、可解释。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
