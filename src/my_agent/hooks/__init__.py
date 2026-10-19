"""模块 G：Hook（§11）。

对外入口：

* :class:`~my_agent.hooks.events.HookEvent` / :class:`~my_agent.hooks.events.HookPayload` /
  :class:`~my_agent.hooks.events.HookOutcome` —— 数据契约。
* :func:`~my_agent.hooks.loader.load_hooks` —— ``hooks.yaml`` 加载（§11.2）。
* :class:`~my_agent.hooks.runner.HookRunner` —— 串行链式执行器（§11.3）。
* :func:`~my_agent.hooks.runner.merge_permission_decision` —— 裁决合并纯函数
  （§11.4 安全不变量，G-5）。
* :func:`~my_agent.hooks.python_hook.load_callable` —— ``type: python`` 本地函数加载
  （仅内置测试用）。
"""

from __future__ import annotations

from my_agent.hooks.events import (
    EDITABLE_EVENTS,
    OUTPUT_EVENTS,
    SPEC_EVENTS,
    TOOL_EVENTS,
    HookEvent,
    HookOutcome,
    HookPayload,
    mutable_fields,
)
from my_agent.hooks.loader import expand_env, expand_mapping, load_hooks, read_hooks_file
from my_agent.hooks.python_hook import DEFAULT_FUNCTION, load_callable, run_python_hook
from my_agent.hooks.runner import (
    HookDecisionMerge,
    HookRunner,
    merge_permission_decision,
    split_command,
)

__all__ = [
    "DEFAULT_FUNCTION",
    "EDITABLE_EVENTS",
    "HookDecisionMerge",
    "HookEvent",
    "HookOutcome",
    "HookPayload",
    "HookRunner",
    "OUTPUT_EVENTS",
    "SPEC_EVENTS",
    "TOOL_EVENTS",
    "expand_env",
    "expand_mapping",
    "load_callable",
    "load_hooks",
    "merge_permission_decision",
    "mutable_fields",
    "read_hooks_file",
    "run_python_hook",
    "split_command",
]
