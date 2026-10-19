"""异常体系（附录 A）。

所有自定义异常继承 :class:`MyAgentError`；每个异常带 ``code``（机器可读）与
``user_message()``（用户可见文案模板）。业务代码**禁止**抛出裸 ``Exception``。

对应规格：``my_agent-开发文档.md`` 附录 A。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ArchiveCorruptedError",
    "ConfigError",
    "ContextOverflowError",
    "EmptyModelResponseError",
    "HookError",
    "JudgeParseError",
    "LLMRequestError",
    "LLMUnavailableError",
    "McpTimeoutError",
    "McpValidationError",
    "MemoryLockError",
    "ModelUnavailableError",
    "MyAgentError",
    "PermanentGrantNotAllowedError",
    "PermissionBypassedError",
    "SkillValidationError",
    "SubagentLimitError",
    "ToolCallParseError",
    "ToolExecutionError",
    "ToolNameConflictError",
    "ToolTimeoutError",
    "UserAbortError",
]


class MyAgentError(Exception):
    """所有 myagent 自定义异常的基类。

    Args:
        *args: 传给 ``Exception`` 的位置参数（通常是一条人类可读消息）。
        detail: 结构化补充信息（如环境变量名、超时秒数），供 ``user_message()`` 拼装。
        **kw: 额外上下文键值，存放在 :attr:`context` 中供审计使用。

    Attributes:
        code: 机器可读错误码。
        detail: 结构化细节。
        context: 任意附加上下文。
    """

    code: str = "myagent_error"
    _TEMPLATE: str = "{detail}"

    def __init__(self, *args: Any, detail: str = "", **kw: Any) -> None:
        if args and not detail:
            detail = str(args[0])
        super().__init__(*args or (detail,))
        self.detail: str = detail
        self.context: dict[str, Any] = dict(kw)
        for key, value in kw.items():
            setattr(self, key, value)

    def user_message(self) -> str:
        """返回给用户看的文案（附录 A 模板）。

        Returns:
            已按 :attr:`detail` 渲染的文案。
        """
        try:
            return self._TEMPLATE.format(detail=self.detail, **self.context)
        except (KeyError, IndexError):
            return self.detail or self.code

    def __str__(self) -> str:  # pragma: no cover - 简单委托
        return self.user_message()


class ConfigError(MyAgentError):
    """配置缺失或非法（如 API Key 未设置、401/403 鉴权失败）。"""

    code = "config_error"
    _TEMPLATE = "配置错误：{detail}"


class LLMUnavailableError(MyAgentError):
    """主对话模型不可用（重试耗尽）。"""

    code = "llm_unavailable"
    _TEMPLATE = "模型不可用：{detail}。请检查网络/密钥，或稍后重试"


class ModelUnavailableError(MyAgentError):
    """判定/小任务模型不可用（超时、5xx、429、解析失败超限）。

    §6.5.3 / U14：所有等价故障统一收敛到本异常，便于一键调整降级路径。
    """

    code = "model_unavailable"
    _TEMPLATE = "⚠ 判定模型不可用（{detail}），已按最保守策略转人工确认"


class JudgeParseError(MyAgentError):
    """判定输出非法 JSON（重试后仍失败）。"""

    code = "judge_parse_error"
    _TEMPLATE = "判定输出无法解析（已重试 {retries} 次），转人工确认"


class LLMRequestError(MyAgentError):
    """请求本身非法（400：参数/schema 错误），**不重试**。"""

    code = "llm_request_error"
    _TEMPLATE = "模型请求被拒绝：{detail}"


class PermissionBypassedError(MyAgentError):
    """工具未经权限流水线执行（C1 内部契约违规）。"""

    code = "permission_bypassed"
    _TEMPLATE = "内部错误：工具 {name} 未经权限检查，已阻止执行（请报告此问题）"


class PermanentGrantNotAllowedError(MyAgentError):
    """对不可永久放行的操作尝试写入缓存（§6.3、B-5）。"""

    code = "permanent_grant_not_allowed"
    _TEMPLATE = "该操作在任何模式下都需要逐次确认，无法永久放行"


class ToolNameConflictError(MyAgentError):
    """工具重名注册。"""

    code = "tool_name_conflict"
    _TEMPLATE = "工具名冲突：{name} 已存在（来源 {source}）"


class ToolExecutionError(MyAgentError):
    """工具执行失败。"""

    code = "tool_execution_error"
    _TEMPLATE = "工具 {name} 执行失败：{detail}"


class ToolTimeoutError(MyAgentError):
    """工具超时。"""

    code = "tool_timeout"
    _TEMPLATE = "工具 {name} 超时（{timeout}s）已终止"


class ToolCallParseError(MyAgentError):
    """模型给出的工具调用参数不是合法 JSON（保留 raw 串供自纠）。"""

    code = "tool_call_parse_error"
    _TEMPLATE = "工具 {name} 的参数不是合法 JSON：{detail}"


class McpValidationError(MyAgentError):
    """MCP 配置校验失败。"""

    code = "mcp_validation_error"
    _TEMPLATE = "MCP 配置校验失败：\n{detail}"


class McpTimeoutError(MyAgentError):
    """MCP 调用超时。"""

    code = "mcp_timeout"
    _TEMPLATE = "MCP 服务 {server} 响应超时（{timeout}s）"


class MemoryLockError(MyAgentError):
    """记忆文件锁冲突。"""

    code = "memory_lock_error"
    _TEMPLATE = "记忆文件被占用（另一进程正在写入），请稍后重试"


class ContextOverflowError(MyAgentError):
    """压缩 + 硬裁剪后仍超预算。"""

    code = "context_overflow"
    _TEMPLATE = "上下文已超出模型窗口，无法继续；请 /clear 或 /compact 后重试"


class ArchiveCorruptedError(MyAgentError):
    """归档文件损坏。"""

    code = "archive_corrupted"
    _TEMPLATE = "归档文件损坏：{path}；该段历史无法回读"


class SkillValidationError(MyAgentError):
    """SKILL.md frontmatter 非法。"""

    code = "skill_validation_error"
    _TEMPLATE = "技能 {name} 定义非法：{detail}"


class SubagentLimitError(MyAgentError):
    """子 Agent 超并发/深度限制。"""

    code = "subagent_limit"
    _TEMPLATE = "子 Agent 已达上限（并发 {n} / 深度 {d}）"


class UserAbortError(MyAgentError):
    """用户中止回合。"""

    code = "user_abort"
    _TEMPLATE = "已中止（用户请求）"


class EmptyModelResponseError(MyAgentError):
    """模型返回空内容且无 tool_calls（§5.3.4 B12）。"""

    code = "empty_model_response"
    _TEMPLATE = "模型返回空响应（已重试 {retries} 次）"


class HookError(MyAgentError):
    """hook 执行失败或返回非法结构（§11）。"""

    code = "hook_error"
    _TEMPLATE = "hook 执行失败：{detail}"
