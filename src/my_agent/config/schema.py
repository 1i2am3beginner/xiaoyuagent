"""配置 Schema（§15.2、§15.3）。

pydantic v2 模型，覆盖 §15.3 默认值表的每一个键。**这是跨模块配置契约**：
所有模块通过 ``Config`` 读取配置，禁止自行读文件。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from my_agent.config import defaults as D
from my_agent.errors import ConfigError
from my_agent.models.permission import SafetyMode

__all__ = [
    "AgentsConfig",
    "Config",
    "ContextConfig",
    "EndpointConfig",
    "HooksConfig",
    "HookSpec",
    "LLMConfig",
    "McpConfig",
    "McpServerConfig",
    "MemoryConfig",
    "Rule",
    "SafetyConfig",
    "SkillsConfig",
    "UIConfig",
]


class _Base(BaseModel):
    """本项目配置模型的公共基类（忽略未知键，便于向后兼容）。"""

    model_config = ConfigDict(extra="ignore")


class Rule(_Base):
    """一条安全规则（正则 + 理由）。"""

    pattern: str = Field(description="正则表达式（不可逆操作为不区分大小写）")
    reason: str = Field(default="", description="展示给用户的理由")
    builtin: bool = Field(default=False, description="是否内置规则（内置项不可被删除）")


class EndpointConfig(_Base):
    """一个 OpenAI 兼容端点的连接配置（§13.1.3）。

    ``api_key_env`` **只存环境变量名，从不存密钥**（§2.2 禁止硬编码 API Key）。
    """

    name: str = Field(default="main", description="逻辑名，如 main")
    base_url: str = Field(default="", description="任意 OpenAI 兼容端点")
    api_key_env: str = Field(default="", description="环境变量名，如 MY_API_KEY")
    model: str = Field(default="", description="模型名（不硬编码在代码里）")
    context_window: int = Field(default=D.MODEL_WINDOW_DEFAULT, gt=0)
    max_output_tokens: int = Field(default=D.MAX_OUTPUT_TOKENS_DEFAULT, gt=0)
    supports_tools: bool = True
    supports_stream: bool = True
    extra_headers: dict[str, str] = Field(default_factory=dict)


class LLMConfig(_Base):
    """模型层配置（§13.4）。"""

    endpoint: str = Field(default="main", description="使用的端点名")
    endpoints: dict[str, EndpointConfig] = Field(default_factory=dict)
    context_window: int | None = Field(default=None, gt=0, description="覆盖端点窗口")
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float = D.MAIN_TEMPERATURE
    judge_temperature: float = D.JUDGE_TEMPERATURE
    log_dir: str | None = Field(default=None, description="LLM 日志目录（默认 ~/.myagent/logs）")

    def resolved_endpoint(self) -> EndpointConfig:
        """解析出实际使用的端点配置。

        Returns:
            生效的 :class:`EndpointConfig`（叠加顶层覆盖项）。

        Raises:
            ConfigError: 端点未配置（附配置指引）。
        """
        cfg = self.endpoints.get(self.endpoint)
        if cfg is None:
            hint = (
                f"未找到端点 {self.endpoint!r}。请在 ~/.myagent/config.toml 中配置 "
                f"[llm.endpoints.{self.endpoint}]（base_url / api_key_env / model）"
            )
            raise ConfigError(hint, detail=hint)
        if self.context_window is not None:
            cfg = cfg.model_copy(update={"context_window": self.context_window})
        if self.max_output_tokens is not None:
            cfg = cfg.model_copy(update={"max_output_tokens": self.max_output_tokens})
        if not cfg.base_url or not cfg.api_key_env or not cfg.model:
            missing = [
                key
                for key, value in (
                    ("base_url", cfg.base_url),
                    ("api_key_env", cfg.api_key_env),
                    ("model", cfg.model),
                )
                if not value
            ]
            hint = (
                f"端点 {self.endpoint!r} 缺少必填项：{', '.join(missing)}。"
                f"请在 ~/.myagent/config.toml 的 [llm.endpoints.{self.endpoint}] 中补齐"
            )
            raise ConfigError(hint, detail=hint)
        return cfg


class SafetyConfig(_Base):
    """安全防护配置（§15.2）。"""

    mode: SafetyMode = "default"
    judge_model_role: str = "judge"
    judge_timeout_s: float = Field(default=D.JUDGE_TIMEOUT_S, gt=0)
    judge_max_retries: int = Field(default=D.JUDGE_MAX_RETRIES, ge=0)
    sensitive_dirs: list[str] = Field(default_factory=lambda: list(D.SENSITIVE_DIR_SEGMENTS))
    global_deny: list[Rule] = Field(default_factory=list)
    global_ask: list[Rule] = Field(default_factory=list)
    irreversible_patterns: list[Rule] = Field(default_factory=list)
    persist_approvals: bool = False
    audit_enabled: bool = True
    pipeline_budget_s: float = Field(default=D.PIPELINE_TOTAL_BUDGET_S, gt=0)

    @field_validator("sensitive_dirs")
    @classmethod
    def _keep_builtin_sensitive_dirs(cls, value: list[str]) -> list[str]:
        """确保内置敏感目录**不可被删除**（§6.3.2）。

        Args:
            value: 用户配置的敏感目录列表。

        Returns:
            并入内置项后的去重列表。
        """
        return list(dict.fromkeys([*D.SENSITIVE_DIR_SEGMENTS, *value]))


class ContextConfig(_Base):
    """上下文与压缩配置（§15.3）。"""

    model_window: int | None = Field(default=None, gt=0)
    reserved_fixed_tokens: int = D.RESERVED_FIXED_TOKENS
    reserved_output_tokens: int = D.RESERVED_OUTPUT_TOKENS
    compress_trigger_ratio: float = Field(default=D.COMPRESS_TRIGGER_RATIO, gt=0, le=1.0)
    keep_recent_large_results: int = D.KEEP_RECENT_LARGE_RESULTS
    large_result_token_threshold: int = D.LARGE_RESULT_TOKEN_THRESHOLD
    large_result_char_threshold: int | None = D.LARGE_RESULT_CHAR_THRESHOLD
    preview_tokens: int = D.PREVIEW_TOKENS
    compact_keep_recent_turns: int = D.COMPACT_KEEP_RECENT_TURNS
    segment_turns: int = D.SEGMENT_TURNS
    summary_max_tokens: int = D.SUMMARY_MAX_TOKENS
    repomap_max_depth: int = D.REPOMAP_MAX_DEPTH
    repomap_max_files: int = D.REPOMAP_MAX_FILES
    repomap_max_packages: int = D.REPOMAP_MAX_PACKAGES
    repomap_ttl_hours: int = D.REPOMAP_TTL_HOURS
    enable_calibration: bool = True
    """【待确认假设 U24】是否启用 token 在线校准。"""


class MemoryConfig(_Base):
    """记忆配置（§15.3）。"""

    inject_max_tokens: int = D.MEMORY_INJECT_MAX_TOKENS
    write_per_turn: bool = True
    max_writes_per_turn: int = D.MAX_MEMORY_WRITES_PER_TURN
    read_external: bool = True
    """【待确认假设 U16】是否兼容只读读取 CLAUDE.md / MYAGENT.md。"""

    global_budget_ratio: float = D.MEMORY_GLOBAL_BUDGET_RATIO
    project_budget_ratio: float = D.MEMORY_PROJECT_BUDGET_RATIO


class SkillsConfig(_Base):
    """skills 配置（§15.3）。"""

    similarity_backend: Literal["trigram", "embedding"] = "trigram"
    """【待确认假设 U8】默认字符 trigram 余弦；``embedding`` 需额外配置。"""

    embedding_provider: str | None = None
    embedding_model: str | None = None
    dedupe_threshold: float = D.DEDUPE_THRESHOLD
    llm_check_range: tuple[float, float] = D.LLM_CHECK_RANGE
    related_range: tuple[float, float] = D.RELATED_RANGE
    router_min_skills: int = D.ROUTER_MIN_SKILLS
    router_min_similar: int = D.ROUTER_MIN_RELATED_PAIRS
    router_mode: Literal["per_cluster", "single_global"] = "per_cluster"
    """【待确认假设 U10】"""

    router_cluster_max: int = D.ROUTER_CLUSTER_MAX
    graph_edge_decay_per_day: float = D.GRAPH_EDGE_DECAY_PER_DAY
    read_external: bool = True
    """是否只读兼容 ``.claude/skills/``（§9.1 优先级 3）。"""


class McpServerConfig(_Base):
    """一个 MCP 服务器配置（§10.2）。

    定义在 config 层（而非 ``mcp/``）以避免 ``config`` 反向依赖 ``mcp``；
    ``my_agent.mcp.schema`` 会重新导出本类，两处是同一个类型。
    """

    name: str = Field(description="必需，唯一")
    transport: Literal["stdio", "http", "sse"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    timeout_s: int = D.MCP_DEFAULT_TIMEOUT_S
    tool_prefix: str | None = None


class McpConfig(_Base):
    """MCP 全局配置（§10.3）。"""

    validate_on_add_only: bool = True
    """上游要求：只在用户添加时校验；启动加载不做 deep 校验（F-2）。"""

    project_file: str = ".myagent/mcp.yaml"
    global_file: str = "mcp.yaml"
    servers: list[McpServerConfig] = Field(default_factory=list)


class HookSpec(_Base):
    """一个 hook 的配置声明（§11.2）。"""

    event: str
    matcher: str | None = Field(default=None, description="工具名正则；缺省 = 全部")
    type: Literal["command", "python"] = "command"
    command: str | None = None
    module: str | None = Field(default=None, description="type=python 时的模块路径（仅测试用）")
    timeout_s: float = D.HOOK_DEFAULT_TIMEOUT_S
    blocking: bool = True
    on_error: Literal["allow", "deny", "abort"] = "allow"
    env: dict[str, str] = Field(default_factory=dict)


class HooksConfig(_Base):
    """hook 配置（§11.1）。"""

    merge_permission_events: bool = D.HOOK_MERGE_PERMISSION_EVENTS
    """【待确认假设 U11】True = 合并为严格 6 个事件。"""

    allow_permission_override: bool = D.HOOK_ALLOW_PERMISSION_OVERRIDE
    """【待确认假设 U12】True = 允许 hook 把 deny 放宽为 allow（**危险**）。"""

    subagent_isolation: bool = False
    """True = 子 Agent 不触发 hook（§12.1）。"""

    project_file: str = ".myagent/hooks.yaml"
    global_file: str = "hooks.yaml"
    hooks: list[HookSpec] = Field(default_factory=list)


class AgentsConfig(_Base):
    """多 Agent 配置（§12）。"""

    enabled: bool = True
    max_concurrent_subagents: int = D.MAX_CONCURRENT_SUBAGENTS
    max_depth: int = D.MAX_SUBAGENT_DEPTH
    max_turns: int = D.SUBAGENT_MAX_TURNS


class UIConfig(_Base):
    """交互层配置（§15.3）。"""

    permission_style: Literal["card", "inline"] = "card"
    stream: bool = D.STREAM_DEFAULT
    emoji: bool = True
    no_color: bool = False
    show_risk_badge: bool = True
    tool_result_preview_lines: int = 10


class Config(_Base):
    """myagent 的完整配置（§15.2）。"""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    def model_window(self) -> int:
        """计算生效的模型上下文窗口。

        Returns:
            ``llm.context_window`` → ``llm.endpoints[endpoint].context_window`` →
            :data:`~my_agent.config.defaults.MODEL_WINDOW_DEFAULT` 的优先级结果。
        """
        if self.context.model_window:
            return self.context.model_window
        if self.llm.context_window:
            return self.llm.context_window
        endpoint = self.llm.endpoints.get(self.llm.endpoint)
        if endpoint is not None:
            return endpoint.context_window
        return D.MODEL_WINDOW_DEFAULT

    def max_output_tokens(self) -> int:
        """计算生效的单次输出上限（§7.2：``min(30k, model_max_output)``）。

        Returns:
            输出 token 上限。
        """
        endpoint = self.llm.endpoints.get(self.llm.endpoint)
        model_max = (
            self.llm.max_output_tokens
            or (endpoint.max_output_tokens if endpoint else None)
            or D.MAX_OUTPUT_TOKENS_DEFAULT
        )
        return min(self.context.reserved_output_tokens, model_max)

    def describe(self) -> dict[str, Any]:
        """生成配置摘要（``/help`` 与 ``--debug`` 用，**不含任何密钥**）。

        Returns:
            摘要字典。
        """
        endpoint = self.llm.endpoints.get(self.llm.endpoint)
        return {
            "mode": self.safety.mode,
            "endpoint": self.llm.endpoint,
            "base_url": endpoint.base_url if endpoint else "",
            "api_key_env": endpoint.api_key_env if endpoint else "",
            "model": endpoint.model if endpoint else "",
            "model_window": self.model_window(),
            "compress_trigger_ratio": self.context.compress_trigger_ratio,
            "skills_backend": self.skills.similarity_backend,
        }
