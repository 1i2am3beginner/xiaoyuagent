"""全部默认值、阈值常量与【待确认假设】注册表。

本文件是**魔法数字与未决假设的唯一入口**（§2.2「可替换假设」、附录 C 最后一条）：
其它模块禁止写字面量阈值，一律 ``from my_agent.config.defaults import X``。

规格：``my_agent-开发文档.md`` §2.2、§5.3.3、§6、§7、§8、§9、§10、§11、§12、§13、§14、§15.3、§20。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "APPROVAL_CACHE_WILDCARD_CHARS",
    "BACKUP_MAX_FILES",
    "BACKUP_MAX_MB",
    "BASH_DEFAULT_TIMEOUT_MS",
    "CLEARED_MARKER",
    "COMPACT_KEEP_RECENT_TURNS",
    "COMPRESS_TRIGGER_RATIO",
    "CONSECUTIVE_FAILURE_LIMIT",
    "DEDUPE_THRESHOLD",
    "DISPATCH_JUDGE_BACKOFF_S",
    "DISPATCH_JUDGE_ENABLED",
    "DISPATCH_JUDGE_FILE_EXCERPT_CHARS",
    "DISPATCH_JUDGE_MAX_FILES",
    "DISPATCH_JUDGE_MAX_OUTPUT_CHARS",
    "DISPATCH_JUDGE_MAX_RETRIES",
    "DISPATCH_JUDGE_TIMEOUT_S",
    "EMPTY_RESPONSE_RETRIES",
    "GRAPH_EDGE_DECAY_PER_DAY",
    "GRAPH_EDGE_MIN",
    "GREP_MAX_INLINE_MATCHES",
    "HOOK_ALLOW_PERMISSION_OVERRIDE",
    "HOOK_DEFAULT_TIMEOUT_S",
    "HOOK_MERGE_PERMISSION_EVENTS",
    "HOOK_TIMEOUT_GRACE_S",
    "JUDGE_ARG_TRUNCATE_TO",
    "JUDGE_MAX_ARG_TOKENS",
    "JUDGE_MAX_RETRIES",
    "JUDGE_TEMPERATURE",
    "JUDGE_TIMEOUT_S",
    "KEEP_RECENT_LARGE_RESULTS",
    "LARGE_RESULT_CHAR_THRESHOLD",
    "LARGE_RESULT_TOKEN_THRESHOLD",
    "LLM_BACKOFF_S",
    "LLM_CHECK_RANGE",
    "LLM_MAX_RETRIES",
    "LOOP_GUARD_WINDOW",
    "LOOP_REPEAT_THRESHOLD",
    "LOOP_RESULT_SIMILARITY",
    "MAX_DISPATCH_JUDGE_PER_TURN",
    "MAIN_TEMPERATURE",
    "MAX_CONCURRENT_SUBAGENTS",
    "MAX_CONCURRENT_TOOLS",
    "MAX_MEMORY_WRITES_PER_TURN",
    "MAX_OUTPUT_TOKENS_DEFAULT",
    "MAX_SUBAGENT_DEPTH",
    "MAX_TOOL_CALLS_PER_TURN",
    "MAX_TOOL_RETRIES",
    "MAX_TURNS_PER_REQUEST",
    "MCP_DEFAULT_TIMEOUT_S",
    "MCP_NAME_RE",
    "MCP_PREFIX_RE",
    "MCP_TERMINATE_GRACE_S",
    "MCP_TIMEOUT_RANGE",
    "MEMORY_BUDGET",
    "MEMORY_DUP_SIMILARITY",
    "MEMORY_GLOBAL_BUDGET_RATIO",
    "MEMORY_INJECT_MAX_TOKENS",
    "MEMORY_LOCK_TIMEOUT_S",
    "MEMORY_PROJECT_BUDGET_RATIO",
    "MEMORY_TEXT_MAX_CHARS",
    "MODEL_WINDOW_DEFAULT",
    "PENDING_ASSUMPTIONS",
    "PIPELINE_TOTAL_BUDGET_S",
    "PREVIEW_HEAD_TOKENS",
    "PREVIEW_TAIL_TOKENS",
    "PREVIEW_TOKENS",
    "PROMPT_BUDGET_TOKENS",
    "READ_DEFAULT_LIMIT",
    "RELATED_RANGE",
    "REPOMAP_BUDGET",
    "REPOMAP_MAX_DEPTH",
    "REPOMAP_MAX_FILES",
    "REPOMAP_MAX_PACKAGES",
    "REPOMAP_TTL_HOURS",
    "RESERVED_FIXED_TOKENS",
    "RESERVED_OUTPUT_TOKENS",
    "ROUTER_CLUSTER_MAX",
    "ROUTER_MIN_RELATED_PAIRS",
    "ROUTER_MIN_SKILLS",
    "ROUTER_MODE",
    "SEGMENT_TURNS",
    "SENSITIVE_DIR_SEGMENTS",
    "SESSION_KEEP_RECENT",
    "SKILLS_MCP_BUDGET",
    "STREAM_DEFAULT",
    "SUBAGENT_MAX_TURNS",
    "SUBAGENT_POLL_INTERVAL_S",
    "SUMMARY_MAX_TOKENS",
    "SYSTEM_PROMPT_BUDGET",
    "TIMEOUT_MAIN_S",
    "TIMEOUT_SMALL_S",
    "TOOL_ARG_MAX_REPAIR_CHARS",
    "TOOL_ARG_REPAIR_ENABLED",
    "TOOL_OVERLAP_JACCARD",
    "TOOL_RETRY_BACKOFF_S",
    "TOOL_RETRY_MAX_ATTEMPTS",
    "TOOLS_PROMPT_BUDGET",
    "USAGE_CALIBRATION_FACTOR",
    "WEB_FETCH_MAX_BYTES",
    "WEB_FETCH_TIMEOUT_S",
    "WRITE_CATEGORIES",
    "assumption_report",
]

# ============================================================================
# §7.2 上下文预算
# ============================================================================
RESERVED_FIXED_TOKENS = 30_000
"""预留给「系统提示词 + 工作目录摘要 + tools + 记忆 + mcp/skills」的固定额度（§7.1）。"""

RESERVED_OUTPUT_TOKENS = 30_000
"""输出预留 + 安全缓冲（§7.1）。"""

COMPRESS_TRIGGER_RATIO = 0.80
"""压缩触发水位。【待确认假设 U2】基数为**有效上下文**（触发点 ≈112k），
若用户的图里 80% 指模型原始窗口，改此值或改 ``budget.effective_context`` 的基数即可。"""

PROMPT_BUDGET_TOKENS = 30_000
"""五件套合计预算上限（= 上文两项之一半，§7.1）。"""

SYSTEM_PROMPT_BUDGET = 2_000
REPOMAP_BUDGET = 8_000
TOOLS_PROMPT_BUDGET = 12_000
MEMORY_BUDGET = 6_000
SKILLS_MCP_BUDGET = 2_000

MODEL_WINDOW_DEFAULT = 200_000
"""默认模型上下文窗口（可被 ``llm.context_window`` 覆盖）。"""

MAX_OUTPUT_TOKENS_DEFAULT = 8_192
"""默认单次输出上限；实际取 ``min(30k, model_max_output)``（§7.2）。"""

USAGE_CALIBRATION_FACTOR = 0.30
"""【待确认假设 U24】在线校准权重：``factor = (1-w)*old + w*actual/estimated``。"""

# ============================================================================
# §5.3.3 内层循环参数
# ============================================================================
MAX_TURNS_PER_REQUEST = 50
"""单次用户输入内最多「模型调用轮次」，防止死循环（B11）。"""

MAX_TOOL_CALLS_PER_TURN = 16
"""单轮 assistant 消息内最多工具调用数（超出截断并在结果中说明）。"""

MAX_TOOL_RETRIES = 1
"""工具自身失败（非权限失败）时的重试次数。"""

CONSECUTIVE_FAILURE_LIMIT = 3
"""连续 N 轮工具全失败 → 注入提示，要求模型停止并解释。"""

STREAM_DEFAULT = True
EMPTY_RESPONSE_RETRIES = 1
"""模型返回空内容且无 tool_calls 时的重试次数（B12）。"""

# ============================================================================
# §6 安全防护 / 权限流水线
# ============================================================================
SENSITIVE_DIR_SEGMENTS: tuple[str, ...] = (".git", ".vscode", ".bin", ".user")
"""敏感目录名（分段精确匹配；可配置追加，**不可删除**）。"""

WRITE_CATEGORIES: tuple[str, ...] = ("write", "execute")
"""视为「潜在写操作」的工具类别（execute 也触发敏感路径检查）。"""

JUDGE_TIMEOUT_S = 4.0
"""判定模型超时阈值（§6.5.3，用户建议区间 3–5s）。"""

JUDGE_MAX_RETRIES = 2
"""判定输出解析失败时的重试次数。"""

PIPELINE_TOTAL_BUDGET_S = 6.0
"""整条流水线的总耗时预算（超时按模型不可用处理）。"""

JUDGE_MAX_ARG_TOKENS = 8_000
"""判定输入中单个字符串值超过该 token 数则截断。"""

JUDGE_ARG_TRUNCATE_TO = 2_000
"""截断后保留的字符数（前缀）+ ``…[已截断]`` 标注。"""

APPROVAL_CACHE_WILDCARD_CHARS: tuple[str, ...] = ("*", "?")
"""含这些通配符的参数**不缓存**（避免误放行，§6.7）。"""

# ============================================================================
# §7.3 工作目录摘要
# ============================================================================
REPOMAP_MAX_DEPTH = 4
REPOMAP_MAX_FILES = 4_000
REPOMAP_MAX_PACKAGES = 20
REPOMAP_TTL_HOURS = 24

# ============================================================================
# §7.4 大结果落盘 / §7.5 裁剪 / §7.7 压缩
# ============================================================================
LARGE_RESULT_TOKEN_THRESHOLD = 50_000
"""【待确认假设 U3】上游「单条调用超过 50k」按 **token** 实现。"""

LARGE_RESULT_CHAR_THRESHOLD: int | None = None
"""【待确认假设 U3】可选的字符口径开关；None = 关闭（只用 token 口径）。"""

PREVIEW_TOKENS = 2_000
PREVIEW_HEAD_TOKENS = 1_500
PREVIEW_TAIL_TOKENS = 500
"""预览 = 头部 1.5k + 尾部 0.5k，中间插省略标记。"""

KEEP_RECENT_LARGE_RESULTS = 5
"""裁剪时保留最近 N 个大工具结果（§7.5）。"""

CLEARED_MARKER = "Old tool result content cleared"
"""被裁剪结果的替换文本，**必须原样**（§7.5、C-4）。"""

COMPACT_KEEP_RECENT_TURNS = 4
"""【待确认假设 U4】全文压缩后保留最近 K 轮原文。"""

SEGMENT_TURNS = 20
"""分段摘要的段大小（轮）。"""

SUMMARY_MAX_TOKENS = 4_000
"""单段摘要上限；超出则要求模型压缩重写一次。"""

# ============================================================================
# §8 记忆
# ============================================================================
MEMORY_INJECT_MAX_TOKENS = 6_000
"""记忆注入预算（全局 + 项目）。"""

MEMORY_GLOBAL_BUDGET_RATIO = 0.40
MEMORY_PROJECT_BUDGET_RATIO = 0.60
"""全局 : 项目 预算比（项目更相关）。"""

MAX_MEMORY_WRITES_PER_TURN = 3
MEMORY_TEXT_MAX_CHARS = 300
MEMORY_DUP_SIMILARITY = 0.95
"""候选与已有条目文本相似度 ≥ 该值直接丢弃（§8.4 步骤 2）。"""

MEMORY_LOCK_TIMEOUT_S = 3.0
"""跨进程文件锁超时（超时抛 ``MemoryLockError``）。"""

# ============================================================================
# §9 skills
# ============================================================================
DEDUPE_THRESHOLD = 0.92
"""≥ 该相似度 → 判为同义技能，去重（不添加）。"""

LLM_CHECK_RANGE: tuple[float, float] = (0.85, 0.92)
"""该区间交 LLM 判定是否同义。"""

RELATED_RANGE: tuple[float, float] = (0.75, 0.85)
"""该区间记「强相关」（供路由）。"""

ROUTER_MIN_SKILLS = 15
ROUTER_MIN_RELATED_PAIRS = 5
"""【待确认假设 U9】触发条件为**严格大于**：skills > 15 且相似对 > 5。"""

ROUTER_CLUSTER_MAX = 8
ROUTER_MODE = "per_cluster"
"""【待确认假设 U10】``per_cluster`` = 每个强相关簇一个大 skill；``single_global`` = 全局唯一。"""

GRAPH_EDGE_DECAY_PER_DAY = 0.01
"""【待确认假设 U25】路由边权每日衰减 1%，低于 ``GRAPH_EDGE_MIN`` 的边被移除。"""

GRAPH_EDGE_MIN = 0.75
TOOL_OVERLAP_JACCARD = 0.6
"""allowed_tools 交集相似对判定阈值（§9.4）。"""

# ============================================================================
# §10 MCP
# ============================================================================
MCP_NAME_RE = r"^[a-zA-Z0-9_-]{1,64}$"
MCP_PREFIX_RE = r"^[a-zA-Z0-9_]{1,32}$"
MCP_TIMEOUT_RANGE: tuple[int, int] = (1, 600)
MCP_DEFAULT_TIMEOUT_S = 30
MCP_TERMINATE_GRACE_S = 3.0
"""关闭子进程：terminate → 等待 N 秒 → kill（§10.5）。"""

# ============================================================================
# §11 hook
# ============================================================================
HOOK_DEFAULT_TIMEOUT_S = 5.0
HOOK_TIMEOUT_GRACE_S = 1.0
HOOK_MERGE_PERMISSION_EVENTS = False
"""【待确认假设 U11】False = 实现 7 个事件；True = 把权限前后合并进工具前后（严格 6 个）。"""

HOOK_ALLOW_PERMISSION_OVERRIDE = False
"""【待确认假设 U12】False = hook **不能**把 deny 放宽为 allow（安全不变量）。"""

# ============================================================================
# §12 多 Agent
# ============================================================================
MAX_CONCURRENT_SUBAGENTS = 4
MAX_SUBAGENT_DEPTH = 1
"""【待确认假设 U13】不允许子 Agent 再派生子 Agent。"""

SUBAGENT_MAX_TURNS = 25
SUBAGENT_POLL_INTERVAL_S = 0.5

# ============================================================================
# §13 模型层
# ============================================================================
LLM_MAX_RETRIES = 3
LLM_BACKOFF_S: tuple[float, ...] = (0.5, 1.0, 2.0)
TIMEOUT_MAIN_S = 120.0
TIMEOUT_SMALL_S = 30.0
MAIN_TEMPERATURE = 0.2
JUDGE_TEMPERATURE = 0.0

# ============================================================================
# §14 工具层
# ============================================================================
MAX_CONCURRENT_TOOLS = 8
READ_DEFAULT_LIMIT = 2_000
BASH_DEFAULT_TIMEOUT_MS = 120_000
GREP_MAX_INLINE_MATCHES = 250
BACKUP_MAX_MB = 200
BACKUP_MAX_FILES = 1_000
WEB_FETCH_MAX_BYTES = 2_000_000
WEB_FETCH_TIMEOUT_S = 30.0

# ============================================================================
# §15 会话持久化
# ============================================================================
SESSION_KEEP_RECENT = 50
"""默认保留最近 N 个会话。"""


# ============================================================================
# 任务分发审判 Agent（新增：判断子任务分发是否过粗/过细）
# ============================================================================
DISPATCH_JUDGE_ENABLED = True
"""是否启用「任务分发审判」——在 ``Task`` 批次真正执行前先审一遍分发计划。"""

DISPATCH_JUDGE_TIMEOUT_S = 20.0
"""审判调用的单次超时。判决质量比权限判定更需要上下文，因此比 4s 宽松。"""

DISPATCH_JUDGE_MAX_RETRIES = 3
"""审判模型失败时的重试次数（**指数退避**，见下）。"""

DISPATCH_JUDGE_BACKOFF_S: tuple[float, ...] = (1.0, 2.0, 4.0)
"""审判重试的指数退避秒数：1s → 2s → 4s（用户明确要求「指数递增」）。

重试耗尽后**不做任何处理（放行）**，并向用户告警「审判模型故障」——
即 fail-open，且必须显式告知用户（不静默失效）。
"""

DISPATCH_JUDGE_MAX_FILES = 6
"""审判 Agent 自行读取的文件数量上限（主 Agent 只传路径清单，不传内容）。"""

DISPATCH_JUDGE_FILE_EXCERPT_CHARS = 2_000
"""每个文件交给审判 Agent 的最大字符数（截断，避免审判上下文爆掉）。"""

DISPATCH_JUDGE_MAX_OUTPUT_CHARS = 1_200
"""单个子任务描述在审判提示词里的最大字符数（防止 prompt 过长）。"""

MAX_DISPATCH_JUDGE_PER_TURN = 1
"""单回合最多审判几次——防止「过细 → 重发 → 再审 → 再过细」的无限循环。"""


# ============================================================================
# 工具调用健壮性（新增：参数容错解析 / 死循环检测 / 工具层网络退避）
# ============================================================================
TOOL_ARG_REPAIR_ENABLED = True
"""是否启用「markdown 剥离 → {} 提取 → json.loads → 容错修复 → schema 校验」解析链。"""

TOOL_ARG_MAX_REPAIR_CHARS = 200_000
"""容错修复的输入长度上限（防止把超大串送进修复器）。"""

LOOP_REPEAT_THRESHOLD = 3
"""【待确认假设 U33】同一 (tool, 参数) 指纹重复达到 N 次后开始做结果相似度判定。"""

LOOP_RESULT_SIMILARITY = 0.95
"""【待确认假设 U33】重复调用时**结果内容**相似度 ≥ 该值即判定陷入死循环。

【待确认假设】用户只说明「内容高度相似」，未给阈值：0.95 为保守初值（与记忆去重同档）。
"""

LOOP_GUARD_WINDOW = 32
"""死循环检测保留的最近指纹数量（避免无界增长）。"""

TOOL_RETRY_MAX_ATTEMPTS = 3
"""工具层（WebFetch / WebSearch / MCP HTTP）网络异常的最大重试次数。"""

TOOL_RETRY_BACKOFF_S: tuple[float, ...] = (0.5, 1.0, 2.0)
"""工具层网络重试的指数退避秒数。"""


# ============================================================================
# §20 / §6.11 【待确认假设】注册表
# ============================================================================
PENDING_ASSUMPTIONS: dict[str, str] = {
    # ---- §6.11 语义冲突（实现按下述选择）----
    "Q1": "dontAsk + 判定模型不可用 → 按模式契约转 deny（不问用户），reason 显式提示",
    "Q2": "bypassPermissions + 判定模型不可用 → 转 ask，**不自动放行**（最保守解释）",
    "Q3": "阶段 0 的所有命中（含 global_ask 名单）一律在阶段 0 终止，bypass 不放行",
    "Q4": "acceptEdits 快速通道覆盖 Write（含新建）/Edit/MultiEdit/NotebookEdit；**不覆盖删除**",
    # ---- §20.1 阻断级 ----
    "U1": "三张流程图未提供 → 按文字描述 + 标准实践实现（§7.4–§7.7）",
    "U2": "「80%」分母 = 有效上下文（触发点 ≈112k）；改 COMPRESS_TRIGGER_RATIO 或预算基数",
    "U3": "「50k」按 token 实现；另留 LARGE_RESULT_CHAR_THRESHOLD 字符口径开关（默认关闭）",
    "U4": "压缩后保留最近 4 轮原文（COMPACT_KEEP_RECENT_TURNS）",
    "U5": "dontAsk 模型不可用 → 转 deny（= Q1）",
    "U6": "bypassPermissions 模型不可用 → 转 ask（= Q2）",
    "U7": "global_ask 类操作在 bypass 下不放行（= Q3）",
    # ---- §20.2 语义歧义 ----
    "U8": "skills 相似度默认字符 trigram 余弦（skills.similarity_backend=trigram），可切 embedding",
    "U9": "「工具相似个数」= 强相关对数 + allowed_tools 重叠对数；触发条件严格 > 15 且 > 5",
    "U10": "路由按每个强相关簇一个大 skill（ROUTER_MODE=per_cluster），可切 single_global",
    "U11": "hook 实现 7 个事件；HOOK_MERGE_PERMISSION_EVENTS=true 时合并为严格 6 个",
    "U12": "after_permission_check 不能把 deny 放宽为 allow（HOOK_ALLOW_PERMISSION_OVERRIDE=false）",
    "U13": "子 Agent 不能再生子 Agent（MAX_SUBAGENT_DEPTH=1）",
    "U14": "「模型不可用」包含超时/连接失败/5xx/429/401-403/解析失败超限，统一收敛到 ModelUnavailableError",
    "U15": "acceptEdits 快速通道覆盖新建与编辑，不覆盖删除（= Q4）",
    "U33": "死循环判定 = 同(tool,参数)指纹重复 ≥3 次 **且** 结果内容相似度 ≥0.95（阈值待确认）",
    "U35": (
        "任务分发审判：过粗→审判 agent 亲自补发（第二条派生路径）；过细→返回补丁由主 agent 重发；"
        "审判模型失败→指数退避重试后**放行并告警用户**（fail-open，不静默失效）"
    ),    "U34": "工具层网络异常（WebFetch/WebSearch/MCP HTTP）同样退避重试 3 次；MCP 超时按 §10.5 不重试",    "U16": "兼容只读读取 CLAUDE.md / MYAGENT.md（memory.read_external=true），永不写入这些文件",
    # ---- §20.3 工程补充项 ----
    "U17": "做写前备份 + /undo（无沙盒 + 无备份 = 误改不可恢复）",
    "U18": "大结果落盘文件会话级保留，超 200MB 淘汰最旧",
    "U19": "「永久放行」默认**不**跨会话持久化（仅会话内）",
    "U20": "审批缓存精确键匹配，不做通配；含通配符的参数不缓存",
    "U21": "权限询问卡展示 diff 预览",
    "U22": "做 /cost 用量统计",
    "U23": "审计日志做打码（redact）",
    "U24": "做上下文 token 在线校准（factor 指数平滑）",
    "U25": "做路由边时间衰减（每日 1%，低于 0.75 移除）",
    "U26": "首启无沙盒风险提示**待定** → 当前实现为：README 顶部醒目提示 + 启动时一行告警（不阻塞）",
    "U27": "NotebookEdit **不纳入**首版（不在工具表注册）",
    "U28": "WebSearch/WebFetch 后端留配置项，未配置则返回明确错误",
    "U29": "模型接入 = OpenAI 官方 SDK 直连（AsyncOpenAI），已确认",
    "U30": "不实现 Anthropic / Gemini 原生协议；在 provider.py 预留第二个 ModelProvider 实现位",
    "U31": "钉 openai>=1.55，uv.lock 锁死；升级只需通过 I-8/I-9/I-10",
    "U32": "代理/自定义 CA 通过 http_client= 注入自定义 httpx.AsyncClient，不改代码",
}


def assumption_report() -> str:
    """生成【待确认假设】的人类可读清单（``myagent --assumptions`` / ``/help`` 用）。

    Returns:
        每行 ``ID  说明`` 的文本，按 ID 排序（Q 在前，U 按编号）。
    """

    def sort_key(item: tuple[str, str]) -> tuple[int, int, str]:
        key = item[0]
        if key.startswith("Q"):
            return (0, int(key[1:]) if key[1:].isdigit() else 0, key)
        if key.startswith("U"):
            return (1, int(key[1:]) if key[1:].isdigit() else 0, key)
        return (2, 0, key)

    lines = ["【待确认假设】当前实现口径（集中定义于 config/defaults.py，便于逐条替换）：", ""]
    for key, value in sorted(PENDING_ASSUMPTIONS.items(), key=sort_key):
        lines.append(f"  {key:<4} {value}")
    return "\n".join(lines)


def assumption_count() -> int:
    """返回已登记的假设条数（供启动告警使用）。

    Returns:
        假设条数。
    """
    return len(PENDING_ASSUMPTIONS)


def as_dict() -> dict[str, Any]:
    """把本模块的全部常量导出为字典（调试与快照测试用）。

    Returns:
        常量名 → 值的字典（不含函数与 ``PENDING_ASSUMPTIONS`` 之外的私有对象）。
    """
    out: dict[str, Any] = {}
    for name in __all__:
        value = globals().get(name)
        if name.startswith("_") or callable(value):
            continue
        out[name] = value
    return out
