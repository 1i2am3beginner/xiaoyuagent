"""阶段 0 内置规则表（§6.3.1 / §6.3.3）。

规格：``my_agent-开发文档.md`` §6.3。三张表都是**内置默认值**：

* `GLOBAL_DENY` / `GLOBAL_ASK`：§6.3.1 的两类名单（正则匹配**完整命令行**或写类工具的目标路径）。
* `IRREVERSIBLE_PATTERNS`：§6.3.3 的七类不可逆操作（正则，**不区分大小写**），每条带中文理由文案。

**可追加不可删除**：项目/用户配置只能往这些表里追加规则，不能删除内置项
（§6.3.2「内置清单可配置追加，不可删除」、§15.1「列表类追加而非覆盖」）。
本模块提供 :func:`merge_rules` / :func:`merge_dirs` 实现该语义，
供 :mod:`my_agent.security.phase0_hard_block` 使用。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from my_agent.config.schema import Rule

__all__ = [
    "GLOBAL_ASK",
    "GLOBAL_DENY",
    "IRREVERSIBLE_PATTERNS",
    "merge_dirs",
    "merge_rules",
]

GLOBAL_DENY: tuple[Rule, ...] = (
    Rule(pattern="rm -rf /", reason="递归删除根目录"),
    Rule(pattern="format [a-zA-Z]:", reason="格式化磁盘"),
    Rule(pattern=r":(){ :\|:& };:", reason="fork bomb"),
    Rule(pattern="shutdown|reboot", reason="关机/重启"),
)
"""§6.3.1 内置全局拒绝名单（命中即 ``deny``，``permanent_eligible=False``）。"""

GLOBAL_ASK: tuple[Rule, ...] = (
    Rule(pattern="git push --force", reason="强制推送会覆盖远端历史"),
    Rule(pattern=r"DROP\s+(TABLE|DATABASE)", reason="破坏性数据库操作"),
)
"""§6.3.1 内置全局询问名单（命中即 ``ask``，``permanent_eligible=False``）。"""

_IRREVERSIBLE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "发布到公共仓库后无法撤回",
        (
            r"\bnpm\s+publish\b",
            r"\byarn\s+publish\b",
            r"\btwine\s+upload\b",
            r"\bcargo\s+publish\b",
            r"\bpoetry\s+publish\b",
            r"\buv\s+publish\b",
        ),
    ),
    (
        "强制杀进程可能丢失未保存数据",
        (
            r"\bkill\s+-9\b",
            r"\bkill\s+-KILL\b",
            r"\btaskkill\b.*\/F",
            r"\bpkill\s+-9\b",
        ),
    ),
    (
        "破坏系统或磁盘数据",
        (
            r"\bmkfs\b",
            r"\bdd\s+.*of=/dev/",
            r"\bdiskpart\b",
            r"\bformat\s+[a-z]:",
            r"\bshutdown\b",
            r"\breboot\b",
        ),
    ),
    (
        "大范围权限变更难以回滚",
        (
            r"\bchmod\s+-R\s+777\b",
            r"\bchown\s+-R\b",
            r"\bicacls\b.*\/grant",
        ),
    ),
    (
        "影响远端生产环境",
        (
            r"\bdocker\s+push\b",
            r"\bkubectl\s+delete\b",
            r"\bterraform\s+(apply|destroy)\b",
            r"\bgit\s+push\s+.*--force",
        ),
    ),
    (
        "重写历史难以恢复",
        (
            r"\bgit\s+filter-branch\b",
            r"\bgit\s+reset\s+--hard\s+origin",
            r"\bgit\s+push\s+--delete\b",
        ),
    ),
    (
        "递归强制删除",
        (
            r"\brm\s+-[a-z]*r[a-z]*f?\b",
            r"\bRemove-Item\b.*-Recurse.*-Force",
            r"\bdel\s+\/s\s+\/q\b",
        ),
    ),
)

IRREVERSIBLE_PATTERNS: tuple[Rule, ...] = tuple(
    Rule(pattern=pattern, reason=reason)
    for reason, patterns in _IRREVERSIBLE_GROUPS
    for pattern in patterns
)
"""§6.3.3 内置不可逆操作清单（命中即 ``ask`` + ``risk="critical"`` + ``permanent_eligible=False``）。"""


def merge_rules(builtin: Sequence[Rule], extra: Iterable[Rule]) -> tuple[Rule, ...]:
    """把配置追加规则合并到内置规则上，保证内置项不可被删除。

    合并语义（§6.3.2、§15.1）：内置项全量保留且**排在前面**；``extra`` 中
    ``pattern`` 与已有项相同者跳过（内置优先），其余按配置顺序追加。

    Args:
        builtin: 内置规则（如 :data:`GLOBAL_DENY`）。
        extra: 配置追加的规则（``config.safety.global_deny`` 等）。

    Returns:
        合并后的规则元组；顺序稳定，重复 ``pattern`` 只保留首次出现者。

    Raises:
        TypeError: ``extra`` 中的元素不是 :class:`~my_agent.config.schema.Rule`。
    """
    merged: list[Rule] = list(builtin)
    seen: set[str] = {rule.pattern for rule in builtin}
    for rule in extra:
        if not isinstance(rule, Rule):  # pragma: no cover - 配置层已由 pydantic 保证
            raise TypeError(f"extra 必须是 Rule，收到 {type(rule).__name__}")
        if rule.pattern in seen:
            continue
        seen.add(rule.pattern)
        merged.append(rule)
    return tuple(merged)


def merge_dirs(builtin: Sequence[str], extra: Iterable[str]) -> tuple[str, ...]:
    """把配置追加的敏感目录合并到内置目录段上（同样不可删除）。

    Args:
        builtin: 内置敏感目录段（:data:`my_agent.config.defaults.SENSITIVE_DIR_SEGMENTS`）。
        extra: 配置追加的目录段（``config.safety.sensitive_dirs``）。

    Returns:
        去重后的目录段元组，内置项在前。
    """
    merged: list[str] = list(builtin)
    seen: set[str] = set(builtin)
    for seg in extra:
        name = seg.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        merged.append(name)
    return tuple(merged)
