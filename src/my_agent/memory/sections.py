"""记忆范围（Section）定义表 —— 规格 §8.2 的**唯一权威**（代码与提示词共用）。

本模块是「哪些章节存在」的单点定义：

- :data:`GLOBAL_SECTIONS` / :data:`PROJECT_SECTIONS`：键 + 一句说明，**逐字来自 §8.2**；
- :data:`SECTION_LABELS`：章节键 → Markdown 二级标题（去掉冒号后半句的中文标题）；
- :data:`GLOBAL_SECTION_PRIORITY` / :data:`PROJECT_SECTION_PRIORITY`：§8.6 的章节优先级（高 → 低）；
- 若干纯函数：标签/键互转、红线判定、提示词用的目录渲染。

规格：``my_agent-开发文档.md`` §8.2、§8.3、§8.6；``CONTRACTS.md`` §8。

【注意】范围选择提示词**只允许**使用 :func:`render_section_catalog` 的输出（只有范围名 + 一句说明），
不得把已有记忆内容拼进提示词（验收 D-1）。
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "GLOBAL_SECTIONS",
    "GLOBAL_SECTION_PRIORITY",
    "PROJECT_SECTIONS",
    "PROJECT_SECTION_PRIORITY",
    "REDLINE_SUFFIX",
    "SECTION_LABELS",
    "Scope",
    "is_redline",
    "label_for",
    "priority_for",
    "priority_index",
    "render_section_catalog",
    "scope_of_section",
    "section_from_label",
    "sections_for",
]

#: 记忆作用域字面量（§8.1）。
Scope = Literal["global", "project"]

#: 红线类章节键的后缀（§8.5 保护规则：此类章节只允许 ADD / DELETE）。
REDLINE_SUFFIX = "redlines"

# --------------------------------------------------------------------------------------
# §8.2 记忆范围定义表（唯一权威，逐字采用）
# --------------------------------------------------------------------------------------

GLOBAL_SECTIONS: dict[str, str] = {
    "user_preference": "用户偏好：交流语言、解释深度、是否要多方案、注释语言、提交信息语言",
    "coding_style": "编码风格偏好：缩进、命名习惯、测试风格、错误处理方式、常用范式",
    "tooling_env": "工具与环境：操作系统、shell、包管理器、编辑器、常用 CLI、常用路径",
    "workflow_pref": "工作流偏好：先计划再改还是直接改、是否自动跑测试、是否允许自动提交、PR/commit 格式",
    "rules_redlines": "长期规则与红线：安全合规、隐私要求、禁止执行的命令、必须人工确认的操作",
    "cross_project_knowledge": "跨项目知识：常用框架/库用法、内部平台 API、常用账号别名、文档入口",
    "behavior_correction": "行为纠正：用户之前纠正过的行为",
}

PROJECT_SECTIONS: dict[str, str] = {
    "overview": "项目概览：目标、业务领域、术语表、主要用户场景",
    "architecture": "架构与目录地图：模块职责、入口文件、数据流、依赖关系、关键服务",
    "stack_commands": "技术栈与命令：语言/框架版本、依赖管理、构建、测试、lint、运行、部署命令",
    "code_conventions": "代码规范与模式：命名、分层、错误处理、状态管理、API 调用、测试写法",
    "contracts": "关键契约：API 接口、数据库 schema、配置项、env 变量名、外部服务",
    "engineering_env": "工程环境：Docker、CI/CD、本地开发步骤、环境变量名、mock/fixture 约定",
    "decisions_history": "决策与历史：ADR、重要设计决策、已知坑、迁移状态、历史失败原因",
    "task_status": "任务状态：当前 feature、TODO、未完成重构、已知失败测试、最近变更摘要",
    "project_redlines": "项目级红线：禁止修改的目录、生成代码规则、必须跑的检查、PR 要求",
}

#: 章节键 → Markdown 标题（§8.3：标题 = 说明表去掉冒号后半句）。
#: 由 :data:`GLOBAL_SECTIONS` / :data:`PROJECT_SECTIONS` 直接推导，保证永不脱节。
SECTION_LABELS: dict[str, str] = {
    key: description.split("：", 1)[0]
    for key, description in {**GLOBAL_SECTIONS, **PROJECT_SECTIONS}.items()
}

# --------------------------------------------------------------------------------------
# §8.6 章节优先级（高 → 低，越前越保留；越靠后越先被裁剪）
# --------------------------------------------------------------------------------------

GLOBAL_SECTION_PRIORITY: tuple[str, ...] = (
    "rules_redlines",
    "behavior_correction",
    "user_preference",
    "workflow_pref",
    "coding_style",
    "tooling_env",
    "cross_project_knowledge",
)

PROJECT_SECTION_PRIORITY: tuple[str, ...] = (
    "project_redlines",
    "contracts",
    "task_status",
    "stack_commands",
    "architecture",
    "code_conventions",
    "overview",
    "engineering_env",
    "decisions_history",
)

_SECTIONS_BY_SCOPE: dict[Scope, dict[str, str]] = {
    "global": GLOBAL_SECTIONS,
    "project": PROJECT_SECTIONS,
}

_PRIORITY_BY_SCOPE: dict[Scope, tuple[str, ...]] = {
    "global": GLOBAL_SECTION_PRIORITY,
    "project": PROJECT_SECTION_PRIORITY,
}


# --------------------------------------------------------------------------------------
# 纯函数
# --------------------------------------------------------------------------------------


def sections_for(scope: Scope) -> dict[str, str]:
    """返回某作用域的范围定义表。

    Args:
        scope: ``"global"`` 或 ``"project"``。

    Returns:
        章节键 → 一句说明 的字典（即 :data:`GLOBAL_SECTIONS` 或 :data:`PROJECT_SECTIONS`）。
    """
    return _SECTIONS_BY_SCOPE[scope]


def priority_for(scope: Scope) -> tuple[str, ...]:
    """返回某作用域的章节优先级（高 → 低，§8.6）。

    Args:
        scope: ``"global"`` 或 ``"project"``。

    Returns:
        章节键元组。
    """
    return _PRIORITY_BY_SCOPE[scope]


def priority_index(scope: Scope, section: str) -> int:
    """返回章节在优先级表中的下标（越小越重要）。

    Args:
        scope: 作用域。
        section: 章节键；未知键也接受。

    Returns:
        下标；未知章节返回 ``len(priority)``（即「最低优先级」）。
    """
    priority = priority_for(scope)
    try:
        return priority.index(section)
    except ValueError:
        return len(priority)


def label_for(section: str) -> str:
    """返回章节键对应的 Markdown 标题。

    Args:
        section: 章节键（如 ``"user_preference"``）。

    Returns:
        中文标题（如 ``"用户偏好"``）；未知键原样返回键名。
    """
    return SECTION_LABELS.get(section, section)


def scope_of_section(section: str) -> Scope | None:
    """判断章节键属于哪个作用域。

    Args:
        section: 章节键。

    Returns:
        所属作用域；两张表都没有该键时返回 ``None``。
    """
    if section in GLOBAL_SECTIONS:
        return "global"
    if section in PROJECT_SECTIONS:
        return "project"
    return None


def section_from_label(scope: Scope, title: str) -> str | None:
    """把 Markdown 二级标题文本还原成章节键（§8.3：``startswith`` 容错匹配）。

    Args:
        scope: 作用域（同名标题在两张表里不会冲突，但仍按作用域限定）。
        title: 标题文本（已去掉 ``## `` 前缀），如 ``"用户偏好（补充）"``。

    Returns:
        章节键；无匹配时返回 ``None``（调用方须**原样保留**该章节）。
    """
    text = title.strip()
    longest: str | None = None
    for key in sections_for(scope):
        # 必须比对**标题**（§8.2 冒号前的部分），而不是整句说明：
        # 说明形如「用户偏好：交流语言、…」，用它做 startswith 永远匹配不上标题「用户偏好」。
        label = SECTION_LABELS.get(key, key)
        if text.startswith(label) and (longest is None or len(label) > len(SECTION_LABELS[longest])):
            longest = key
    return longest


def is_redline(section: str) -> bool:
    """判断章节是否为红线类（§8.5：键以 ``redlines`` 结尾）。

    Args:
        section: 章节键。

    Returns:
        是红线类返回 ``True``。
    """
    return section.endswith(REDLINE_SUFFIX)


def render_section_catalog(scope: Scope) -> str:
    """渲染提示词用的范围目录（**只有范围名 + 一句说明**，验收 D-1）。

    Args:
        scope: 作用域。

    Returns:
        每个范围一行 ``- <键>：<说明>`` 的文本。
    """
    return "\n".join(f"- {key}：{description}" for key, description in sections_for(scope).items())
