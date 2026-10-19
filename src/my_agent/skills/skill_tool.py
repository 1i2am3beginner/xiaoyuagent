"""Skill 调用工具（§9.6）。

====================== ============================ ========== ==================================
工具                    参数                          权限类别    行为
====================== ============================ ========== ==================================
``Skill``              ``name``, ``args?``           **write**   返回 ``SKILL.md`` 正文
``SkillList``          ``filter?``                   read        列出 skills（名 + 一行描述 + related）
====================== ============================ ========== ==================================

``Skill`` 被判为 write 类的原因：技能正文可能指示模型执行写操作，让权限流水线在
``plan`` 模式下拒绝它更安全（U8 备注）。**不自动执行脚本**——只返回「请用 Read/Bash
执行 scripts/xxx」的指令。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from my_agent.models.tools import ToolContext, ToolResult, ToolSpec

if TYPE_CHECKING:
    from my_agent.skills.registry import SkillRegistry

__all__ = ["SKILL_LIST_SPEC", "SKILL_SPEC", "SkillListTool", "SkillTool"]

SKILL_DESCRIPTION = (
    "读取一个技能（skill）的完整操作说明，并按其指引执行任务。\n"
    "何时用：用户请求与某个技能的描述匹配时；需要该技能的详细步骤、脚本路径或决策规则时。\n"
    "何时不用：只想知道有哪些技能可用时用 SkillList；已有足够信息可直接完成任务时不要读技能——"
    "读正文会占用上下文。反例：用户只是让你改一行代码，不要为此先读一个「代码审查」技能。\n"
    "副作用声明：本工具本身只读文本，但技能正文可能指示你执行写类操作（会修改磁盘文件）；"
    "后续写操作仍需经过权限确认。"
)

SKILL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "技能名（kebab-case，如 pdf-form-filler）"},
        "args": {
            "type": "object",
            "description": "可选参数；技能正文里可用 {args} 占位",
        },
    },
    "required": ["name"],
}

SKILL_LIST_DESCRIPTION = (
    "列出当前可用的技能（名字 + 一行描述 + 强相关技能）。\n"
    "何时用：不确定该用哪个技能时；用户问到「你能做什么」时。\n"
    "何时不用：已经知道技能名、要读它的正文时用 Skill；也不要用它替代真正的任务执行。\n"
    "副作用声明：只读，不修改任何文件。"
)

SKILL_LIST_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "filter": {"type": "string", "description": "只列出名字或描述含该关键词的技能"},
    },
    "required": [],
}

SKILL_SPEC = ToolSpec(
    name="Skill",
    description=SKILL_DESCRIPTION,
    parameters=SKILL_PARAMETERS,
    category="write",
    is_read_only=False,
    is_destructive=False,
    source="skill",
)

SKILL_LIST_SPEC = ToolSpec(
    name="SkillList",
    description=SKILL_LIST_DESCRIPTION,
    parameters=SKILL_LIST_PARAMETERS,
    category="read",
    is_read_only=True,
    is_destructive=False,
    source="skill",
)


class SkillTool:
    """``Skill`` 工具：返回技能正文（渐进披露）。"""

    def __init__(self, registry: SkillRegistry) -> None:
        """初始化。

        Args:
            registry: skill 注册表。
        """
        self.registry = registry
        self.spec = SKILL_SPEC

    def schema(self) -> dict[str, Any]:
        """返回发往 Provider 的 function schema。

        Returns:
            OpenAI 兼容 schema。
        """
        return self.spec.to_schema()

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """读取指定技能的正文。

        Args:
            args: ``name`` 与可选 ``args``。
            ctx: 执行上下文。

        Returns:
            :class:`~my_agent.models.tools.ToolResult`：``content`` 为 SKILL.md 正文。
        """
        name = str(args.get("name") or "").strip()
        if not name:
            return ToolResult(
                call_id=ctx.call_id, tool="Skill", ok=False, error="缺少 name 参数"
            )
        meta = self.registry.get(name)
        if meta is None:
            available = ", ".join(item.name for item in self.registry.list_skills()) or "（无）"
            return ToolResult(
                call_id=ctx.call_id,
                tool="Skill",
                ok=False,
                error=f"技能 {name} 不存在。可用技能：{available}",
            )
        body = meta.body()
        if not body:
            return ToolResult(
                call_id=ctx.call_id,
                tool="Skill",
                ok=False,
                error=f"技能 {name} 的 SKILL.md 正文为空或不可读：{meta.path}",
            )

        extra_args = args.get("args")
        if isinstance(extra_args, dict) and extra_args:
            body = _substitute(body, extra_args)

        parts = [f"# 技能：{meta.name}", "", body.strip()]
        scripts = meta.path / "scripts"
        if scripts.is_dir():
            files = sorted(p.name for p in scripts.iterdir() if p.is_file())
            if files:
                parts.extend(
                    [
                        "",
                        "---",
                        "该技能附带脚本（**不会自动执行**，需要时用 Read 查看、用 Bash 运行）：",
                        *[f"- {meta.path / 'scripts' / name}" for name in files],
                    ]
                )
        return ToolResult(
            call_id=ctx.call_id, tool="Skill", ok=True, content="\n".join(parts)
        )


class SkillListTool:
    """``SkillList`` 工具：列出可用技能。"""

    def __init__(self, registry: SkillRegistry) -> None:
        """初始化。

        Args:
            registry: skill 注册表。
        """
        self.registry = registry
        self.spec = SKILL_LIST_SPEC

    def schema(self) -> dict[str, Any]:
        """返回发往 Provider 的 function schema。

        Returns:
            OpenAI 兼容 schema。
        """
        return self.spec.to_schema()

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """列出技能。

        Args:
            args: 可选 ``filter``。
            ctx: 执行上下文。

        Returns:
            :class:`~my_agent.models.tools.ToolResult`。
        """
        needle = str(args.get("filter") or "").strip().lower()
        skills = self.registry.list_skills()
        if needle:
            skills = [
                meta
                for meta in skills
                if needle in meta.name.lower() or needle in meta.description.lower()
            ]
        if not skills:
            hint = f"（过滤词：{needle}）" if needle else ""
            return ToolResult(
                call_id=ctx.call_id,
                tool="SkillList",
                ok=True,
                content=f"没有匹配的技能{hint}。技能目录：{self.registry.skills_root()}",
            )
        lines = [f"共 {len(skills)} 个技能：", ""]
        for meta in skills:
            flag = " [router]" if meta.is_router else ""
            related = f"  · 强相关：{', '.join(meta.related)}" if meta.related else ""
            lines.append(f"- {meta.name}{flag}: {meta.description}{related}")
        return ToolResult(
            call_id=ctx.call_id, tool="SkillList", ok=True, content="\n".join(lines)
        )


def _substitute(text: str, values: dict[str, Any]) -> str:
    """把 ``{key}`` 占位替换为参数值（未知键保持原样，不抛异常）。

    Args:
        text: 技能正文。
        values: 参数。

    Returns:
        替换后的文本。
    """
    out = text
    for key, value in values.items():
        out = out.replace("{" + str(key) + "}", str(value))
    return out
