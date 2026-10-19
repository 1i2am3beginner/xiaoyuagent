"""路由层（大 Skill）生成（§9.5）。

上游：*采用一个大 Skill 下分发其余具有强相关的 skills，LLM 直接调用大 skill，
具体调用什么 skills 由大 skill 决定。*

本文档实现（U10）：**每个强相关簇生成一个路由大 skill**（``ROUTER_MODE =
"per_cluster"``），孤立节点不进路由、保持可被直接调用。切 ``single_global``
则只生成一个全局路由。

触发条件（§9.4 / U9）：``len(skills) > 15`` **且** 相似对 ``> 5``（严格大于）。
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from my_agent.config import defaults as D
from my_agent.errors import MyAgentError

if TYPE_CHECKING:
    from my_agent.llm.client import LLMClient
    from my_agent.skills.graph import RelatedGraph
    from my_agent.skills.registry import SkillMeta, SkillRegistry

__all__ = ["router_slug", "rebuild", "should_build", "similar_pair_count"]

_LOG = logging.getLogger(__name__)

ROUTER_ROLE = "router_gen"
ROUTER_HEADER = "<!-- generated-by: myagent router v1; do not edit -->"
ROUTER_PREFIX = "router-"
SLUG_RE = re.compile(r"[^a-z0-9]+")


def similar_pair_count(skills: list[SkillMeta], graph: RelatedGraph) -> int:
    """计算「工具相似个数」（§9.4 / U9 的精确定义）。

    = 强相关对数 ∪ ``allowed_tools`` 交集 Jaccard ≥ 阈值的对数（去重后）。

    Args:
        skills: 全部 skills。
        graph: 强相关图。

    Returns:
        相似对数量。
    """
    tools = {meta.name: list(meta.allowed_tools) for meta in skills}
    return graph.similar_count(tools)


def should_build(skills: list[SkillMeta], graph: RelatedGraph) -> bool:
    """判断是否满足路由层触发条件（§9.4）。

    Args:
        skills: 全部 skills。
        graph: 强相关图。

    Returns:
        满足 ``len(skills) > 15`` 且相似对 ``> 5`` 时 ``True``。
    """
    if len(skills) <= D.ROUTER_MIN_SKILLS:
        return False
    return similar_pair_count(skills, graph) > D.ROUTER_MIN_RELATED_PAIRS


def router_slug(names: list[str]) -> str:
    """把一组子技能名压成一个路由 skill 的 slug。

    Args:
        names: 子技能名列表。

    Returns:
        形如 ``document-tools`` 的 slug（kebab-case）。
    """
    if len(names) == 1:
        return names[0]
    joined = "-".join(sorted(names))
    slug = SLUG_RE.sub("-", joined.lower()).strip("-")
    if len(slug) <= 40:
        return slug
    return slug[:40].rstrip("-")


async def rebuild(
    registry: SkillRegistry,
    skills: list[SkillMeta] | None = None,
    *,
    llm: LLMClient | None = None,
    force: bool = False,
) -> list[SkillMeta]:
    """重建路由层（``/skills route``、``/skills dedupe`` 与添加后触发共用）。

    Args:
        registry: skill 注册表。
        skills: 参与聚类的 skills；缺省取注册表当前内容。
        llm: 模型客户端（生成 description/正文）；``None`` 时用确定性模板。
        force: 覆盖用户手工编辑过的路由文件。

    Returns:
        本次生成/更新的路由 :class:`~my_agent.skills.registry.SkillMeta` 列表。
    """

    all_skills = list(skills if skills is not None else registry.list_skills())
    graph = registry.graph()
    mode = getattr(registry.config.skills, "router_mode", D.ROUTER_MODE)

    if not all_skills:
        return []

    if mode == "single_global":
        groups = [([meta for meta in all_skills if not meta.is_router], ["all"])]
    else:
        clusters = graph.clusters(max_size=registry.config.skills.router_cluster_max)
        groups = []
        for cluster in clusters:
            members = [registry.get(name) for name in cluster]
            concrete = [meta for meta in members if meta is not None and not meta.is_router]
            if len(concrete) >= 2:
                groups.append((concrete, cluster))

    built: list[SkillMeta] = []
    for members, _cluster_names in groups:
        slug = router_slug(sorted(meta.name for meta in members))
        body = await _generate(registry, members, llm=llm)
        target = registry.skills_root() / f"{ROUTER_PREFIX}{slug}"
        skill_file = target / "SKILL.md"
        if skill_file.is_file() and not force:
            existing = skill_file.read_text(encoding="utf-8", errors="replace")
            if ROUTER_HEADER not in existing:
                _LOG.warning("路由文件被手工编辑，跳过覆盖（加 force 可强制）：%s", skill_file)
                continue
        try:
            target.mkdir(parents=True, exist_ok=True)
            skill_file.write_text(body, encoding="utf-8")
        except OSError as exc:
            _LOG.warning("写入路由 skill 失败：%s（%s）", skill_file, exc)
            continue

        try:
            meta = registry._parse(target, skill_file, "project")
        except MyAgentError as exc:
            _LOG.warning("解析新生成的路由 skill 失败：%s", exc.user_message())
            continue
        meta.is_router = True
        meta.children = sorted(m.name for m in members)
        registry._skills[meta.name] = meta
        built.append(meta)

    if built:
        _LOG.info("路由层重建完成：%s", ", ".join(meta.name for meta in built))
    else:
        _LOG.info("路由层未变化（%s 个簇，%s 模式）", len(groups), mode)
    return built


async def _generate(
    registry: SkillRegistry, members: list[SkillMeta], *, llm: LLMClient | None
) -> str:
    """生成一个路由 SKILL.md 内容。

    Args:
        registry: 注册表（取配置）。
        members: 该簇的子技能。
        llm: 模型客户端。

    Returns:
        完整 SKILL.md 文本（含前导生成标记）。
    """
    names = sorted(meta.name for meta in members)
    slug = router_slug(names)
    router_name = f"{ROUTER_PREFIX}{slug}"
    children_detail = "\n".join(
        f"- {meta.name}: {meta.description}" for meta in sorted(members, key=lambda m: m.name)
    )

    if llm is not None:
        from my_agent.config.defaults import TIMEOUT_SMALL_S
        from my_agent.llm.prompts import ROUTER_GEN_PROMPT, render

        prompt = render(
            ROUTER_GEN_PROMPT,
            router_name=router_name,
            router_description_placeholder="（请填写何时使用本路由技能）",
            children_list=str(names),
            children_detail=children_detail,
        )
        try:
            from my_agent.skills.dedupe import _run_sync

            result = _run_sync(
                llm.complete_role(
                    ROUTER_ROLE,
                    user=prompt,
                    max_tokens=1_500,
                    temperature=0.0,
                    timeout_s=TIMEOUT_SMALL_S,
                )
            )
            text = (getattr(result, "text", "") or "").strip()
            if text:
                return _ensure_header(text, names)
        except Exception as exc:  # 模型不可用 → 退化为确定性模板
            _LOG.warning("路由 skill 生成调用失败，改用确定性模板：%s", exc)

    return _deterministic_body(router_name, members)


def _ensure_header(text: str, names: list[str]) -> str:
    """确保生成文本带生成标记（§9.5 步骤 6）。

    Args:
        text: 模型输出。
        names: 子技能名。

    Returns:
        带标记的文本。
    """
    header = ROUTER_HEADER
    if ROUTER_MARKER_IN(text):
        return text
    if text.lstrip().startswith("---"):
        parts = text.split("\n---", 1)
        if len(parts) == 2:
            return f"{header}\n{parts[0]}\n---{parts[1]}"
    return f"{header}\n{text}\n\n<!-- children: {', '.join(names)} -->"


def ROUTER_MARKER_IN(text: str) -> bool:  # noqa: N802 - 与常量同名的判定函数
    """判断文本是否已含生成标记。

    Args:
        text: 待检查文本。

    Returns:
        含标记时 ``True``。
    """
    return "generated-by: myagent router" in text


def _deterministic_body(router_name: str, members: list[SkillMeta]) -> str:
    """模型不可用时的确定性路由正文（§7.7.4 同款兜底思路）。

    Args:
        router_name: 路由 skill 名。
        members: 子技能。

    Returns:
        完整 SKILL.md 文本。
    """
    names = sorted(meta.name for meta in members)
    lines = [
        ROUTER_HEADER,
        "---",
        f"name: {router_name}",
        f"description: 强相关技能组（{', '.join(names)}）。当用户请求涉及这些技能的领域时使用；"
        "本路由技能会指出应调用哪个子技能。",
        "is_router: true",
        f"children: {names}",
        "---",
        "",
        "# 说明",
        "",
        "你是本技能组的路由者。根据用户请求，选择并调用下列子技能之一（可多次）：",
        "",
    ]
    for meta in sorted(members, key=lambda m: m.name):
        lines.append(f"- {meta.name}：{meta.description}")
    lines.extend(
        [
            "",
            "# 决策规则",
            "",
            "1. 先判断用户请求的**产出物类型**（文档/表格/代码/数据），命中哪个子技能的描述就直接调用它。",
            "2. 若一个请求需要多个子技能配合，按「读取 → 处理 → 导出」的顺序依次调用。",
            "3. 若没有任何子技能匹配，不要调用本路由技能下的任何子技能，改为直接处理或询问用户。",
            "4. 不要臆造子技能不具备的能力。",
            "",
            "用 Skill 工具按名字读取所选子技能的完整说明后再执行。",
            "",
        ]
    )
    return "\n".join(lines)


def router_paths(registry: SkillRegistry) -> list[Any]:
    """返回当前已生成的路由 skill 路径。

    Args:
        registry: 注册表。

    Returns:
        路由 SKILL.md 路径列表。
    """
    return [meta.path / "SKILL.md" for meta in registry.list_skills() if meta.is_router]
