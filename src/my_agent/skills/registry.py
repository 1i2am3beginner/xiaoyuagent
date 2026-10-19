"""Skill 注册表（§9.1）。

**Skill = 一个目录**：``<skills_root>/<skill-name>/SKILL.md``，frontmatter 声明
``name`` / ``description`` / ``when_to_use`` / ``allowed_tools`` / ``version``。

存储优先级（同名时高优先级覆盖低优先级）：

=========== ============================================ ==================
优先级       路径                                          用途
=========== ============================================ ==================
1           ``<project>/.myagent/skills/``                项目私有 skills
2           ``$MYAGENT_HOME/skills/``                     全局 skills
3           ``<project>/.claude/skills/``、``~/.claude``  **只读**兼容
=========== ============================================ ==================

注入上下文时只给 ``name + description``（≤2k），正文按需用 ``Skill`` 工具读取
（渐进披露，省 token）。
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, Field

from my_agent.errors import MyAgentError, SkillValidationError

if TYPE_CHECKING:
    from my_agent.config.schema import Config
    from my_agent.llm.client import LLMClient
    from my_agent.skills.graph import RelatedGraph

__all__ = ["AddResult", "SkillMeta", "SkillRegistry"]

_LOG = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"
NAME_MAX_LEN = 64
DESCRIPTION_MAX_LEN = 1024
ROUTER_MARKER = "generated-by: myagent router"


class SkillMeta(BaseModel):
    """一个 skill 的元信息（§9.1）。"""

    name: str
    description: str
    when_to_use: str | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    path: Path
    source: Literal["project", "global", "external"] = "project"
    content_hash: str = ""
    is_router: bool = False
    related: list[str] = Field(default_factory=list)
    version: str | None = None
    children: list[str] = Field(default_factory=list, description="路由 skill 的子技能")

    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}

    def one_line(self) -> str:
        """生成注入上下文用的一行描述。

        Returns:
            形如 ``- name: description`` 的文本。
        """
        text = " ".join(self.description.split())
        if len(text) > 120:
            text = text[:120] + "…"
        return f"- {self.name}: {text}"

    def body(self) -> str:
        """读取 SKILL.md 的正文（去掉 frontmatter）。

        Returns:
            正文 Markdown；读取失败返回空串。
        """
        try:
            raw = (self.path / SKILL_FILENAME).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return _split_frontmatter(raw)[1]


class AddResult(BaseModel):
    """添加 skill 的结果（§9.2 步骤 5）。"""

    added: bool
    reason: str
    deduped_with: str | None = None
    related: list[str] = Field(default_factory=list)
    router_rebuilt: bool = False
    skill: SkillMeta | None = None


class SkillRegistry:
    """skills 的装载、查询、添加与注入。"""

    def __init__(
        self,
        *,
        project_root: Path,
        home: Path,
        config: Config,
        llm: LLMClient | None = None,
    ) -> None:
        """初始化。

        Args:
            project_root: 项目根目录。
            home: 全局目录（``~/.myagent``）。
            config: 生效配置。
            llm: 模型客户端（同义判定与路由生成用；缺省则跳过需要模型的分支）。
        """
        self.project_root = Path(project_root)
        self.home = Path(home)
        self.config = config
        self.llm = llm
        self._skills: dict[str, SkillMeta] = {}
        self._graph: RelatedGraph | None = None

    # ------------------------------------------------------------------ 路径
    def skills_root(self) -> Path:
        """返回项目级 skills 目录（``/skills add`` 的落点）。

        Returns:
            项目级 skills 根目录。
        """
        return self.project_root / ".myagent" / "skills"

    def _source_roots(self) -> list[tuple[Path, Literal["project", "global", "external"]]]:
        """按优先级返回 (目录, 来源) 列表（高优先级在前）。

        Returns:
            三元组列表。
        """
        roots: list[tuple[Path, Literal["project", "global", "external"]]] = [
            (self.skills_root(), "project"),
            (self.home / "skills", "global"),
        ]
        if self.config.skills.read_external:
            roots.append((self.project_root / ".claude" / "skills", "external"))
            roots.append((Path.home() / ".claude" / "skills", "external"))
        return roots

    def graph(self) -> RelatedGraph:
        """返回强相关图（惰性加载）。

        Returns:
            :class:`~my_agent.skills.graph.RelatedGraph`。
        """
        if self._graph is None:
            from my_agent.skills.graph import RelatedGraph

            self._graph = RelatedGraph(self.skills_root() / "graph.json")
            self._graph.load()
        return self._graph

    # ------------------------------------------------------------------ 装载
    def load_all(self) -> list[SkillMeta]:
        """扫描全部 skills 目录并建立注册表（高优先级同名覆盖）。

        Returns:
            按名字排序的 :class:`SkillMeta` 列表；单个非法 skill 只告警并跳过。
        """
        found: dict[str, SkillMeta] = {}
        for root, source in self._source_roots():
            if not root.is_dir():
                continue
            for entry in sorted(root.iterdir()):
                skill_file = entry / SKILL_FILENAME
                if not entry.is_dir() or not skill_file.is_file():
                    continue
                try:
                    meta = self._parse(entry, skill_file, source)
                except MyAgentError as exc:
                    _LOG.warning("跳过非法 skill %s：%s", entry.name, exc.user_message())
                    continue
                found.setdefault(meta.name, meta)

        graph = self.graph()
        for name, meta in found.items():
            meta.related = graph.related_of(name)
        self._skills = found
        return self.list_skills()

    def list_skills(self) -> list[SkillMeta]:
        """返回当前注册表内容。

        Returns:
            按名字排序的列表。
        """
        return [self._skills[name] for name in sorted(self._skills)]

    def get(self, name: str) -> SkillMeta | None:
        """按名取 skill。

        Args:
            name: skill 名。

        Returns:
            :class:`SkillMeta` 或 ``None``。
        """
        return self._skills.get(name)

    def remove(self, name: str) -> None:
        """从注册表移除（不删除磁盘文件）。

        Args:
            name: skill 名。
        """
        self._skills.pop(name, None)

    # ------------------------------------------------------------------ 解析
    def _parse(
        self, directory: Path, skill_file: Path, source: Literal["project", "global", "external"]
    ) -> SkillMeta:
        """解析一个 SKILL.md。

        Args:
            directory: skill 目录。
            skill_file: ``SKILL.md`` 路径。
            source: 来源。

        Returns:
            :class:`SkillMeta`。

        Raises:
            SkillValidationError: frontmatter 缺失或字段非法。
        """
        try:
            raw = skill_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SkillValidationError(f"读取失败：{exc}", name=directory.name, detail=str(exc)) from exc

        front, body = _split_frontmatter(raw)
        if front is None:
            raise SkillValidationError(
                "缺少 YAML frontmatter（--- 包裹）",
                name=directory.name,
                detail="missing frontmatter",
            )
        name = str(front.get("name") or "").strip()
        description = str(front.get("description") or "").strip()
        if not name:
            raise SkillValidationError("frontmatter 缺少 name", name=directory.name, detail="no name")
        if len(name) > NAME_MAX_LEN:
            raise SkillValidationError(
                f"name 超过 {NAME_MAX_LEN} 字符", name=name, detail="name too long"
            )
        if not description:
            raise SkillValidationError(
                "frontmatter 缺少 description", name=name, detail="no description"
            )
        if len(description) > DESCRIPTION_MAX_LEN:
            raise SkillValidationError(
                f"description 超过 {DESCRIPTION_MAX_LEN} 字符", name=name, detail="desc too long"
            )

        allowed = front.get("allowed_tools") or []
        if not isinstance(allowed, list):
            allowed = []
        children = front.get("children") or []
        if not isinstance(children, list):
            children = []
        is_router = bool(front.get("is_router")) or ROUTER_MARKER in raw

        return SkillMeta(
            name=name,
            description=description,
            when_to_use=(str(front["when_to_use"]) if front.get("when_to_use") else None),
            allowed_tools=[str(item) for item in allowed],
            path=directory,
            source=source,
            content_hash=hashlib.sha1(body.encode("utf-8")).hexdigest(),
            is_router=is_router,
            version=(str(front["version"]) if front.get("version") else None),
            children=[str(item) for item in children],
        )

    # ------------------------------------------------------------------ 添加
    def add(self, path: Path, *, force: bool = False) -> AddResult:
        """添加一个 skill，执行 §9.2 的三级相似度流程。

        Args:
            path: skill 目录（含 SKILL.md）或 SKILL.md 自身。
            force: 同名不同内容时是否覆盖。

        Returns:
            :class:`AddResult`。

        Raises:
            SkillValidationError: 目录或 frontmatter 非法。
        """
        directory = path if path.is_dir() else path.parent
        skill_file = directory / SKILL_FILENAME
        if not skill_file.is_file():
            raise SkillValidationError(
                f"找不到 {SKILL_FILENAME}", name=directory.name, detail=str(skill_file)
            )
        meta = self._parse(directory, skill_file, "project")

        existing = self._skills.get(meta.name)
        if existing is not None:
            if existing.content_hash == meta.content_hash:
                return AddResult(
                    added=False,
                    reason=f"已存在同名且内容相同的 skill：{meta.name}",
                    deduped_with=meta.name,
                )
            if not force:
                raise SkillValidationError(
                    "同名 skill 已存在且内容不同；请改名或显式 force 覆盖",
                    name=meta.name,
                    detail=str(existing.path),
                )

        outcome = self._similarity_flow(meta)
        if not outcome.added:
            return outcome

        target = self.skills_root() / meta.name
        try:
            if directory.resolve() != target.resolve():
                if target.exists():
                    shutil.rmtree(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(directory, target)
        except OSError as exc:
            raise SkillValidationError(
                f"复制 skill 失败：{exc}", name=meta.name, detail=str(exc)
            ) from exc

        meta.path = target
        meta.source = "project"
        self._skills[meta.name] = meta
        graph = self.graph()
        for other in outcome.related:
            graph.add_edge(meta.name, other, sim=0.0)
            peer = self._skills.get(other)
            if peer is not None and meta.name not in peer.related:
                peer.related.append(meta.name)
        graph.save()
        outcome.skill = meta

        outcome.router_rebuilt = self._maybe_rebuild_router()
        return outcome

    def _similarity_flow(self, meta: SkillMeta) -> AddResult:
        """执行三级相似度分桶（§9.2 步骤 2–3）。

        Args:
            meta: 待添加的 skill。

        Returns:
            :class:`AddResult`；``added=False`` 表示被去重。
        """
        from my_agent.skills.dedupe import bucket, is_synonym

        if not self._skills:
            return AddResult(added=True, reason="技能库为空，直接添加")

        backend = self._backend()
        new_text = _similarity_text(meta)
        best: tuple[float, str] = (0.0, "")
        for name, other in self._skills.items():
            sim = backend.similarity(new_text, _similarity_text(other))
            if sim > best[0]:
                best = (sim, name)

        sim, other_name = best
        kind = bucket(sim)
        if kind == "dedupe":
            return AddResult(
                added=False,
                reason=f"与 <{other_name}> 为同义技能（sim={sim:.2f}），已跳过",
                deduped_with=other_name,
            )
        if kind == "llm_check":
            other = self._skills[other_name]
            same = is_synonym(self.llm, meta, other, config=self.config)
            if same:
                return AddResult(
                    added=False,
                    reason=f"LLM 判定与 <{other_name}> 同义（sim={sim:.2f}），已跳过",
                    deduped_with=other_name,
                )
            return AddResult(
                added=True,
                reason=f"与 <{other_name}> 相似但经 LLM 判定不同义（sim={sim:.2f}）",
                related=[other_name],
            )
        if kind == "related":
            return AddResult(
                added=True,
                reason=f"与 <{other_name}> 强相关（sim={sim:.2f}）",
                related=[other_name],
            )
        return AddResult(added=True, reason="无相近技能，直接添加")

    def _backend(self) -> Any:
        """构造相似度后端（§9.2 的 L1/L2）。

        Returns:
            :class:`~my_agent.skills.similarity.SimilarityBackend` 实现。
        """
        from my_agent.skills.similarity import get_backend

        return get_backend(self.config)

    def _maybe_rebuild_router(self) -> bool:
        """按 §9.4 触发条件决定是否重建路由层（同步轻量检查）。

        Returns:
            是否触发了重建检查。
        """
        from my_agent.skills.router import should_build

        return should_build(self.list_skills(), self.graph())

    # ------------------------------------------------------------------ 注入
    def block_for_context(self) -> str:
        """生成注入系统提示词的 skills 索引块（§9.1）。

        路由 skill **替代**其子技能条目；子技能仍可被显式调用。

        Returns:
            Markdown 块；无 skill 时返回空串。
        """
        skills = self.list_skills()
        if not skills:
            return ""
        routers = [meta for meta in skills if meta.is_router]
        replaced: set[str] = set()
        for router in routers:
            replaced.update(router.children)
        visible = [meta for meta in skills if meta.name not in replaced]
        lines = ["## 可用技能（skills）", ""]
        for meta in visible:
            lines.append(meta.one_line())
        lines.append("")
        lines.append("需要某个技能的完整说明时，用 Skill 工具按名字读取正文。")
        return "\n".join(lines)


def _similarity_text(meta: SkillMeta) -> str:
    """拼出用于相似度比较的文本（name + description + when_to_use）。

    Args:
        meta: skill 元信息。

    Returns:
        归一化前的原始文本。
    """
    parts = [meta.name, meta.description]
    if meta.when_to_use:
        parts.append(meta.when_to_use)
    return " ".join(parts)


def _split_frontmatter(raw: str) -> tuple[dict[str, Any] | None, str]:
    """拆分 SKILL.md 的 YAML frontmatter 与正文。

    Args:
        raw: 文件全文。

    Returns:
        ``(frontmatter 字典或 None, 正文)``。
    """
    text = raw.lstrip("\ufeff")
    if not text.startswith("---"):
        return None, text
    parts = text.split("\n", 1)
    if len(parts) < 2:
        return None, text
    rest = parts[1]
    end = rest.find("\n---")
    if end == -1:
        return None, text
    front_text = rest[:end]
    body = rest[end + 4 :].lstrip("\n")
    try:
        data = yaml.safe_load(front_text)
    except yaml.YAMLError:
        return None, text
    if not isinstance(data, dict):
        return None, text
    return data, body
