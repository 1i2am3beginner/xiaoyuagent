"""强相关图 ``related``（§9.3、§9.4；U9 / U25）。

上游规则：``0.75 ≤ sim < 0.85`` 的两个 skill 互为**强相关**，用于后续路由：

* 强相关是**双向**的（``a.related`` 含 ``b`` ⇒ ``b.related`` 含 ``a``）。
* 落盘 ``<project>/.myagent/skills/graph.json``，形状**逐字**采用 §9.3：

  .. code-block:: json

     {"nodes": {"pdf-form-filler": {"desc": "..."}},
      "edges": [{"a": "pdf-form-filler", "b": "excel-export", "sim": 0.81, "kind": "related"}]}

* ``related_pair_count`` = ``edges`` 中 ``kind == "related"`` 的对数。
* ``similar_count`` = ``related_pair_count`` ∪ ``allowed_tools`` 重叠对（Jaccard ≥
  :data:`~my_agent.config.defaults.TOOL_OVERLAP_JACCARD`，去重），用于 §9.4 触发条件。
* 边权时间衰减（【待确认假设 U25】，上游未提）：每 24h 衰减
  :data:`~my_agent.config.defaults.GRAPH_EDGE_DECAY_PER_DAY`，低于
  :data:`~my_agent.config.defaults.GRAPH_EDGE_MIN` 的边被移除。

**衰减的时间基准（实现说明）**：为了保持 §9.3 规定的 JSON 形状**不多一个键**，
本模块不把时间戳写进 ``edges``，而是以 ``graph.json`` 的**上次落盘时间（mtime）**
作为衰减起点：``load()`` 时按 ``(now - mtime)`` 的天数衰减，并在确有变化时立即回写
（回写即刷新 mtime，保证「每 24h 只衰减一次」，不会重复计息）。

本模块不做任何权限判定，也不读写 skill 正文，只维护图。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from my_agent.config.defaults import (
    GRAPH_EDGE_DECAY_PER_DAY,
    GRAPH_EDGE_MIN,
    ROUTER_CLUSTER_MAX,
    TOOL_OVERLAP_JACCARD,
)

__all__ = [
    "GRAPH_FILE_NAME",
    "RELATED_KIND",
    "SECONDS_PER_DAY",
    "GraphEdge",
    "GraphNode",
    "RelatedGraph",
    "pair_key",
]

_LOG = logging.getLogger(__name__)

#: 图文件的文件名（位于 ``<project>/.myagent/skills/``）。
GRAPH_FILE_NAME = "graph.json"

#: 一条边的种类：目前只有「强相关」一种（§9.3）。
RELATED_KIND = "related"

#: 一天的秒数（单位换算常量，不是可调阈值）。
SECONDS_PER_DAY = 86_400.0

#: 衰减累计小于该值时不改写文件（避免每次 load 都因为浮点噪声回写）。
_DECAY_EPSILON = 1e-6

#: 落盘时 ``sim`` 保留的小数位（避免浮点尾巴污染 diff）。
_SIM_DECIMALS = 6


def pair_key(a: str, b: str) -> tuple[str, str]:
    """把无向边的两个端点规范化为字典序元组（保证 ``(a, b)`` 与 ``(b, a)`` 同键）。

    Args:
        a: 端点 A 的 skill 名。
        b: 端点 B 的 skill 名。

    Returns:
        排序后的 ``(小, 大)`` 元组。
    """
    return (a, b) if a <= b else (b, a)


class GraphNode(BaseModel):
    """图节点（一个 skill 的轻量快照，不存正文）。"""

    model_config = ConfigDict(extra="ignore")

    desc: str = Field(default="", description="skill 的 description（便于人工查看 graph.json）")


class GraphEdge(BaseModel):
    """一条**无向**强相关边。

    ``a`` 与 ``b`` 恒为字典序（见 :func:`pair_key`），因此同一条边只存一份。
    """

    model_config = ConfigDict(extra="ignore")

    a: str = Field(description="端点 A（字典序较小者）")
    b: str = Field(description="端点 B（字典序较大者）")
    sim: float = Field(description="当时的相似度（会被时间衰减）")
    kind: str = Field(default=RELATED_KIND, description="边种类；目前只有 related")

    def other(self, name: str) -> str:
        """返回边的另一端。

        Args:
            name: 已知的一端。

        Returns:
            另一端的 skill 名；``name`` 不在边上时返回空串。
        """
        if name == self.a:
            return self.b
        if name == self.b:
            return self.a
        return ""

    def key(self) -> tuple[str, str]:
        """返回该边的规范化键。"""
        return pair_key(self.a, self.b)


class RelatedGraph:
    """强相关图：加载 / 保存 / 增量更新 / 衰减 / 聚类（§9.3、§9.5）。

    Args:
        path: ``graph.json`` 的完整路径。
        decay_per_day: 每日衰减比例（默认 :data:`GRAPH_EDGE_DECAY_PER_DAY`）。
        edge_min: 低于该权重的边被移除（默认 :data:`GRAPH_EDGE_MIN`）。
        overlap_jaccard: ``allowed_tools`` 相似对的 Jaccard 阈值
            （默认 :data:`TOOL_OVERLAP_JACCARD`）。

    Attributes:
        nodes: skill 名 → :class:`GraphNode`。
        edges: 全部边（无向，每条只存一份）。
    """

    def __init__(
        self,
        path: Path,
        *,
        decay_per_day: float = GRAPH_EDGE_DECAY_PER_DAY,
        edge_min: float = GRAPH_EDGE_MIN,
        overlap_jaccard: float = TOOL_OVERLAP_JACCARD,
    ) -> None:
        self._path = Path(path)
        self._decay_per_day = float(decay_per_day)
        self._edge_min = float(edge_min)
        self._overlap_jaccard = float(overlap_jaccard)
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        #: 衰减起点（= 上次落盘时间）；未落盘时为构造时刻。
        self._epoch: float = time.time()
        self._decay_applied = False

    # ------------------------------------------------------------------ 基础属性
    @property
    def path(self) -> Path:
        """``graph.json`` 的路径。"""
        return self._path

    @property
    def decay_applied(self) -> bool:
        """上一次 :meth:`decay` 是否真的改动了边权（供 ``load()`` 决定是否回写）。"""
        return self._decay_applied

    @classmethod
    def for_project(cls, project_root: Path) -> RelatedGraph:
        """按项目根构造（路径 = ``<project>/.myagent/skills/graph.json``）。

        Args:
            project_root: 项目根目录。

        Returns:
            尚未加载的空图对象。
        """
        return cls(Path(project_root) / ".myagent" / "skills" / GRAPH_FILE_NAME)

    # ------------------------------------------------------------------ 读写
    def load(self) -> RelatedGraph:
        """从磁盘加载图，并按「距上次落盘的天数」做一次时间衰减。

        文件不存在或损坏时按空图处理（记 warning，不抛异常——图是缓存性质的数据，
        丢失只会退化为「无强相关」，不应阻断启动）。

        Returns:
            ``self``（便于链式调用）。
        """
        if not self._path.exists():
            self._epoch = time.time()
            return self
        self._epoch = self._path.stat().st_mtime
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _LOG.warning("graph.json 读取失败（按空图处理）：%s", exc)
            return self
        self._restore(raw)
        removed = self.decay()
        if removed or self._decay_applied:
            self.save()
        return self

    def save(self) -> None:
        """把图写回磁盘（先写临时文件再 ``replace``，避免半截 JSON）。

        Raises:
            OSError: 目录不可创建或写入失败。
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(f"{self._path.name}.tmp")
        tmp.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._path)
        self._epoch = time.time()

    def to_json(self) -> dict[str, Any]:
        """导出 §9.3 规定的 JSON 形状（键序稳定，便于 diff 与测试）。

        Returns:
            ``{"nodes": {name: {"desc": str}}, "edges": [{a, b, sim, kind}, ...]}``。
        """
        return {
            "nodes": {
                name: {"desc": node.desc} for name, node in sorted(self.nodes.items())
            },
            "edges": [
                {
                    "a": edge.a,
                    "b": edge.b,
                    "sim": round(edge.sim, _SIM_DECIMALS),
                    "kind": edge.kind,
                }
                for edge in sorted(self.edges, key=lambda item: item.key())
            ],
        }

    def _restore(self, raw: Any) -> None:
        """用磁盘上的字典重建内存状态（容忍缺字段与脏数据）。"""
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        if isinstance(raw, dict):
            raw_nodes = raw.get("nodes")
            if isinstance(raw_nodes, dict):
                for name, payload in raw_nodes.items():
                    if not isinstance(name, str):
                        continue
                    desc = payload.get("desc", "") if isinstance(payload, dict) else ""
                    nodes[name] = GraphNode(desc=str(desc))
            raw_edges = raw.get("edges")
            if isinstance(raw_edges, list):
                seen: set[tuple[str, str]] = set()
                for item in raw_edges:
                    if not isinstance(item, dict):
                        continue
                    a, b = item.get("a"), item.get("b")
                    if not isinstance(a, str) or not isinstance(b, str) or a == b:
                        continue
                    key = pair_key(a, b)
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        sim = float(item.get("sim", 0.0))
                    except (TypeError, ValueError):
                        continue
                    kind = item.get("kind", RELATED_KIND)
                    edges.append(
                        GraphEdge(a=key[0], b=key[1], sim=sim, kind=str(kind))
                    )
        self.nodes = nodes
        self.edges = edges

    # ------------------------------------------------------------------ 增量更新
    def set_node(self, name: str, desc: str = "") -> None:
        """新增或更新一个节点（不触发存盘，由调用方在批量更新后 :meth:`save`）。

        Args:
            name: skill 名。
            desc: skill 的 description。
        """
        self.nodes[name] = GraphNode(desc=desc)

    def add_edge(self, a: str, b: str, sim: float, *, kind: str = RELATED_KIND) -> GraphEdge:
        """新增/更新一条**双向**边（同一条边只存一份，自环被忽略）。

        Args:
            a: 端点 A。
            b: 端点 B。
            sim: 相似度（写入前按 :data:`GRAPH_EDGE_MIN` 语义由调用方判断，本方法照存）。
            kind: 边种类，默认 ``"related"``。

        Returns:
            写入后的边对象；``a == b`` 时返回一个 ``sim=0`` 的占位边且不写入。
        """
        if a == b:
            return GraphEdge(a=a, b=b, sim=0.0, kind=kind)
        key = pair_key(a, b)
        for name in key:
            self.nodes.setdefault(name, GraphNode())
        for edge in self.edges:
            if edge.key() == key:
                edge.sim = float(sim)
                edge.kind = kind
                return edge
        edge = GraphEdge(a=key[0], b=key[1], sim=float(sim), kind=kind)
        self.edges.append(edge)
        return edge

    def remove_edge(self, a: str, b: str) -> bool:
        """删除一条边。

        Args:
            a: 端点 A。
            b: 端点 B。

        Returns:
            删除成功返回 ``True``；边不存在返回 ``False``。
        """
        key = pair_key(a, b)
        before = len(self.edges)
        self.edges = [edge for edge in self.edges if edge.key() != key]
        return len(self.edges) != before

    def remove_node(self, name: str) -> None:
        """删除节点及其全部关联边（``remove`` 后的增量更新）。

        Args:
            name: skill 名。
        """
        self.nodes.pop(name, None)
        self.edges = [edge for edge in self.edges if name not in (edge.a, edge.b)]

    # ------------------------------------------------------------------ 查询
    def edge_between(self, a: str, b: str) -> GraphEdge | None:
        """取两个 skill 之间的边。

        Args:
            a: 端点 A。
            b: 端点 B。

        Returns:
            边对象；不存在返回 ``None``。
        """
        key = pair_key(a, b)
        for edge in self.edges:
            if edge.key() == key:
                return edge
        return None

    def related_of(self, name: str) -> list[str]:
        """列出与 ``name`` 强相关的其它 skill 名（字典序）。

        Args:
            name: skill 名。

        Returns:
            强相关 skill 名列表。
        """
        out = [
            edge.other(name)
            for edge in self.edges
            if edge.kind == RELATED_KIND and name in (edge.a, edge.b)
        ]
        return sorted(item for item in out if item)

    def related_pairs(self) -> set[tuple[str, str]]:
        """返回全部强相关对（规范化键集合）。

        Returns:
            ``{(a, b), ...}``，去重后的无序对集合。
        """
        return {edge.key() for edge in self.edges if edge.kind == RELATED_KIND}

    def related_pair_count(self) -> int:
        """强相关对数（§9.3 的 ``related_pair_count``）。

        Returns:
            ``kind == "related"`` 的边数。
        """
        return len(self.related_pairs())

    def tool_overlap_pairs(
        self, tools: Mapping[str, Sequence[str]]
    ) -> set[tuple[str, str]]:
        """计算 ``allowed_tools`` 重叠产生的相似对（§9.4）。

        判定：两个 skill 的工具白名单**交集非空**且 ``Jaccard ≥ 阈值``。

        Args:
            tools: skill 名 → ``allowed_tools`` 列表。

        Returns:
            规范化后的相似对集合。
        """
        names = sorted(name for name in tools if tools[name])
        out: set[tuple[str, str]] = set()
        for index, first in enumerate(names):
            set_a = set(tools[first])
            for second in names[index + 1 :]:
                set_b = set(tools[second])
                union = set_a | set_b
                if not union:
                    continue
                jaccard = len(set_a & set_b) / len(union)
                if jaccard >= self._overlap_jaccard:
                    out.add(pair_key(first, second))
        return out

    def tool_overlap_pair_count(self, tools: Mapping[str, Sequence[str]]) -> int:
        """``allowed_tools`` 重叠相似对的个数（去重、不含已是强相关的对）。

        Args:
            tools: skill 名 → ``allowed_tools`` 列表。

        Returns:
            仅由工具重叠贡献的相似对数。
        """
        return len(self.tool_overlap_pairs(tools) - self.related_pairs())

    def similar_count(self, tools: Mapping[str, Sequence[str]]) -> int:
        """「工具相似个数」（§9.4 / U9）= 强相关对 ∪ 工具重叠对，去重计数。

        Args:
            tools: skill 名 → ``allowed_tools`` 列表。

        Returns:
            去重后的相似对总数。
        """
        return len(self.related_pairs() | self.tool_overlap_pairs(tools))

    # ------------------------------------------------------------------ 衰减
    def decay(self, *, now: float | None = None) -> int:
        """按距上次落盘的天数衰减全部边权，并移除低于阈值的边（U25）。

        公式：``sim ← sim × (1 - decay_per_day) ** elapsed_days``。

        Args:
            now: 覆盖「当前时间」（测试注入用）；``None`` = :func:`time.time`。

        Returns:
            被移除的边数。
        """
        current = time.time() if now is None else float(now)
        elapsed_days = (current - self._epoch) / SECONDS_PER_DAY
        if elapsed_days <= 0.0:
            self._decay_applied = False
            return 0
        factor = (1.0 - self._decay_per_day) ** elapsed_days
        if factor >= 1.0 or (1.0 - factor) < _DECAY_EPSILON:
            self._decay_applied = False
            return 0
        kept: list[GraphEdge] = []
        removed = 0
        for edge in self.edges:
            edge.sim = edge.sim * factor
            if edge.sim < self._edge_min:
                removed += 1
                continue
            kept.append(edge)
        self.edges = kept
        self._epoch = current
        self._decay_applied = True
        return removed

    # ------------------------------------------------------------------ 聚类
    def clusters(
        self,
        *,
        max_size: int = ROUTER_CLUSTER_MAX,
        exclude: Iterable[str] = (),
    ) -> list[list[str]]:
        """贪心归并强相关图，得到路由聚类（§9.5 步骤 1）。

        算法：按边权从高到低遍历强相关边，把两个分量归并——仅当归并后分量大小
        ``≤ max_size``；已同属一个分量或会超限的边跳过。孤立节点（无强相关）
        **不进路由**，因此只返回成员数 ``≥ 2`` 的簇。

        Args:
            max_size: 单个簇的最大成员数（默认 :data:`ROUTER_CLUSTER_MAX`）。
            exclude: 不参与聚类的节点（如已生成的路由大 skill）。

        Returns:
            簇列表，每项是字典序排列的 skill 名列表；按（成员数降序，首名升序）稳定排序。
        """
        blocked = set(exclude)
        parent: dict[str, str] = {
            name: name for name in self.nodes if name not in blocked
        }
        size: dict[str, int] = dict.fromkeys(parent, 1)

        def find(name: str) -> str:
            root = name
            while parent[root] != root:
                root = parent[root]
            while parent[name] != root:
                parent[name], name = root, parent[name]
            return root

        candidates = [
            edge
            for edge in self.edges
            if edge.kind == RELATED_KIND and edge.a in parent and edge.b in parent
        ]
        for edge in sorted(candidates, key=lambda item: (-item.sim, item.key())):
            root_a, root_b = find(edge.a), find(edge.b)
            if root_a == root_b:
                continue
            if size[root_a] + size[root_b] > max_size:
                continue
            keeper, merged = sorted((root_a, root_b))
            parent[merged] = keeper
            size[keeper] = size[root_a] + size[root_b]
            size.pop(merged, None)

        groups: dict[str, list[str]] = {}
        for name in sorted(parent):
            groups.setdefault(find(name), []).append(name)
        result = [members for members in groups.values() if len(members) >= 2]
        result.sort(key=lambda members: (-len(members), members[0]))
        return result
