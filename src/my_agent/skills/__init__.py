"""Skills 与 skills 路由（§9）。"""

from __future__ import annotations

from my_agent.skills.dedupe import Bucket, bucket, is_synonym
from my_agent.skills.graph import GraphEdge, GraphNode, RelatedGraph
from my_agent.skills.registry import AddResult, SkillMeta, SkillRegistry
from my_agent.skills.router import rebuild, should_build, similar_pair_count
from my_agent.skills.similarity import (
    EmbeddingBackend,
    SimilarityBackend,
    TrigramCosineBackend,
    get_backend,
)
from my_agent.skills.skill_tool import SkillListTool, SkillTool

__all__ = [
    "AddResult",
    "Bucket",
    "EmbeddingBackend",
    "GraphEdge",
    "GraphNode",
    "RelatedGraph",
    "SimilarityBackend",
    "SkillListTool",
    "SkillMeta",
    "SkillRegistry",
    "SkillTool",
    "TrigramCosineBackend",
    "bucket",
    "get_backend",
    "is_synonym",
    "rebuild",
    "should_build",
    "similar_pair_count",
]
