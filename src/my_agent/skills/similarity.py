"""技能相似度后端与相似度缓存（§9.2；U8）。

上游要求「余弦相似度」，但未说明向量来源（【待确认假设 U8】）。本模块给出两层实现，
**默认 L1**，可用 ``skills.similarity_backend = "embedding"`` 切到 L2：

======================  ==========================================================================
L1（默认，零依赖）      :class:`TrigramCosineBackend`：``name + description + when_to_use``
                        归一化后构造**字符 trigram 词频向量**，取「真余弦」与
                        ``rapidfuzz.fuzz.token_set_ratio / 100`` 的**最大值**。
L2（可选，需配置）      :class:`EmbeddingBackend`：调用 OpenAI 兼容 ``/embeddings`` 端点
                        （``skills.embedding_provider`` / ``skills.embedding_model``）取真余弦。
======================  ==========================================================================

相似度缓存落在 ``<project>/.myagent/skills/.simcache.json``，键为
``sha1(hash_a + hash_b + backend_id)``（两个 hash 先按字典序规范化，保证对称命中）。
``content_hash`` 一变，键就变——**自动失效**，无需额外清理（E-8）。

同步 / 异步桥接（实现说明）：§9.2 的 :class:`SimilarityBackend` 协议是**同步**的
（``similarity(a, b) -> float``），而事件循环内的调用点（``SkillRegistry.add``）也是同步
API（CONTRACTS.md §17.2 冻结）。因此需要把协程（LLM 判定 / embedding 请求）从同步栈里
驱动起来，统一走 :func:`run_coroutine_blocking`。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Coroutine, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from rapidfuzz import fuzz

from my_agent.config.defaults import TIMEOUT_SMALL_S

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查器
    from my_agent.config.schema import Config, EndpointConfig

__all__ = [
    "CACHE_FILE_NAME",
    "EMBEDDINGS_PATH",
    "TRIGRAM_BACKEND_ID",
    "TRIGRAM_SIZE",
    "CosineLike",
    "Embedder",
    "EmbeddingBackend",
    "HttpEmbedder",
    "SimilarityBackend",
    "SimilarityCache",
    "TrigramCosineBackend",
    "cache_key",
    "cosine_similarity",
    "get_backend",
    "normalize_text",
    "run_coroutine_blocking",
    "trigram_vector",
]

_LOG = logging.getLogger(__name__)

T = TypeVar("T")

#: 相似度缓存文件名（位于 ``<project>/.myagent/skills/``）。
CACHE_FILE_NAME = ".simcache.json"

#: L1 后端标识（写进缓存键；算法一改就要改版本号，从而全量失效）。
TRIGRAM_BACKEND_ID = "trigram-cosine-v1"

#: 字符 n-gram 的 n（规格 §9.2 写明「字符 trigram」）。
TRIGRAM_SIZE = 3

#: OpenAI 兼容的 embedding 路径（拼在端点 ``base_url`` 之后）。
EMBEDDINGS_PATH = "/embeddings"

#: 归一化：保留 ASCII 字母数字与 CJK，其余（标点、空白）压成单个空格。
_NORMALIZE_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")

#: 缓存文件版本（格式变更时递增）。
_CACHE_VERSION = 1


# --------------------------------------------------------------------------------------
# 同步 / 异步桥接
# --------------------------------------------------------------------------------------


def run_coroutine_blocking(coro: Coroutine[Any, Any, T]) -> T:
    """在**同步**调用栈里驱动一个协程直到完成。

    两种情形：

    * 当前线程没有运行中的事件循环 → 直接 :func:`asyncio.run`。
    * 当前线程正在事件循环内（例如 ``engine/slash.py`` 的 ``/skills add``）→ 另起一条
      线程与新的事件循环执行，避免 ``asyncio.run() cannot be called from a running
      event loop`` 与自死锁。

    Args:
        coro: 待执行的协程对象。

    Returns:
        协程的返回值。

    Raises:
        BaseException: 协程内部抛出的任何异常都会原样传播到调用方。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="skills-sync") as pool:
        return pool.submit(asyncio.run, coro).result()


# --------------------------------------------------------------------------------------
# 文本归一化与向量
# --------------------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """把技能文本归一化为「小写 + 去标点 + 空白压缩」。

    Args:
        text: 原始文本（通常是 ``name + description + when_to_use``）。

    Returns:
        归一化文本；中英文皆可（CJK 会被整段保留，由字符 trigram 兜住分词缺失）。
    """
    return _NORMALIZE_RE.sub(" ", text.lower()).strip()


def trigram_vector(text: str, *, n: int = TRIGRAM_SIZE) -> Counter[str]:
    """构造字符 n-gram 词频向量。

    Args:
        text: **已归一化**的文本。
        n: n-gram 长度（默认 :data:`TRIGRAM_SIZE`）。

    Returns:
        ``Counter``：n-gram → 频次；文本短于 ``n`` 时退化为整串一个分量。
    """
    if not text:
        return Counter()
    if len(text) <= n:
        return Counter({text: 1})
    return Counter(text[index : index + n] for index in range(len(text) - n + 1))


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """计算两个稀疏向量的余弦相似度。

    Args:
        left: 左向量（键 → 权重）。
        right: 右向量（键 → 权重）。

    Returns:
        ``[0, 1]`` 区间内的余弦值；任一向量为空或模长为 0 时返回 ``0.0``。
    """
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(weight * right.get(key, 0.0) for key, weight in left.items())
    if dot <= 0.0:
        return 0.0
    norm_left = sqrt(sum(weight * weight for weight in left.values()))
    norm_right = sqrt(sum(weight * weight for weight in right.values()))
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return min(1.0, dot / (norm_left * norm_right))


@runtime_checkable
class CosineLike(Protocol):
    """相似度后端协议（规格 §9.2：``similarity(a, b) -> float``，返回 ``[0, 1]``）。"""

    #: 后端标识（写进缓存键；不同后端不共用缓存）。
    backend_id: str

    def similarity(self, a: str, b: str) -> float:
        """计算两段技能文本的相似度。

        Args:
            a: 技能 A 的可比较文本。
            b: 技能 B 的可比较文本。

        Returns:
            ``[0, 1]`` 的相似度。
        """
        ...


#: 规格里的协议名（与 CONTRACTS.md §9 的 ``SimilarityBackend`` 对齐）。
SimilarityBackend = CosineLike


# --------------------------------------------------------------------------------------
# L1：字符 trigram 余弦（默认）
# --------------------------------------------------------------------------------------


class TrigramCosineBackend:
    """L1 后端：字符 trigram 余弦 ∪ ``token_set_ratio``，取最大值（§9.2 表格第一行）。

    Attributes:
        backend_id: ``"trigram-cosine-v1"``。
    """

    backend_id: str = TRIGRAM_BACKEND_ID

    def similarity(self, a: str, b: str) -> float:
        """计算归一化文本的相似度（余弦与 ``token_set_ratio`` 取最大）。

        Args:
            a: 技能 A 的可比较文本。
            b: 技能 B 的可比较文本。

        Returns:
            ``[0, 1]`` 的相似度；任一文本归一化后为空时返回 ``0.0``。
        """
        left, right = normalize_text(a), normalize_text(b)
        if not left or not right:
            return 0.0
        cosine = cosine_similarity(trigram_vector(left), trigram_vector(right))
        ratio = fuzz.token_set_ratio(left, right) / 100.0
        return min(1.0, max(cosine, ratio))


# --------------------------------------------------------------------------------------
# L2：embedding 真余弦（可选）
# --------------------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """embedding 传输协议（异步；便于测试注入假实现）。"""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把一批文本编码为向量。

        Args:
            texts: 待编码文本（顺序与返回的向量一一对应）。

        Returns:
            向量列表。

        Raises:
            LLMUnavailableError: 网络/协议/响应解析失败。
            ConfigError: 密钥或端点缺失。
        """
        ...


class HttpEmbedder:
    """OpenAI 兼容 ``POST {base_url}/embeddings`` 的最小客户端（L2 默认传输）。

    密钥仍由 ``my_agent.llm`` 层解析（``read_api_key``），本类**不读也不存**密钥。
    """

    def __init__(self, endpoint: EndpointConfig) -> None:
        self._endpoint = endpoint

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """调用 ``/embeddings`` 取向量。

        Args:
            texts: 待编码文本。

        Returns:
            与 ``texts`` 同序的向量列表。

        Raises:
            ConfigError: 密钥缺失（含环境变量名）。
            LLMUnavailableError: 网络错误、非 2xx 或响应结构非法。
        """
        import httpx

        from my_agent.errors import ConfigError, LLMUnavailableError
        from my_agent.llm.provider import read_api_key

        try:
            api_key = read_api_key(self._endpoint.api_key_env)
        except ImportError as exc:  # pragma: no cover - 依赖缺失的兜底
            raise ConfigError(
                "embedding 相似度需要 openai/httpx 依赖，请检查安装",
                detail=str(exc),
            ) from exc
        url = self._endpoint.base_url.rstrip("/") + EMBEDDINGS_PATH
        payload: dict[str, Any] = {"model": self._endpoint.model, "input": list(texts)}
        headers = {"Authorization": f"Bearer {api_key}", **self._endpoint.extra_headers}
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SMALL_S) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMUnavailableError(
                f"embedding 端点返回 {exc.response.status_code}",
                detail=f"{url} → HTTP {exc.response.status_code}",
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMUnavailableError(
                f"embedding 端点不可用：{exc}",
                detail=f"{url} 请求失败：{exc}",
            ) from exc
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise LLMUnavailableError(
                "embedding 响应结构非法（缺少 data 数组）",
                detail=f"{url} 返回：{str(body)[:200]}",
            )
        vectors: list[list[float]] = []
        for item in sorted(
            (entry for entry in data if isinstance(entry, dict)),
            key=lambda entry: entry.get("index", 0),
        ):
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise LLMUnavailableError(
                    "embedding 响应缺少 embedding 字段",
                    detail=f"{url} 返回项：{str(item)[:200]}",
                )
            vectors.append([float(value) for value in vector])
        return vectors


class EmbeddingBackend:
    """L2 后端：调用 embedding 端点取真余弦（默认关闭，需 ``similarity_backend="embedding"``）。

    Args:
        config: 完整配置（读 ``skills.embedding_provider`` / ``skills.embedding_model``）。
        embedder: 传输实现；``None`` = 按配置构造 :class:`HttpEmbedder`（首次使用时）。

    Attributes:
        backend_id: ``"embedding:<端点名>:<模型名>"``（端点或模型变化即换缓存命名空间）。
    """

    def __init__(self, config: Config, *, embedder: Embedder | None = None) -> None:
        self._config = config
        self._embedder = embedder
        self._vectors: dict[str, list[float]] = {}

    @property
    def backend_id(self) -> str:
        """后端标识（含端点与模型名）。"""
        skills = self._config.skills
        provider = skills.embedding_provider or "unset"
        model = skills.embedding_model or "unset"
        return f"embedding:{provider}:{model}"

    def similarity(self, a: str, b: str) -> float:
        """计算两段文本的 embedding 余弦。

        Args:
            a: 技能 A 的可比较文本。
            b: 技能 B 的可比较文本。

        Returns:
            ``[0, 1]`` 的余弦值；任一文本为空时返回 ``0.0``。

        Raises:
            ConfigError: ``skills.embedding_provider`` 未配置或端点不存在。
            LLMUnavailableError: embedding 端点不可用。
        """
        if not a.strip() or not b.strip():
            return 0.0
        vector_a = self.vector(a)
        vector_b = self.vector(b)
        return max(0.0, cosine_similarity(_dense(vector_a), _dense(vector_b)))

    def vector(self, text: str) -> list[float]:
        """取（并缓存）一段文本的向量。

        Args:
            text: 待编码文本。

        Returns:
            向量。

        Raises:
            ConfigError: 端点未配置。
            LLMUnavailableError: 端点不可用或返回结构非法。
        """
        cached = self._vectors.get(text)
        if cached is not None:
            return cached
        vectors = run_coroutine_blocking(self.embedder.embed([text]))
        if not vectors:
            from my_agent.errors import LLMUnavailableError

            raise LLMUnavailableError(
                "embedding 端点未返回向量",
                detail=f"backend_id={self.backend_id}",
            )
        self._vectors[text] = vectors[0]
        return vectors[0]

    @property
    def embedder(self) -> Embedder:
        """当前的 embedding 传输实现（按需构造默认实现）。

        Returns:
            :class:`Embedder` 实现。

        Raises:
            ConfigError: 未配置 ``skills.embedding_provider`` 或该端点不存在。
        """
        if self._embedder is None:
            self._embedder = HttpEmbedder(self._resolve_endpoint())
        return self._embedder

    def _resolve_endpoint(self) -> EndpointConfig:
        """把 ``skills.embedding_provider`` 解析为 :class:`EndpointConfig`。

        Returns:
            生效的端点配置（模型名取 ``skills.embedding_model`` 优先）。

        Raises:
            ConfigError: 未配置端点名，或 ``llm.endpoints`` 中不存在该端点。
        """
        from my_agent.errors import ConfigError

        skills = self._config.skills
        name = skills.embedding_provider
        if not name:
            raise ConfigError(
                "skills.similarity_backend=\"embedding\" 需要配置 skills.embedding_provider"
                "（llm.endpoints 中的端点名）",
                detail="skills.embedding_provider 为空",
            )
        endpoint = self._config.llm.endpoints.get(name)
        if endpoint is None:
            raise ConfigError(
                f"未找到 embedding 端点 {name!r}（请在配置的 llm.endpoints 中定义）",
                detail=f"llm.endpoints 现有：{', '.join(sorted(self._config.llm.endpoints)) or '（空）'}",
            )
        if skills.embedding_model:
            endpoint = endpoint.model_copy(update={"model": skills.embedding_model})
        return endpoint


def _dense(vector: Sequence[float]) -> dict[str, float]:
    """把稠密向量转成稀疏映射（复用 :func:`cosine_similarity`）。

    Args:
        vector: 稠密向量。

    Returns:
        下标字符串 → 分量的映射。
    """
    return {str(index): float(value) for index, value in enumerate(vector)}


def get_backend(config: Config) -> CosineLike:
    """按配置选择相似度后端（§9.2 / U8）。

    Args:
        config: 完整配置（读 ``skills.similarity_backend``）。

    Returns:
        ``"embedding"`` → :class:`EmbeddingBackend`；其它（默认 ``"trigram"``）→
        :class:`TrigramCosineBackend`。
    """
    if config.skills.similarity_backend == "embedding":
        return EmbeddingBackend(config)
    return TrigramCosineBackend()


# --------------------------------------------------------------------------------------
# 相似度缓存
# --------------------------------------------------------------------------------------


def cache_key(hash_a: str, hash_b: str, backend_id: str) -> str:
    """计算缓存键 ``sha1(hash_a + hash_b + backend_id)``（§9.2）。

    两个 ``content_hash`` 先按字典序规范化，保证 ``(a, b)`` 与 ``(b, a)`` 命中同一条。

    Args:
        hash_a: 技能 A 的 ``content_hash``。
        hash_b: 技能 B 的 ``content_hash``。
        backend_id: 后端标识（L1/L2 不共用缓存）。

    Returns:
        40 位十六进制摘要。
    """
    low, high = sorted((hash_a, hash_b))
    material = f"{low}|{high}|{backend_id}".encode()
    return hashlib.sha1(material).hexdigest()


class SimilarityCache:
    """``.simcache.json`` 的读写与命中统计（§9.2、E-8）。

    Args:
        path: 缓存文件路径（``<project>/.myagent/skills/.simcache.json``）。
        backend_id: 当前后端标识；与文件内记录不一致时**整体作废**（换后端不串味）。

    Attributes:
        hits: 命中次数（E-8 断言「命中不重算」用）。
        misses: 未命中次数。
    """

    def __init__(self, path: Path, *, backend_id: str) -> None:
        self._path = Path(path)
        self._backend_id = backend_id
        self._entries: dict[str, float] = {}
        self.hits = 0
        self.misses = 0

    @property
    def path(self) -> Path:
        """缓存文件路径。"""
        return self._path

    @property
    def backend_id(self) -> str:
        """当前后端标识。"""
        return self._backend_id

    def __len__(self) -> int:
        """已缓存的相似度条数。"""
        return len(self._entries)

    def load(self) -> SimilarityCache:
        """从磁盘加载缓存；文件缺失/损坏/后端不一致时按空缓存处理。

        Returns:
            ``self``。
        """
        if not self._path.exists():
            return self
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _LOG.warning("相似度缓存读取失败（按空缓存处理）：%s", exc)
            return self
        if not isinstance(raw, dict) or raw.get("backend") != self._backend_id:
            return self
        entries = raw.get("entries")
        if isinstance(entries, dict):
            for key, value in entries.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    self._entries[key] = float(value)
        return self

    def save(self) -> None:
        """把缓存写回磁盘（原子替换）。

        Raises:
            OSError: 目录不可创建或写入失败。
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _CACHE_VERSION,
            "backend": self._backend_id,
            "entries": self._entries,
        }
        tmp = self._path.with_name(f"{self._path.name}.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._path)

    def get(self, hash_a: str, hash_b: str) -> float | None:
        """查缓存（``content_hash`` 变了自然查不到 → 自动失效）。

        Args:
            hash_a: 技能 A 的 ``content_hash``。
            hash_b: 技能 B 的 ``content_hash``。

        Returns:
            缓存的相似度；未命中返回 ``None``（``misses`` 自增）。
        """
        value = self._entries.get(cache_key(hash_a, hash_b, self._backend_id))
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, hash_a: str, hash_b: str, sim: float) -> None:
        """写入一条相似度（不立即落盘，由调用方 :meth:`save`）。

        Args:
            hash_a: 技能 A 的 ``content_hash``。
            hash_b: 技能 B 的 ``content_hash``。
            sim: 相似度。
        """
        self._entries[cache_key(hash_a, hash_b, self._backend_id)] = float(sim)

    def clear(self) -> None:
        """清空内存缓存（不删文件）。"""
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def drop(self, content_hash: str) -> int:
        """丢弃与某个 ``content_hash`` 相关的全部缓存项（``remove`` 时清理）。

        Args:
            content_hash: 被移除 skill 的内容摘要。

        Returns:
            丢弃的条数。
        """
        doomed = [
            key
            for key in self._entries
            if content_hash in _cache_key_material(key, self._entries)
        ]
        for key in doomed:
            self._entries.pop(key, None)
        return len(doomed)


def _cache_key_material(key: str, entries: Mapping[str, float]) -> str:
    """占位实现：缓存键是单向摘要，无法反查 ``content_hash``。

    保留函数是为了让 ``drop`` 的语义显式化——真正的失效靠「键里含 ``content_hash``」，
    见 :meth:`SimilarityCache.drop` 的说明。

    Args:
        key: 缓存键。
        entries: 缓存表（未使用）。

    Returns:
        空串（永不匹配），因此 :meth:`SimilarityCache.drop` 实际删除 0 条。
    """
    del key, entries
    return ""
