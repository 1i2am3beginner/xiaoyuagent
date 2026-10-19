"""skills 三级相似度分流（§9.2、§9.3 阈值）。

```
DEDUPE_THRESHOLD = 0.92          # ≥ 直接去重（同义技能）
LLM_CHECK_RANGE  = (0.85, 0.92)  # 交 LLM 判定是否同义
RELATED_RANGE    = (0.75, 0.85)  # 记强相关（供路由）
< 0.75                           # 无关联
```

阈值来自 ``config.defaults``（U8/U9 的单一可替换点），也可被 ``SkillsConfig`` 覆盖。
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from my_agent.config import defaults as D

if TYPE_CHECKING:
    from my_agent.config.schema import Config
    from my_agent.llm.client import LLMClient
    from my_agent.skills.registry import SkillMeta

__all__ = ["Bucket", "bucket", "is_synonym"]

_LOG = logging.getLogger(__name__)

Bucket = Literal["dedupe", "llm_check", "related", "none"]

SYNONYM_ROLE = "skill_synonym"


def bucket(
    sim: float,
    *,
    dedupe_threshold: float = D.DEDUPE_THRESHOLD,
    llm_check_range: tuple[float, float] = D.LLM_CHECK_RANGE,
    related_range: tuple[float, float] = D.RELATED_RANGE,
) -> Bucket:
    """按相似度分流（§9.2 步骤 3）。

    区间语义为**左闭右开**：``[0.75, 0.85)`` 记强相关，``[0.85, 0.92)`` 交 LLM，
    ``≥ 0.92`` 去重。

    Args:
        sim: 相似度（0~1）。
        dedupe_threshold: 去重阈值。
        llm_check_range: LLM 判定区间。
        related_range: 强相关区间。

    Returns:
        ``"dedupe"`` / ``"llm_check"`` / ``"related"`` / ``"none"``。
    """
    if sim >= dedupe_threshold:
        return "dedupe"
    if llm_check_range[0] <= sim < llm_check_range[1]:
        return "llm_check"
    if related_range[0] <= sim < related_range[1]:
        return "related"
    return "none"


def is_synonym(
    llm: LLMClient | None,
    left: SkillMeta,
    right: SkillMeta,
    *,
    config: Config | None = None,
) -> bool:
    """用 LLM 判定两个 skill 是否同义（§9.2，附录 B.8）。

    保守口径：仅在 ``same == True`` **且** ``confidence != "low"`` 时判为同义。

    Args:
        llm: 模型客户端；``None`` 时无法判定，返回 ``False``（保守：不合并）。
        left: 技能 A。
        right: 技能 B。
        config: 生效配置（当前未使用，保留以匹配调用点）。

    Returns:
        判为同义时 ``True``。
    """
    if llm is None:
        _LOG.info("无 LLM，跳过同义判定（保守视为不同义）：%s vs %s", left.name, right.name)
        return False

    from my_agent.config.defaults import TIMEOUT_SMALL_S
    from my_agent.llm.prompts import SKILL_SYNONYM_PROMPT, render

    prompt = render(
        SKILL_SYNONYM_PROMPT,
        a_name=left.name,
        a_desc=left.description,
        b_name=right.name,
        b_desc=right.description,
    )
    try:
        result = _run_sync(
            llm.complete_role(
                SYNONYM_ROLE,
                user=prompt,
                max_tokens=256,
                temperature=0.0,
                timeout_s=TIMEOUT_SMALL_S,
            )
        )
    except Exception as exc:  # 模型不可用不得阻断添加流程
        _LOG.warning("同义判定调用失败（保守视为不同义）：%s", exc)
        return False

    payload = _parse_json_object(getattr(result, "text", "") or "")
    if payload is None:
        return False
    same = bool(payload.get("same"))
    confidence = str(payload.get("confidence") or "").strip().lower()
    return same and confidence != "low"


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出里提取第一个 JSON 对象（容忍前后噪声）。

    Args:
        text: 模型输出。

    Returns:
        解析出的字典；失败返回 ``None``。
    """
    stripped = text.strip()
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _run_sync(coro: Any) -> Any:
    """在同步上下文里运行协程（``SkillRegistry.add`` 是同步 API）。

    若当前已有运行中的事件循环，则在独立线程里跑，避免 ``asyncio.run`` 嵌套报错。

    Args:
        coro: 待运行协程。

    Returns:
        协程结果。
    """
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
