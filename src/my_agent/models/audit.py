"""审计类记录模型（§7.7.1、§8.4 步骤 6）。

规格书引用了 ``MemoryWriteRecord`` 与 ``CompactEvent`` 但未给出定义，
此处按引用处字段补齐（见 ``CONTRACTS.md`` §2）。
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["CompactEvent", "MemoryWriteRecord"]


class MemoryWriteRecord(BaseModel):
    """一条记忆写入的审计记录（§8.4 步骤 6）。

    写入 ``<project>/.myagent/audit/memory-YYYY-MM-DD.jsonl``；全局作用域的写入
    也记在本项目审计里，附 ``scope="global"``。
    """

    ts: float = Field(default_factory=time.time)
    session_id: str
    turn: int
    candidate: str
    scope: Literal["global", "project"] = "project"
    section: str = ""
    why: str = Field(default="", description="上游「解决 2」要求的「因为选择了哪个选项」")
    action: Literal["ADD", "UPDATE", "DELETE", "SKIP"] = "SKIP"
    target_id: str | None = None
    before: str | None = None
    after: str | None = None
    confidence: str = "medium"

    model_config = {"extra": "ignore"}


class CompactEvent(BaseModel):
    """一次全文压缩的事件记录（§7.7.1 步骤 6）。"""

    ts: float = Field(default_factory=time.time)
    session_id: str
    turn_start: int
    turn_end: int
    archive_path: str = ""
    messages_archived: int = 0
    mode: Literal["llm", "rule_based"] = "llm"
    summary_tokens: int = 0
    summary_ids: list[str] = Field(default_factory=list)
    watermark_before: float = 0.0
    watermark_after: float = 0.0
    keep_recent_turns: int = 0
    note: str = ""

    model_config = {"extra": "ignore"}
