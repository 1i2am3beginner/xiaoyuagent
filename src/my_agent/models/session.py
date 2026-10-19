"""会话模型（§4.4）。

``SessionState.history`` 是**唯一真相源**（C4）：压缩/裁剪只操作上下文视图，
永不删除这里的原始记录。
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from my_agent.models.messages import Message
from my_agent.models.permission import SafetyMode

__all__ = ["SessionState", "TurnRecord"]


class TurnRecord(BaseModel):
    """一个用户回合的记账。"""

    index: int
    user_input: str
    assistant_final: str = ""
    tool_calls: list[str] = Field(default_factory=list, description="call_id 列表")
    started_at: float = Field(default_factory=time.time)
    ended_at: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0

    model_config = {"extra": "ignore"}


class SessionState(BaseModel):
    """一个 CLI 会话的完整状态。"""

    session_id: str
    cwd: str
    mode: SafetyMode = "default"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    turn_index: int = 0
    history: list[Message] = Field(default_factory=list, description="唯一真相源（C4）")
    turns: list[TurnRecord] = Field(default_factory=list)
    context_watermark: float = Field(default=0.0, description="最近一次 token 水位（0~1）")
    compact_count: int = 0
    archive_files: list[str] = Field(default_factory=list)
    pending_memory_candidates: list[str] = Field(default_factory=list)
    usage_totals: dict[str, int] = Field(default_factory=dict, description="prompt/completion/total")

    # ---- 【补齐】会话级辅助字段（§8.1/§7.3 需要）----
    project_root: str = ""
    title: str = ""
    repomap_path: str | None = None
    token_factor: float = Field(default=1.0, description="§7.2 在线校准比例因子（U24）")
    last_compact_turn: int = Field(default=0, description="最近一次压缩发生在第几轮")
    mode_switches: int = 0

    model_config = {"extra": "ignore"}

    def to_memory_summary(self) -> str:
        """生成简短会话摘要（`/session list` 用）。

        Returns:
            形如 ``sess-abc · 12 轮 · mode=default`` 的单行文本。
        """
        label = self.title or self.session_id
        return f"{label} · {self.turn_index} 轮 · mode={self.mode}"
