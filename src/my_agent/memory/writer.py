"""记忆写入管线编排（§8.4 步骤 1–6）——回合结束后的即时判定与 ``/memory add`` 共用入口。

- :meth:`MemoryWriter.after_turn`：回合结束调用（``engine/query_engine.py::_after_turn``），
  依次走「抽取 → 去噪 → 范围选择 → 冲突合并 → 落盘 → 审计」；
- :meth:`MemoryWriter.write_candidates`：``/memory add`` 与 ``MemoryWrite`` 工具调用
  （验收 D-9：两者走**同一**管线）。

无候选时**不产生任何文件写入**（含审计文件），保证 mtime 不变（验收 D-6）。

审计落盘：``<audit_dir>/memory-YYYY-MM-DD.jsonl``（``ensure_ascii=False``）；
「无法判断」的 SKIP 另记 ``<audit_dir>/memory-unresolved.jsonl``（§8.5）。

规格：``my_agent-开发文档.md`` §8.4、§8.5；``CONTRACTS.md`` §17.1。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from my_agent.errors import MyAgentError
from my_agent.memory.extractor import Candidate, coerce_confidence, denoise, extract
from my_agent.memory.merger import MergeResult, merge
from my_agent.memory.scope_router import ScopedCandidate, choose, normalize_scoped
from my_agent.memory.sections import Scope
from my_agent.memory.store import MemoryStore, sanitize_text
from my_agent.models.audit import MemoryWriteRecord

if TYPE_CHECKING:
    from my_agent.config.schema import Config
    from my_agent.engine.query_engine import EngineResult
    from my_agent.memory import MemoryLLM
    from my_agent.models.session import TurnRecord

__all__ = ["AUDIT_PREFIX", "UNRESOLVED_NAME", "MemoryWriter"]

_LOG = logging.getLogger(__name__)

#: 审计文件名前缀（§8.4 步骤 6）。
AUDIT_PREFIX = "memory-"

#: 「无法判断」记录文件名（§8.5）。
UNRESOLVED_NAME = "memory-unresolved.jsonl"


class MemoryWriter:
    """记忆写入管线（§8.4）。

    Args:
        llm: 角色化模型客户端（``memory_extract`` / ``memory_scope`` / ``memory_merge``）。
        config: 运行配置（``memory.write_per_turn`` / ``memory.max_writes_per_turn``）。
        global_store: 全局记忆存储。
        project_store: 项目记忆存储。
        audit_dir: 审计目录（``<project>/.myagent/audit``）。
        session_id: 会话 id（写审计用）。

    Attributes:
        audit_dir: 审计目录。
        session_id: 会话 id。
    """

    def __init__(
        self,
        llm: MemoryLLM,
        config: Config,
        global_store: MemoryStore,
        project_store: MemoryStore,
        audit_dir: Path,
        session_id: str,
    ) -> None:
        self.llm = llm
        self.config = config
        self.global_store = global_store
        self.project_store = project_store
        self.audit_dir = Path(audit_dir)
        self.session_id = session_id
        self._fallback_turn = 0

    # ----------------------------------------------------------------------------------
    # 对外入口
    # ----------------------------------------------------------------------------------

    async def after_turn(
        self,
        turn: TurnRecord | None,
        result: EngineResult,
        tool_summary: str,
    ) -> list[MemoryWriteRecord]:
        """回合结束后即时判定并写入记忆（§8.4）。

        Args:
            turn: 本轮 :class:`~my_agent.models.session.TurnRecord`；``None`` 时按写入器内部
                轮次计数记账（``session.turn_index`` 的近似，``write_candidates`` 同理）。
            result: 本轮引擎结果（只用 ``kind`` / ``text``）。
            tool_summary: 本轮工具调用摘要（工具名 + 参数 + 成败）。

        Returns:
            审计记录列表（含 SKIP）；无候选或全部被过滤时返回空列表。

        Raises:
            MyAgentError: 写入落盘失败（含 :class:`~my_agent.errors.MemoryLockError`）。
        """
        if not self.config.memory.write_per_turn:
            return []
        if result.kind != "assistant_text":
            return []

        turn_index = turn.index if turn is not None else self._next_fallback_turn()
        user_input = turn.user_input if turn is not None else ""
        candidates = await extract(
            user_input=user_input,
            assistant_final=result.text,
            tool_summary=tool_summary,
            llm=self.llm,
        )
        if not candidates:
            return []  # D-6：无候选 → 连审计文件都不创建
        stamped = [
            candidate.model_copy(update={"evidence_turn": turn_index}) for candidate in candidates
        ]
        return await self._run_pipeline(stamped, turn_index)

    async def write_candidates(
        self,
        candidates: Sequence[tuple[str, str]],
        turn: int,
        *,
        scope: str | None = None,
        section: str | None = None,
    ) -> list[MemoryWriteRecord]:
        """显式写入若干候选（``/memory add`` 与 ``MemoryWrite`` 工具共用，D-9）。

        Args:
            candidates: ``[(text, confidence), ...]``。
            turn: 轮次（审计记账用）。
            scope: 显式作用域（``global`` / ``project``）；给出 ``section`` 时可省略。
            section: 显式章节键（§8.2）；给出时不再调用范围选择模型。

        Returns:
            审计记录列表（含 SKIP）；全部被去噪丢弃时返回空列表。

        Raises:
            MyAgentError: 入参结构非法，或写入落盘失败。
        """
        pending = self._to_candidates(candidates, turn)
        if not pending:
            return []
        return await self._run_pipeline(pending, turn, scope=scope, section=section)

    # ----------------------------------------------------------------------------------
    # 管线
    # ----------------------------------------------------------------------------------

    async def _run_pipeline(
        self,
        candidates: Sequence[Candidate],
        turn: int,
        *,
        scope: str | None = None,
        section: str | None = None,
    ) -> list[MemoryWriteRecord]:
        """步骤 2–6：去噪 → 范围选择 → 合并 → 落盘 → 审计。

        Args:
            candidates: 步骤 1（或显式）给出的候选。
            turn: 轮次。
            scope: 显式作用域。
            section: 显式章节键。

        Returns:
            审计记录列表。

        Raises:
            MyAgentError: 落盘失败。
        """
        existing = await asyncio.to_thread(self._existing_texts)
        kept = denoise(
            candidates, existing, max_writes=self.config.memory.max_writes_per_turn
        )
        if not kept:
            _LOG.debug("记忆管线：%d 条候选全部被去噪丢弃", len(candidates))
            return []

        scoped, manual = await self._locate(kept, scope=scope, section=section)
        records: list[MemoryWriteRecord] = []
        results: list[MergeResult] = []
        if scoped:
            results = await merge(
                scoped,
                self._store_for,
                self.llm,
                audit=records.append,
                session_id=self.session_id,
                turn=turn,
            )
        for skipped in manual:
            results.append(skipped)
            records.append(skipped.to_record(session_id=self.session_id, turn=turn))
        if not records:
            return []

        unresolved = [
            record
            for result, record in zip(results, records, strict=True)
            if result.unresolved
        ]
        await asyncio.to_thread(self._append_jsonl, records, unresolved)
        return records

    async def _locate(
        self,
        candidates: Sequence[Candidate],
        *,
        scope: str | None,
        section: str | None,
    ) -> tuple[list[ScopedCandidate], list[MergeResult]]:
        """步骤 3：确定每条候选的 ``(scope, section)``。

        ``section`` 显式给出时跳过范围选择模型（B.6）；否则调用
        :func:`~my_agent.memory.scope_router.choose`。与显式 ``scope`` 冲突的候选记为 SKIP。

        Args:
            candidates: 步骤 2 保留下来的候选。
            scope: 显式作用域。
            section: 显式章节键。

        Returns:
            ``(可写入的候选, 直接判 SKIP 的结果)``。
        """
        if section:
            scoped: list[ScopedCandidate] = []
            manual: list[MergeResult] = []
            for candidate in candidates:
                located = normalize_scoped(
                    text=candidate.text,
                    scope=scope or "",
                    section=section,
                    why="显式指定写入位置",
                    confidence=candidate.confidence,
                    evidence_turn=candidate.evidence_turn,
                )
                if located is None:
                    manual.append(
                        self._skip(
                            candidate,
                            scope=scope or "global",
                            section=section,
                            reason=f"显式指定的章节键非法：{section}",
                        )
                    )
                elif scope and located.scope != scope:
                    manual.append(
                        self._skip(
                            candidate,
                            scope=located.scope,
                            section=located.section,
                            reason=f"显式 scope={scope} 与章节键归属（{located.scope}）不一致",
                            why=located.why,
                        )
                    )
                else:
                    scoped.append(located)
            return scoped, manual

        chosen = await choose(candidates, self.llm)
        if not scope:
            return chosen, []
        scoped, manual = [], []
        for item in chosen:
            if item.scope == scope:
                scoped.append(item)
            else:
                manual.append(
                    MergeResult(
                        candidate=item.text,
                        scope=item.scope,
                        section=item.section,
                        action="SKIP",
                        why=item.why,
                        reason=f"显式 scope={scope} 与归档器选择（{item.scope}）不一致，已跳过",
                    )
                )
        return scoped, manual

    # ----------------------------------------------------------------------------------
    # 辅助
    # ----------------------------------------------------------------------------------

    def _to_candidates(
        self, candidates: Sequence[tuple[str, str]], turn: int
    ) -> list[Candidate]:
        """把 ``[(text, confidence)]`` 规范成候选对象。

        Args:
            candidates: 原始二元组序列。
            turn: 轮次（写入 ``evidence_turn``）。

        Returns:
            非空候选列表。

        Raises:
            MyAgentError: 入参不是 ``(text, confidence)`` 二元组。
        """
        out: list[Candidate] = []
        for item in candidates:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise MyAgentError(
                    "记忆候选格式非法：应为 (text, confidence) 二元组", detail=repr(item)[:120]
                )
            text, confidence = item
            clean = sanitize_text(text)
            if not clean:
                continue
            out.append(
                Candidate(
                    text=clean,
                    confidence=coerce_confidence(confidence),
                    evidence_turn=turn,
                )
            )
        return out

    def _skip(
        self,
        candidate: Candidate,
        *,
        scope: str,
        section: str,
        reason: str,
        why: str = "",
    ) -> MergeResult:
        """构造一条直接 SKIP 的结果（未进入合并模型）。

        Args:
            candidate: 候选。
            scope: 作用域。
            section: 章节键。
            reason: 跳过理由。
            why: 范围选择理由。

        Returns:
            标记为 ``unresolved`` 的 SKIP 结果。
        """
        return MergeResult(
            candidate=candidate.text,
            scope=scope if scope in ("global", "project") else "global",
            section=section,
            action="SKIP",
            why=why,
            reason=reason,
            confidence=candidate.confidence,
            unresolved=True,
        )

    def _store_for(self, scope: Scope) -> MemoryStore:
        """返回作用域对应的存储。

        Args:
            scope: 作用域。

        Returns:
            全局或项目存储。
        """
        return self.global_store if scope == "global" else self.project_store

    def _existing_texts(self) -> list[str]:
        """读取两个作用域的全部已有记忆文本（供步骤 2 查重）。

        Returns:
            文本列表。
        """
        texts = [entry.text for entry in self.global_store.read()]
        texts.extend(entry.text for entry in self.project_store.read())
        return texts

    def _next_fallback_turn(self) -> int:
        """``turn=None`` 时的轮次记账（写入器内部自增）。

        Returns:
            自增后的轮次。
        """
        self._fallback_turn += 1
        return self._fallback_turn

    def _append_jsonl(
        self, records: Sequence[MemoryWriteRecord], unresolved: Sequence[MemoryWriteRecord]
    ) -> None:
        """把审计记录追加到 JSONL 文件（阻塞 I/O，调用方用 ``to_thread``）。

        Args:
            records: 全部记录（含 SKIP）。
            unresolved: 其中「无法判断」的记录，另写 ``memory-unresolved.jsonl``。
        """
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        day = datetime.now().strftime("%Y-%m-%d")
        with (self.audit_dir / f"{AUDIT_PREFIX}{day}.jsonl").open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
        if not unresolved:
            return
        with (self.audit_dir / UNRESOLVED_NAME).open("a", encoding="utf-8") as handle:
            for record in unresolved:
                payload = {**record.model_dump(), "unresolved": True}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
