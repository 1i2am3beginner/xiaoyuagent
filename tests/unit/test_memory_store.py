"""``memory/store.py`` 单元测试：Markdown 契约解析、外科式写入、并发锁。

覆盖验收：D-6（无候选不写文件在管线测试里验证写入路径）、D-7（未知章节保真）、
D-10（锁冲突 → ``MemoryLockError`` 且文件不损坏）。
"""

from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path

import pytest

from my_agent.config.defaults import MEMORY_LOCK_TIMEOUT_S, MEMORY_TEXT_MAX_CHARS
from my_agent.errors import MemoryLockError, MyAgentError
from my_agent.memory.store import MemoryEntry, MemoryStore, sanitize_text

LEGACY_FILE = """# 全局记忆（myagent）
<!-- managed-by: myagent; 可手工编辑；条目 id 请勿修改 -->

## 用户偏好
- (2026-09-14, conf=high) 交流语言：中文
- [m-abc123] (2026-09-10, conf=medium) 直接给推荐方案

## 我手写的自定义章节
这是我自己加的内容，任何自动写入都不得改动它。
- 一行长得像条目但其实不是的文本
"""


def _store(workdir: Path, text: str | None = None) -> MemoryStore:
    """构造一个全局作用域 store（可选预置文件内容）。"""
    path = workdir / "memory.md"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    return MemoryStore(path, scope="global")


def test_read_missing_file_returns_empty(workdir: Path) -> None:
    store = _store(workdir)
    assert store.read() == []
    assert store.path == workdir / "memory.md"


def test_read_assigns_id_to_legacy_entry_and_backs_up(workdir: Path) -> None:
    store = _store(workdir, LEGACY_FILE)
    entries = store.read()

    assert [entry.text for entry in entries] == ["交流语言：中文", "直接给推荐方案"]
    legacy = entries[0]
    assert re.fullmatch(r"m-[0-9a-z]{6}", legacy.id)
    assert legacy.date == "2026-09-14"
    assert legacy.confidence == "high"
    assert legacy.section == "user_preference"

    backup = store.path.with_name(store.path.name + ".bak")
    assert backup.read_text(encoding="utf-8") == LEGACY_FILE
    rewritten = store.path.read_text(encoding="utf-8")
    assert f"[{legacy.id}] (2026-09-14, conf=high) 交流语言：中文" in rewritten
    assert "## 我手写的自定义章节" in rewritten  # 未知章节一并保留


def test_add_update_delete_roundtrip(workdir: Path) -> None:
    store = _store(workdir)
    entry = store.add("包管理用 uv", "tooling_env", "high")

    text = store.path.read_text(encoding="utf-8")
    assert text.startswith("# 全局记忆（myagent）")
    assert "<!-- managed-by: myagent" in text
    assert "## 工具与环境" in text
    assert entry.to_line() in text
    assert entry.date == date.today().isoformat()

    updated = store.update(entry.id, "包管理统一用 uv，不用 pip", "tooling_env", "medium")
    assert updated.id == entry.id
    assert updated.text == "包管理统一用 uv，不用 pip"
    assert updated.confidence == "medium"
    assert len(store.read()) == 1
    assert "不用 pip" in store.path.read_text(encoding="utf-8")

    removed = store.delete(entry.id)
    assert removed is not None and removed.id == entry.id
    assert store.delete(entry.id) is None
    assert store.read() == []


def test_update_unknown_id_raises(workdir: Path) -> None:
    store = _store(workdir)
    with pytest.raises(MyAgentError):
        store.update("m-zzzzzz", "x", "tooling_env", "high")


def test_unknown_section_and_comments_survive_automatic_write(workdir: Path) -> None:
    store = _store(workdir, LEGACY_FILE)
    store.read()  # 触发补 id 回写
    store.add("新偏好", "user_preference", "high")

    text = store.path.read_text(encoding="utf-8")
    assert "## 我手写的自定义章节" in text
    assert "这是我自己加的内容，任何自动写入都不得改动它。" in text
    assert "- 一行长得像条目但其实不是的文本" in text
    assert "<!-- managed-by: myagent; 可手工编辑；条目 id 请勿修改 -->" in text


def test_write_entries_leaves_other_sections_untouched(workdir: Path) -> None:
    store = _store(workdir, LEGACY_FILE)
    store.read()  # 先补 id，使「用户偏好」成为规范托管行
    untouched = store.sections()["user_preference"]

    store.write_entries(
        [
            MemoryEntry(
                id="m-000001",
                text="函数必须写类型标注",
                date="2026-01-01",
                confidence="high",
                section="coding_style",
                scope="global",
            )
        ]
    )

    text = store.path.read_text(encoding="utf-8")
    assert "## 编码风格偏好" in text
    assert "函数必须写类型标注" in text
    # write_entries 未提及的章节（含用户手写章节）原样保留
    for entry in untouched:
        assert entry.to_line() in text
    assert "## 我手写的自定义章节" in text
    assert "- 一行长得像条目但其实不是的文本" in text


def test_ids_are_unique_and_sections_grouped(workdir: Path) -> None:
    store = _store(workdir)
    for index in range(5):
        store.add(f"第 {index} 条", "coding_style", "medium")

    entries = store.read()
    assert len({entry.id for entry in entries}) == 5
    assert all(re.fullmatch(r"m-[0-9a-z]{6}", entry.id) for entry in entries)
    grouped = store.sections()
    assert list(grouped) == ["coding_style"]
    assert len(grouped["coding_style"]) == 5


def test_text_is_sanitized_to_single_line_and_capped(workdir: Path) -> None:
    store = _store(workdir)
    entry = store.add("第一行\n第二行\t带制表符", "coding_style", "high")
    assert entry.text == "第一行 第二行 带制表符"

    long_text = "字" * (MEMORY_TEXT_MAX_CHARS + 50)
    entry = store.add(long_text, "coding_style", "high")
    assert len(entry.text) == MEMORY_TEXT_MAX_CHARS
    assert sanitize_text("  a \n b  ") == "a b"


def test_unknown_section_rejected(workdir: Path) -> None:
    store = _store(workdir)
    with pytest.raises(MyAgentError):
        store.add("x", "overview", "high")  # overview 属于项目作用域


def test_d10_lock_conflict_raises_memory_lock_error(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("my_agent.memory.store.MEMORY_LOCK_TIMEOUT_S", 0.3)
    store = _store(workdir)
    store.add("锁冲突前的条目", "coding_style", "high")
    before = store.path.read_text(encoding="utf-8")
    lock = store.path.with_name(store.path.name + ".lock")
    lock.write_text("another-process", encoding="utf-8")

    started = time.monotonic()
    with pytest.raises(MemoryLockError):
        store.add("锁冲突期间的条目", "coding_style", "high")
    elapsed = time.monotonic() - started

    assert elapsed >= 0.3
    assert MEMORY_LOCK_TIMEOUT_S == 3.0  # 默认阈值仍来自 defaults（§8.3）
    assert store.path.read_text(encoding="utf-8") == before  # 文件未被写坏
    assert "锁冲突期间的条目" not in store.path.read_text(encoding="utf-8")
    leftovers = [p.name for p in workdir.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_lock_released_after_write(workdir: Path) -> None:
    store = _store(workdir)
    store.add("条目", "coding_style", "high")
    assert not store.path.with_name(store.path.name + ".lock").exists()
