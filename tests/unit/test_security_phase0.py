"""阶段 0 硬拦截单测（§6.3 / B-2、B-3、B-4、B-5）。"""

from __future__ import annotations

from typing import Any

import pytest

from my_agent.config.schema import Config, Rule, SafetyConfig
from my_agent.errors import ConfigError, PermanentGrantNotAllowedError
from my_agent.models.tools import ToolCategory, ToolInvocation, ToolSpec
from my_agent.security.cache import ApprovalCache
from my_agent.security.patterns import GLOBAL_ASK, GLOBAL_DENY, merge_rules
from my_agent.security.phase0_hard_block import (
    hard_block,
    hits_global_list,
    hits_irreversible,
    hits_sensitive_path,
)

CWD = "C:/proj"


def make_spec(
    name: str = "Write",
    *,
    category: ToolCategory = "write",
    is_read_only: bool = False,
    is_destructive: bool = False,
) -> ToolSpec:
    """构造一个用于测试的工具声明。"""
    return ToolSpec(
        name=name,
        description=f"{name} 工具（测试用）",
        parameters={"type": "object", "properties": {}},
        category=category,
        is_read_only=is_read_only,
        is_destructive=is_destructive,
    )


def make_inv(
    tool: str = "Write",
    arguments: dict[str, Any] | None = None,
    *,
    category: ToolCategory = "write",
    is_read_only: bool = False,
    is_destructive: bool = False,
    reason: str | None = None,
) -> ToolInvocation:
    """构造一个用于测试的工具调用。"""
    return ToolInvocation(
        call_id="call-1",
        tool=tool,
        arguments=arguments or {},
        reason=reason,
        spec=make_spec(
            tool,
            category=category,
            is_read_only=is_read_only,
            is_destructive=is_destructive,
        ),
    )


def make_config(**safety: Any) -> Config:
    """构造带自定义安全配置的 Config。"""
    return Config(safety=SafetyConfig(**safety))


# --------------------------------------------------------------------------- 敏感路径


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (".git/config", ".git"),
        ("/home/u/proj/.git/config", ".git"),
        ("./.git/config", ".git"),
        ("src/.vscode/settings.json", ".vscode"),
        ("a/.bin/tool", ".bin"),
        (".user/secret", ".user"),
        (r"C:\proj\.git\config", ".git"),
        (r".\.git\config", ".git"),
        (r"a\b\.vscode\settings.json", ".vscode"),
    ],
)
def test_hits_sensitive_path_positive(path: str, expected: str) -> None:
    """命中敏感目录段（含 Windows 反斜杠路径）。"""
    assert hits_sensitive_path(path, CWD) == expected


@pytest.mark.parametrize(
    "path",
    [".gitignore", "a/gitignore", "notes/.gitignore", "my.git/config", "git/config", "", "."],
)
def test_b3_gitignore_is_not_sensitive(path: str) -> None:
    """B-3：``.gitignore`` 等相似名**不**是敏感路径（分段精确匹配）。"""
    assert hits_sensitive_path(path, CWD) is None


def test_b4_windows_backslash_path_hits() -> None:
    """B-4：Windows 风格 ``.\\.git\\config`` 必须命中（先归一为 ``/`` 再分段）。"""
    assert hits_sensitive_path(r".\.git\config", CWD) == ".git"


def test_hits_sensitive_path_relative_resolved_against_cwd() -> None:
    """相对路径按 cwd 解析后再匹配一次（更保守）。"""
    assert hits_sensitive_path("config", f"{CWD}/.git/hooks") == ".git"


def test_hits_sensitive_path_extra_dirs_append_only() -> None:
    """配置可追加敏感目录，内置项仍生效。"""
    assert hits_sensitive_path("secrets/key.pem", CWD, extra_dirs=["secrets"]) == "secrets"
    assert hits_sensitive_path(".git/config", CWD, extra_dirs=["secrets"]) == ".git"


def test_b2_read_of_sensitive_path_does_not_hit() -> None:
    """B-2：**读** ``.git/config`` 不触发敏感路径（仅写操作触发）。"""
    read_inv = make_inv(
        "Read",
        {"file_path": ".git/config"},
        category="read",
        is_read_only=True,
    )
    assert hard_block(read_inv, config=make_config(), cwd=CWD) is None

    write_tool_but_read_only = make_inv(
        "Write",
        {"file_path": ".git/config"},
        category="write",
        is_read_only=True,
    )
    assert hard_block(write_tool_but_read_only, config=make_config(), cwd=CWD) is None


def test_sensitive_write_asks_without_permanent_grant() -> None:
    """写 ``.git/config`` → ask 且 ``permanent_eligible=False``（阶段 0 层）。

    B-1 的五模式参数化版本见 ``test_security_pipeline.py::test_b1_...``。
    """
    inv = make_inv("Write", {"file_path": ".git/config", "content": "x"})
    outcome = hard_block(inv, config=make_config(), cwd=CWD)
    assert outcome is not None
    assert outcome.kind == "sensitive_path"
    assert outcome.decision == "ask"
    assert outcome.permanent_eligible is False
    assert ".git" in outcome.detail


def test_sensitive_path_applies_to_execute_category() -> None:
    """``execute`` 视为潜在写操作（WRITE_CATEGORIES）。"""
    inv = make_inv(
        "Bash",
        {"command": "echo hi > .git/config"},
        category="execute",
        is_read_only=False,
    )
    outcome = hard_block(inv, config=make_config(), cwd=CWD)
    assert outcome is not None
    assert outcome.kind == "sensitive_path"


def test_sensitive_path_extra_dir_from_config() -> None:
    """config.safety.sensitive_dirs 追加的目录同样触发 ask。"""
    inv = make_inv("Write", {"file_path": "secrets/token.txt"})
    outcome = hard_block(inv, config=make_config(sensitive_dirs=["secrets"]), cwd=CWD)
    assert outcome is not None
    assert outcome.kind == "sensitive_path"


# --------------------------------------------------------------------------- 不可逆操作


@pytest.mark.parametrize(
    ("command", "reason_part"),
    [
        ("npm publish", "发布到公共仓库后无法撤回"),
        ("npm  publish --access public", "发布到公共仓库后无法撤回"),
        ("kill -9 4321", "强制杀进程可能丢失未保存数据"),
        ("kill -KILL 4321", "强制杀进程可能丢失未保存数据"),
        ("taskkill /PID 1 /F", "强制杀进程可能丢失未保存数据"),
        ("rm -rf ./build", "递归强制删除"),
        ("git push origin main --force", "影响远端生产环境"),
        ("git filter-branch --all", "重写历史难以恢复"),
        ("chmod -R 777 /", "大范围权限变更难以回滚"),
        ("mkfs.ext4 /dev/sda1", "破坏系统或磁盘数据"),
        ("docker push repo/img:latest", "影响远端生产环境"),
        ("terraform destroy -auto-approve", "影响远端生产环境"),
    ],
)
def test_b5_irreversible_commands_ask(command: str, reason_part: str) -> None:
    """B-5：不可逆命令 → ask（``risk="critical"``、``permanent_eligible=False``）。"""
    inv = make_inv("Bash", {"command": command}, category="execute")
    outcome = hard_block(inv, config=make_config(), cwd=CWD)
    assert outcome is not None
    assert outcome.kind == "irreversible"
    assert outcome.decision == "ask"
    assert outcome.risk == "critical"
    assert outcome.permanent_eligible is False
    assert reason_part in outcome.detail


def test_b5_permanent_grant_refused_by_cache() -> None:
    """B-5：对不可永久放行的操作调用 ``cache.put`` → PermanentGrantNotAllowedError。"""
    cache = ApprovalCache()
    with pytest.raises(PermanentGrantNotAllowedError):
        cache.put("Bash", {"command": "npm publish"}, permanent_eligible=False)
    assert len(cache) == 0


def test_irreversible_is_case_insensitive() -> None:
    """§6.3.3：不可逆清单不区分大小写。"""
    assert hits_irreversible("NPM PUBLISH", []) is not None
    assert hits_irreversible("KILL -9 1", []) is not None


def test_irreversible_extra_rules_append_only() -> None:
    """配置追加的不可逆规则生效，且内置项不可删除。"""
    extra = [Rule(pattern=r"\bmy-deploy\b", reason="自定义高风险部署")]
    assert hits_irreversible("my-deploy prod", extra) is not None
    assert hits_irreversible("npm publish", extra) is not None


def test_irreversible_no_match_returns_none() -> None:
    """普通命令不命中。"""
    assert hits_irreversible("ls -la", []) is None
    assert hits_irreversible("", []) is None


# --------------------------------------------------------------------------- 全局名单


@pytest.mark.parametrize("command", ["rm -rf /", "format C:", "shutdown -h now", "reboot"])
def test_global_deny_blocks(command: str) -> None:
    """§6.3.1：全局 deny 名单 → ``deny``。"""
    inv = make_inv("Bash", {"command": command}, category="execute")
    outcome = hard_block(inv, config=make_config(), cwd=CWD)
    assert outcome is not None
    assert outcome.kind == "global_deny"
    assert outcome.decision == "deny"
    assert outcome.permanent_eligible is False


@pytest.mark.parametrize("command", ["git push --force", "DROP TABLE users"])
def test_global_ask_asks(command: str) -> None:
    """§6.3.1：全局 ask 名单 → ``ask``。"""
    inv = make_inv("Bash", {"command": command}, category="execute")
    outcome = hard_block(inv, config=make_config(), cwd=CWD)
    assert outcome is not None
    assert outcome.kind == "global_ask"
    assert outcome.decision == "ask"
    assert outcome.permanent_eligible is False


def test_global_lists_are_append_only() -> None:
    """内置名单不可删除，配置只能追加（§6.3.2 / §15.1）。"""
    cfg = make_config(global_deny=[Rule(pattern="danger-zone", reason="自定义拒绝")])
    custom = hard_block(
        make_inv("Bash", {"command": "danger-zone"}, category="execute"), config=cfg, cwd=CWD
    )
    builtin = hard_block(
        make_inv("Bash", {"command": "rm -rf /"}, category="execute"), config=cfg, cwd=CWD
    )
    assert custom is not None and custom.kind == "global_deny"
    assert builtin is not None and builtin.kind == "global_deny"
    assert len(merge_rules(GLOBAL_DENY, [Rule(pattern="rm -rf /", reason="dup")])) == len(GLOBAL_DENY)
    assert len(merge_rules(GLOBAL_DENY, [Rule(pattern="new", reason="x")])) == len(GLOBAL_DENY) + 1


def test_global_list_matches_write_target_path() -> None:
    """§6.3.1：写类工具的目标路径也在匹配范围内。"""
    cfg = make_config(global_deny=[Rule(pattern=r"secret\.env$", reason="禁止写入密钥文件")])
    inv = make_inv("Write", {"file_path": "config/secret.env"})
    outcome = hard_block(inv, config=cfg, cwd=CWD)
    assert outcome is not None
    assert outcome.kind == "global_deny"


def test_hits_global_list_empty_text() -> None:
    """空文本不匹配。"""
    assert hits_global_list("", GLOBAL_ASK) is None
    assert hits_global_list("ls", GLOBAL_DENY) is None


def test_invalid_regex_raises_config_error() -> None:
    """用户配置的非法正则必须显式报错，不静默忽略。"""
    with pytest.raises(ConfigError):
        hits_global_list("x", [Rule(pattern="([", reason="bad")])


def test_phase0_precedence_deny_before_sensitive_path() -> None:
    """求值顺序：0.1 全局名单 → 0.2 敏感路径 → 0.3 不可逆。"""
    cfg = make_config(global_deny=[Rule(pattern="rm -rf /", reason="递归删除根目录")])
    inv = make_inv("Bash", {"command": "rm -rf / .git/config"}, category="execute")
    outcome = hard_block(inv, config=cfg, cwd=CWD)
    assert outcome is not None and outcome.kind == "global_deny"


def test_no_hit_returns_none() -> None:
    """普通写操作不触发阶段 0。"""
    inv = make_inv("Write", {"file_path": "src/app.py", "content": "print('hi')"})
    assert hard_block(inv, config=make_config(), cwd=CWD) is None


def test_as_trace_shape() -> None:
    """阶段 0 痕迹字段齐全（B-17 依赖）。"""
    inv = make_inv("Write", {"file_path": ".git/config"})
    outcome = hard_block(inv, config=make_config(), cwd=CWD)
    assert outcome is not None
    trace = outcome.as_trace()
    assert trace["hit"] == "sensitive_path"
    assert trace["detail"]
    assert trace["permanent_eligible"] is False
    assert trace["path"] == ".git/config"
