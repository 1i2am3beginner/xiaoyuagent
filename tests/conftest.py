"""pytest 全局配置与共享 fixture（§18.1）。

两件事：

1. 把项目根加入 ``sys.path``，使 ``tests.fakes.*`` 可被各层测试 import。
2. 把 ``tempfile`` 的临时根目录重定向到**工作区内的全新目录**。

   原因：在某些沙箱/受管环境下系统临时目录（``%TEMP%``）不可写，且**已存在**的
   临时目录可能被拒绝删除或复用。因此这里每次会话都用一个随机命名的全新目录，
   并显式设置 ``TMPDIR``/``TEMP``/``TMP``，让 ``tmp_path`` 与 ``tempfile.mkdtemp``
   都落在可写位置。
"""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 每次会话一个全新目录（不复用、不删除旧目录，规避沙箱的目录级拒绝）。
_TEMP_ROOT = ROOT / ".tmp" / f"session-{uuid.uuid4().hex[:8]}"


def _prepare_temp_root() -> Path | None:
    """创建本次会话的临时根目录并写入环境变量。

    Returns:
        可写的临时根目录；无法创建时返回 ``None``（此时沿用系统默认）。
    """
    try:
        _TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        probe = _TEMP_ROOT / "probe.txt"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return None
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ[key] = str(_TEMP_ROOT)
    return _TEMP_ROOT


_TEMP_ROOT_OK = _prepare_temp_root()
if _TEMP_ROOT_OK is not None:
    # pytest 的 basetemp 会解析成 <TEMPROOT>/pytest-of-<user>；该目录一旦已存在，
    # 受管环境会拒绝复用，因此指向本次会话的**全新**子目录（需先建好父目录）。
    _pytest_root = _TEMP_ROOT_OK / "pytest"
    try:
        _pytest_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        _pytest_root = _TEMP_ROOT_OK
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(_pytest_root)


@pytest.fixture(scope="session")
def temp_root() -> Path:
    """返回本次会话可写的临时根目录。

    Returns:
        临时根目录（回退到 ``.tmp`` 下的固定路径）。
    """
    return _TEMP_ROOT_OK or (ROOT / ".tmp")


@pytest.fixture
def workdir(temp_root: Path) -> Iterator[Path]:
    """为单个测试提供一个**全新**的可写目录。

    与 ``tmp_path`` 的区别：目录名唯一且不依赖 pytest 的 basetemp 复用逻辑。

    Args:
        temp_root: 会话级临时根。

    Yields:
        本测试专属的可写目录。
    """
    path = temp_root / f"case-{uuid.uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=True)
    yield path


@pytest.fixture
def project_root(workdir: Path) -> Path:
    """构造一个隔离的「项目根」目录（含 ``.git`` 标记）。

    Args:
        workdir: 本测试专属目录。

    Returns:
        项目根路径。
    """
    (workdir / ".git").mkdir(exist_ok=True)
    return workdir


@pytest.fixture
def home_dir(temp_root: Path) -> Path:
    """构造一个隔离的 ``MYAGENT_HOME``。

    Args:
        temp_root: 会话级临时根。

    Returns:
        全局目录路径。
    """
    home = temp_root / f"home-{uuid.uuid4().hex[:8]}"
    home.mkdir(parents=True, exist_ok=True)
    return home


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, home_dir: Path) -> Iterator[dict[str, str]]:
    """把 ``MYAGENT_HOME`` 与测试用密钥指向隔离目录。

    Args:
        monkeypatch: pytest 的 monkeypatch。
        home_dir: 隔离的全局目录。

    Yields:
        本次测试生效的环境变量字典。
    """
    env = {
        "MYAGENT_HOME": str(home_dir),
        "MYAGENT_TEST_API_KEY": "test-key-not-a-secret",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    yield env


@pytest.fixture
def make_config(isolated_env: dict[str, str]):
    """返回一个构造 :class:`~my_agent.config.schema.Config` 的工厂。

    Args:
        isolated_env: 隔离环境变量。

    Returns:
        工厂函数：``make_config(**overrides) -> Config``。
    """

    def factory(**overrides: object):
        from my_agent.config.schema import Config, EndpointConfig, LLMConfig

        endpoint = EndpointConfig(
            name="main",
            base_url="http://127.0.0.1:9/v1",
            api_key_env="MYAGENT_TEST_API_KEY",
            model="test-model",
            context_window=200_000,
            max_output_tokens=8_192,
        )
        llm = LLMConfig(endpoint="main", endpoints={"main": endpoint})
        config = Config(llm=llm)
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    return factory
