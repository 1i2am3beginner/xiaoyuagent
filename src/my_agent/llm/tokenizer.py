"""Token 计数（§7.2）。

统一口径：**所有预算判断必须使用同一个 ``TokenCounter`` 实例**（§7.2），避免混用
导致水位漂移。首选 :class:`TiktokenCounter`（按模型选 encoding，导入或取 encoding
失败自动回退），兜底 :class:`HeuristicCounter`（CJK ×1.0 + 其它字符 /4，工具 JSON
每键 +8）。

``tiktoken`` 是可选依赖（``pyproject.toml`` 的 ``extras``），**懒加载**：未安装时
本模块照常可用，只是 :func:`get_counter` 返回启发式计数器。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from my_agent.models.messages import Message

__all__ = [
    "CJK_TOKENS_PER_CHAR",
    "DEFAULT_ENCODING",
    "OTHER_CHARS_PER_TOKEN",
    "TOOL_JSON_TOKENS_PER_KEY",
    "HeuristicCounter",
    "TiktokenCounter",
    "TokenCounter",
    "get_counter",
]

_LOG = logging.getLogger(__name__)

#: 每个 CJK（含全角标点、假名、谚文）字符折算的 token 数。
CJK_TOKENS_PER_CHAR = 1.0

#: 非 CJK 字符每多少个折算 1 个 token。
OTHER_CHARS_PER_TOKEN = 4.0

#: 工具调用 JSON 每个键额外计入的开销（引号、冒号、结构符号）。
TOOL_JSON_TOKENS_PER_KEY = 8

#: tiktoken 取不到模型对应 encoding 时的回退名。
DEFAULT_ENCODING = "cl100k_base"

#: 消息结构的固定开销（role / 分隔符），与 OpenAI 官方 cookbook 的估算口径一致。
_MESSAGE_OVERHEAD_TOKENS = 4


@runtime_checkable
class TokenCounter(Protocol):
    """Token 计数器协议（§7.2）。"""

    def count_text(self, text: str) -> int:
        """统计一段文本的 token 数。

        Args:
            text: 待统计文本。

        Returns:
            token 数（非负）。
        """
        ...

    def count_messages(self, msgs: list[Message]) -> int:
        """统计一组消息的 token 数（含角色开销与工具调用 JSON）。

        Args:
            msgs: 消息列表。

        Returns:
            token 数（非负）。
        """
        ...


def is_cjk(char: str) -> bool:
    """判断字符是否按「1 字符 ≈ 1 token」计（CJK 及全角区）。

    Args:
        char: 单个字符。

    Returns:
        ``True`` = CJK 汉字 / 假名 / 谚文 / 全角标点等。

    Raises:
        TypeError: ``char`` 不是长度为 1 的字符串。
    """
    if len(char) != 1:
        raise TypeError(f"is_cjk 需要单个字符，收到 {char!r}")
    code = ord(char)
    return (
        0x1100 <= code <= 0x11FF  # 谚文字母
        or 0x2E80 <= code <= 0x303F  # CJK 部首 + CJK 符号与标点
        or 0x3040 <= code <= 0x30FF  # 平假名 + 片假名
        or 0x3130 <= code <= 0x318F  # 谚文兼容字母
        or 0x3400 <= code <= 0x4DBF  # CJK 扩展 A
        or 0x4E00 <= code <= 0x9FFF  # CJK 统一表意文字
        or 0xA960 <= code <= 0xA97F  # 谚文字母扩展 A
        or 0xAC00 <= code <= 0xD7AF  # 谚文音节
        or 0xF900 <= code <= 0xFAFF  # CJK 兼容表意文字
        or 0xFF00 <= code <= 0xFFEF  # 全角字符
        or 0x20000 <= code <= 0x3FFFF  # CJK 扩展 B 及以后
    )


class HeuristicCounter:
    """启发式计数器：CJK ×1.0 + 其它字符 /4，工具 JSON 每键 +8（§7.2）。

    这是**兜底**实现：无第三方依赖、纯确定性，用于 ``tiktoken`` 不可用或端点未知的
    场合。误差由 §7.2 的在线校准（``USAGE_CALIBRATION_FACTOR``）在运行中收敛。
    """

    def count_text(self, text: str) -> int:
        """按字符类别估算 token 数。

        Args:
            text: 待统计文本。

        Returns:
            估算 token 数；空文本为 0（向上取整，保证非空文本至少 1）。
        """
        if not text:
            return 0
        cjk = sum(1 for char in text if is_cjk(char))
        other = len(text) - cjk
        estimate = cjk * CJK_TOKENS_PER_CHAR + other / OTHER_CHARS_PER_TOKEN
        return max(1, int(estimate + 0.9999))

    def count_messages(self, msgs: list[Message]) -> int:
        """估算消息列表的 token 数。

        每条消息计入固定角色开销；``tool_calls`` 的参数按「每键 +8」追加。

        Args:
            msgs: 消息列表。

        Returns:
            估算 token 数。
        """
        total = 0
        for msg in msgs:
            total += _MESSAGE_OVERHEAD_TOKENS
            total += self.count_text(msg.content or "")
            if msg.name:
                total += self.count_text(msg.name)
            for call in msg.tool_calls:
                total += TOOL_JSON_TOKENS_PER_KEY
                total += self.count_text(call.name)
                total += count_json_keys(call.arguments)
        return total


def count_json_keys(obj: Any) -> int:
    """递归统计 JSON 结构里的键数量（供「每键 +8」口径使用）。

    Args:
        obj: 任意已解析的 JSON 结构。

    Returns:
        键的总数（含嵌套）。
    """
    if isinstance(obj, Mapping):
        return sum(1 + count_json_keys(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(count_json_keys(item) for item in obj)
    return 0


class TiktokenCounter:
    """基于 ``tiktoken`` 的精确计数器（§7.2 首选实现）。

    ``tiktoken`` **懒加载**：模块导入、``encoding_for_model``、``get_encoding`` 任一步
    失败都会自动回退到 :class:`HeuristicCounter`，因此本类在未安装 tiktoken 的环境下
    依然可用（不抛异常、不中断预算计算）。
    """

    def __init__(self, model: str | None = None) -> None:
        """初始化计数器（此刻并**不**导入 tiktoken）。

        Args:
            model: 模型名；``None``/未知模型会回退到 :data:`DEFAULT_ENCODING`。
        """
        self.model = model or ""
        self.fallback = HeuristicCounter()
        self._loader: Callable[[], Any] | None = None
        self._encoder: Any | None = None

    @property
    def using_tiktoken(self) -> bool:
        """当前是否真的用上了 tiktoken（回退后为 ``False``）。

        Returns:
            ``True`` = encoder 已成功加载。
        """
        return self._encoder is not None

    def count_text(self, text: str) -> int:
        """统计文本 token 数；encoder 不可用时回退到启发式。

        Args:
            text: 待统计文本。

        Returns:
            token 数（非负）。
        """
        if not text:
            return 0
        encoder = self._get_encoder()
        if encoder is None:
            return self.fallback.count_text(text)
        try:
            return len(encoder.encode(text))
        except Exception:  # noqa: BLE001 - 第三方编码器异常一律回退，不中断预算
            _LOG.warning("tiktoken 编码失败，已回退到启发式计数", exc_info=True)
            self._encoder = None
            return self.fallback.count_text(text)

    def count_messages(self, msgs: list[Message]) -> int:
        """统计消息列表 token 数。

        Args:
            msgs: 消息列表。

        Returns:
            token 数（非负）。
        """
        total = 0
        for msg in msgs:
            total += _MESSAGE_OVERHEAD_TOKENS
            total += self.count_text(msg.content or "")
            if msg.name:
                total += self.count_text(msg.name)
            for call in msg.tool_calls:
                total += TOOL_JSON_TOKENS_PER_KEY
                total += self.count_text(call.name)
                total += self.count_text(_dumps(call.arguments))
        return total

    def _get_encoder(self) -> Any | None:
        """惰性加载 tiktoken encoder。

        Returns:
            encoder 对象；tiktoken 不可用时为 ``None``（此后不再重试加载）。
        """
        if self._encoder is not None:
            return self._encoder
        if self._loader is None:
            self._loader = _load_tiktoken_encoder(self.model)
        self._encoder = self._loader()
        return self._encoder


def _load_tiktoken_encoder(model: str) -> Callable[[], Any]:
    """构造 encoder 加载器（把 import 推迟到首次计数时）。

    Args:
        model: 模型名。

    Returns:
        无参可调用对象：成功返回 encoder，失败返回 ``None``。
    """

    def load() -> Any:
        try:
            import tiktoken
        except ImportError:
            _LOG.info("tiktoken 未安装 → 使用启发式 token 计数（§7.2 兜底）")
            return None
        if model:
            try:
                return tiktoken.encoding_for_model(model)
            except (KeyError, ValueError):
                _LOG.info("tiktoken 不认识模型 %s → 回退到 %s", model, DEFAULT_ENCODING)
        try:
            return tiktoken.get_encoding(DEFAULT_ENCODING)
        except (KeyError, ValueError):
            _LOG.warning("tiktoken 取不到 encoding %s → 使用启发式计数", DEFAULT_ENCODING)
            return None

    return load


def get_counter(model: str | None = None) -> TokenCounter:
    """返回该模型适用的计数器（§7.2）。

    已安装 ``tiktoken`` 时返回 :class:`TiktokenCounter`（内部仍带启发式回退），
    否则直接返回 :class:`HeuristicCounter`。

    Args:
        model: 模型名；``None`` = 使用默认 encoding 或启发式。

    Returns:
        满足 :class:`TokenCounter` 协议的计数器。

    Raises:
        TypeError: ``model`` 既不是 ``str`` 也不是 ``None``。
    """
    if model is not None and not isinstance(model, str):
        raise TypeError(f"get_counter 需要模型名或 None，收到 {type(model).__name__}")
    if _tiktoken_available():
        return TiktokenCounter(model)
    return HeuristicCounter()


def _tiktoken_available() -> bool:
    """检查 ``tiktoken`` 是否可导入（只探测，不加载 encoding）。

    Returns:
        ``True`` = 已安装。
    """
    import importlib.util

    try:
        return importlib.util.find_spec("tiktoken") is not None
    except (ImportError, ValueError):  # pragma: no cover - 防御性
        return False


def count_tool_schema(schema: Sequence[Mapping[str, Any]], counter: TokenCounter) -> int:
    """统计工具 schema 列表的 token 数（工具 JSON 每键 +8 口径）。

    Args:
        schema: ``{"type": "function", "function": {...}}`` 列表。
        counter: 计数器实例。

    Returns:
        估算 token 数。
    """
    total = 0
    for item in schema:
        total += TOOL_JSON_TOKENS_PER_KEY
        total += counter.count_text(_dumps(dict(item)))
    return total


def _dumps(value: Any) -> str:
    """把任意结构序列化为紧凑 JSON 文本。

    Args:
        value: 待序列化对象。

    Returns:
        JSON 字符串；不可序列化时退化为 ``str(value)``。
    """
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)
