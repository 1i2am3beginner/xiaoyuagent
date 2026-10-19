"""工具协议、基类与共享 I/O 辅助（§14.1）。

本模块只定义「所有工具长什么样」，不含任何具体工具：

- :class:`Tool`：工具协议（``spec`` + ``async run``），MCP/子 Agent 工具也按它适配。
- :class:`BaseTool`：内置工具基类，统一 ``schema()`` / ``ok()`` / ``fail()``。
- 共享小工具：路径解析、二进制嗅探、按行读取、原子写。

安全不变量（C1）：工具本身**不做**权限判定；执行入口只有
:class:`~my_agent.tools.registry.ToolRegistry`，由 ``ctx.permission_checked`` 双保险拦截
「未经权限流水线」的调用。

规格：``my_agent-开发文档.md`` §14.1、§14.3。
"""

from __future__ import annotations

import codecs
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from my_agent.models.tools import ToolContext, ToolResult, ToolSpec

__all__ = [
    "BaseTool",
    "Tool",
    "as_positive_int",
    "atomic_write_bytes",
    "atomic_write_text",
    "count_lines",
    "decode_text",
    "display_path",
    "is_binary_data",
    "read_text_normalized",
    "resolve_path",
    "to_posix",
]

#: 嗅探二进制时检查的头部字节数（NUL 字节命中即判定为二进制）。
BINARY_SNIFF_BYTES = 8_192

#: 文本解码候选编码（按顺序尝试；``utf-8`` 失败后回退到中文/西文常见编码）。
TEXT_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "gbk", "cp1252")


@runtime_checkable
class Tool(Protocol):
    """工具协议（§14.1）。

    Attributes:
        spec: 工具的静态声明（名字、描述、参数 Schema、类别、是否只读）。
    """

    spec: ToolSpec

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """执行工具。

        Args:
            args: 模型给出的参数（已解析成 dict，但**取值可能非法**，需自行校验）。
            ctx: 执行上下文（cwd / project_root / artifact_dir / call_id 等）。

        Returns:
            工具结果。业务性失败（文件不存在、命令非零退出等）应返回
            ``ok=False`` 的 :class:`ToolResult` 而不是抛异常（§14.4 失败隔离）。
        """
        ...


class BaseTool:
    """内置工具基类，统一 ``spec`` / ``schema()`` / ``ok()`` / ``fail()``。

    子类只需设置类属性 ``spec`` 并实现 :meth:`run`。

    Attributes:
        spec: 工具的静态声明，子类必须提供。
    """

    spec: ToolSpec

    def schema(self) -> dict[str, Any]:
        """返回发往 Provider 的 function schema。

        Returns:
            ``{"type": "function", "function": {...}}``（§14.1）。
        """
        return self.spec.to_schema()

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """执行工具；基类不提供实现。

        Args:
            args: 模型给出的参数。
            ctx: 执行上下文。

        Returns:
            工具结果。

        Raises:
            NotImplementedError: 子类未覆盖本方法。
        """
        del args, ctx  # 子类覆盖后才使用；基类只负责统一报错
        raise NotImplementedError(f"工具 {self.spec.name} 未实现 run()")

    def ok(
        self,
        ctx: ToolContext,
        content: str,
        *,
        exit_code: int | None = None,
        truncated: bool = False,
        duration_ms: int = 0,
    ) -> ToolResult:
        """构造成功结果。

        Args:
            ctx: 执行上下文（提供 ``call_id``）。
            content: 给模型看的文本。
            exit_code: 命令类工具的退出码。
            truncated: 内容是否被截断。
            duration_ms: 耗时毫秒。

        Returns:
            ``ok=True`` 的 :class:`ToolResult`。
        """
        return ToolResult(
            call_id=ctx.call_id,
            tool=self.spec.name,
            ok=True,
            content=content,
            exit_code=exit_code,
            truncated=truncated,
            duration_ms=duration_ms,
        )

    def fail(
        self,
        ctx: ToolContext,
        error: str,
        *,
        content: str = "",
        exit_code: int | None = None,
        duration_ms: int = 0,
    ) -> ToolResult:
        """构造失败结果。

        Args:
            ctx: 执行上下文（提供 ``call_id``）。
            error: 失败原因（给模型看，要求可操作）。
            content: 可选的附加输出（如命令的部分输出）。
            exit_code: 命令类工具的退出码。
            duration_ms: 耗时毫秒。

        Returns:
            ``ok=False`` 的 :class:`ToolResult`。
        """
        return ToolResult(
            call_id=ctx.call_id,
            tool=self.spec.name,
            ok=False,
            content=content,
            error=error,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )


def resolve_path(raw: str, ctx: ToolContext) -> Path:
    """把工具参数里的路径解析成绝对路径。

    相对路径一律相对 ``ctx.cwd``（**不**相对进程 cwd，避免子 Agent/多会话串味）。

    Args:
        raw: 参数中的路径字符串（``~`` 会被展开）。
        ctx: 执行上下文。

    Returns:
        规范化后的绝对路径（不做存在性检查，也不做沙盒限制——沙盒由权限流水线负责）。
    """
    expanded = Path(raw).expanduser()
    if not expanded.is_absolute():
        expanded = Path(ctx.cwd) / expanded
    return Path(expanded).resolve()


def to_posix(path: Path) -> str:
    """把路径转成正斜杠形式（用于匹配 glob/gitignore 模式）。

    Args:
        path: 任意路径。

    Returns:
        使用 ``/`` 分隔的字符串。
    """
    return path.as_posix()


def display_path(path: Path, ctx: ToolContext) -> str:
    """生成给模型/用户看的路径（项目内用相对路径，项目外保留绝对路径）。

    Args:
        path: 目标路径。
        ctx: 执行上下文（提供 ``project_root`` 与 ``cwd``）。

    Returns:
        相对项目根的正斜杠路径；不在项目内时返回原绝对路径。
    """
    resolved = Path(path).resolve()
    for root in (ctx.project_root, ctx.cwd):
        if not root:
            continue
        try:
            return to_posix(resolved.relative_to(Path(root).resolve()))
        except ValueError:
            continue
    return str(resolved)


def as_positive_int(raw: Any, default: int) -> int | None:
    """把工具参数解析为正整数。

    Args:
        raw: 参数原值（``None`` 表示未提供）。
        default: 未提供时使用的默认值。

    Returns:
        解析出的正整数；``None`` 表示参数非法（调用方应返回参数错误）。
    """
    if raw is None:
        return default
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, str) and raw.strip().lstrip("+").isdigit():
        value = int(raw.strip())
        return value if value > 0 else None
    return None


def count_lines(text: str) -> int:
    """统计文本行数（与 ``Read`` 的口径一致：末尾换行不算新的一行）。

    Args:
        text: 文本内容。

    Returns:
        行数；空文本为 0。
    """
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def is_binary_data(data: bytes) -> bool:
    """判断字节串是否像二进制内容。

    Args:
        data: 待判定的字节串（通常只传文件头部）。

    Returns:
        头部含 NUL 字节时返回 ``True``（与 git/grep 的启发式一致）。
    """
    return b"\x00" in data[:BINARY_SNIFF_BYTES]


def decode_text(data: bytes) -> str:
    """把字节串解码为文本。

    Args:
        data: 文件内容字节串。

    Returns:
        解码后的文本；按 :data:`TEXT_ENCODINGS` 顺序尝试，最后兜底 ``cp1252``。

    Raises:
        UnicodeDecodeError: 所有候选编码都失败（极少见，说明内容确实不是文本）。
    """
    for encoding in TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    # 明确抛错而不是静默丢弃：调用方（Read）会把它翻译成「疑似二进制」错误。
    return codecs.decode(data, "utf-8")  # pragma: no cover - 兜底路径


def read_text_normalized(path: Path) -> tuple[str, str]:
    """读取文本文件并把换行归一化为 ``\\n``。

    Args:
        path: 目标文件。

    Returns:
        ``(归一化文本, 原换行符)``；原换行符为 ``"\\r\\n"`` 或 ``"\\n"``。

    Raises:
        OSError: 读取失败（不存在、权限、是目录等）。
        UnicodeDecodeError: 无法按候选编码解码（疑似二进制）。
    """
    text = decode_text(path.read_bytes())
    newline = "\r\n" if "\r\n" in text else "\n"
    if newline == "\r\n":
        text = text.replace("\r\n", "\n")
    return text, newline


def atomic_write_bytes(path: Path, data: bytes) -> int:
    """原子写入字节串（同目录临时文件 + ``os.replace``）。

    Args:
        path: 目标文件（父目录必须已存在）。
        data: 待写入内容。

    Returns:
        写入的字节数。

    Raises:
        OSError: 写入或替换失败（临时文件会被清理）。
    """
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(data)
            handle.flush()
        Path(tmp_name).replace(path)
    except OSError:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise
    return len(data)


def atomic_write_text(
    path: Path,
    text: str,
    *,
    newline: str = "\n",
    encoding: str = "utf-8",
) -> int:
    """原子写入文本（同目录临时文件 + ``os.replace``）。

    Args:
        path: 目标文件（父目录必须已存在）。
        text: 内容（内部一律使用 ``\\n``）。
        newline: 落盘换行符，``"\\r\\n"`` 时逐行还原为 CRLF。
        encoding: 编码，默认 utf-8。

    Returns:
        写入的字节数。

    Raises:
        OSError: 写入或替换失败（临时文件会被清理）。
    """
    payload = text.replace("\n", "\r\n") if newline == "\r\n" else text
    return atomic_write_bytes(path, payload.encode(encoding))
