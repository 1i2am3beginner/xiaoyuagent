"""归档回读工具 ``ArchiveSearch``（§7.7.3）。

上游要求「全文压缩不是简单丢弃，会把完整对话历史保存，留有**专门的 tool** 去读取」——
本模块就是那个 tool。它在**正常状态下也注册**（无归档时返回「本会话暂无归档历史」），
避免模型产生「工具不存在」的困惑。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from my_agent.errors import ArchiveCorruptedError, MyAgentError
from my_agent.models.tools import ToolContext, ToolResult, ToolSpec

if TYPE_CHECKING:
    from my_agent.context.archive import ArchiveStore

__all__ = ["ARCHIVE_SEARCH_PARAMETERS", "ARCHIVE_SEARCH_SPEC", "ArchiveSearchTool"]

ARCHIVE_SEARCH_DESCRIPTION = (
    "检索本会话已被压缩/归档的历史对话。当用户提到早前讨论过、但现在不在上下文中的内容时使用。\n"
    "何时用：用户说「刚才说的那个」「之前那个文件」而你在当前上下文里找不到；"
    "或压缩提示里给出的归档文件需要定位具体轮次时。\n"
    "何时不用：内容仍在当前上下文里（直接看即可）；要找的是**磁盘上的代码**——那用 Grep/Glob。"
    "反例：用户让你找项目里的某个函数定义，不要用本工具，用 Grep。\n"
    "副作用声明：只读，不修改任何文件。至少给出 query / turn_start+turn_end / file_glob 之一。"
)

#: §7.7.3 的参数 schema（逐字采用）。
ARCHIVE_SEARCH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "关键词（不区分大小写）"},
        "turn_start": {"type": "integer", "description": "轮次区间起点（含）"},
        "turn_end": {"type": "integer", "description": "轮次区间终点（含）"},
        "file_glob": {
            "type": "string",
            "description": "只搜涉及该路径模式的消息，如 **/auth.py",
        },
        "limit": {"type": "integer", "default": 20, "maximum": 100},
        "context_lines": {"type": "integer", "default": 2},
    },
    "required": [],
}

ARCHIVE_SEARCH_SPEC = ToolSpec(
    name="ArchiveSearch",
    description=ARCHIVE_SEARCH_DESCRIPTION,
    parameters=ARCHIVE_SEARCH_PARAMETERS,
    category="read",
    is_read_only=True,
    is_destructive=False,
    source="builtin",
)

NO_CRITERIA = (
    "参数错误：至少给出一个检索条件（query / turn_start+turn_end / file_glob）。"
)
NO_ARCHIVE = "本会话暂无归档历史。"


class ArchiveSearchTool:
    """``ArchiveSearch`` 工具的 Tool 协议实现。"""

    def __init__(self, archive_store: ArchiveStore) -> None:
        """初始化。

        Args:
            archive_store: 归档存储。
        """
        self.archive = archive_store
        self.spec = ARCHIVE_SEARCH_SPEC

    def schema(self) -> dict[str, Any]:
        """返回发往 Provider 的 function schema。

        Returns:
            OpenAI 兼容 schema。
        """
        return self.spec.to_schema()

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """在归档 JSONL 中检索历史消息。

        Args:
            args: ``query`` / ``turn_start`` / ``turn_end`` / ``file_glob`` /
                ``limit`` / ``context_lines``。
            ctx: 执行上下文（提供 ``session_id``）。

        Returns:
            :class:`~my_agent.models.tools.ToolResult`：命中消息的紧凑视图；
            无归档时 ``ok=True`` 且内容为「本会话暂无归档历史」。
        """
        query = _opt_str(args.get("query"))
        file_glob = _opt_str(args.get("file_glob"))
        turn_start = _opt_int(args.get("turn_start"))
        turn_end = _opt_int(args.get("turn_end"))
        if query is None and file_glob is None and turn_start is None and turn_end is None:
            return ToolResult(
                call_id=ctx.call_id, tool="ArchiveSearch", ok=False, error=NO_CRITERIA
            )

        limit = min(max(_opt_int(args.get("limit")) or 20, 1), 100)
        context_lines = max(_opt_int(args.get("context_lines")) or 2, 0)

        try:
            index = self.archive.index(ctx.session_id)
        except MyAgentError as exc:
            return ToolResult(
                call_id=ctx.call_id, tool="ArchiveSearch", ok=False, error=exc.user_message()
            )
        if not index:
            return ToolResult(
                call_id=ctx.call_id, tool="ArchiveSearch", ok=True, content=NO_ARCHIVE
            )

        try:
            text = self.archive.search(
                ctx.session_id,
                query=query,
                turn_start=turn_start,
                turn_end=turn_end,
                file_glob=file_glob,
                limit=limit,
                context_lines=context_lines,
            )
        except ArchiveCorruptedError as exc:
            return ToolResult(
                call_id=ctx.call_id, tool="ArchiveSearch", ok=False, error=exc.user_message()
            )
        except MyAgentError as exc:
            return ToolResult(
                call_id=ctx.call_id, tool="ArchiveSearch", ok=False, error=exc.user_message()
            )

        return ToolResult(
            call_id=ctx.call_id, tool="ArchiveSearch", ok=True, content=text or "未命中任何归档消息。"
        )


def _opt_str(value: Any) -> str | None:
    """把可选参数规整为字符串或 ``None``。

    Args:
        value: 原始值。

    Returns:
        去空白后的字符串；空值返回 ``None``。
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_int(value: Any) -> int | None:
    """把可选参数规整为整数或 ``None``。

    Args:
        value: 原始值。

    Returns:
        整数；无法解析时 ``None``。
    """
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
