"""``WebSearch`` / ``WebFetch`` 工具：网络检索与抓取（§14.2、【待确认假设 U28】）。

- ``WebSearch`` 走**可配置后端**；未配置 → 返回明确错误，**绝不编造结果**（U28）。
- ``WebFetch``：HTTP GET + 大小上限 :data:`~my_agent.config.defaults.WEB_FETCH_MAX_BYTES`
  + 超时 :data:`~my_agent.config.defaults.WEB_FETCH_TIMEOUT_S`，HTML 转纯文本，非 2xx 明确报错。
- 两者都是只读工具（可与其它只读工具并发，§14.4）。
- **网络异常重试**（工具层补齐模型层的同等纪律）：超时 / 连接失败 / 408 / 429 / 5xx →
  指数退避重试 :data:`~my_agent.config.defaults.TOOL_RETRY_MAX_ATTEMPTS` 次
  （见 :mod:`my_agent.retry`）；4xx（如 404）**立即失败**，重试无用。
  重试**耗尽后仍返回** ``ToolResult(ok=False)``——工具层失败绝不外泄成未捕获异常（§14.4）。

配置读取刻意做成**宽容查找**（pydantic 模型 / dict / 任意对象都支持），因为配置 schema
由 §15.2 定义，``web`` 段是 U28 新增的可选段；缺失即视为「未配置」。
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx

from my_agent.config.defaults import (
    TOOL_RETRY_BACKOFF_S,
    TOOL_RETRY_MAX_ATTEMPTS,
    WEB_FETCH_MAX_BYTES,
    WEB_FETCH_TIMEOUT_S,
)
from my_agent.models.tools import ToolContext, ToolResult, ToolSpec
from my_agent.retry import retryable_status, with_network_retry
from my_agent.tools.base import BaseTool

__all__ = ["WebFetchTool", "WebSearchTool"]

#: 模块级可注入睡眠（测试 monkeypatch 本对象即可跳过真实退避等待）。
_retry_sleep: Callable[[float], Awaitable[None]] | None = None

#: 每个 query 期望的结果条数（工具级默认，非规格阈值）。
SEARCH_RESULTS_PER_QUERY = 5

#: 抓取输出中保留的最大字符数（防止一个超长正文撑爆上下文）。
FETCH_MAX_CHARS = 100_000

_WEB_SEARCH_DESCRIPTION = """调用已配置的搜索后端执行网络搜索，返回每条结果的标题、链接与摘要。

何时用：需要外部最新信息（版本号、官方文档、报错含义）而本地代码无法回答时。
何时不用：① 不要用它搜本地代码——用 Grep/Glob；② 未配置搜索后端时它只会返回明确错误，
   此时**不要**据此编造结果，应改为询问用户或改用 WebFetch 抓取已知 URL；
③ 不要用它抓取整页正文——用 WebFetch。
副作用：无（只读，不修改磁盘文件；会发起网络请求）。"""

_WEB_FETCH_DESCRIPTION = """对指定 URL 发起 HTTP GET，把返回内容（HTML 自动转纯文本）取回。

何时用：已经知道具体文档/网页地址，需要其正文内容时。
何时不用：① 不要用它做检索——用 WebSearch；② 不要用它读本地文件——用 Read；
③ 不要用它抓取超大文件、需登录或依赖 JS 渲染的页面（有大小上限、超时与纯文本转换限制）。
副作用：无（只读，不修改磁盘文件；会发起网络请求）。"""


def _lookup(config: Any, *path: str) -> Any:
    """宽容地逐层读取配置。

    Args:
        config: 配置对象（pydantic 模型 / dict / 任意对象，可为 ``None``）。
        *path: 属性名或键名序列。

    Returns:
        命中的值；任一环节缺失时 ``None``。
    """
    current: Any = config
    for key in path:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


class WebSearchTool(BaseTool):
    """可配置后端的网络搜索（未配置即明确报错）。

    Args:
        config: 全局配置；读取 ``web.search_backend`` / ``web.search_url`` /
            ``web.search_api_key_env`` / ``web.search_method`` / ``web.search_results_path``
            （亦兼容 ``web_search_backend`` 之类的扁平键与同名环境变量）。
    """

    spec = ToolSpec(
        name="WebSearch",
        description=_WEB_SEARCH_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "一个或多个搜索关键词",
                }
            },
            "required": ["queries"],
        },
        category="network",
        is_read_only=True,
    )

    def __init__(self, config: Any = None) -> None:
        self.config = config

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """执行搜索。

        Args:
            args: ``queries``（字符串数组；也接受单个 ``query`` 字符串）。
            ctx: 执行上下文（仅用于结果记账）。

        Returns:
            ``ok=True`` 且 ``content`` 为格式化结果；未配置后端或后端失败时 ``ok=False``。
        """
        queries = self._queries(args)
        if queries is None:
            return self.fail(ctx, "缺少必填参数 queries（应为非空字符串数组）")

        backend = self._setting("search_backend", "MYAGENT_WEB_SEARCH_BACKEND", default="")
        backend = (backend or "").strip().lower()
        if backend in ("", "none", "off", "disabled"):
            return self.fail(
                ctx,
                "WebSearch 未配置搜索后端（U28）：请先在配置中设置 web.search_backend 与 "
                "web.search_url（以及 web.search_api_key_env 指向的密钥环境变量）。"
                "在此之前请勿凭记忆编造搜索结果；已知具体网址时可用 WebFetch 抓取。",
            )

        url_template = self._setting("search_url", "MYAGENT_WEB_SEARCH_URL", default="")
        if backend == "tavily":
            url_template = url_template or "https://api.tavily.com/search"
            key_env = self._setting(
                "search_api_key_env", "MYAGENT_WEB_SEARCH_API_KEY_ENV", default="TAVILY_API_KEY"
            )
        else:
            key_env = self._setting(
                "search_api_key_env", "MYAGENT_WEB_SEARCH_API_KEY_ENV", default=""
            )
        if not url_template:
            return self.fail(
                ctx,
                f"WebSearch 后端 {backend!r} 缺少 web.search_url 配置，无法发起请求（U28）。",
            )

        results_path = self._setting(
            "search_results_path", "MYAGENT_WEB_SEARCH_RESULTS_PATH", default="results"
        )
        method = str(
            self._setting("search_method", "MYAGENT_WEB_SEARCH_METHOD", default="")
            or ("GET" if "{query}" in url_template else "POST")
        ).upper()
        api_key = os.environ.get(key_env, "") if key_env else ""

        async def attempt() -> list[str]:
            """整批查询的一次尝试（每次重试都重建客户端，天然重发全部请求）。

            Returns:
                每个关键词渲染后的结果文本块。

            Raises:
                httpx.HTTPError: 网络/协议错误或可重试的非 2xx 状态。
                ValueError: 响应不是合法 JSON 或结构不符合预期。
            """
            blocks: list[str] = []
            async with httpx.AsyncClient(timeout=WEB_FETCH_TIMEOUT_S) as client:
                for query in queries:
                    payload = await self._search_one(
                        client,
                        query=query,
                        url_template=url_template,
                        method=method,
                        api_key=api_key,
                        results_path=results_path or "results",
                    )
                    blocks.append(payload)
            return blocks

        retries = 0

        def note_retry(_attempt: int, _delay: float, _exc: BaseException) -> None:
            """统计实际重试次数（用于最终错误文案）。

            Args:
                _attempt: 重试序号（0 起）。
                _delay: 本次退避秒数。
                _exc: 触发重试的异常。
            """
            nonlocal retries
            retries += 1

        try:
            blocks = await with_network_retry(
                attempt,
                attempts=TOOL_RETRY_MAX_ATTEMPTS,
                schedule=TOOL_RETRY_BACKOFF_S,
                on_retry=note_retry,
                sleep=_retry_sleep,
                label="WebSearch",
            )
        except httpx.HTTPStatusError as exc:
            return self.fail(
                ctx,
                f"搜索请求失败：后端返回 HTTP {exc.response.status_code}"
                f"（已重试 {retries} 次仍失败）。",
            )
        except httpx.HTTPError as exc:
            return self.fail(ctx, f"搜索请求失败（已重试 {retries} 次）：{exc}")
        except (ValueError, KeyError) as exc:
            return self.fail(
                ctx, f"搜索后端返回结构无法解析（{exc}）；请检查 web.search_results_path"
            )
        return self.ok(ctx, "\n\n".join(blocks))

    def _setting(self, key: str, env_key: str, *, default: str) -> str:
        """读取一个搜索配置项（配置优先，环境变量兜底）。

        Args:
            key: ``config.web.<key>`` 或扁平 ``config.web_<key>``。
            env_key: 兜底环境变量名。
            default: 都缺失时的默认值。

        Returns:
            配置字符串。
        """
        value = _lookup(self.config, "web", key)
        if value is None:
            value = _lookup(self.config, f"web_{key}")
        if value is None:
            value = os.environ.get(env_key)
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _queries(args: dict[str, Any]) -> list[str] | None:
        """解析 ``queries`` 参数。

        Args:
            args: 工具参数。

        Returns:
            去空后的关键词列表；非法时 ``None``。
        """
        raw = args.get("queries")
        if raw is None and isinstance(args.get("query"), str):
            raw = [args["query"]]
        if not isinstance(raw, list):
            return None
        queries = [item.strip() for item in raw if isinstance(item, str) and item.strip()]
        return queries or None

    async def _search_one(
        self,
        client: httpx.AsyncClient,
        *,
        query: str,
        url_template: str,
        method: str,
        api_key: str,
        results_path: str,
    ) -> str:
        """执行单个关键词的搜索。

        Args:
            client: httpx 客户端。
            query: 关键词。
            url_template: 后端 URL（可含 ``{query}`` 占位符）。
            method: ``GET`` 或 ``POST``。
            api_key: API Key（无则不带鉴权头）。
            results_path: 结果数组在响应 JSON 中的点分路径。

        Returns:
            渲染后的结果文本。

        Raises:
            httpx.HTTPError: 网络/协议错误；或**可重试**的非 2xx（408 / 429 / 5xx）——
                这一类交给 :func:`~my_agent.retry.with_network_retry` 退避重试。
            ValueError: 响应不是合法 JSON、结构不符合预期，或**不可重试**的非 2xx
                （如 404：重试无用，立即失败）。
        """
        url = url_template.replace("{query}", httpx.QueryParams({"q": query}).get("q", query))
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if method == "GET":
            response = await client.get(url, headers=headers)
        else:
            response = await client.post(
                url, headers=headers, json={"query": query, "max_results": SEARCH_RESULTS_PER_QUERY}
            )
        if response.status_code < 200 or response.status_code >= 300:
            if retryable_status(response.status_code):
                # 交给 with_network_retry：抛可重试的 HTTPStatusError（含 response）。
                response.raise_for_status()
            raise ValueError(f"后端返回 HTTP {response.status_code}")
        data = json.loads(response.text)
        items = self._dig(data, results_path)
        if not isinstance(items, list):
            raise ValueError(f"结果路径 {results_path!r} 不是数组")
        lines = [f"搜索：{query}（后端 {type(client).__name__}，{len(items)} 条）"]
        for index, item in enumerate(items, start=1):
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or item.get("name") or "(无标题)")
            link = str(item.get("url") or item.get("link") or item.get("href") or "")
            snippet = str(
                item.get("snippet") or item.get("content") or item.get("description") or ""
            )
            lines.append(f"{index}. {title}\n   {link}\n   {snippet}".rstrip())
        if len(lines) == 1:
            lines.append("（后端未返回可解析的结果条目）")
        return "\n".join(lines)

    @staticmethod
    def _dig(data: Any, path: str) -> Any:
        """按点分路径取值（如 ``data.items``）。

        Args:
            data: 解析后的 JSON。
            path: 点分路径；空串表示整体。

        Returns:
            命中的值；缺失时 ``None``。
        """
        current = data
        for key in filter(None, path.split(".")):
            if isinstance(current, Mapping):
                current = current.get(key)
            elif isinstance(current, list) and key.isdigit():
                index = int(key)
                current = current[index] if index < len(current) else None
            else:
                return None
        return current


class _HtmlTextExtractor(HTMLParser):
    """把 HTML 粗略转成纯文本（够用即可，不引入额外依赖）。"""

    SKIP_TAGS: frozenset[str] = frozenset({"script", "style", "noscript", "svg", "head"})
    BLOCK_TAGS: frozenset[str] = frozenset(
        {
            "p",
            "div",
            "br",
            "li",
            "tr",
            "td",
            "th",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "section",
            "article",
            "header",
            "footer",
            "table",
            "ul",
            "ol",
            "pre",
            "blockquote",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: str = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """处理开始标签。

        Args:
            tag: 标签名。
            attrs: 属性列表。
        """
        del attrs
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """处理结束标签。

        Args:
            tag: 标签名。
        """
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        """处理文本节点。

        Args:
            data: 文本片段。
        """
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data.strip()
            return
        self.parts.append(data)

    def text(self) -> str:
        """返回清洗后的纯文本（压掉多余空行与行尾空白）。

        Returns:
            纯文本。
        """
        raw = "".join(self.parts)
        lines = [line.strip() for line in raw.splitlines()]
        kept: list[str] = []
        for line in lines:
            if not line and kept and not kept[-1]:
                continue
            kept.append(line)
        return "\n".join(kept).strip()


def html_to_text(html: str) -> tuple[str, str]:
    """把 HTML 转成纯文本。

    Args:
        html: 原始 HTML。

    Returns:
        ``(标题, 正文)``。
    """
    parser = _HtmlTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.title, parser.text()


class WebFetchTool(BaseTool):
    """抓取 URL 并转纯文本（有大小与超时限制）。

    Args:
        config: 全局配置（当前仅用于保持构造签名一致；限额来自 ``config/defaults.py``）。
    """

    spec = ToolSpec(
        name="WebFetch",
        description=_WEB_FETCH_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要抓取的 http/https 地址"},
                "prompt": {
                    "type": "string",
                    "description": "可选：想从页面里了解什么（仅用于结果说明）",
                },
            },
            "required": ["url"],
        },
        category="network",
        is_read_only=True,
    )

    def __init__(self, config: Any = None) -> None:
        self.config = config

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """抓取 URL。

        Args:
            args: ``url``（可选 ``prompt``）。
            ctx: 执行上下文（仅用于结果记账）。

        Returns:
            ``ok=True`` 且 ``content`` 为纯文本正文；URL 非法、非 2xx 或网络失败时 ``ok=False``。
            网络异常会先按指数退避重试（超时 / 连接失败 / 408 / 429 / 5xx），
            重试耗尽后仍返回 ``ok=False``——**绝不抛出未捕获异常**。
        """
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return self.fail(ctx, "缺少必填参数 url（应为 http/https 地址）")
        url = url.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return self.fail(ctx, f"URL 非法：{url}（只支持 http:// 或 https:// 绝对地址）")

        retries = 0

        def note_retry(_attempt: int, _delay: float, _exc: BaseException) -> None:
            """统计实际重试次数（用于最终错误文案）。

            Args:
                _attempt: 重试序号（0 起）。
                _delay: 本次退避秒数。
                _exc: 触发重试的异常。
            """
            nonlocal retries
            retries += 1

        try:
            status, content_type, body, truncated = await self._fetch(url, on_retry=note_retry)
        except httpx.TimeoutException:
            return self.fail(
                ctx,
                f"抓取超时（>{WEB_FETCH_TIMEOUT_S:g}s，已重试 {retries} 次）：{url}",
            )
        except httpx.HTTPStatusError as exc:
            return self.fail(
                ctx,
                f"抓取失败：{url} 返回 HTTP {exc.response.status_code}"
                f"（非 2xx，已重试 {retries} 次）",
            )
        except httpx.HTTPError as exc:
            return self.fail(ctx, f"抓取失败：{url}（已重试 {retries} 次，{exc}）")

        if status < 200 or status >= 300:
            return self.fail(ctx, f"抓取失败：{url} 返回 HTTP {status}（非 2xx）")
        if truncated:
            notice = (
                f"\n\n[内容超过 {WEB_FETCH_MAX_BYTES} 字节已截断；如需后续部分，请改用其它方式获取]"
            )
        else:
            notice = ""
        if "html" in content_type:
            title, text = html_to_text(body)
            header = f"标题：{title}\n来源：{url}\n\n" if title else f"来源：{url}\n\n"
            return self.ok(ctx, self._clip(header + text) + notice, truncated=truncated)
        return self.ok(ctx, self._clip(f"来源：{url}\n\n{body}") + notice, truncated=truncated)

    @staticmethod
    def _clip(text: str) -> str:
        """限制返回文本长度。

        Args:
            text: 原文本。

        Returns:
            超长时截断并附说明。
        """
        if len(text) <= FETCH_MAX_CHARS:
            return text
        return text[:FETCH_MAX_CHARS] + f"\n\n[正文超过 {FETCH_MAX_CHARS} 字符已截断]"

    @staticmethod
    async def _fetch(
        url: str,
        *,
        on_retry: Callable[[int, float, BaseException], None] | None = None,
    ) -> tuple[int, str, str, bool]:
        """发起 HTTP GET（网络异常按 :data:`~my_agent.config.defaults.TOOL_RETRY_MAX_ATTEMPTS` 重试）。

        Args:
            url: 目标地址。
            on_retry: 可选重试回调（签名 ``(attempt, delay, exc)``），供调用方审计重试次数。

        Returns:
            ``(状态码, content-type, 文本正文, 是否截断)``。

        Raises:
            httpx.HTTPError: 网络/协议错误（含超时）；或**可重试**的非 2xx
                （408 / 429 / 5xx）在重试耗尽后抛出。两者都由 :meth:`run` 翻译成
                ``ToolResult(ok=False)``。
        """
        return await with_network_retry(
            lambda: WebFetchTool._fetch_once(url),
            attempts=TOOL_RETRY_MAX_ATTEMPTS,
            schedule=TOOL_RETRY_BACKOFF_S,
            on_retry=on_retry,
            sleep=_retry_sleep,
            label="WebFetch",
        )

    @staticmethod
    async def _fetch_once(url: str) -> tuple[int, str, str, bool]:
        """发起一次 HTTP GET 并按字节上限读取响应（不做重试）。

        Args:
            url: 目标地址。

        Returns:
            ``(状态码, content-type, 文本正文, 是否截断)``。

        Raises:
            httpx.HTTPError: 网络/协议错误（含超时）。
            httpx.HTTPStatusError: 状态码可重试（408 / 429 / 5xx）时抛出，
                交由 :meth:`_fetch` 的退避逻辑处理；其余非 2xx 原样返回状态码。
        """
        async with httpx.AsyncClient(follow_redirects=True, timeout=WEB_FETCH_TIMEOUT_S) as client:
            async with client.stream("GET", url, headers={"User-Agent": "myagent/0.1"}) as response:
                content_type = response.headers.get("content-type", "").lower()
                chunks: list[bytes] = []
                size = 0
                truncated = False
                async for chunk in response.aiter_bytes():
                    remaining = WEB_FETCH_MAX_BYTES - size
                    if len(chunk) >= remaining:
                        chunks.append(chunk[:remaining])
                        truncated = True
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                data = b"".join(chunks)
                encoding = response.encoding or "utf-8"
                try:
                    text = data.decode(encoding, errors="replace")
                except LookupError:
                    text = data.decode("utf-8", errors="replace")
                if (response.status_code < 200 or response.status_code >= 300) and retryable_status(
                    response.status_code
                ):
                    # 408 / 429 / 5xx：抛可重试异常（含 response），交给退避逻辑。
                    response.raise_for_status()
                return response.status_code, content_type, text, truncated
