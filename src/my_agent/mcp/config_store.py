"""MCP 配置文件读写（§10.2、§10.3）。

配置文件两个位置：

* 项目级 ``<project>/.myagent/mcp.yaml``（``project_root_for`` 定位，优先）
* 全局 ``~/.myagent/mcp.yaml``（``myagent_home`` 定位，``$MYAGENT_HOME`` 可覆盖）

本模块负责「配置能被解析」这一层（V1 结构 + ``${ENV_VAR}`` 展开）与**写路径**；
规则校验（V2–V13）在 :mod:`my_agent.mcp.validator`。

**写路径纪律（F-3）**：:meth:`McpConfigStore.save` 只接受「校验通过」的报告；
只要报告里有一条 ``errors``，就在**触碰文件之前**抛
:class:`~my_agent.errors.McpValidationError`，配置文件保持字节级不变。

规格：``my_agent-开发文档.md`` §10.2、§10.3。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from my_agent.config.loader import myagent_home, project_root_for
from my_agent.config.schema import McpServerConfig
from my_agent.errors import McpValidationError
from my_agent.mcp.schema import expand_env_placeholders, expand_server
from my_agent.mcp.validator import ValidationReport, validate_document

__all__ = [
    "MCP_FILENAME",
    "PROJECT_DIR_NAME",
    "McpConfigStore",
    "expand_env_placeholders",
    "parse_mcp_yaml",
]

_LOG = logging.getLogger(__name__)

#: 配置文件名（§10.2）。
MCP_FILENAME = "mcp.yaml"

#: 项目运行时目录名（与 ``config.loader.PROJECT_DIR_NAME`` 同值）。
PROJECT_DIR_NAME = ".myagent"


def parse_mcp_yaml(text: str, *, source: str = MCP_FILENAME) -> Any:
    """解析 ``mcp.yaml`` 文本（§10.3.1 V1 的前半条）。

    Args:
        text: 文件内容。
        source: 文件名（进消息文案）。

    Returns:
        YAML 解析结果（空文件 → ``{}``）。

    Raises:
        McpValidationError: YAML 不可解析；``detail`` 含行号，形如
            ``mcp.yaml 解析失败：第 3 行 …``，``context["rule"] == "V1"``。
    """
    name = Path(source).name
    if not text.strip():
        return {}
    try:
        return yaml.safe_load(text)
    except yaml.MarkedYAMLError as exc:
        problem = exc.problem or "YAML 语法错误"
        mark = getattr(exc, "problem_mark", None)
        if mark is not None:
            detail = f"{name} 解析失败：第 {mark.line + 1} 行 {problem}"
        else:
            detail = f"{name} 解析失败：{problem}"
        raise McpValidationError(detail=detail, rule="V1", source=str(source)) from exc
    except yaml.YAMLError as exc:
        raise McpValidationError(
            detail=f"{name} 解析失败：{exc}", rule="V1", source=str(source)
        ) from exc


class McpConfigStore:
    """``mcp.yaml`` 的定位、读取、校验与写入。

    Args:
        project_root: 项目根目录；``None`` 时由 ``cwd`` 经 ``project_root_for`` 推导。
        home: 全局目录；``None`` 时用 :func:`my_agent.config.loader.myagent_home`。
        cwd: 推导项目根用的起始目录（默认 ``Path.cwd()``）。
        env: 环境变量映射（``${VAR}`` 展开与 ``$MYAGENT_HOME`` 用；默认 ``os.environ``）。
    """

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        home: Path | None = None,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._env = env
        self._project_root = Path(project_root) if project_root else project_root_for(cwd or Path.cwd())
        self._home = Path(home) if home else myagent_home(env)

    # ------------------------------------------------------------------ 路径

    @property
    def project_root(self) -> Path:
        """项目根目录。"""
        return self._project_root

    @property
    def home(self) -> Path:
        """全局配置目录（``~/.myagent``）。"""
        return self._home

    @property
    def project_path(self) -> Path:
        """项目级配置文件：``<project>/.myagent/mcp.yaml``。"""
        return self._project_root / PROJECT_DIR_NAME / MCP_FILENAME

    @property
    def home_path(self) -> Path:
        """全局配置文件：``~/.myagent/mcp.yaml``。"""
        return self._home / MCP_FILENAME

    def paths(self) -> list[Path]:
        """按优先级返回全部候选文件（项目级在前）。

        Returns:
            存在的文件按项目 → 全局排序；都不存在时仍返回两个候选路径。
        """
        candidates = [self.project_path, self.home_path]
        existing = [path for path in candidates if path.exists()]
        return existing or candidates

    def default_write_path(self) -> Path:
        """``/mcp add`` 的默认写入目标（项目级）。

        Returns:
            项目级配置文件路径。
        """
        return self.project_path

    # ------------------------------------------------------------------ 读取

    def read_document(self, path: Path) -> Any:
        """读取并解析一个 ``mcp.yaml``。

        Args:
            path: 文件路径。

        Returns:
            解析结果；文件不存在 → ``{}``。

        Raises:
            McpValidationError: YAML 不可解析（V1）。
        """
        file_path = Path(path)
        if not file_path.exists():
            return {}
        text = file_path.read_text(encoding="utf-8")
        return parse_mcp_yaml(text, source=file_path.name)

    def entries_of(self, document: Any, *, source: str = MCP_FILENAME) -> list[Mapping[str, Any]]:
        """取出顶层 ``servers`` 列表（§10.3.1 V1 的后半条）。

        Args:
            document: :meth:`read_document` 的结果。
            source: 文件名（进消息文案）。

        Returns:
            ``servers`` 原始条目列表。

        Raises:
            McpValidationError: 顶层不是对象、``servers`` 不是列表，
                或某个条目不是对象。
        """
        name = Path(source).name
        if document is None:
            return []
        if not isinstance(document, Mapping):
            raise McpValidationError(
                detail=f"{name} 顶层必须是 servers 列表（当前是 {type(document).__name__}）",
                rule="V1",
                source=str(source),
            )
        entries = document.get("servers")
        if entries is None:
            return []
        if not isinstance(entries, (list, tuple)):
            raise McpValidationError(
                detail=(
                    f"{name} 顶层必须是 servers 列表"
                    f"（servers 当前是 {type(entries).__name__}）"
                ),
                rule="V1",
                source=str(source),
            )
        out: list[Mapping[str, Any]] = []
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, Mapping):
                raise McpValidationError(
                    detail=(
                        f"{name} 的 servers 第 {index} 项必须是对象"
                        f"（当前是 {type(entry).__name__}）"
                    ),
                    rule="V1",
                    source=str(source),
                )
            out.append(entry)
        return out

    def build(self, entry: Mapping[str, Any], *, source: str = MCP_FILENAME) -> McpServerConfig:
        """把原始条目构造成 :class:`McpServerConfig`。

        Args:
            entry: ``servers`` 里的一个原始条目。
            source: 文件名（进消息文案）。

        Returns:
            配置对象。

        Raises:
            McpValidationError: 结构非法（V2/V3 级问题）。
        """
        name = entry.get("name") if isinstance(entry.get("name"), str) else str(Path(source).name)
        try:
            return McpServerConfig.model_validate(dict(entry))
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            location = ".".join(str(part) for part in first.get("loc", ())) or "配置"
            raise McpValidationError(
                detail=f"服务器 {location} 非法：{first.get('msg', '结构不合法')}",
                rule="V2",
                server=name,
            ) from exc

    def read_file(
        self,
        path: Path,
        *,
        env: Mapping[str, str] | None = None,
        existing_names: Mapping[str, str] | Iterable[str] | None = None,
        existing_servers: Iterable[str] | None = None,
    ) -> tuple[list[McpServerConfig], ValidationReport]:
        """读取一个文件并做**静态**校验（不连接、不握手）。

        Args:
            path: 文件路径。
            env: 覆盖实例默认的环境变量映射。
            existing_names: 已占用工具名（V11）。
            existing_servers: 已存在服务器名（V2 唯一性）。

        Returns:
            ``(可加载的服务器列表, 报告)``。

        Raises:
            McpValidationError: YAML 不可解析或顶层结构非法（V1）。
        """
        document = self.read_document(path)
        entries = self.entries_of(document, source=str(path))
        servers, report = validate_document(
            {"servers": list(entries)},
            source=Path(path).name,
            env=self._env if env is None else env,
            existing_names=existing_names,
            existing_servers=existing_servers,
        )
        return servers, report

    def load_report(
        self,
        *,
        existing_names: Mapping[str, str] | Iterable[str] | None = None,
    ) -> tuple[list[McpServerConfig], ValidationReport]:
        """加载项目级 + 全局配置（启动路径：**只做静态检查**，F-2）。

        同名服务器以**项目级优先**；解析失败的文件只记告警并跳过，绝不阻塞启动。

        Args:
            existing_names: 已占用工具名（内置工具名，V11）。

        Returns:
            ``(服务器列表, 合并报告)``。
        """
        merged: dict[str, McpServerConfig] = {}
        report = ValidationReport()
        for path in (self.project_path, self.home_path):
            if not path.exists():
                continue
            try:
                servers, file_report = self.read_file(
                    path, existing_names=existing_names, existing_servers=set(merged)
                )
            except McpValidationError as exc:
                _LOG.warning("跳过 MCP 配置文件 %s：%s", path, exc.detail)
                report.add_error("V1", exc.detail, hint=f"来源：{path}")
                continue
            report.merge(file_report)
            bad = {issue.server for issue in file_report.errors}
            for server in servers:
                if server.name in bad:
                    _LOG.warning("跳过非法 MCP 服务器 %s（来源 %s）", server.name, path)
                    continue
                merged.setdefault(server.name, server)
        return list(merged.values()), report

    def load(
        self, *, existing_names: Mapping[str, str] | Iterable[str] | None = None
    ) -> list[McpServerConfig]:
        """加载全部可用服务器（忽略校验失败项，只记日志）。

        Args:
            existing_names: 已占用工具名（内置工具名，V11）。

        Returns:
            服务器列表（项目级优先、同名去重）。
        """
        servers, _report = self.load_report(existing_names=existing_names)
        return servers

    # ------------------------------------------------------------------ 展开 / 序列化 / 写入

    def expand(
        self, server: McpServerConfig, *, env: Mapping[str, str] | None = None
    ) -> tuple[McpServerConfig, list[tuple[str, str]]]:
        """展开 ``headers`` / ``env`` 里的 ``${ENV_VAR}``（§10.2）。

        Args:
            server: 服务器配置。
            env: 覆盖实例默认的环境变量映射。

        Returns:
            ``(展开后的配置, [(变量名, 位置), ...])``。
        """
        return expand_server(server, env=self._env if env is None else env)

    def dump(self, servers: Sequence[McpServerConfig]) -> str:
        """把服务器列表序列化为 ``mcp.yaml`` 文本。

        Args:
            servers: 服务器配置列表。

        Returns:
            YAML 文本（UTF-8、保留键序、省略 ``None`` 字段）。
        """
        document = {
            "servers": [server.model_dump(exclude_none=True) for server in servers],
        }
        return yaml.safe_dump(document, allow_unicode=True, sort_keys=False)

    def save(
        self,
        servers: Sequence[McpServerConfig],
        *,
        path: Path | None = None,
        validation: ValidationReport | None = None,
    ) -> Path:
        """写入配置文件（**校验通过才允许落地**，F-3）。

        Args:
            servers: 要写入的服务器列表。
            path: 目标文件（默认项目级 ``<project>/.myagent/mcp.yaml``）。
            validation: 本次 ``/mcp add`` 的校验报告；含 ``errors`` 时**拒绝写入**
                并在抛异常前保持文件不变。

        Returns:
            实际写入的路径。

        Raises:
            McpValidationError: ``validation`` 含错误，或写入失败。
        """
        if validation is not None and validation.errors:
            raise McpValidationError(
                detail=validation.error_text() or "校验未通过，未写入配置",
                rule="V3-写保护",
                path=str(path or self.project_path),
            )
        target = Path(path) if path else self.project_path
        text = self.dump(servers)
        tmp = target.with_name(f"{target.name}.tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(target)
        except OSError as exc:
            raise McpValidationError(
                detail=f"写入 {target} 失败：{exc}", path=str(target)
            ) from exc
        return target

    def merge_servers(
        self, existing: Sequence[McpServerConfig], incoming: Sequence[McpServerConfig]
    ) -> list[McpServerConfig]:
        """把新服务器并入既有列表（同名 **incoming 覆盖** 并保持原位置）。

        Args:
            existing: 既有服务器列表。
            incoming: 新加入的服务器列表。

        Returns:
            合并后的列表。
        """
        merged: list[McpServerConfig] = list(existing)
        index = {server.name: position for position, server in enumerate(merged)}
        for server in incoming:
            position = index.get(server.name)
            if position is None:
                index[server.name] = len(merged)
                merged.append(server)
            else:
                merged[position] = server
        return merged

    def remove_server(
        self, servers: Sequence[McpServerConfig], name: str
    ) -> list[McpServerConfig]:
        """从列表中按名字删除服务器。

        Args:
            servers: 服务器列表。
            name: 服务器名。

        Returns:
            删除后的新列表。
        """
        return [server for server in servers if server.name != name]
