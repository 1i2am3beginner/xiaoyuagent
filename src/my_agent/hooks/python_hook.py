"""``type: python`` 的本地函数 hook 加载器（§11.2）。

> 规格原文：``type: command``（子进程）| ``python``（本地函数，**仅内置测试用**）。
> 因此本模块是**测试/内置**通道：它让 ``hooks.yaml`` 能直接指向一个 Python 可调用体，
> 免去为每个测试写一个脚本文件。

支持的 ``module`` 写法：

* 点分模块路径：``"my_hooks.deny_write:handler"``（模块会被导入）
* 文件路径：``"hooks/my_check.py:handler"`` / ``"hooks/my_check.py"``
  （相对 ``cwd`` 或 ``project_root`` 解析；省略 ``:func`` 时默认取 ``hook``）

任何无法解析的情况一律抛 :class:`~my_agent.errors.HookError`（带清晰指引），
由 :class:`~my_agent.hooks.runner.HookRunner` 按 ``on_error`` 策略处理。
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from my_agent.errors import HookError

__all__ = ["DEFAULT_FUNCTION", "load_callable", "run_python_hook"]

#: ``module`` 未写 ``:func`` 时使用的默认函数名。
DEFAULT_FUNCTION = "hook"

#: 已加载模块缓存（键 = 绝对文件路径），避免重复导入触发重复副作用。
_MODULE_CACHE: dict[str, Any] = {}


def _split_target(target: str) -> tuple[str, str]:
    """把 ``"mod:func"`` 拆成 ``(module, function)``。

    Args:
        target: ``HookSpec.module`` 的原始值。

    Returns:
        ``(module_part, function_name)``；未写 ``:`` 时函数名取 :data:`DEFAULT_FUNCTION`。

    Raises:
        HookError: ``target`` 为空，或 ``:`` 两侧有空串。
    """
    text = target.strip()
    if not text:
        raise HookError(
            "python hook 的 module 为空；请写成 'pkg.mod:func' 或 'hooks/x.py:func'",
            detail="module 为空",
        )
    module_part, sep, func = text.partition(":")
    module_part = module_part.strip()
    func = func.strip() or DEFAULT_FUNCTION
    if not module_part or (sep and not func):
        raise HookError(
            f"python hook 目标 {target!r} 非法；应形如 'pkg.mod:func' 或 'hooks/x.py:func'",
            detail=f"非法目标 {target!r}",
        )
    return module_part, func


def _load_from_file(
    path: Path, func_name: str, *, origin: str, cwd: Path, project_root: Path | None
) -> Callable[..., Any]:
    """从 ``.py`` 文件加载可调用体。

    Args:
        path: 文件路径（可能不存在）。
        func_name: 函数名。
        origin: 日志/报错用的原始 ``module`` 串。
        cwd: 相对路径基准。
        project_root: 相对路径的第二基准。

    Returns:
        解析出的可调用体。

    Raises:
        HookError: 文件不存在、导入失败或函数缺失/不可调用。
    """
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(cwd / path)
        if project_root is not None:
            candidates.append(project_root / path)
    resolved = next((item for item in candidates if item.is_file()), None)
    if resolved is None:
        searched = "、".join(str(item) for item in candidates)
        raise HookError(
            f"python hook {origin!r} 指向的文件不存在（已查找：{searched}）",
            detail=f"文件不存在：{origin}",
        )

    key = str(resolved.resolve())
    module = _MODULE_CACHE.get(key)
    if module is None:
        spec = importlib.util.spec_from_file_location(f"_myagent_hook_{resolved.stem}", resolved)
        if spec is None or spec.loader is None:
            raise HookError(
                f"python hook {origin!r} 无法作为模块导入（{resolved}）",
                detail=f"无法导入 {resolved}",
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except (ImportError, OSError, SyntaxError) as exc:
            sys.modules.pop(spec.name, None)
            raise HookError(
                f"python hook {origin!r} 导入失败：{exc}",
                detail=f"导入失败：{exc}",
            ) from exc
        _MODULE_CACHE[key] = module
    return _extract_callable(module, func_name, origin=origin)


def _load_from_dotted(
    module_name: str, func_name: str, *, origin: str, project_root: Path | None
) -> Callable[..., Any]:
    """按点分模块路径加载可调用体。

    Args:
        module_name: 例如 ``"my_hooks.check"``。
        func_name: 函数名。
        origin: 日志/报错用的原始 ``module`` 串。
        project_root: 导入前临时加入 ``sys.path`` 的目录（存在时）。

    Returns:
        解析出的可调用体。

    Raises:
        HookError: 模块不可导入，或函数缺失/不可调用。
    """
    added = False
    if project_root is not None and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        added = True
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise HookError(
            f"python hook 模块 {module_name!r} 不可导入：{exc}（module={origin!r}）",
            detail=f"模块不可导入：{module_name}",
        ) from exc
    finally:
        if added:
            try:
                sys.path.remove(str(project_root))
            except ValueError:  # pragma: no cover - 并发移除时的兜底
                pass
    return _extract_callable(module, func_name, origin=origin)


def _extract_callable(module: Any, func_name: str, *, origin: str) -> Callable[..., Any]:
    """从模块对象取出可调用体。

    Args:
        module: 已导入的模块。
        func_name: 函数名。
        origin: 日志/报错用的原始 ``module`` 串。

    Returns:
        可调用体。

    Raises:
        HookError: 属性不存在或不可调用。
    """
    func = getattr(module, func_name, None)
    if func is None:
        raise HookError(
            f"python hook {origin!r} 中找不到函数 {func_name!r}",
            detail=f"缺少函数 {func_name}",
        )
    if not callable(func):
        raise HookError(
            f"python hook {origin!r} 的 {func_name!r} 不可调用（{type(func).__name__}）",
            detail=f"{func_name} 不可调用",
        )
    return func


def load_callable(
    target: str, *, cwd: Path, project_root: Path | None = None
) -> Callable[..., Any]:
    """解析 ``HookSpec.module`` 为可调用体。

    Args:
        target: ``"pkg.mod:func"`` 或 ``"hooks/x.py:func"``。
        cwd: 相对文件路径的基准目录。
        project_root: 点分模块导入的相对基准 / 文件路径的第二基准。

    Returns:
        解析出的可调用体。

    Raises:
        HookError: 目标非法、文件/模块不存在、函数缺失或不可调用。
    """
    module_part, func_name = _split_target(target)
    path = Path(module_part)
    looks_like_path = (
        path.suffix == ".py" or "/" in module_part or "\\" in module_part or path.is_absolute()
    ) or (cwd / module_part).is_file()
    if looks_like_path:
        return _load_from_file(
            path, func_name, origin=target, cwd=cwd, project_root=project_root
        )
    return _load_from_dotted(module_part, func_name, origin=target, project_root=project_root)


async def run_python_hook(
    target: str,
    data: dict[str, Any],
    *,
    cwd: Path,
    project_root: Path | None = None,
    timeout_s: float,
) -> Any:
    """加载并执行一个 ``type: python`` hook。

    兼容同步与异步 hook：异步定义会被 ``await``；同步定义放入线程池执行，避免阻塞事件循环。

    Args:
        target: ``HookSpec.module``。
        data: 完整 stdin JSON 结构（``event`` / ``session_id`` / ``turn_index`` / ``cwd`` /
            ``payload``）。
        cwd: 相对文件路径基准。
        project_root: 点分模块导入基准。
        timeout_s: 单次执行超时（秒）。

    Returns:
        hook 的返回值（``None`` / ``dict`` / :class:`~my_agent.hooks.events.HookOutcome`）。

    Raises:
        HookError: 目标无法解析，或执行超时。
        Exception: hook 自身抛出的异常原样向上传播（由 runner 按 ``on_error`` 处理）。
    """
    func = load_callable(target, cwd=cwd, project_root=project_root)
    kwargs: dict[str, Any] = {
        "event": data.get("event", ""),
        "session_id": data.get("session_id", ""),
        "turn_index": data.get("turn_index", 0),
        "cwd": data.get("cwd", str(cwd)),
        "payload": data.get("payload", {}),
    }
    if inspect.iscoroutinefunction(func):
        return await asyncio.wait_for(func(**kwargs), timeout=timeout_s)
    # 同步 hook：放线程池执行，超时后主事件循环立即返回（线程无法强杀，仅记录超时）。
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, lambda: func(**kwargs)), timeout=timeout_s
    )
