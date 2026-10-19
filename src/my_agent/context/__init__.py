"""上下文管理与压缩（§7 模块 C）。

对外只暴露稳定接口，内部实现分文件：

- :mod:`my_agent.context.budget` —— 有效上下文 / 水位 / 五件套预算与超预算降级（§7.2、§7.6）
- :mod:`my_agent.context.repomap` —— 工作目录摘要（§7.3）
- :mod:`my_agent.context.preview` —— 大工具结果落盘与预览（§7.4）
- :mod:`my_agent.context.pruner` —— 大结果裁剪（§7.5）
- :mod:`my_agent.context.assembler` —— 组装与 ``/context`` 报告（§7.6、§7.7.5）
- :mod:`my_agent.context.compactor` —— 全文压缩（§7.7）
- :mod:`my_agent.context.archive` —— 归档与回读（§7.7.2、§7.7.3）

典型用法（engine）：``RepoMap.load_or_generate`` → ``ContextAssembler.build`` →
（水位 ≥ 80% 时）``Compactor.maybe_compact`` → 工具结果经 ``post_tool_result`` 回填。
"""

from __future__ import annotations

from my_agent.context.archive import ArchiveIndex, ArchiveIndexEntry, ArchiveStore
from my_agent.context.assembler import (
    BuildResult,
    ContextAssembler,
    format_context_report,
)
from my_agent.context.budget import (
    BLOCK_LABELS,
    DROP_ORDER,
    FittedPrompt,
    PromptBudget,
    effective_context,
    fit_to_budget,
    needs_compaction,
    watermark,
)
from my_agent.context.compactor import Compactor
from my_agent.context.preview import artifact_path, head_tail_preview, post_tool_result
from my_agent.context.pruner import prune_large_results
from my_agent.context.repomap import RepoMap

__all__ = [
    "ArchiveIndex",
    "ArchiveIndexEntry",
    "ArchiveStore",
    "BLOCK_LABELS",
    "BuildResult",
    "Compactor",
    "ContextAssembler",
    "DROP_ORDER",
    "FittedPrompt",
    "PromptBudget",
    "RepoMap",
    "artifact_path",
    "effective_context",
    "fit_to_budget",
    "format_context_report",
    "head_tail_preview",
    "needs_compaction",
    "post_tool_result",
    "prune_large_results",
    "watermark",
]
