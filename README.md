# my\_agent

本地运行的**编程 CLI Agent**：在终端里与模型对话，自主读写代码、执行命令、调用外部工具，
并在**每一次工具调用前**经过一条四阶段权限流水线（硬拦截 → 前置修正 → 大模型判定 → 后置修正），
保证「可控、可审计、可解释」。

对标 Claude Code 精简版。不是通用聊天机器人，不是 Web 服务，不是多租户系统。

***

> ##  安全警告（务必先读）
>
> **本工具是个人练习与学习项目，并没有沙盒隔离，所以请勿在生产环境运行。** 请勿在开发机、宿主机、任何有重要数据的机器上直接执行。 建议丢进 **Docker / 虚拟机 / 一次性容器** 里再考虑运行。
>
> 它能够读写你的文件、执行任意 shell 命令。请谨慎批准写/执行类操作。
> 项目**不做**沙盒，补偿手段只有两层：
>
> 1. **四阶段权限流水线**（唯一安全闸门）；
> 2. **写前备份 +** **`/undo`**（对应「没有沙盒则必须有回滚」）。
>
> 遇到敏感路径（`.git/`、`.vscode/`、`.bin/`、`.user/`）与不可逆操作（`npm publish`、
> `kill -9`、`rm -rf`、强推历史等）时，**任何模式下都会强制询问**，且不提供「永久放行」。

***

## 安装

需要 Python 3.11+（开发机 3.13.5）与 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
uv run myagent --help
```

若 `uv` 因缓存目录无权限而失败，把缓存指到工作区内：

```bash
# PowerShell
$env:UV_CACHE_DIR="$PWD\.uvcache"; uv sync
```

## 配置模型端点（必需）

**代码中不硬编码任何厂商名与模型名**，端点一律走配置。首次启动若未配置会抛
`ConfigError` 并提示环境变量名。

配置文件优先级（高 → 低，§15.1）：
CLI 参数 → 环境变量 `MYAGENT_*` → `<project>/.myagent/config.toml` → `~/.myagent/config.toml` → 内置默认。

示例 `~/.myagent/config.toml`：

```toml
[llm]
endpoint = "main"
context_window = 200000
max_output_tokens = 8192
temperature = 0.2          # 主对话
judge_temperature = 0.0    # 判定/小任务角色（确定性优先）

[llm.endpoints.main]
base_url = "https://your-endpoint.example.com/v1"  # 任意 OpenAI 兼容端点
api_key_env = "MY_API_KEY"                          # 只存环境变量名，不存密钥
model = "your-model-name"
```

支持任何 OpenAI 兼容端点（自建网关 / vLLM / SGLang / LM Studio / Ollama 等）。
本地端点无需真实密钥，把 `api_key_env` 指向一个填了占位值的环境变量即可。

> ⚠️ **不承诺** Anthropic 原生 `/v1/messages` 与 Gemini 原生协议直连；本方案走
> OpenAI 兼容协议（`base_url` 一项即可切换端点）。如需原生协议，在 `llm/provider.py`
> 内新增一个 `ModelProvider` 实现即可，上层代码无需改动（C7）。

## 使用

```bash
uv run myagent                      # 交互式 REPL
uv run myagent "帮我看下这个报错"     # 一次性提问
uv run myagent --print "..."        # 非交互：输出后退出，遇 ask 一律拒绝（绝不默认放行）
uv run myagent --mode plan          # 指定安全模式
uv run myagent --continue           # 继续最近会话
uv run myagent --resume <session_id>
uv run myagent --cwd <path>         # 指定工作目录
uv run myagent --config <path>
uv run myagent --debug
uv run myagent --assumptions        # 打印全部【待确认假设】及其当前实现口径
```

### 五种安全模式

| 模式                  | 正常行为                          | 模型不可用时                   | 是否调判定模型        |
| ------------------- | ----------------------------- | ------------------------ | -------------- |
| `default`           | 所有工具调用都需要用户确认                 | 全部转 `ask` + 显式提示         | 否（阶段 1 直接 ask） |
| `acceptEdits`       | 工作目录内编辑走确定性快速通道，不调模型；其余交模型    | 全部转 `ask` + 显式提示         | 视参数路径而定        |
| `plan`              | 写类/破坏类 `deny`（附理由），读类 `allow` | 不受影响                     | 否              |
| `dontAsk`           | 调用模型，但只接受 `allow`/`deny`      | 转 `deny`（按模式契约，§6.11 Q1） | 是              |
| `bypassPermissions` | 调用模型，绝大部分 `ask` 直接允许          | 转 `ask`（不自动放行，§6.11 Q2）  | 是              |

> 阶段 0 硬拦截优先于一切模式修正：敏感路径、不可逆操作、`global_ask` 名单在任何模式下
> 都会询问用户，`bypassPermissions` 也不例外。

### `/` 命令

`/help` `/mode` `/clear` `/compact` `/context` `/cost` `/memory` `/skills` `/mcp`
`/hook` `/session` `/run` `/init` `/undo` `/exit`

`/` 命令由外层 `QueryEngine` 处理，**不进入**内层 ReAct 循环。
（例外：`/compact`、`/init` 等本身需要 LLM 的命令会调用模型。）

## 架构

```
L5 交互层   cli.py / ui/*             Typer 命令、rich 渲染、权限问答
L4 编排层   engine/query_engine.py    会话生命周期、/ 命令、压缩与归档
            engine/query.py           内层 ReAct 循环
L3 能力层   security/ context/ memory/ skills/ mcp/ hooks/ agents/ tools/
L2 模型层   llm/provider.py llm/client.py llm/tokenizer.py
L1 基础设施 config/ models/ errors.py engine/session_store.py
```

单向依赖 L5 → L4 → L3 → L2 → L1，禁止反向 import。`security` 是**唯一**拥有
「批准/拒绝工具调用」权力的模块。

三层循环协议见 `CONTRACTS.md`（并行开发的冻结接口契约）。

## 开发

```bash
uv run pytest -q                  # 全部测试
uv run ruff check src tests       # lint
uv run ruff format src tests      # 格式化
```

测试分层（§18.1）：

| 层   | 目录                     | 模型如何被替换                                 |
| --- | ---------------------- | --------------------------------------- |
| 单元  | `tests/unit/`          | 不触网                                     |
| 集成  | `tests/integration/`   | 假 Provider（实现内部协议，**不依赖** **`openai`**） |
| 端到端 | `tests/e2e/`（规划中，尚未落地） | 本地假模型 HTTP 服务                           |

集成与 e2e 全部使用假 Provider 与本地假模型服务，**不消耗真实 token、不需要 API Key**。

## 目录约定

- 项目级运行时目录：`<project>/.myagent/`（含 `memory.md`、`skills/`、`sessions/`、
  `artifacts/`、`archive/`、`backups/`、`audit/`、`todos.json`、`repomap.md`）。
- 全局目录：`~/.myagent/`（可用 `MYAGENT_HOME` 覆盖）。
- 建议把 `.myagent/` 加入项目 `.gitignore`。

## 未决项

项目中 34 条【待确认假设】与 4 条语义冲突**当前按临时决定实现**，并全部集中隔离在
`src/my_agent/config/defaults.py`（含 `PENDING_ASSUMPTIONS` 注册表），便于逐条替换。
运行 `myagent --assumptions` 查看完整清单。
