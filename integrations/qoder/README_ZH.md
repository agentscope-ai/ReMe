# Qoder 的 ReMe 记忆插件

[English](./README.md)

Qoder 支持把 Skills、MCP、Hooks 和配置打包成插件。本目录提供 ReMe 的 Qoder 整包集成：
自动召回、完成轮次记录、MCP 记忆工具，以及 `reme-memory` Skill。

## 环境要求

- Qoder CLI 1.1.41 或更新版本，并支持插件。
- Qoder 执行 Hook 和 MCP Server 的环境中可以调用 `python3`，且 Python 版本为 3.11 或更新版本。
- 同一个 Python 环境中安装 FastMCP 3.1 或更新版本；`reme-ai[core]` 会一并安装。
- 单独运行一个 ReMe 服务，提供 MCP endpoint 以及 `search`、`auto_memory` HTTP Job。

## 1. 准备 ReMe

在本仓库中安装 ReMe 和 Python 依赖。请使用 Qoder 能通过 `python3` 调用的同一个解释器。

macOS 或 Linux：

```bash
cd /path/to/ReMe
python3 -m pip install -e ".[core]"
```

Windows PowerShell：

```powershell
cd E:\path\to\ReMe
python -m pip install -e ".[core]"
```

自动生成记忆需要模型凭据。请按照 [快速开始](../../README_ZH.md#环境变量配置) 配置
`LLM_API_KEY` 和 `LLM_BASE_URL`。

在另一个终端启动 ReMe，并保持服务运行：

```bash
reme start
```

默认 HTTP 服务在同一端口提供 Job API 和 streamable HTTP MCP：

- MCP：`http://127.0.0.1:2333/mcp`
- HTTP Job API：从 `mcp_url` 推导为 `http://127.0.0.1:2333`

如果确实使用拆分端口部署，才需要显式设置 `api_url`。

## 2. 安装插件

插件从仓库内的本地 marketplace 发布。安装或升级时需要使用这个 ReMe checkout。

macOS 或 Linux：

```bash
cd /path/to/ReMe
reme_root="$(pwd)"
qodercli plugins validate "$reme_root/integrations/qoder/reme" --strict
qodercli plugins marketplace add "$reme_root/integrations/qoder"
qodercli plugins install reme@reme-marketplace --scope user
```

Windows PowerShell：

```powershell
cd E:\path\to\ReMe
$remeRoot = (Resolve-Path .).Path
qodercli plugins validate "$remeRoot\integrations\qoder\reme" --strict
qodercli plugins marketplace add "$remeRoot\integrations\qoder"
qodercli plugins install reme@reme-marketplace --scope user
```

也可以直接从插件目录安装：

```bash
qodercli plugins install /path/to/ReMe/integrations/qoder/reme --scope user
```

## 3. 配置

用户配置位于 `~/.qoder/reme/config.json`，也可以通过 `QODER_CONFIG_DIR` 改配置根。

macOS 或 Linux：

```bash
mkdir -p "$HOME/.qoder/reme"
cp integrations/qoder/reme/config.example.json "$HOME/.qoder/reme/config.json"
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force "$HOME\.qoder\reme"
Copy-Item .\integrations\qoder\reme\config.example.json "$HOME\.qoder\reme\config.json"
```

关键默认配置如下：

```json
{
  "mcp_url": "http://127.0.0.1:2333/mcp",
  "api_url": "",
  "memory_interval": 1
}
```

当 `mcp_url` 以 `/mcp` 结尾时，空的 `api_url` 会自动推导为同一服务地址。只有拆分端口或
反向代理部署才需要设置 `api_url`。仓库中的示例把 `memory_interval` 设为 `1`，用于每轮立即写入；
代码默认值为 `5`。

| 配置项 | 默认值 | 含义 |
| --- | --- | --- |
| `mcp_url` | `http://127.0.0.1:2333/mcp` | ReMe MCP 地址 |
| `api_url` | `""` | 可选的 HTTP Job API 地址；为空时从 `mcp_url` 推导 |
| `auto_recall` | `true` | 用户提问前自动召回 |
| `auto_memory` | `true` | 采集并提交完成轮次 |
| `recall_limit` | `5` | 召回结果上限 |
| `recall_min_score` | `0` | 最低召回分数 |
| `recall_timeout` | `5` | 前台召回超时，最多 10 秒 |
| `request_timeout` | `600` | 记忆写入超时，最多 600 秒 |
| `memory_interval` | `5` | 多少轮完成后批量写入 |
| `shutdown_timeout` | `2` | 会话结束时尽力排空，最多 2 秒 |
| `context_max_chars` | `8000` | 注入的召回上下文上限 |
| `timezone` | `Asia/Shanghai` | 按天分批使用的时区 |

安装后重启 Qoder，或执行 `/plugins reload`。修改 `mcp_url` 后需要重新连接 MCP；Hook 配置会在
下次调用时生效。

## 工作方式

`UserPromptSubmit` 执行有上限的召回，并作为不受信任的历史证据注入。`Stop` 先同步捕获本轮
user/最终 assistant 文本到本地队列，再由并行异步 Hook 调用 `POST {api_url}/auto_memory`。
`SessionStart` 重试残留队列，`SessionEnd` 做短时排空。队列按服务地址隔离，并带进程锁和
`.done` 回执；稳定的消息 ID 保证重复 Stop 不会重复写入。

Qoder 使用 Claude 风格 JSONL transcript。工具输出、推理、元数据消息、子代理和注入的
`<reme-context>` 都会被排除。当连续两轮最终回复相同时，插件会等待当前轮的新 transcript
行，而不会把上一轮的相同回复当成新轮次。

## 从手工 Hook 迁移

安装整包后，应从 `settings.json` 删除手工添加的 ReMe `UserPromptSubmit`、`Stop`、
`SessionStart`、`SessionEnd` Hook 和手工 `mcpServers.reme`。否则同一轮会被重复采集。
Qoder、Orca、记忆库门禁等其他 Hook 不要删除。

## 验收

首先确认 Qoder 发现了插件资源：

```bash
qodercli plugins list --json
```

`reme` 条目应当已启用，并显示 1 个 Skill、1 个 `reme` MCP Server 和 5 个 Hook 注册项。
`qodercli mcp list` 不会列出插件内置的 MCP Server，因此不能作为连接是否正常的判断依据。

新开一个 Qoder 会话，让 Agent 实际调用 ReMe 的 `health_check` 和 `version` MCP 工具。然后把
`memory_interval` 设为 `1`，完成一轮值得记忆的对话，并同时确认：

1. `~/.qoder/reme/hooks.log` 中出现不包含正文的 `memory_saved`；
2. `~/.qoder/reme/queue/` 下出现对应的 `.done` 回执；
3. 新的 `session/dialog/qoder-*.jsonl` 中包含该轮 user 和最终 assistant 两行。

最后新开第二个 Qoder 会话，询问第一轮中的事实，确认自动召回成功后再认为安装完成。

`auto_dream` 仍由 ReMe 服务自身调度，插件不会额外启动调度器。
