# ReMe 记忆插件 for ZCode

[English](./README.md)

ZCode 支持打包 skill、MCP server 和生命周期 Hook 的插件。本目录把 ReMe 的自动召回、完成轮次记录、
MCP 记忆工具和 `reme-memory` skill 打包成一个 ZCode 插件。

## 环境要求

- 支持插件与 Hook 的 ZCode 桌面版。插件同时接受两种字段写法（`session_id`/`sessionId`、
  `turnId`/`turn_id`），并从 `last_assistant_message` 或 `responseText` 读取最终回复文本。
- 启动 ZCode 的环境里需要 Python 3.11 或更新版本，以及 FastMCP 3.1 或更新版本。
- 一个正在运行的 ReMe 服务，同时暴露 MCP 端点和 HTTP job `search`、`auto_memory`。

Windows、macOS、Linux 均受支持。Hook 脚本只用 Python 标准库，唯一的平台差异是投递进程的脱离方式
（Windows 用 `creationflags`，其他平台用 `start_new_session`）。下面的命令同时给出 Windows PowerShell
和 macOS/Linux 两种写法。

## 1. 准备 ReMe

用 ZCode 能访问到的解释器安装 ReMe 及其 Python 依赖：

macOS / Linux：

```bash
cd /path/to/ReMe
python3 -m pip install -e ".[core]"
```

Windows PowerShell：

```powershell
cd E:\path\to\ReMe
python -m pip install -e ".[core]"
```

自动记忆生成需要模型凭据，按 [Quick Start](../../README.md#environment-variables) 配置
`LLM_API_KEY` 和 `LLM_BASE_URL`。

在另一个终端启动 ReMe 并保持运行。显式指定 workspace，记忆就不依赖你从哪个目录启动：

macOS / Linux：

```bash
reme start workspace_dir="$HOME/reme-workspace"
```

Windows PowerShell：

```powershell
reme start workspace_dir="$HOME\reme-workspace"
```

默认的 `reme start` 在 `http://127.0.0.1:2333` 同时提供 Job API 和 streamable HTTP MCP。分端口部署属于
显式覆盖：把 HTTP Job API 起在另一个端口，并在配置里填写对应的 `api_url`。

## 2. 安装插件

1. 打开 **Plugin Marketplace → Add → Add Plugin Marketplace**。
2. 添加本目录的绝对路径：`E:\path\to\ReMe\integrations\zcode`（或 `/path/to/ReMe/integrations/zcode`）。
3. 打开 **Personal → ReMe Memory → Install**。
4. 安装后新开一个 ZCode 任务，已有任务可能仍持有旧的组件目录。

marketplace 源是 `integrations/zcode/marketplace.json`，可安装的插件源是 `integrations/zcode/reme`。

## 3. 配置

插件读取 `%USERPROFILE%\.zcode\reme\config.json`；设置了 `ZCODE_HOME` 时读取
`${ZCODE_HOME}/reme/config.json`。

macOS / Linux：

```bash
mkdir -p "$HOME/.zcode/reme"
cp integrations/zcode/reme/config.example.json "$HOME/.zcode/reme/config.json"
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force "$HOME\.zcode\reme"
Copy-Item integrations\zcode\reme\config.example.json "$HOME\.zcode\reme\config.json"
```

随附的示例内容：

```json
{
  "mcp_url": "http://127.0.0.1:2333/mcp",
  "api_url": "",
  "memory_interval": 1
}
```

`api_url` 留空时，插件会从以 `/mcp` 结尾的 `mcp_url` 推导出 Job API 地址，也就是默认的单端口部署方式。
只有分端口或反向代理部署才需要填写 `api_url`。示例把 `memory_interval` 设为 `1`，便于每轮立即写入；
代码默认值是 `5`。

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `mcp_url` | `http://127.0.0.1:2333/mcp` | ReMe MCP 端点 |
| `api_url` | `""` | 可选 HTTP Job API 覆盖；留空时由 `mcp_url` 推导 |
| `auto_recall` | `true` | 用户提问前检索 |
| `auto_memory` | `true` | 捕获并提交已完成轮次 |
| `recall_limit` | `5` | 召回条数上限 |
| `recall_min_score` | `0` | 召回最低分 |
| `recall_timeout` | `5` | 前台超时，最多 10 秒 |
| `request_timeout` | `600` | 记忆写入超时，最多 600 秒 |
| `memory_interval` | `5` | 每批提交的完成轮次数 |
| `context_max_chars` | `8000` | 召回上下文最大字符数 |
| `timezone` | `Asia/Shanghai` | 每日批处理时区 |

未知字段和非法值都会校验失败。关闭 `auto_memory` 会保留待投递的队列文件，同时停止自动捕获和重试，
包括已经启动的投递进程。改变端点不会把旧的排队对话发给新服务。改动 `mcp_url` 之后需要重连 MCP：
ZCode 在插件加载时读取 server 设置，请重载插件或新开任务，不要复用已经打开的任务。

## 生命周期

ZCode 支持 `SessionStart`、`UserPromptSubmit`、`PreToolUse`、`PermissionRequest`、`PostToolUse`、
`PostToolUseFailure` 和 `Stop`。

ZCode 的 Hook 文档里有命令级的 `async` 字段，本插件不设置它。`Stop` Hook 本身是同步执行的，因为它必须
读取最终回复并可靠地入队，然后启动一个脱离的投递进程再返回。脱离与 Hook 执行器无关：它在任何版本上都能
让 `Stop` 的前台耗时可控，包括支持异步命令 Hook 的版本；持久队列也仍然能覆盖之后才退出或被杀掉的进程。

- `UserPromptSubmit` 按 `session_id` 和 `turn_id` 缓存用户文本，然后做有上限的召回。召回结果作为不可信的
  历史证据注入。
- `Stop` 把缓存的用户文本与 ZCode 的最终回复合成一条持久队列记录，再启动脱离进程提交队列批次，让 Hook
  不必等待模型抽取就能返回。
- `SessionStart` 启动一次脱离的强制 flush，清掉之前会话残留的批次。

ZCode 的 Stop transcript 只包含最终回复，所以插件不依赖那个临时文件。工具事件、子 agent、reasoning 和注入的
`<reme-context>` 都不会被当作源对话文本。

队列文件按服务端点分命名空间，并用会话级锁保护。`.done` 回执避免重复的 Stop 再次提交已经确认的轮次。
失败或超时的写入会留在队列里，等下一次 Stop 或 SessionStart 再试。如果投递进行中又来了新的投递请求，
当前这一轮会被标记为需要再次扫描，因此投递期间捕获的会话不会被丢掉。

## 从手工 Hook 迁移

如果之前把 ReMe 直接加进了 `%USERPROFILE%\.zcode\cli\config.json`，安装本插件后请删掉 `hooks.events`
里的 ReMe 条目和手工添加的 `reme` MCP server。ZCode、Orca 等无关 Hook 保持不动。两套配置同时存在会把同一轮
对话记录两次。

## 验证

先验证连接，再验证记忆：

1. 在 **Settings → Plugins** 确认插件已安装，**Settings → MCP** 里能看到 `reme` server。
2. 新开一个 ZCode 任务，让它调用 ReMe 的 `health_check` 和 `version` MCP 工具。两者都必须有返回；失败就先修
   `mcp_url` 或服务，不要继续往下测。

然后把 `memory_interval` 设为 `1`，新开任务，让它记住一个合成的事实。成功需要同时满足：

1. `%USERPROFILE%\.zcode\reme\hooks.log` 里出现 `prompt_cached`，随后是 `memory_saved`；
2. 对应的队列 `.done` 回执存在；
3. 新增一个 `session/dialog/zcode-*.jsonl` 文件，含一条 user 和一条最终 assistant 记录；
4. 之后再开一个任务，能召回该事实并给出源路径。

如果插件已启用但组件缺失，先新开任务，再改文件。

`auto_dream` 仍由 ReMe 服务负责，插件不会另起调度器。
