# Codex 的 ReMe 长期记忆插件

[English](./README.md)

本插件在用户提交消息前自动召回记忆，在完成对话后分批记录，同时提供显式 MCP 查询工具。
长期记忆保存在用户拥有的 ReMe workspace。插件与安装文档独立维护在
`integrations/codex/`，可直接从仓库安装。

```text
UserPromptSubmit → search → 不可信历史上下文 → 助手回答
Stop → 完成的用户/助手文本 → 本地重试队列 → auto_memory → daily Markdown
```

## 环境要求

- 宿主环境中有 Python 3.11+，可通过 `python3` 执行；插件仅使用标准库。
- Codex CLI 0.145.0 或兼容版本，支持插件、`UserPromptSubmit`、`Stop`、同步命令 Hook 和 `last_assistant_message`。
- 已启动 ReMe HTTP 服务，开放 `search`、`auto_memory`、`health_check`，并启用 MCP。
- ReMe 服务端有可用于自动记忆提取的模型配置。默认 BM25 召回不要求 Embedding。

## 启动 ReMe

```bash
pip install "reme-ai[core]"
reme start workspace_dir=/absolute/path/to/workspace service.backend=http
```

默认地址为 `http://127.0.0.1:2333`，MCP 路径为 `/mcp`。HTTP 服务没有内建 API Key 鉴权，
应保持 loopback 监听或部署受保护的代理。不同 workspace 隔离记忆；不同宿主使用同一 workspace 时会共享召回。

## 安装插件

添加仓库内的本地 marketplace，再安装插件：

```bash
codex plugin marketplace add /absolute/path/to/ReMe/integrations/codex
codex plugin add reme@reme-codex
```

Codex App 也可在插件目录中选择该本地 marketplace。安装后新建线程。
在 CLI 的 `/hooks` 中检查并信任插件 Hook；仅启用插件不会自动信任 Hook。通过 `/mcp` 确认 ReMe 连接。
修改源码后刷新本地 marketplace 并重新安装，新建线程，重新审核有变化的 Hook。

自动记录会把完成的用户/助手文本发送到配置的 ReMe 服务；仅需召回时，将 `auto_memory` 设为 `false`。

## 配置

可选配置位于 `${CODEX_HOME:-$HOME/.codex}/reme/config.json`。从仓库根目录复制示例：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/reme"
cp integrations/codex/plugins/reme/config.example.json "${CODEX_HOME:-$HOME/.codex}/reme/config.json"
```

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `auto_recall` | `true` | 用户提交消息时自动检索 |
| `auto_memory` | `true` | 自动捕获与提交；关闭后也不重试已有队列 |
| `recall_limit` | `5` | 检索结果上限 |
| `recall_min_score` | `0` | 自动召回最低分数 |
| `recall_timeout` | `5` | 前台 HTTP 超时秒数，最大 10 |
| `request_timeout` | `600` | 写入 HTTP 超时秒数，最大 600 |
| `memory_interval` | `5` | 每批完成的对话轮数；设为 1 则逐轮提交 |
| `shutdown_timeout` | `2` | 退出时尽力刷新队列的总秒数，最大 2 |
| `context_max_chars` | `8000` | 召回正文字符上限 |
| `timezone` | `Asia/Shanghai` | 批次按此时区分日，应与 ReMe workspace 一致 |

未知字段或无效值会使配置校验失败；配置修改在下一次 Hook 调用生效。
插件 `.mcp.json` 中的 `reme.url` 是 MCP 和 Hook 共用的唯一服务地址，必须以 `/mcp` 结尾。
自定义端口或代理时，在安装前修改源码中的此文件，再刷新或重新安装插件，不要修改版本化安装缓存。
不再提供单独的 `REME_HOST` / `REME_PORT` Hook 地址覆盖。

## 生命周期与失败行为

- `UserPromptSubmit` 在短超时范围内同步调用 `search`，用 `<reme-context>` 包裹有长度限制的结果，明确其为不可信历史数据。失败不阻止用户消息。
- 同步 `Stop` Hook 从本地 transcript 提取完成轮次的用户与最终助手文本，通过 JSON Job API 分批调用 `auto_memory`。工具输出、推理内容和注入的 ReMe 上下文不作为对话来源。服务端无需读取宿主文件。
- 使用带宿主命名空间的稳定会话和消息 ID，重复 Stop 去重；每个会话串行提交，跨日拆分批次。排除子 Agent 事件和 Hook 续写。插件不启动独立守护进程，也不自动下载依赖或启动 ReMe。
- 队列与确认收据保存在宿主的 `reme/queue/`，按服务地址隔离；失败或超时不会丢弃队列。`SessionStart` 同步重试，`SessionEnd` 在短时间预算内同步尝试刷新。
- 退出刷新是尽力而为：宿主在捕获前被强制结束时可能丢失该轮。HTTP 确认丢失时可能重复提取；稳定消息 ID 避免重复存储来源，但记忆提取是至少一次语义。未满批次或未及时完成的任务留待下次会话处理。
- 队列包含原始对话文本，请在交付完成前保留。`reme/hooks.log` 仅记录事件及异常类型。关闭 `auto_memory` 保留队列且停止重试。
- `auto_dream` 整理继续由 ReMe 服务管理。与 OpenClaw 常驻 Gateway 不同，短生命周期 Hook 不另建定时调度器。

Codex 0.145.0 会跳过设置了 `async` 的 hook，因此本适配器使用同步 hook。达到批次阈值时的写入和启动时的重试，最多会等待 `request_timeout` 秒；Claude Code 则使用其独立的原生异步适配器。

## 验证与排查

1. 对配置的服务运行 `reme health_check`，通过 Codex `/mcp` 确认连接。
2. 快速验证时把 `memory_interval` 设为 `1`，新建会话，要求记住一条合成事实，例如“Juniper 项目每周四评审”。
3. 等待写入，检查 `reme/hooks.log` 的 `memory_saved`，以及 ReMe workspace 的 `daily/` 笔记。
4. 再新建一个独立会话，询问该事实，并确认回答附有 ReMe 来源路径。

`reme-memory` Skill 支持显式检索、阅读和健康检查。MCP 工具缺失也可能源于插件未启用或
`service.jobs` 限制，不能直接判断服务未启动。旧宿主不支持 Hook 时，可继续使用 MCP 和 Skill 手动查询。

记录要求 Hook 提供 `transcript_path` 和 `last_assistant_message`，读取 rollout 的 `event_msg`
（`user_message` 与最终 `agent_message`），不读取私有推理。
Codex transcript 不是稳定公开接口；未知格式或缺失文件会跳过并记录状态，不猜测对话。
召回正常但未记录时，检查 `capture_skipped`、`hook_failed`。

在 Codex 0.145.0 中，无人值守的 `codex exec` 无法收集工具审批时，MCP 调用可能返回 `user cancelled MCP tool call`。请在交互会话中批准具体调用来验证显式工具。自动 Hook 使用 Job API，不依赖 MCP 工具审批。

## 源码与验证

在 ReMe 仓库运行 `pytest tests/unit/test_coding_agent_plugins.py -v`。测试使用隔离目录和模拟服务。
宿主行为以 [Codex Hook 文档](https://learn.chatgpt.com/docs/hooks) 为准。
ReMe 在 [agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe) 开发，采用 Apache-2.0 许可证。
