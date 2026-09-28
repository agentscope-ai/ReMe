# Claude Code 的 ReMe 长期记忆插件

[English](./README.md)

先启动 ReMe 服务，再在 Claude Code 中安装 ReMe 插件，即可使用长期记忆。这是本宿主支持的集成方式。
插件统一安装自动召回、完成轮次记录、MCP 工具和内置 `reme-memory` Skill；ReMe 服务负责工作区中的记忆存储与整理。

插件与本地 marketplace 维护在 `integrations/claude_code/`，从仓库安装。
使用插件期间，保持 ReMe 服务运行。

```text
启动 ReMe 服务 → 安装 ReMe 插件 → 在 Claude Code 中使用记忆
```

## 环境要求

- 宿主环境中有 Python 3.11+，可通过 `python3` 执行；插件仅使用标准库。
- 当前 Claude Code 版本支持插件、`UserPromptSubmit`、`Stop`、原生异步命令 Hook 和 `last_assistant_message`。
- 已启动 ReMe HTTP 服务，开放 `search`、`auto_memory`、`health_check`，并启用 MCP。
- ReMe 服务端有可用于自动记忆提取的模型配置。默认 BM25 召回不要求 Embedding。

## 1. 启动 ReMe 服务

```bash
pip install "reme-ai[core]"
reme start workspace_dir=/absolute/path/to/workspace service.backend=http
```

默认地址为 `http://127.0.0.1:2333`，MCP 路径为 `/mcp`。HTTP 服务没有内建 API Key 鉴权，
应保持 loopback 监听或部署受保护的代理。不同 workspace 隔离记忆；不同宿主使用同一 workspace 时会共享召回。

## 2. 安装 ReMe 插件

在 Claude Code 中使用仓库的绝对路径：

```text
/plugin marketplace add /absolute/path/to/ReMe/integrations/claude_code
/plugin install reme@reme-marketplace
```

重启 Claude Code 或重新加载已安装的插件，然后新建会话。插件安装会一并加载 Hook、MCP 连接和内置 Skill。
更新后重新安装或加载插件。

## 3. 在 Claude Code 中使用记忆

在新会话中询问“检查 ReMe 是否可用”，由已安装插件的工具检查服务连接；也可通过 `/mcp` 查看插件连接状态。
之后正常使用 Claude Code：插件在消息提交前自动召回相关记忆，每完成五轮对话分批记录一次。
需要主动查询时，可以询问“之前关于 Juniper 项目做了什么决定？请附上记忆来源”。

自动记录会把完成的用户/助手文本发送到配置的 ReMe 服务；仅需召回时，将 `auto_memory` 设为 `false`。

## 可选配置

使用默认服务地址时，安装插件后即可使用。需要调整下列默认行为时，再创建配置文件。

可选配置位于 `${CLAUDE_CONFIG_DIR:-$HOME/.claude}/reme/config.json`。从仓库根目录复制示例：

```bash
mkdir -p "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/reme"
cp integrations/claude_code/reme/config.example.json "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/reme/config.json"
```

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `auto_recall` | `true` | 用户提交消息时自动检索 |
| `auto_memory` | `true` | 自动捕获与提交；关闭后也不重试已有队列 |
| `recall_limit` | `5` | 检索结果上限 |
| `recall_min_score` | `0` | 自动召回最低分数 |
| `recall_timeout` | `5` | 前台 HTTP 超时秒数，最大 10 |
| `request_timeout` | `600` | 后台 HTTP 超时秒数，最大 600 |
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
- 原生异步 `Stop` Hook 从本地 transcript 提取完成轮次的用户与最终助手文本，通过 JSON Job API 分批调用 `auto_memory`。工具输出、推理内容和注入的 ReMe 上下文不作为对话来源。服务端无需读取宿主文件。
- 使用带宿主命名空间的稳定会话和消息 ID，重复 Stop 去重；每个会话串行提交，跨日拆分批次。排除子 Agent 事件和 Hook 续写。插件不启动独立守护进程，也不自动下载依赖或启动 ReMe。
- 队列与确认收据保存在宿主的 `reme/queue/`，按服务地址隔离；失败或超时不会丢弃队列。`SessionStart` 后台重试，`SessionEnd` 在短时间预算内同步尝试刷新。
- 退出刷新是尽力而为：宿主在捕获前被强制结束时可能丢失该轮。HTTP 确认丢失时可能重复提取；稳定消息 ID 避免重复存储来源，但记忆提取是至少一次语义。未满批次或未及时完成的任务留待下次会话处理。
- 队列包含原始对话文本，请在交付完成前保留。`reme/hooks.log` 仅记录事件及异常类型。关闭 `auto_memory` 保留队列且停止重试。
- `auto_dream` 整理继续由 ReMe 服务管理。与 OpenClaw 常驻 Gateway 不同，短生命周期 Hook 不另建定时调度器。

## 验证与排查

1. 在 Claude Code 中让已安装的 ReMe 插件检查连接，通过 `/mcp` 确认插件连接状态。
2. 快速验证时把 `memory_interval` 设为 `1`，新建会话，要求记住一条合成事实，例如“Juniper 项目每周四评审”。
3. 等待后台写入，检查 `reme/hooks.log` 的 `memory_saved`，以及 ReMe workspace 的 `daily/` 笔记。
4. 再新建一个独立会话，询问该事实，并确认回答附有 ReMe 来源路径。

自动记忆和显式检索、阅读、健康检查均由已安装的插件提供。工具缺失时，检查插件是否已安装并启用、
ReMe 服务是否运行，以及 `service.jobs` 是否开放所需 Job。Hook 无法加载时，请使用满足插件要求的宿主版本，
再重新安装或加载插件。

记录要求 Hook 提供 `transcript_path` 和 `last_assistant_message`，读取标准用户/助手 JSONL，
不读取私有推理；未知格式或缺失文件会跳过并记录状态，不猜测对话。
召回正常但未记录时，检查 `capture_skipped`、`hook_failed`。

## 从旧 Claude Code 插件迁移

0.2 版本用自动召回与客户端完成轮次捕获替换旧的脱离宿主 Stop 进程，新记录调用 `auto_memory`。
服务端 `auto_memory_cc` 仍保留兼容，已有 `session/claude_code/` transcript 和笔记不改写。
新来源使用标准 `{session_dir}/dialog/claude-code-<hash>.jsonl` 与带命名空间的会话 ID。
不回放旧对话；请移除手工安装的重复 Stop Hook。日志从插件目录迁移到宿主 `reme/hooks.log`。

端到端验证时，应保持会话打开，直到出现 `memory_saved`。短暂的 `claude -p` 进程可能在异步 Hook 收到写入确认前结束；已捕获的任务会在下一次会话重试。仅看到 Markdown 笔记不代表队列已经收到确认。

## 源码与验证

在 ReMe 仓库运行 `pytest tests/unit/test_coding_agent_plugins.py -v`。测试使用隔离目录和模拟服务。
宿主行为以 [Claude Code Hook 文档](https://code.claude.com/docs/en/hooks) 为准。
ReMe 在 [agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe) 开发，采用 Apache-2.0 许可证。
