# Codex 的 ReMe 长期记忆插件

[English](./README.md)

先启动 ReMe 服务，再在 Codex 中安装 ReMe 插件，即可使用长期记忆。这是本宿主支持的集成方式。
插件统一安装自动召回、完成轮次记录、MCP 工具和内置 `reme-memory` Skill；ReMe 服务负责工作区中的记忆存储与整理。

插件与本地 marketplace 维护在 `integrations/codex/`，从仓库安装。
使用插件期间，保持 ReMe 服务运行。

```text
启动 ReMe 服务 → 安装 ReMe 插件 → 在 Codex 中使用记忆
```

## 环境要求

- 宿主环境中有 Python 3.11+，可通过 `python3` 执行，并已安装 FastMCP >= 3.4.2（ReMe 自带此依赖）。
- Codex CLI 0.145.0 或兼容版本，支持插件、`UserPromptSubmit`、`Stop`、同步命令 Hook 和 `last_assistant_message`。
- 原生配置界面需要支持 `openai/settings` MCP 扩展的图形宿主；CLI 0.145.0 支持 Hook 和工具，但不渲染该界面。
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

添加仓库内的本地 marketplace，再安装插件：

```bash
codex plugin marketplace add /absolute/path/to/ReMe/integrations/codex
codex plugin add reme@reme-codex
```

启动 Codex，在 `/hooks` 中检查并信任 ReMe 的四个 Hook（`SessionStart`、`SessionEnd`、
`UserPromptSubmit`、`Stop`），然后新建会话。
插件安装会一并加载 Hook、MCP 连接和内置 Skill。更新后重新安装插件，并审核有变化的 Hook。

本插件使用受支持的 `.codex-plugin/plugin.json` 与 `.mcp.json` 布局。
在 Codex 0.159.2 和桌面运行时 0.158.0-alpha.2.1 中，根目录 `plugin.json` 会选择 portable
加载器，导致本插件的 Hook 被遗漏，显式声明 Hook 路径也不能解决。确认 portable 加载器能通过
原生 Hook 发现验证前，保留 Codex 布局。

## 3. 在 Codex 中使用记忆

在新会话中询问“检查 ReMe 是否可用”，由已安装插件的工具检查服务连接；也可通过 `/mcp` 查看插件连接状态。
之后正常使用 Codex：插件在消息提交前自动召回相关记忆，每完成五轮对话分批记录一次。
需要主动查询时，可以询问“之前关于 Juniper 项目做了什么决定？请附上记忆来源”。

自动记录会把完成的用户/助手文本发送到配置的 ReMe 服务；仅需召回时，将 `auto_memory` 设为 `false`。

## 通过 MCP 设置界面配置

在支持 `openai/settings` 的宿主中，打开已安装的 ReMe MCP Server 设置。
原生表单提供服务地址、自动召回与记录开关、批次大小、召回限制、超时和时区。
保存后点击 **Check connection**，会在已保存地址调用 `health_check`，不发送对话内容。
ReMe 服务离线时仍可读取和修改设置，因此可以在此修正错误地址。

配置原子保存到 `${CODEX_HOME:-$HOME/.codex}/reme/config.json`，位于插件缓存之外，升级后保留。
下一次 MCP 工具调用和 Hook 调用直接使用新配置，无需重新连接或重装。
正在执行的调用继续使用原来的地址与配置。关闭 `auto_memory` 会暂停自动捕获与队列重试，显式工具仍可使用。

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `mcp_url` | `http://127.0.0.1:2333/mcp` | 显式工具和自动 Hook 共用的 MCP 地址，支持自定义路径 |
| `auto_recall` | `true` | 用户提交消息时自动检索 |
| `auto_memory` | `true` | 自动捕获与提交；关闭后也不重试已有队列 |
| `recall_limit` | `5` | 检索结果上限 |
| `recall_min_score` | `0` | 自动召回最低分数 |
| `recall_timeout` | `5` | 前台 MCP 超时秒数，最大 10 |
| `request_timeout` | `600` | 写入与 MCP 请求超时秒数，最大 600 |
| `memory_interval` | `5` | 每批完成的对话轮数；设为 1 则逐轮提交 |
| `shutdown_timeout` | `2` | 退出时尽力刷新队列的总秒数，最大 2 |
| `context_max_chars` | `8000` | 召回正文字符上限 |
| `timezone` | `Asia/Shanghai` | 批次按此时区分日，应与 ReMe workspace 一致 |

JSON 配置仍由用户掌控，也可直接编辑，只需写入要覆盖的选项。
完整示例见 [config.example.json](plugins/reme/config.example.json)；从文件移除某项即可恢复其默认值。
未知字段和无效值会使校验失败，Hook 跳过并记录 `hook_failed`，不会回退到其他服务。
界面保存无效值时会返回错误，保留原配置文件。

已有配置仍可读取。旧 `api_url` 字段不再参与请求，下次在界面保存时会移除；所有操作统一使用 `mcp_url`。
标准 `/mcp` 地址继续使用原来的队列和收据标识。自定义旧 `api_url` 下的待提交记录原样保留，
不会自动迁移目标地址。切换地址不会把旧对话发往新服务；切回原地址可继续重试该地址的队列。

宿主管理的 MCP Server 和短生命周期 Hook 都依赖宿主 `python3` 环境中的 FastMCP。
如果 ReMe 运行在其他环境，请执行 `python3 -m pip install "fastmcp>=3.4.2"`。插件不会启动 ReMe 服务。
原生表单遵循固定版本的
[OpenAI MCP Extensions 设置协议](https://github.com/openai/mcp-extensions/blob/node-v0.1.0/docs/spec.md#structured-settings)。
不支持该扩展的宿主仍可使用工具和 Hook，但不会显示原生设置表单。

## 生命周期与失败行为

- `UserPromptSubmit` 在短超时范围内同步调用 `search`，用 `<reme-context>` 包裹有长度限制的结果，明确其为不可信历史数据。失败不阻止用户消息。
- 同步 `Stop` Hook 从本地 transcript 提取完成轮次的用户与最终助手文本，通过 MCP 分批调用 `auto_memory`。工具输出、推理内容和注入的 ReMe 上下文不作为对话来源。服务端无需读取宿主文件。
- 使用带宿主命名空间的稳定会话和消息 ID，重复 Stop 去重；每个会话串行提交，跨日拆分批次。排除子 Agent 事件和 Hook 续写。插件不启动独立守护进程，也不自动下载依赖或启动 ReMe。
- 队列与确认收据保存在宿主的 `reme/queue/`，按服务地址隔离；失败或超时不会丢弃队列。`SessionStart` 同步重试，`SessionEnd` 在短时间预算内同步尝试刷新。
- 退出刷新是尽力而为：宿主在捕获前被强制结束时可能丢失该轮。MCP 确认丢失时可能重复提取；稳定消息 ID 避免重复存储来源，但记忆提取是至少一次语义。未满批次或未及时完成的任务留待下次会话处理。
- 队列包含原始对话文本，请在交付完成前保留。`reme/hooks.log` 仅记录事件及异常类型。关闭 `auto_memory` 保留队列且停止重试。
- `auto_dream` 整理继续由 ReMe 服务管理。与 OpenClaw 常驻 Gateway 不同，短生命周期 Hook 不另建定时调度器。

Codex 0.145.0 会跳过设置了 `async` 的 hook，因此本适配器使用同步 hook。达到批次阈值时的写入和启动时的重试，最多会等待 `request_timeout` 秒；Claude Code 则使用其独立的原生异步适配器。

## 验证与排查

1. 在 `/hooks` 中确认 ReMe 的四个 Hook 均已加载、启用且信任，再在 MCP 设置中点击
   **Check connection**。健康结果只验证服务连接，不代表 Hook 已加载或执行。
2. 快速验证时把 `memory_interval` 设为 `1`，新建会话，要求记住一条合成事实，例如“Juniper 项目每周四评审”。
3. 等待写入，检查 `reme/hooks.log` 的 `memory_saved`，以及 ReMe workspace 的 `daily/` 笔记。
4. 再新建一个独立会话，询问该事实，并确认回答附有 ReMe 来源路径。

自动记忆和显式检索、阅读、健康检查均由已安装的插件提供。工具缺失时，检查插件是否已安装并启用、
ReMe 服务是否运行，以及 `service.jobs` 是否开放所需 Job。`/hooks` 中没有 ReMe 时，重新安装当前插件并
重启宿主；已列出但待审核时，信任当前 Hook 定义。已信任的 Hook 执行失败时，检查 Hook 环境中的
`python3` 是否为 Python 3.11+，并已安装 FastMCP。

记录要求 Hook 提供 `transcript_path` 和 `last_assistant_message`，读取 rollout 的 `event_msg`：
旧版 `user_message` / 最终 `agent_message`，以及新版 `item_completed` 中的
`UserMessage` / 最终 `AgentMessage` 文本块。排除 `response_item` 中的注入上下文和重复消息，不读取私有推理。
Codex transcript 不是稳定公开接口；未知格式或缺失文件会跳过并记录状态，不猜测对话。
召回正常但未记录时，检查 `capture_skipped`、`hook_failed`。

在 Codex 0.145.0 中，无人值守的 `codex exec` 无法收集工具审批时，MCP 调用可能返回 `user cancelled MCP tool call`。请在交互会话中批准具体调用来验证显式工具。已信任的自动 Hook 自身作为 MCP 客户端调用服务，不逐次请求模型工具审批。

## 源码与验证

在 ReMe 仓库运行 `pytest tests/unit/test_coding_agent_plugins.py tests/unit/test_codex_plugin_mcp.py -v`。测试使用隔离目录和模拟服务。
使用支持 `hooks/list` 的运行时，执行
`REME_CODEX_BIN=/path/to/codex pytest tests/integration/test_codex_plugin_install.py -v`，
可在隔离配置目录安装插件，验证信任前的 Hook 发现行为，不调用模型或执行 Hook。
宿主行为以 [Codex Hook 文档](https://learn.chatgpt.com/docs/hooks) 为准。
ReMe 在 [agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe) 开发，采用 Apache-2.0 许可证。
