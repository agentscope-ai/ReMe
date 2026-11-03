# ZCode 的 ReMe 记忆插件

[English](./README.md)

ZCode 支持把 Skills、MCP 和生命周期 Hook 打包成插件。本目录提供 ReMe 的 ZCode 整包集成：
自动召回、完成轮次记录、MCP 记忆工具，以及 `reme-memory` Skill。

## 环境要求

- 支持插件和 Hook 的当前 ZCode 桌面端。
- 启动 ZCode 的环境中有 Python 3.11+，命令名为 `python3`，并安装 FastMCP 3.1+。
- ReMe 服务正在运行，并暴露 MCP 端点以及 `search`、`auto_memory` 两个 HTTP Job。

默认拆分服务为：

- MCP：`http://127.0.0.1:2333/mcp`
- Hook HTTP Job API：`http://127.0.0.1:2334`

## 安装

1. 打开 **插件市场 → 添加 → 添加插件市场**。
2. 添加本仓库的绝对目录：`E:\path\to\ReMe\integrations\zcode`。
3. 打开 **个人 → ReMe 记忆 → 安装**。
4. 安装后新建一个 ZCode 任务。旧任务可能继续使用安装前的组件清单。

市场清单是 `integrations/zcode/marketplace.json`，可安装插件源码位于
`integrations/zcode/reme`。

## 配置

插件读取 `%USERPROFILE%\.zcode\reme\config.json`；设置 `ZCODE_HOME` 后则读取
`${ZCODE_HOME}\reme\config.json`：

```powershell
New-Item -ItemType Directory -Force "$HOME\.zcode\reme"
Copy-Item integrations\zcode\reme\config.example.json "$HOME\.zcode\reme\config.json"
```

`mcp_url` 控制 MCP 工具连接，`api_url` 控制 Hook 使用的 HTTP Job API。把
`memory_interval` 设为 `1` 可在每轮结束后立即写入。`auto_recall` 控制提醒前召回，
`auto_memory` 控制完成轮次采集；超时、上下文上限和时区配置与 Claude Code、Qoder 集成一致。

未知字段和非法取值会显式失败。关闭 `auto_memory` 会保留待提交队列并停止自动捕获与重试。
修改服务地址后，旧队列不会发送到新服务。

## 工作方式

ZCode 仅支持 `SessionStart`、`UserPromptSubmit`、`PreToolUse`、`PermissionRequest`、
`PostToolUse`、`PostToolUseFailure` 和 `Stop`。当前 `async` 字段没有后台执行效果，因此插件
不依赖它。

- `UserPromptSubmit` 按 `session_id` 与 `turn_id` 暂存用户文本，再同步执行有界召回。
- `Stop` 将暂存用户文本与 ZCode 的最终 assistant 回复配成一轮，先写本地队列，再启动脱离宿主
  的后台进程提交批次，使 Hook 不必等待模型抽取完成。
- `SessionStart` 启动一次强制排空，重试以前会话留下的批次。

ZCode 的 Stop transcript 只包含最终 assistant 文本，因此插件不依赖该临时文件。工具事件、
子代理、推理内容和注入的 `<reme-context>` 都不会作为对话源文本。

队列按服务地址隔离，并用每会话文件锁串行化。`.done` 回执保证重复 Stop 不会重复提交；失败或
超时的写入会留在队列中，等待下一次 Stop 或 SessionStart。

## 从手工 Hook 迁移

如果曾直接在 `%USERPROFILE%\.zcode\cli\config.json` 中配置 ReMe，安装整包后应删除
`hooks.events` 里的 ReMe 条目和手工 `reme` MCP 服务。ZCode、Orca 和其他无关 Hook 不要删除。
两套安装同时运行会让同一轮被采集两次。

## 验收

把 `memory_interval` 设为 `1`，新建 ZCode 任务并让它记住一个合成事实。成功时应满足：

1. `%USERPROFILE%\.zcode\reme\hooks.log` 中先出现 `prompt_cached`，再出现 `memory_saved`；
2. 对应队列中存在 `.done` 回执；
3. `session/dialog/zcode-*.jsonl` 新增一条 user 和最终 assistant 记录；
4. 稍后新任务能够召回该事实并引用来源路径。

在 **设置 → 插件** 确认插件已安装，在 **设置 → MCP** 确认 `reme` 服务已连接。插件已启用但
组件缺失时，先新建任务再检查文件。

`auto_dream` 仍由 ReMe 服务自身调度，插件不会额外启动调度器。
