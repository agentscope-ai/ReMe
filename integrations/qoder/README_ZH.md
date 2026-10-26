# Qoder 的 ReMe 记忆插件

[English](./README.md)

Qoder 支持把 Skills、MCP、Hooks 和配置打包成插件。本目录提供 ReMe 的 Qoder 整包集成：
自动召回、每轮完成的对话采集、MCP 记忆工具，以及 `reme-memory` Skill。

## 安装

```powershell
qodercli plugins marketplace add E:\path\to\ReMe\integrations\qoder
qodercli plugins install reme@reme-marketplace --scope user
```

也可以直接从插件目录安装：

```powershell
qodercli plugins install E:\path\to\ReMe\integrations\qoder\reme --scope user
```

安装前可运行：

```powershell
qodercli plugins validate E:\path\to\ReMe\integrations\qoder\reme --strict
```

## 配置

用户配置位于 `%USERPROFILE%\.qoder\reme\config.json`，也可以通过 `QODER_CONFIG_DIR` 改配置根。
默认拆分服务为：

- MCP：`http://127.0.0.1:2333/mcp`
- Hook HTTP Job API：`http://127.0.0.1:2334`

把 `memory_interval` 设为 `1` 可在每轮结束后立即写入。`auto_recall` 控制提醒前召回，
`auto_memory` 控制完成轮次采集；超时与上下文上限和 Claude Code 集成一致。

## 工作方式

`UserPromptSubmit` 同步召回并注入不受信任的历史证据；`Stop` 先同步捕获本轮 user/最终
assistant 文本到本地队列，再由并行异步 Hook 调用 `POST {api_url}/auto_memory`。
`SessionStart` 重试残留队列，`SessionEnd` 做短时排空。

队列按服务地址隔离，带进程锁和 `.done` 回执；工具输出、推理、子代理、注入的
`<reme-context>` 都会被排除。Qoder 使用 Claude 风格 JSONL transcript，Hook 支持等到最后一行
完整落盘后再捕获。

## 从手工 Hook 迁移

安装整包后，应从 `settings.json` 删除手工添加的 ReMe `UserPromptSubmit`、`Stop`、
`SessionStart`、`SessionEnd` Hook 和手工 `mcpServers.reme`。否则同一轮会被重复采集。
Qoder、Orca、记忆库门禁等其他 Hook 不要删除。

## 验收

```powershell
qodercli plugins list
qodercli mcp list
```

新开一个 Qoder 会话后，应同时看到 `%USERPROFILE%\.qoder\reme\hooks.log` 中的
`memory_saved`、队列 `.done` 回执，以及新的 `session/dialog/qoder-*.jsonl` user/assistant 两行。
`auto_dream` 仍由 ReMe 服务自身调度，插件不会额外启动调度器。
