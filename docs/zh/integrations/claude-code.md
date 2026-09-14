---
title: Claude Code 集成
description: 通过 MCP、reme-memory Skill 和 Stop Hook 将 Claude Code 连接到 ReMe。
---

# Claude Code 集成

ReMe 的 Claude Code 插件负责召回和会话捕获：所有 Claude Code 窗口共享一个 ReMe HTTP MCP 服务，Daily 到 digest 的
整理、watcher 和 dream cron 也只在这个服务中运行一份。

## 能力

- 通过 MCP 使用 `search`、`traverse`、`daily_list`、`frontmatter_read`、`read`、`version`、`health_check` 和
  `auto_memory_cc` 等工具；
- `reme-memory` Skill 区分语义、图关系和状态三种查询，再用 `read` 读取命中内容并保留 workspace-relative 来源路径；
- Skill 可通过 `version` 和 `health_check` 检查共享服务；工具缺失时应提示启动 ReMe，不得猜测历史记忆；
- Stop Hook 只把 Claude Code `session_id` 传给服务端，服务端从本地 transcript 解析会话；
- POSIX 系统上的记录会在脱离 Claude Code 的后台进程中进行，不延迟退出；服务不可用时记录日志并结束。

## 部署模型

插件连接到用户预先启动的共享 HTTP MCP 服务，不为每个 Claude Code 窗口创建 ReMe。这样所有窗口共享一个 workspace、一组 watcher 和一次 dream cron。

## 准备 ReMe

```bash
pip install "reme-ai[core]"
```

在稳定目录配置 LLM 环境，然后启动：

```bash
reme start service.backend=http
```

默认 JSON Job API 和 MCP 地址分别位于同一个 `127.0.0.1:2333` 服务，MCP 路径是 `/mcp`。使用其他端口时，必须同步修改插件的 `.mcp.json`。

默认搜索使用 BM25；只有启用向量检索时才需要 Embedding 配置。

请勿将默认 HTTP 服务直接暴露到不可信网络；跨主机使用时应在反向代理层提供认证和 TLS，并限制可调用 Job。

自动捕获还要求 ReMe 服务进程能读取 Claude Code transcript。它默认在服务端的 `~/.claude/projects`
下查找会话；因此 ReMe 应与 Claude Code 运行在同一主机，或者将 transcript 挂载/同步到服务端，并在启动
ReMe 时用 `CLAUDE_CONFIG_DIR` 指向对应目录。如果服务端无法访问 transcript，远程 MCP 召回等功能仍可使用，但
`auto_memory_cc` 会因没有消息而跳过，不会生成记忆。

## 安装插件

在 Claude Code 中运行：

```text
/plugin marketplace add ./integrations/claude_code
/plugin install reme@reme-marketplace
```

重启 Claude Code，再运行 `/mcp`，确认 `reme` server 和工具已经连接。

## Hook 与路径

- MCP 配置：`integrations/claude_code/reme/.mcp.json`；
- 自动记忆 Hook：`integrations/claude_code/reme/hooks/auto_memory.py`；
- Hook 日志：`integrations/claude_code/reme/logs/auto_memory_hook.log`；
- 默认 transcript 根目录：`~/.claude/projects`；
- 启动 ReMe 服务时，可通过 `CLAUDE_CONFIG_DIR` 修改服务端的 transcript 根目录；
- 可通过 `REME_HOST`、`REME_PORT` 覆盖 Hook 使用的服务地址。

Hook 需要 `python3` 位于 `PATH`。MCP 工具名前缀可能随 Claude Code 版本包含 server segment；Skill 使用 `mcp__reme__*` 匹配这一差异。

Hook 是 best-effort 的：它不会因记忆服务故障而阻塞 Claude Code。服务端会保留 transcript 处理进度；同一会话重复触发 Stop 且没有新消息时，
`auto_memory_cc` 会跳过重复的记忆生成和打标。

## 验证

1. `reme health_check` 返回健康；
2. Claude Code `/mcp` 显示 ReMe；
3. `reme-memory` 能召回一条已存在记忆；
4. 完成测试会话后，Hook 日志没有错误；
5. 对应内容出现在当天 daily note 中。

英文原始部署说明位于 `integrations/claude_code/README.md`。
