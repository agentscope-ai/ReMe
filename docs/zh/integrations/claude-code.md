---
title: Claude Code 集成
description: 通过 MCP、reme-memory Skill 和 Stop Hook 将 Claude Code 连接到 ReMe。
---

# Claude Code 集成

ReMe 的 Claude Code 插件提供长期记忆召回，并在每次会话结束后异步记录对话。Daily 到 digest 的整理仍由共享的 ReMe 服务负责。

## 能力

- 通过 MCP 使用 `search`、`traverse`、`daily_list`、`frontmatter_read`、`read`、`auto_memory_cc` 等工具；
- `reme-memory` Skill 在回答前召回长期记忆并保留来源路径；
- Stop Hook 只把 Claude Code `session_id` 传给服务端，服务端从本地 transcript 解析会话；
- 记录在脱离 Claude Code 的后台进程中进行，不延迟退出；服务不可用时记录日志并结束。

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
- 可通过 `CLAUDE_CONFIG_DIR` 修改 transcript 根目录；
- 可通过 `REME_HOST`、`REME_PORT` 覆盖 Hook 使用的服务地址。

Hook 需要 `python3` 位于 `PATH`。MCP 工具名前缀可能随 Claude Code 版本包含 server segment；Skill 使用 `mcp__reme__*` 匹配这一差异。

## 验证

1. `reme health_check` 返回健康；
2. Claude Code `/mcp` 显示 ReMe；
3. `reme-memory` 能召回一条已存在记忆；
4. 完成测试会话后，Hook 日志没有错误；
5. 对应内容出现在当天 daily note 中。

英文原始部署说明位于 `integrations/claude_code/README.md`。
