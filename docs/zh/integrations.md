---
title: Agent 集成
description: 通过 CLI、HTTP、MCP、Skill 和宿主适配器把 ReMe 接入 Agent。
---

# Agent 集成

ReMe 把记忆能力放在独立服务和用户拥有的 workspace 中。Agent 可以通过标准接口调用同一套记忆，而不必把存储逻辑绑定到某一个模型或宿主。

## 选择接入方式

| 场景 | 推荐方式 |
|---|---|
| 本机脚本或 Hook | ReMe CLI |
| 应用后端 | HTTP Client |
| 支持工具协议的 Agent | MCP |
| DeepSeek Harness | [`@agentscope-ai/reme-dsh-plugin`](./integrations/dsh.md) profile bundle |
| OpenClaw | [`@agentscope-ai/reme-openclaw-plugin`](./integrations/openclaw.md) |
| Claude Code | [共享 HTTP MCP + Skill + Stop Hook](./integrations/claude-code.md) |
| Hermes Agent | Memory provider adapter |
| Codex 或其他 coding agent | `reme_memory` Skill 或 MCP |

## 通用接入循环

一个完整但可控的 Agent 记忆循环通常包含：

1. 会话开始或回答前，用 `search` 找到相关记忆；
2. 对高价值结果使用 `read`，必要时用 `traverse` 展开关系；
3. 在回答中保留 workspace-relative 来源路径；
4. 会话结束后，把原始消息交给 `auto_memory`；
5. 由后台或定时任务把 daily 内容整理到 digest。

搜索不到内容时应明确返回空结果，不应把模型推测当成历史记忆。

## MCP

默认 HTTP 服务在 `http://127.0.0.1:2333/mcp` 提供 streamable HTTP MCP。常用工具包括：

- `search`
- `read`
- `traverse`
- `list`
- `auto_memory`
- `proactive_read`

根据宿主风险模型，可以用 `service.jobs` 只暴露只读工具，或将写入工具放在单独配置中。

## CLI 和 Skill

仓库中的 `skills/reme_memory/SKILL.md` 描述了一个通用 Agent 工作流，包括安装检查、服务发现、检索、读取和写入边界。它适合能够执行本地命令的 Agent。

Skill 不应：

- 未经允许安装或升级 Python 环境；
- 发现端口冲突后停止未知进程；
- 把召回的工具结果再次写入对话来源；
- 将密钥或敏感信息写入记忆。

## DeepSeek Harness

安装自包含的 [DeepSeek Harness 插件](./integrations/dsh.md)：

```bash
dsh plugin --profile web add @agentscope-ai/reme-dsh-plugin
```

发布页：[Awesome DSH Plugin](https://awesome-dsh-plugin.com/p/agentscope-ai/ReMe--integrations-dsh/) 和
[npm](https://www.npmjs.com/package/@agentscope-ai/reme-dsh-plugin)。

插件会在新的根 Agent 会话中注入长期记忆使用指引，并提供只读 `reme_search` 工具；它不会把所有历史记忆预先塞入上下文。
已完成的用户/助手对话可以分批在后台交给 `auto_memory`，并由带时区的计划任务调用 `auto_dream` 整理 daily note。

DSH 设置可配置服务地址、指引语言、搜索数量、捕获间隔、根 Agent 过滤和整理计划。ReMe Status 页面包含 Overview、
Auto Memory、Memory Consolidation、Components、Journal 和 Personal Knowledge Base 六个视图。运行时计数只用于诊断，
workspace 中的 Markdown 仍是持久事实来源。

## OpenClaw

安装独立发布的 [OpenClaw 插件](./integrations/openclaw.md)：

```bash
openclaw plugins install clawhub:@agentscope-ai/reme-openclaw-plugin
```

发布页：[ClawHub](https://clawhub.ai/agentscope-ai/plugins/reme-openclaw-plugin) 和
[npm](https://www.npmjs.com/package/@agentscope-ai/reme-openclaw-plugin)。该插件拥有自己的宿主适配 ReMe HTTP 边界和发布周期。

## Claude Code

[Claude Code 插件](./integrations/claude-code.md) 默认让所有 Claude Code 窗口连接同一个
`http://127.0.0.1:2333/mcp` ReMe HTTP 进程。`reme-memory` Skill 会在语义 `search`、图关系 `traverse` 和状态查询
`daily_list` / `frontmatter_read` 之间选择，再读取并引用相关 workspace 路径。

会话 Stop 时，Hook 只把 Claude Code `session_id` 交给服务端 `auto_memory_cc` Job。在 POSIX 系统上，它会脱离可能耗时的
模型调用，让 Claude Code 立即停止；服务不可达等 best-effort 失败只写入插件日志，不阻塞宿主。ReMe 会解析本地 transcript；
重复 Stop 且没有新消息时，不会重复生成记忆。

## Hermes Agent

`integrations/hermes_agent/` 提供 HTTP 和 Embedded 双模式 memory provider：模型调用前检索相关记忆，每轮结束后异步调用 `auto_memory`，并通过 Hermes 通用配置面板展示设置。完整配置见该目录 README。

## 生产接入建议

- 明确选择稳定、绝对的 `workspace_dir`；
- 启动前复用 `reme find_reme` 发现的服务；
- 以 `reme help` 为当前 Job 契约；
- 为写入动作设置超时和失败日志；
- 不因记忆服务暂时不可用而阻塞宿主的核心回答流程；
- 对远程访问使用认证、TLS 和最小 Job allowlist。
