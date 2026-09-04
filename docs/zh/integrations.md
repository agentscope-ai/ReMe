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
| TypeScript Agent | `@agentscope-ai/reme` |
| Claude Code | MCP + Skill + Stop Hook |
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
- `proactive`

根据宿主风险模型，可以用 `service.jobs` 只暴露只读工具，或将写入工具放在单独配置中。

## CLI 和 Skill

仓库中的 `skills/reme_memory/SKILL.md` 描述了一个通用 Agent 工作流，包括安装检查、服务发现、检索、读取和写入边界。它适合能够执行本地命令的 Agent。

Skill 不应：

- 未经允许安装或升级 Python 环境；
- 发现端口冲突后停止未知进程；
- 把召回的工具结果再次写入对话来源；
- 将密钥或敏感信息写入记忆。

## TypeScript、OpenClaw 与 DeepSeek Harness

统一 TypeScript 客户端和宿主适配器见 [TypeScript Agent 集成](./integrations/typescript.md)。它包含：

- HTTP Client；
- DeepSeek Harness adapter；
- OpenClaw adapter；
- 构建与发布检查。

## Claude Code

仓库的 `integrations/claude_code/` 提供：

- streamable HTTP MCP 配置；
- `reme-memory` Skill；
- 会话停止时调用 `auto_memory_cc` 的 Hook。

完整安装步骤以仓库中的 `integrations/claude_code/README.md` 为准。

## Hermes Agent

`integrations/hermes_agent/` 提供 memory provider：模型调用前检索相关记忆，每轮结束后异步调用 `auto_memory`。完整配置见该目录 README。

## 生产接入建议

- 明确选择稳定、绝对的 `workspace_dir`；
- 启动前复用 `reme find_reme` 发现的服务；
- 以 `reme help` 为当前 Job 契约；
- 为写入动作设置超时和失败日志；
- 不因记忆服务暂时不可用而阻塞宿主的核心回答流程；
- 对远程访问使用认证、TLS 和最小 Job allowlist。
