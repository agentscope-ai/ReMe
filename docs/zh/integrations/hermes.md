---
title: Hermes Agent 集成
description: 使用 ReMe memory provider 在 Hermes 调用模型前召回、每轮结束后异步记录。
---

# Hermes Agent 集成

Hermes memory provider 连接到一个已经运行的 ReMe HTTP 服务，在每次模型调用前召回相关记忆，并在用户/助手回合完成后异步调用 `auto_memory`。

## Workspace 隔离

ReMe 的搜索范围是一个完整 workspace。多个 Hermes profile 指向同一个 workspace 时会共享召回结果；需要隔离时，为每个 profile 使用独立 workspace 和端点。

```bash
reme start \
  workspace_dir="/absolute/path/to/reme-hermes-default" \
  service.backend=http \
  service.host=127.0.0.1 \
  service.port=2333
```

自动记忆需要 LLM；默认 BM25 搜索不需要 Embedding。

## 安装与配置

```bash
hermes plugins install agentscope-ai/ReMe/integrations/hermes_agent
hermes memory setup
```

选择 `reme`，接受默认的 `http://127.0.0.1:2333`，或输入上一步使用的端点。Setup 会先调用 `health_check`，只有新端点健康时才替换现有 provider 配置。

配置存放在 `$HERMES_HOME/reme.json`：

```json
{
  "endpoint": "http://127.0.0.1:2333",
  "request_timeout": 600.0,
  "recall_timeout": 5.0,
  "health_timeout": 2.0,
  "health_retry_seconds": 30.0,
  "shutdown_timeout": 30.0,
  "recall_limit": 5
}
```

运行 `hermes memory status` 检查安装和配置。新的 Hermes 会话还会重新检查端点健康状态。

## 生命周期和失败行为

- `prefetch` 调用 ReMe `search`，Hermes 将结果放入受保护的 memory context；
- `sync_turn` 把完成的回合加入串行后台写队列，再调用 `auto_memory`；
- cron、flush 和 subagent context 不写入对话记忆；
- 健康检查失败后，在 cooldown 结束前暂停召回和记录；
- 召回与写入有独立 cooldown，单项失败不会关闭另一项；
- 召回使用较短超时，避免慢搜索长期阻塞模型调用；
- shutdown 会在有限时间内排空写队列，ReMe 服务仍由用户独立管理。

英文权威安装说明位于 `integrations/hermes_agent/README.md`。
