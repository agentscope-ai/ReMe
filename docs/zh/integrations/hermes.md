---
title: Hermes Agent 集成
description: 使用 HTTP 或 Embedded ReMe memory provider，在模型调用前召回、每轮结束后异步记录。
---

# Hermes Agent 集成

ReMe 的 Hermes memory provider 支持两种运行方式：

- HTTP（默认）：连接独立运行的 ReMe 服务，Hermes 环境不需要安装 ReMe SDK；
- Embedded：在 Hermes Python 进程内创建 ReMe `Application`，不需要服务进程和端口。

两种模式都会在模型调用前执行 `search`，并在用户/助手回合完成后把 `auto_memory` 放入串行后台队列。

## 安装与配置

```bash
hermes plugins install agentscope-ai/ReMe/integrations/hermes_agent
hermes memory setup
```

Hermes Dashboard 会展示 provider 的模式相关字段和高级 recall、health、write、shutdown 设置；切换模式后只显示对应的
HTTP endpoint 或 Embedded workspace 字段。

配置主路径为 `$HERMES_HOME/reme/config.json`。新文件未包含的字段仍会从旧 `$HERMES_HOME/reme.json` 继承，新文件中的
值优先；后续 CLI 保存会写入完整的新配置，但不会删除旧文件。

## HTTP 模式

为当前 Hermes profile 启动独立 workspace：

```bash
reme start \
  workspace_dir="/absolute/path/to/reme-hermes-default" \
  service.backend=http \
  service.host=127.0.0.1 \
  service.port=2333
```

配置示例：

```json
{
  "mode": "http",
  "endpoint": "http://127.0.0.1:2333"
}
```

终端 setup 会在保存前调用 `health_check`。HTTP action 接口没有本集成专用的认证头，不要直接暴露到公网；跨主机使用时
应放在可信网络、SSH tunnel 或带认证的反向代理后。

## Embedded 模式

先把 ReMe 安装到 Hermes 使用的同一个 Python 环境：

```bash
pip install "reme-ai[core]"
```

然后配置独立 workspace：

```json
{
  "mode": "embedded",
  "workspace_dir": "~/.reme-hermes-default",
  "reme_config": "default"
}
```

插件在专用 asyncio loop thread 上构造并启动 `reme.Application`，直接执行 `health_check`、`search` 和 `auto_memory`。
关闭时会在有界时间内排空写队列、调用 `Application.close()`、停止 loop 并 join thread；不会调用
`Application.run_app()`，因此不会监听端口。

## Workspace 隔离

ReMe 搜索覆盖整个 workspace。多个 Hermes profile 指向同一个 workspace 时会共享召回结果；除非明确需要共享，否则为
每个 profile 配置不同 workspace。HTTP 模式通常也使用不同端口。

自动记忆需要可用的 LLM 配置；默认 BM25 搜索不需要 Embedding，除非选用的 ReMe 配置启用了依赖 Embedding 的向量检索。

## 生命周期和失败行为

- `prefetch` 只返回 ReMe answer，受保护的 memory context 包装由 Hermes 统一添加；
- 有召回内容时，Hermes UI 会显示 ReMe recall indicator 和可获得的结果数量；
- `sync_turn` 只提交最新完整回合，并用 profile 与 session 共同生成安全 ID；
- cron、flush 和 subagent context 不写入对话记忆；
- backend 健康、召回和写入使用独立 cooldown；错误只记录警告，不中断 Hermes 对话；
- 写队列目前位于内存，进程异常退出时尚未完成的写入可能丢失。

运行 `hermes memory status` 检查插件状态，然后启动新的 Hermes 会话。完整字段、真实截图和故障排查见
[`integrations/hermes_agent/README_ZH.md`](../../../integrations/hermes_agent/README_ZH.md)。
