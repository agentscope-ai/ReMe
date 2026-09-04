# OpenClaw 的 ReMe 长期记忆插件

[English](./openclaw.md)

ReMe 为 OpenClaw 提供文件原生的长期记忆，并将持久记忆保存在用户拥有的 workspace 中。插件使用 OpenClaw
原生的生命周期、Hook、工具接口和 memory slot。

## 插件能力

- 注册 `reme_search`，支持显式检索记忆。
- 在根 Agent 的对话运行前自动召回相关记忆。
- 在后台分批捕获已完成的用户/助手对话。
- 按 workspace 时区运行可选的每日 `auto_dream` 记忆整理。
- 默认排除子 Agent、Cron、Heartbeat、Memory 和 Overflow 触发的运行。
- 使用 `<reme-context>` 包裹召回内容，并将其标记为不可信历史数据。

## 环境要求

- OpenClaw `2026.7.1` 或更高版本。
- 对应受支持主版本线上的 Node.js `22.22.3+`、`24.15.0+` 或 `25.9.0+`。
- 已启动 ReMe HTTP 服务，并提供 `search`、`auto_memory` 和 `auto_dream` Job。

## 启动 ReMe

安装 ReMe 并启动本地 HTTP 服务：

```bash
pip install "reme-ai[core]"
reme start workspace_dir=/absolute/path/to/workspace
```

默认地址为 `http://127.0.0.1:2333`。ReMe HTTP 服务不使用 API Key 认证，因此除非前面部署了受保护的代理边界，
否则应仅监听 loopback 或其他可信网络。

## 安装 OpenClaw 插件

明确指定从 ClawHub 安装：

```bash
openclaw plugins install clawhub:@agentscope-ai/reme
```

如果已经启用了其他记忆插件，请将 `plugins.slots.memory` 设为 `reme`。对话访问和 Prompt 注入权限仍由
OpenClaw 管理；策略要求显式授权时，需要为 ReMe 开启相应权限。插件不会修改 Gateway 配置。

## 配置

| 配置项                | 默认值                  | 作用                                 |
| --------------------- | ----------------------- | ------------------------------------ |
| `endpoint`            | `http://127.0.0.1:2333` | ReMe HTTP 服务地址                   |
| `language`            | `en`                    | 记忆指引语言：`en` 或 `zh`           |
| `autoRecall`          | `true`                  | 在根 Agent 对话运行前自动召回        |
| `searchLimit`         | `5`                     | 搜索结果上限                         |
| `recallMinScore`      | `0`                     | 自动召回的最低搜索分数               |
| `autoMemoryEnabled`   | `true`                  | 捕获已完成的对话                     |
| `autoMemoryInterval`  | `5`                     | 每完成多少轮提交一次                 |
| `autoDreamEnabled`    | `true`                  | 启用每日记忆整理                     |
| `dreamCron`           | `0 23 * * *`            | workspace 时区下的每日计划           |
| `dreamHint`           | 空字符串                | 传给 `auto_dream` 的可选指引         |
| `rootAgentsOnly`      | `true`                  | 不为子 Agent 注入指引或捕获对话      |
| `timezone`            | `Asia/Shanghai`         | 每日批次和计划使用的 IANA 时区       |
| `requestTimeoutMs`    | `10000`                 | 自动召回和显式搜索超时               |
| `backgroundTimeoutMs` | `3600000`               | 自动记忆和 Auto Dream 超时           |
| `shutdownTimeoutMs`   | `5000`                  | Gateway 退出时尽力排空任务的时间预算 |

配置修改会从后续运行开始生效。失败的自动记忆批次会保留等待重试，Gateway 退出时会在 `shutdownTimeoutMs` 范围内
尝试刷新未完成任务。

## 源码与许可证

ReMe 在 [agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe) 开发，并使用 Apache-2.0 许可证发布。
