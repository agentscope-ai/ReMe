# 面向 TypeScript Agent 的 ReMe

[English](./README.md)

`@agentscope-ai/reme` 将 DeepSeek Harness 和 OpenClaw 连接到 ReMe 本地优先、文件原生的长期记忆，并提供不依赖宿主的 ReMe HTTP 客户端。

![DeepSeek Harness 中的 ReMe 状态页](./figures/dsh/reme-status-overview.png)

## 核心能力

- 注入长期记忆使用指引，并提供显式的 `reme_search` 检索。
- 通过后台 `auto_memory` 批次沉淀已完成的对话。
- 按 workspace 时区运行可选的每日 `auto_dream` 记忆整理。
- 将持久记忆保存在用户拥有的 `daily` 和 `digest` Markdown 文件中。
- 使用各宿主原生的生命周期、工具、设置和关停 Hook。
- 自动记忆会排除插件上下文和工具结果，避免循环写回。

## 使用文档

| 宿主             | English                     | 中文                                 |
| ---------------- | --------------------------- | ------------------------------------ |
| DeepSeek Harness | [Guide](./docs/dsh.md)      | [使用指南](./docs/dsh.zh-CN.md)      |
| OpenClaw         | [Guide](./docs/openclaw.md) | [使用指南](./docs/openclaw.zh-CN.md) |

## 快速开始

启动 ReMe：

```bash
pip install "reme-ai[core]"
reme start workspace_dir=/absolute/path/to/workspace
```

安装对应宿主的适配器：

```bash
# DeepSeek Harness
dsh plugin --profile web add @agentscope-ai/reme

# OpenClaw
openclaw plugins install clawhub:@agentscope-ai/reme
```

默认地址为 `http://127.0.0.1:2333`。ReMe HTTP 不使用 API Key 认证；除非前面有受保护的代理，否则应只监听 loopback 或其他可信网络。

## 客户端库

```ts
import { ReMeClient, formatReMeContext } from "@agentscope-ai/reme";
```

宿主适配器分别从 `@agentscope-ai/reme/dsh` 和 `@agentscope-ai/reme/openclaw` 导出。环境要求、完整配置、截图、排障和发布行为请查看对应的宿主文档。
