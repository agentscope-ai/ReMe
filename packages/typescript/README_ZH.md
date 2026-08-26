# 面向 TypeScript Agent 的 ReMe

`@agentscope-ai/reme` 提供统一的 ReMe HTTP 客户端，以及 DeepSeek Harness（DSH）和 OpenClaw 适配器。每个适配器都使用宿主原生的生命周期与工具接口；导入包根入口不会加载任何宿主适配器。

使用前需要启动 ReMe HTTP 服务，并确保所选适配器需要的 `search`、`auto_memory` 和 `auto_dream` 任务可用：

```bash
reme start workspace_dir=/absolute/path/to/workspace
```

默认服务地址为 `http://127.0.0.1:2333`。所有入口均支持 `REME_URL`，也支持组合使用 `REME_HOST` 和 `REME_PORT`。ReMe HTTP 服务不使用 API Key 认证。

## DeepSeek Harness

将本包安装为 DSH profile bundle：

```bash
dsh plugin --profile web add @agentscope-ai/reme
```

安装后可在 **设置 → 插件 → 插件配置 → ReMe Memory** 中配置服务地址、记忆指引语言、自动记忆、每日记忆整理和超时时间，并查看服务健康状态。每日记忆整理和每日批次统一使用配置的 workspace 时区。

完整配置项和 `cordis.patch.yml` 示例请参阅[英文文档](./README.md#deepseek-harness)。

## OpenClaw

OpenClaw `2026.7.1` 或更高版本可以直接安装本包。当前 SDK 和 Gateway 支持各主版本线上的
Node.js `22.22.3+`、`24.15.0+` 或 `25.9.0+`：

```bash
openclaw plugins install @agentscope-ai/reme
```

当其他记忆插件已启用时，请将 `plugins.slots.memory` 设为 `reme`。适配器使用最新的
`before_prompt_build` Hook，注册 `reme_search` Action，并为根 Agent 注入记忆使用指引和相关历史。
已完成的用户/助手消息按会话、日期分批提交给 `auto_memory`；失败批次会保留重试，Gateway 退出时会在有限时间内刷新。
插件还会按 workspace 时区运行一份每日 `auto_dream` 计划。默认不会处理子 Agent、Cron、Heartbeat、Memory 或 Overflow 触发的运行。

主要配置包括 `language`、`autoRecall`、`searchLimit`、`autoMemoryEnabled`、`autoMemoryInterval`、
`autoDreamEnabled`、`dreamCron`、`dreamHint`、`rootAgentsOnly` 和 `timezone`。默认每 5 轮写入一次记忆，
每日 23:00（`Asia/Shanghai`）执行 Auto Dream。OpenClaw 的会话访问和 Prompt 注入权限仍由宿主侧配置，
本适配器不会修改 Gateway 设置。

完整配置项请参阅[英文文档](./README.md#openclaw)。

## 客户端库

仅需要 HTTP 客户端时，可以从包根入口导入：

```ts
import { ReMeClient, formatReMeContext } from "@agentscope-ai/reme";
```

宿主适配器分别通过 `@agentscope-ai/reme/dsh` 和 `@agentscope-ai/reme/openclaw` 提供。

## 开发与发布检查

```bash
cd packages/typescript
npm ci
npm run format:check
npm run lint
npm run typecheck
npm test
npm run test:package
```

正式版本 `0.1.0` 使用 npm 的 `latest` distribution tag 发布。
