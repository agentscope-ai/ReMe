# 面向 TypeScript Agent 的 ReMe

`@agentscope-ai/reme` 提供统一的 ReMe HTTP 客户端，以及 DeepSeek Harness（DSH）和 OpenClaw 适配器。每个适配器都使用宿主原生的生命周期与工具接口；导入包根入口不会加载任何宿主适配器。

使用前需要启动 ReMe HTTP 服务，并确保所选适配器需要的 `search`、`auto_memory` 和 `auto_dream` 任务可用：

```bash
reme start workspace_dir=/absolute/path/to/workspace
```

默认服务地址为 `http://127.0.0.1:2333`。所有入口均支持 `REME_URL`，也支持组合使用 `REME_HOST` 和 `REME_PORT`；还可以通过 `REME_API_KEY` 配置可选的 Bearer Token。

## DeepSeek Harness

将本包安装为 DSH profile bundle：

```bash
dsh plugin --profile web add @agentscope-ai/reme
```

安装后可在 **设置 → 插件 → 插件配置 → ReMe Memory** 中配置服务地址、记忆指引语言、自动记忆、每日记忆整理和超时时间，并查看服务健康状态。`apiKey` 属于部署级密钥，不会显示在设置页面中。

完整配置项和 `cordis.patch.yml` 示例请参阅[英文文档](./README.md#deepseek-harness)。

## OpenClaw

OpenClaw `2026.3.12` 或更高版本可以直接安装本包：

```bash
openclaw plugins install @agentscope-ai/reme
```

当其他记忆插件已启用时，请将 `plugins.slots.memory` 设为 `reme`。适配器会注册 `reme_search`，在用户触发的 Agent 运行前检索长期记忆，并将最后一组已完成的用户/助手消息提交给 `auto_memory`。

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
