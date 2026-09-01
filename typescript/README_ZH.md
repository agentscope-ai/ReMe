# 面向 TypeScript Agent 的 ReMe

`@agentscope-ai/reme` 提供统一的 ReMe HTTP 客户端，以及 DeepSeek Harness（DSH）和 OpenClaw 适配器。每个适配器都使用宿主原生的生命周期与工具接口；导入包根入口不会加载任何宿主适配器。

使用前需要启动 ReMe HTTP 服务，并确保所选适配器需要的 `search`、`auto_memory` 和 `auto_dream` 任务可用：

```bash
reme start workspace_dir=/absolute/path/to/workspace
```

默认服务地址为 `http://127.0.0.1:2333`。所有入口均支持 `REME_URL`，也支持组合使用 `REME_HOST` 和 `REME_PORT`。ReMe HTTP 服务不使用 API Key 认证。

## 环境要求

- 已启动 ReMe HTTP 服务，并提供所选适配器需要的 Job。
- 对应受支持主版本线上的 Node.js `22.22.3+`、`24.15.0+` 或 `25.9.0+`。
- 只有使用宿主专用入口时才需要 DeepSeek Harness 或 OpenClaw；包根 client 不依赖宿主 runtime。

## DeepSeek Harness

将本包安装为 DSH profile bundle：

```bash
dsh plugin --profile web add @agentscope-ai/reme
```

安装后可在 **设置 → 插件 → 插件配置 → ReMe Memory** 中配置服务地址、记忆指引语言、自动记忆、每日记忆整理和超时时间，并查看服务健康状态。每日记忆整理和每日批次统一使用配置的 workspace 时区。

Bundle 会在隔离的 `remeMemory` realm 中加载 `@agentscope-ai/reme/dsh`，注入长期记忆指引，注册
`reme_search`，将主 Agent 已完成的对话提交给 `auto_memory`，并按可选的每日计划运行 `auto_dream`。召回的
plugin context 和工具结果不会再次进入自动记忆。

如需直接修改 profile，可以替换 `cordis.patch.yml` 中对应条目：

```yaml
- id: reme-memory
  config:
    - id: reme-memory-runtime
      name: "@agentscope-ai/reme/dsh"
      config:
        endpoint: http://127.0.0.1:2333
        language: zh
        timezone: Asia/Shanghai
        autoMemoryInterval: 5
        autoDreamEnabled: true
        dreamCron: "0 23 * * *"
```

DSH Web profile 也可以在 **设置 → 插件 → 插件配置 → ReMe Memory** 中编辑同一组配置。设置会保存到 DSH
用户设置文档，并从后续请求和捕获开始生效；修改每日计划会重新调度下一次任务，修改语言只影响新会话。
仅供测试的 `dreamIntervalMs` 不属于用户设置。

| 配置项                | 默认值                  | 作用                            |
| --------------------- | ----------------------- | ------------------------------- |
| `endpoint`            | `http://127.0.0.1:2333` | ReMe HTTP 服务地址              |
| `language`            | `en`                    | 记忆指引语言：`en` 或 `zh`      |
| `autoMemoryEnabled`   | `true`                  | 捕获主 Agent 已完成的对话       |
| `autoMemoryInterval`  | `5`                     | 每完成多少轮提交一次            |
| `autoDreamEnabled`    | `true`                  | 启用每日记忆整理                |
| `dreamCron`           | `0 23 * * *`            | workspace 时区下的每日计划      |
| `dreamHint`           | 空字符串                | 传给 `auto_dream` 的可选指引    |
| `rootAgentsOnly`      | `true`                  | 不为子 Agent 注入指引或捕获对话 |
| `searchLimit`         | `5`                     | `reme_search` 返回结果上限      |
| `requestTimeoutMs`    | `10000`                 | 搜索请求超时                    |
| `backgroundTimeoutMs` | `3600000`               | 自动记忆和 Auto Dream 超时      |
| `shutdownTimeoutMs`   | `5000`                  | 退出时尽力排空任务的时间预算    |
| `timezone`            | `Asia/Shanghai`         | 每日批次和计划使用的 IANA 时区  |

ReMe 配置卡会按需调用 `health_check` 和 `status`，显示 ReMe 版本、组件健康状态、chunk/index 数量、进程 RSS
与组件内存估算；它也可以显示脱敏后的 `app_config`，并手动触发一次 `auto_dream`。页面首次打开或用户主动刷新时才会
更新诊断，不会持续轮询。浏览器必须能够访问所配置的 ReMe HTTP 地址，且 ReMe 服务需要允许 DSH 页面所在的 origin。

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

召回内容包裹在 `<reme-context>` 中并标记为不可信历史数据。OpenClaw 插件配置如下：

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

OpenClaw 的会话访问和 Prompt 注入权限仍由宿主侧配置；策略要求显式授权时，需要为 ReMe 开启相应权限。本适配器不会修改
Gateway 配置。

## 客户端库

仅需要 HTTP 客户端时，可以从包根入口导入：

```ts
import { ReMeClient, formatReMeContext } from "@agentscope-ai/reme";
```

宿主适配器分别通过 `@agentscope-ai/reme/dsh` 和 `@agentscope-ai/reme/openclaw` 提供。

## 开发与发布检查

```bash
cd typescript
npm ci
npm run format:check
npm run lint
npm run typecheck
npm test
npm run test:package
```

`npm pack` 会先构建 TypeScript，并且只打包 `dist`、DSH patch、OpenClaw manifest 和中英文 README。
`npm run pack:clawhub -- <输出目录>` 会使用相同的运行时文件生成 ClawHub 产物，并将仅包含 OpenClaw
说明的 `README_OPENCLAW.md` 和 `README_OPENCLAW_ZH.md` 放到包内 README 路径。

稳定版本使用 npm 的 `latest` distribution tag 发布，预发布版本使用 `next`。
