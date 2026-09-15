# 钉钉插件

[English](README.md)

钉钉插件为 ReMe 提供 Markdown 主动通知和长期运行的 Agent 对话桥接。本目录是一个独立 Python distribution；
`reme.plugins` entry point 暴露只包含 backend 的 `plugin.yaml`，注册 `dingtalk_markdown_send_step` 和
`dingtalk_wait_step`。Application 自行决定是否以及如何使用这些 backend；仅启用插件不会暴露发送 endpoint，也不会
启动长期连接。

## 快速开始

安装 ReMe 和插件：

```bash
python -m pip install "reme-ai[core]>=0.4.1.12"
reme plugins install plugins/dingtalk
reme plugins validate dingtalk
```

配置企业内部应用机器人的凭据和目标群会话 ID：

```dotenv
DINGTALK_APP_KEY=your-app-key
DINGTALK_APP_SECRET=your-app-secret
DINGTALK_ROBOT_CODE=your-robot-code
DINGTALK_CONVERSATION_IDS=cid-group-one,cid-group-two
```

仓库内置的 `cookbook` 配置会组合钉钉、Auto Fin 和 Daily Paper 三个独立安装的插件：

```bash
reme plugins install plugins/auto-fin
reme plugins install plugins/daily_paper
reme start config=cookbook
```

`cookbook.yaml` 负责跨插件路由：在自动标签之后追加通知 Step，让手动 Job 与 cron Job 复用同一 pipeline，并显式以
background Job 启动钉钉 Agent bridge。两个业务插件仍可完全脱离钉钉使用。

## Markdown 发送 Job 示例

需要直接发送时，可以在 Application 配置中添加一个私有 one-shot Job：

```yaml
extends: default
plugins: [dingtalk]

jobs:
  dingtalk_send:
    backend: base
    enable_serve: false
    parameters:
      type: object
      properties:
        markdown_path:
          type: string
      required: [markdown_path]
    steps:
      - backend: dingtalk_markdown_send_step
        app_key: ${DINGTALK_APP_KEY}
        app_secret: ${DINGTALK_APP_SECRET}
        robot_code: ${DINGTALK_ROBOT_CODE}
        conversation_ids: ${DINGTALK_CONVERSATION_IDS:-}
        timeout: 15
```

无需通过 HTTP 或 MCP 暴露即可运行：

```bash
reme start config=/path/to/dingtalk.yaml \
  job=dingtalk_send \
  markdown_path=daily/2026-09-15/report.md
```

`markdown_path` 必须指向 workspace 内真实存在的 `.md` 文件。消息会去掉 frontmatter，再按配置顺序串行发送到每个
非空的逗号分隔会话 ID。未配置会话 ID 时，发送会无副作用成功跳过；一旦配置了接收方，凭据缺失、路径无效、正文
为空、token 获取失败或部分接收方发送失败都会明确令 Job 失败。响应 metadata 会记录配置数与成功数，但日志不会
输出凭据或会话 ID。

## Agent 对话桥接

桥接会为每个“发送者 + 会话”维护独立 Agent session，支持 ReMe 的 session 命令，并以 Markdown 返回 Agent 的最终
回复。可在继承内置默认配置的应用文件中添加以下后台 Job：

```yaml
extends: default
plugins: [dingtalk]

jobs:
  dingtalk_agent:
    backend: background
    supervisor: true
    close_timeout: 10
    steps:
      - backend: dingtalk_wait_step
        agent_wrapper: default
        app_key: ${DINGTALK_APP_KEY}
        app_secret: ${DINGTALK_APP_SECRET}
        robot_code: ${DINGTALK_ROBOT_CODE}
        worker_count: 4
        builtin_tools: false
        job_tools: [search, read]
```

`reme/config/cookbook.yaml` 已启用同样的后台 Job。独立使用时通过 `reme start config=/path/to/dingtalk.yaml` 启动。
后台 Job 不会暴露为 HTTP endpoint 或 MCP tool。不同 session key
可以并发处理，同一 key 的消息保持串行。Session ID 只存放在 `ApplicationContext.metadata`，重启 ReMe 后会重新建立
钉钉到 Agent 的 session 映射。模型访问与权限由配置的 `agent_wrapper` 决定，只应授予机器人必需的内置工具和 Job。

## 开发验证

在仓库根目录运行：

```bash
reme plugins install ./plugins/dingtalk --editable
reme plugins validate dingtalk
python -m pytest plugins/dingtalk -v
```

测试会 mock 钉钉 HTTP、WebSocket 和 Agent 边界，不访问外部服务。
