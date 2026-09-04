---
title: 基础配置
description: ReMe 配置文件、环境变量、命令行覆盖和核心组件配置。
---

# 基础配置

ReMe 使用 YAML 或 JSON 描述 Service、Job 和 Component。默认配置位于 `reme/config/default.yaml`；启动时可以选择其他配置，再用命令行覆盖其中的字段。

## 配置优先级

配置按下面的顺序合并，靠后的值优先：

1. 已启用插件提供的 `application_defaults`。
2. 选中的配置文件；未指定时使用内置 `default`。
3. 命令行 dot notation 覆盖。

```bash
reme start
reme start config=demo
reme start config=/absolute/path/to/app.yaml
reme start service.port=8181 workspace_dir=/data/reme
```

`config` 支持内置配置名以及 `.yaml`、`.yml`、`.json` 文件。覆盖采用深度合并，不会因为修改 `service.port` 而丢失 `service` 下的其他字段。

## 值的解析

CLI 参数使用 `key=value`，前导 `-` 或 `--` 也会被接受：

```bash
reme start --service.port=8181 --service.web_enabled=false
```

值支持：

- `null`、布尔值、整数和浮点数；
- JSON 数组和对象；
- JSON 引号字符串；
- 普通字符串。

类似 `007` 的前导零字符串不会被转换成数字。需要保留 `true`、`false` 等字面字符串时，使用 JSON 引号：`value='"true"'`。

## 环境变量

配置文件会递归展开两种表达式：

```yaml
api_key: ${LLM_API_KEY}
base_url: ${LLM_BASE_URL:-https://example.com/v1}
```

`${VAR}` 在变量未定义时会报错；`${VAR:-default}` 使用默认值。ReMe 还会从命令启动目录向上查找 `.env`，最多检查五级父目录。

不要把密钥提交到配置文件或 Git。推荐把密钥放在 `.env` 或进程环境中。

## Application 字段

| 字段 | 默认值 | 作用 |
|---|---|---|
| `app_name` | `ReMe` | 应用显示名称 |
| `workspace_dir` | `.reme` | 用户拥有的 workspace 根目录，会规范化为绝对路径 |
| `metadata_dir` | `metadata` | 索引、图谱和 catalog 等派生状态 |
| `session_dir` | `session` | Agent 对话记录；标准 transcript 位于 `session/dialog` |
| `mem_session_dir` | `mem_session` | Agent wrapper 的会话和配置 |
| `resource_dir` | `resource` | 外部资料 |
| `daily_dir` | `daily` | Daily memory |
| `digest_dir` | `digest` | 长期整理后的记忆 |
| `timezone` | `Asia/Shanghai` | Cron、日期和梦境流程使用的 IANA 时区 |
| `language` | 空 | LLM 交互默认语言 |
| `plugins` | `[]` | 为当前 Application 启用的已安装插件 |
| `service` | HTTP | 服务端配置 |
| `jobs` | 默认 Job | Job 名到 Job 配置的映射 |
| `components` | 默认组件 | 按类型和名称组织的组件配置 |

`session_dir` 必须保持 workspace-relative。其他 workspace 子目录也应使用清晰、稳定的相对名称。

## LLM 配置

默认 LLM 使用 OpenAI-compatible 接口：

```yaml
components:
  as_llm:
    default:
      backend: openai
      model: qwen3.7-plus
      context_size: 200000
      credential:
        api_key: ${LLM_API_KEY:-}
        base_url: ${LLM_BASE_URL:-}
```

可注册的内置 backend 包括 `openai`、`anthropic`、`dashscope`、`deepseek`、`gemini`、`moonshot`、`ollama` 和 `xai`。实际字段由对应 AgentScope model wrapper 决定。

基础文件操作、BM25 检索、wikilink 遍历不需要 LLM。`auto_memory`、`auto_resource`、`auto_dream` 等演化流程需要可用 LLM。

## Embedding 配置

向量检索默认关闭。只设置 `EMBEDDING_API_KEY` 不会自动启用它；还需要同时启用 `as_embedding`、`embedding_store`，并把它连接到 `file_store`：

```yaml
components:
  as_embedding:
    default:
      backend: openai
      model: text-embedding-v4
      dimensions: 1024
      credential:
        api_key: ${EMBEDDING_API_KEY}
        base_url: ${EMBEDDING_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
  embedding_store:
    default:
      backend: local
      as_embedding: default
  file_store:
    default:
      backend: local
      embedding_store: default
      keyword_index: default
      file_graph: default
```

修改 embedding 模型或维度后，应重新构建 embedding 索引。

## Service 和 Job

最小 HTTP 配置：

```yaml
service:
  backend: http
  host: 127.0.0.1
  port: 2333
  web_enabled: true
  mcp_enabled: true
  mcp_path: /mcp
```

Job 由 backend、参数 schema 和顺序执行的 Step 组成：

```yaml
jobs:
  example:
    backend: base
    description: Example job
    parameters:
      type: object
      properties:
        text: { type: string }
      required: [text]
    steps:
      - backend: example_step
```

设置 `enable_serve: false` 可以保留内部 Job、禁止 Service 暴露。后台和 Cron Job始终不会作为请求端点暴露。

## 查看生效配置

服务启动后运行：

```bash
reme app_config
```

返回的是已合并、已校验并隐藏密钥后的配置。排查覆盖顺序或插件配置时，应以它为准，而不是只查看某一个 YAML 文件。

完整字段定义以 `reme/schema/application_config.py` 和 `reme/config/default.yaml` 为准。
