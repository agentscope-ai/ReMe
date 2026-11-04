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

在一个终端选择一条启动命令，并保持服务运行：

```bash
reme start
reme start config=demo
reme start config=cookbook
reme start config=/absolute/path/to/app.yaml
reme start service.port=8181 workspace_dir=/data/reme
```

`config` 支持内置配置名以及 `.yaml`、`.yml`、`.json` 文件。覆盖采用深度合并，不会因为修改 `service.port` 而丢失 `service` 下的其他字段。
可选的 `cookbook` 变体继承 `default`，并组合独立安装的 Auto Fin、Daily Paper 与钉钉插件；配置加载前必须提供三个
钉钉应用凭据环境变量。该配置还会启用 `text-embedding-v4` 向量检索，默认通过 AgentScope 使用
`${LLM_MODEL_NAME:-qwen3.8-max}`，并让钉钉桥接通过 Claude Code 复用 `LLM_MODEL_NAME` 与 `LLM_API_KEY`。

在另一个终端调用 `reme <job-name>`，例如 `reme help`；CLI 会发现正在运行的服务配置。

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

类似 `007` 的前导零字符串以及 `nan`、`inf`、`Infinity` 不会被转换成数字。需要保留 `true`、`false` 等字面字符串时，使用 JSON 引号：`value='"true"'`。

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

## Job 启停

`jobs.<name>.enabled` 默认为 `true`，适用于所有 Job 类型。禁用的 Job 保留配置，但不启动、不暴露接口，
通过 `Application.run_job()` / `run_stream_job()` 调用时会报错。由宿主插件负责调度时，可以关闭 ReMe 的 Dream cron：

```bash
reme start jobs.dream_cron.enabled=false
```

重启服务后生效。独立的 `auto_dream` 接口仍可调用，其他 Job 继续运行。
`enable_serve` 独立控制接口暴露：`enabled=true, enable_serve=false` 允许本地调用。
Background 和 cron Job 始终不暴露服务接口。

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

基础文件操作、BM25 检索、wikilink 遍历和 `proactive_read` 不需要 LLM。`auto_memory`、`auto_resource`、
`auto_dream` 和 proactive refresh 等演化流程需要可用 LLM。

## 决策模型配置

`as_decision_model` 是不依赖供应商的概率决策模型接口；当前的 `agentscope` 适配器通过 AgentScope `2.0.11` 调用 TypeSafe API。
它接受共享的文本或 JSON object 与命名问题，返回二分类概率、选项分类或评分结果。

安装可选依赖：

```bash
pip install "reme-ai[core,decision]"
# 从源码安装时：
pip install -e ".[core,decision]"
```

`core` 不包含决策模型 SDK，`full` 包含该 extra。只设置环境变量不会自动启用组件；默认配置中的决策模型示例保持注释。
组件启动不导入可选决策模型模块、不创建客户端，首次决策调用才检查依赖和凭证。客户端由组件复用，关闭时先等待正在执行的请求结束，再释放客户端。

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `DECISION_MODEL_BACKEND` | `agentscope` | 决策模型后端 |
| `DECISION_MODEL_NAME` | `null`（适配器默认值） | 模型名或别名 |
| `DECISION_MODEL_API_KEY` | 空 | TypeSafe API key，决策调用时必须提供 |
| `DECISION_MODEL_BASE_URL` | `null` | 可选 API 地址；空值使用 SDK 默认地址 |

启用组件的配置示例：

```yaml
components:
  as_decision_model:
    default:
      backend: ${DECISION_MODEL_BACKEND:-agentscope}
      model: ${DECISION_MODEL_NAME:-null}
      credential:
        api_key: ${DECISION_MODEL_API_KEY:-}
        base_url: ${DECISION_MODEL_BASE_URL:-null}
      timeout: 30
      max_retries: 2
      retry_delay: 0.5
```

`timeout` 是单次请求的超时秒数；`max_retries` 是首次请求之外的重试次数；`retry_delay` 是初始退避秒数。
这三个字段通过 YAML 或 CLI 覆盖配置，重试由 SDK 执行。

内置 `decision_demo` 配置通过 `extends: default` 继承默认 HTTP/MCP 服务、组件和 Job，并增加决策模型与
可通过服务调用的 `decide` Job。默认后台任务也会启动。先在一个终端启动服务：

```bash
export DECISION_MODEL_API_KEY=your-typesafe-api-key
reme start config=decision_demo \
  components.as_decision_model.default.timeout=15
```

再在另一个终端调用，客户端会发现正在运行的服务配置：

```bash
reme decide state="我的账户被重复扣款，请尽快处理"
```

调用会把 `state` 和问题发送给 TypeSafe。省略 `questions` 时，demo 同时判断是否紧急、选择客服类别和评估优先级。
可以完整替换演示问题，也可以传入 JSON object：

```bash
reme decide \
  state='{"request":"我的账户被重复扣款","channel":"email"}' \
  questions='{"urgent":{"type":"binary","instructions":"是否需要紧急处理？"}}'
```

问题的 `type` 支持 `binary`、`choice`、`score`。`binary` 可提供 `criteria.true` / `criteria.false` 描述；
`choice` 需要非空 `criteria` object，key 是类别，value 是描述或 `null`；`score` 需要非空的有序描述列表，
分数从零开始。问题可提供 `instructions`。显式传入空的 `questions` 会报错，不会退回 demo 默认问题。

`reme decide` 显示标准 Response JSON，决策结果位于 `answer`，包含 `model`、`content`、
`usage`、响应标识和时间等信息。`content` 按问题名组织，保留概率与置信度；评分分布的整数 key 在 JSON 中变为字符串。
Step 同时写入请求内的 `context.data["decision_result"]`，可供后续 Step 消费，不自动写入记忆文件。
SDK 错误沿用 BaseJob 的 `success=false` 契约。

在自己的 Job 中使用 `decision_demo_step`，通过 Step 配置 `as_decision_model: <组件名>` 选择命名组件。
新增供应商时，实现 `BaseAsDecisionModel` 并注册 backend；Job 按组件名选择模型。当前适配器使用 AgentScope classifier
schema 定义问题与答案。接口见 `reme/components/as_decision_model/`。`model` 为 null 时使用适配器默认模型。

工具历史压缩使用同一决策模型接口：先启动 `reme start config=decision_compaction`，再在另一终端调用
`reme compact_context`，传入 `messages` 以及需要的保护和预算选项。返回值是建议的快照，归档和应用由调用方负责。
请求 token 预算使用预留开销的估算值，与宿主模型的 tokenizer 独立。

完整策略、SVG 图解及真实模型用例见[基于决策模型的上下文压缩](./context_compaction.md)。

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
