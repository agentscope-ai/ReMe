# Proactive

`proactive` 是 ReMe 的主动记忆读取接口。它不重新分析 daily，也不调用 LLM，只读取 `auto_dream` 写出的当天兴趣主题：

```text
daily/<date>/interests.yaml
```

上层 Agent 可以用它获取“今天值得主动关注什么”，再决定是否提醒、追问、推荐下一步或生成主动洞察。

`interests.yaml` 由 [Auto Dream](./auto_dream.md) 的 Topics 阶段生成；`proactive` 只负责读取和暴露结果。

## 配置入口

默认配置在 `reme/config/default.yaml`：

```yaml
proactive:
  backend: base
  description: "Proactive: read daily/<date>/interests.yaml and expose the latest user-interest topics."
  parameters:
    date:
      type: string
      default: ""
    include_content:
      type: boolean
      default: true
  steps:
    - backend: proactive_step
```

参数含义：

| 参数                | 作用                                     |
|-------------------|----------------------------------------|
| `date`            | 要读取的日期，格式为 `YYYY-MM-DD`。为空时使用应用时区中的今天。 |
| `include_content` | 是否在 metadata 中返回 YAML 原文，默认 `true`。    |

## 输入契约

典型格式如下：

```yaml
date: 2026-06-20
topic_count: 3
diversity_days: 7
topics:
  - title: 记忆检索链路的质量回归
    reason: 用户近期持续修改 search、node_search 和 dream 集成链路。
    evidence: daily/2026-06-20/session.md
    keywords:
      - memory search
      - auto dream
    paths:
      - daily/2026-06-20/session.md
```

只有 `topics` 列表会被解析成结构化结果。每个 topic 至少需要 `title` 和 `reason`；`evidence`、`keywords`、`paths` 是辅助字段。

## 返回结果

成功读取时，`proactive_step` 会把结果写入标准 response metadata：

| 字段        | 说明                                     |
|-----------|----------------------------------------|
| `date`    | 实际读取的日期。                               |
| `path`    | `daily/<date>/interests.yaml`。         |
| `topics`  | 解析后的 topic 列表。                         |
| `content` | YAML 原文；仅在 `include_content=true` 时返回。 |
| `skipped` | 文件不存在时为 `true`。                        |
| `error`   | 读取或解析异常。                               |
| `summary` | 简短摘要。                                  |

文件存在且解析成功时，answer 类似：

```text
Read 3 proactive topic(s) from daily/2026-06-20/interests.yaml
```

文件不存在时不会报错，而是成功返回 skipped：

```text
Skipped: interests file not found at daily/2026-06-20/interests.yaml
```

这让上层 Agent 可以把“今天还没有 dream 结果”当作正常空状态处理。

## 运行方式

CLI：

```bash
reme proactive date=2026-06-20
```

不返回 YAML 原文：

```bash
reme proactive date=2026-06-20 include_content=false
```

## 与 auto_dream 的关系

`proactive` 是 `auto_dream` 的下游读取步骤：

```text
daily notes
  -> auto_dream
  -> daily/<date>/interests.yaml
  -> proactive
  -> upper-level agent
```

职责边界如下。更完整的 Extract、Integrate、Topics、Finish 说明见 [Auto Dream](./auto_dream.md)：

| 模块                   | 职责                                     |
|----------------------|----------------------------------------|
| `dream_extract_step` | 从 changed daily 输入抽取 topic candidates。 |
| `dream_topics_step`  | 去重、筛选并写入 `interests.yaml`。             |
| `proactive_step`     | 读取 `interests.yaml`，暴露给上层 Agent。       |

`proactive` 不修改任何文件，不更新 catalog，也不负责判断是否应该主动打扰用户。它只提供当天主题材料；是否推送、何时推送、用什么语气推送，应由调用方根据产品策略决定。

## 失败模式

| 场景                   | 行为                                         |
|----------------------|--------------------------------------------|
| `interests.yaml` 不存在 | `success=true`，`skipped=true`，`topics=[]`。 |
| YAML 无法读取或解析异常       | `success=false`，answer 返回错误摘要。             |
| YAML 存在但没有合法 topics  | `success=true`，`topics=[]`。                |

因此推荐调用方先检查 `success`，再检查 `skipped`，最后检查 `topics` 是否为空。
