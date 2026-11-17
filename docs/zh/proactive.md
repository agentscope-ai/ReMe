# Proactive

`proactive_read` 是 ReMe 的主动记忆读取接口。它不重新分析 daily，也不调用 LLM，只读取独立 proactive refresh
流程写出的兴趣主题：

```text
daily/<date>/interests.yaml
```

上层 Agent 可以用它获取“今天值得主动关注什么”，再决定是否提醒、追问、推荐下一步或生成主动洞察。

`interests.yaml` 由 proactive refresh writer 链路生成，默认通过 `proactive_refresh_cron` 定时执行；
`proactive_read` 只负责读取和暴露结果。Auto Dream 是独立的 daily-to-digest 流程，不读取或写入 proactive 状态。

## 配置入口

默认配置在 `reme/config/default.yaml`。同一组 refresh steps 有两个入口：用于维护和调试的本地 one-shot job，
以及按应用时区每天 18:00 执行的定时任务：

```yaml
proactive_refresh:
  backend: base
  enable_serve: false
  steps: &proactive_refresh_steps
    - backend: proactive_extract_step
      file_catalog: proactive
      scan_days: 2
      carry_forward_days: 14
      max_carry_forward_topics: 20
      llm_timeout_seconds: 300
      max_chars_per_file: 60000
      max_total_chars: 300000
    - backend: proactive_topics_step
      known_threshold: 0.85
      known_threshold_calibrated_for: text-embedding-v4@1024
      min_push_confidence: 0.5
      max_topics: 10
    - backend: proactive_plan_step
    - backend: proactive_agenda_step
    - backend: proactive_finish_step
      file_catalog: proactive

proactive_refresh_cron:
  backend: cron
  cron: "0 18 * * *"
  steps: *proactive_refresh_steps
```

`default.yaml` 也通过这个 YAML anchor 共享同一组 steps。读取 job 为：

```yaml
proactive_read:
  backend: base
  description: "Proactive: read daily/<date>/interests.yaml and expose the latest user-interest topics."
  parameters:
    type: object
    properties:
      date:
        type: string
        default: ""
      include_content:
        type: boolean
        default: true
      horizon_days:
        type: integer
        default: 1
      min_confidence:
        type: number
        default: 0.4
  steps:
    - backend: proactive_step
      min_confidence: 0.4
```

参数含义：

| 参数              | 作用                                                                        |
|-------------------|-----------------------------------------------------------------------------|
| `date`            | 要读取的日期，格式为 `YYYY-MM-DD`。为空时使用应用时区中的今天。              |
| `include_content` | 是否在 answer 和 metadata 中返回 YAML 原文，默认 `true`。                    |
| `horizon_days`    | 读取单日曝光文件，或在更宽时间窗下读取 truth source；默认 `1`。              |
| `min_confidence`  | 返回 topic 的最低置信度，默认 `0.4`；旧版 topic 使用 `0.5`。                 |

### Refresh 成本、文件与关闭方式

`metadata.proactive.duration_ms` 已移除。服务运行期间，在另一个终端使用
`reme job_status name=proactive_refresh_cron` 查询整条 Job 的开始和结束时间。

没有 daily Markdown 发生变化时，refresh 会在调用 LLM 前结束，也不会生成新的曝光文件。有变化素材时，extract
通常调用一次 LLM；回复不可用时最多重试一次。存在 push candidates 时，plan 会再调用一次；候选数大于一个时，
agenda 会再调用一次。因此默认链路单轮最多调用四次 LLM。

Refresh 会维护可重建的 truth source `daily/_proactive.yaml`，写入 `daily/<date>/interests.yaml`，并推进独立的
`proactive` file catalog。Auto Dream 不读取或写入这些 proactive 产物。

如需关闭自动 refresh，请使用不包含 `proactive_refresh_cron` job 的显式应用配置。若仍需按需维护，可保留
本地 one-shot 的 `proactive_refresh`；它设置了 `enable_serve: false`，不会暴露到 HTTP 或 MCP。

## 输入契约

当前 proactive refresh 生成的文件如下：

```yaml
version: 2
date: 2026-06-20
generated_at: 2026-06-20T18:00:00+08:00
push: true
topics:
  - id: b2120f3573cb
    title: 记忆检索链路的质量回归
    reason: 用户近期持续修改 search、node_search 和 dream 集成链路。
    kind: follow_up
    confidence: 0.86
    first_seen: 2026-06-20
    last_evidence_at: 2026-06-20
    evidence: daily/2026-06-20/session.md
    paths:
      - daily/2026-06-20/session.md
agenda:
  - topic_id: b2120f3573cb
    title: 记忆检索链路的质量回归
    scenario_type: resume_task
    opener: 下次发布前先回顾最近的检索回归。
    next_action: 对比失败查询与上一个索引快照。
    preconditions: []
    delivery: in_conversation
    linked_memory: []
    order_reason: 证据较新且下一步明确。
suppressed: []
```

当前 v2 topic 包含稳定 ID、类型、置信度、证据日期和来源路径。读取器仍兼容包含 `title`、`reason`、`evidence`、
`keywords`、`paths` 的 v1 文件；缺少 v2 置信度时按 `0.5` 处理。

## 返回结果

成功读取时，`proactive_step` 会在主要 answer 中返回 `summary` 和 `topics`；当 `include_content=true` 时还会返回
`content`。相同的结果字段也会保留在标准 response metadata 中：

| 字段      | 说明                                            |
|-----------|-------------------------------------------------|
| `date`    | 实际读取的日期。                                |
| `path`    | `daily/<date>/interests.yaml`。                 |
| `topics`  | 解析后的 topic 列表。                           |
| `content` | YAML 原文；仅在 `include_content=true` 时返回。 |
| `skipped` | 文件不存在时为 `true`。                         |
| `error`   | 读取或解析异常。                                |
| `summary` | 简短摘要。                                      |
| `agenda`  | 当日主动议程（可选，仅 v2 文件携带时返回）。    |

当当天的 `interests.yaml` 由 proactive refresh 链路生成并携带议程时，answer 会附带
`agenda` 字段：按顺序排列的当日议程条目，每条包含 `topic_id`、`title`、`scenario_type`、
`opener`（自然口吻的开场白）、`next_action`（最小可执行动作）、`preconditions`、
`delivery`、`linked_memory` 与 `order_reason`。读侧会过滤已解决或低于 `min_confidence`
的主题对应的议程条目；文件不含议程时该字段不出现。

文件存在且解析成功时，answer 是结构化数据，例如：

```json
{
  "summary": "Read 1 proactive topic(s) from daily/2026-06-20/interests.yaml",
  "topics": [
    {
      "id": "b2120f3573cb",
      "title": "记忆检索链路的质量回归",
      "reason": "用户最近反复修改了 search、node_search 和 dream integration。",
      "kind": "follow_up",
      "confidence": 0.86,
      "first_seen": "2026-06-20",
      "last_evidence_at": "2026-06-20",
      "evidence": "daily/2026-06-20/session.md",
      "paths": ["daily/2026-06-20/session.md"]
    }
  ],
  "agenda": [
    {
      "topic_id": "b2120f3573cb",
      "title": "记忆检索链路的质量回归",
      "scenario_type": "resume_task",
      "opener": "下次发布前先回顾最近的检索回归。",
      "next_action": "对比失败查询与上一个索引快照。",
      "preconditions": [],
      "delivery": "in_conversation",
      "linked_memory": [],
      "order_reason": "证据较新且下一步明确。"
    }
  ],
  "content": "date: 2026-06-20\n..."
}
```

当 `include_content=false` 时，answer 不包含 `content` 字段。文件缺失和读取失败仍分别返回明确的
`Skipped: ...` 和 `Error: ...` 消息。

文件不存在时不会报错，而是成功返回 skipped：

```text
Skipped: interests file not found at daily/2026-06-20/interests.yaml
```

这让上层 Agent 可以把“今天还没有 proactive refresh 结果”当作正常空状态处理。

## 运行方式

### 启动服务并读取 topics

在一个终端启动服务，并保持运行；默认 cron 会按配置刷新 topics：

```bash
reme start config=default
```

在另一个终端读取生成的 topics：

```bash
reme proactive_read date=2026-06-20
```

不返回 YAML 原文：

```bash
reme proactive_read date=2026-06-20 include_content=false
```

### 无服务单次 refresh

维护或调试需要立即执行 refresh 时，通过正常应用生命周期在本地运行一次：

```bash
reme start config=default job=proactive_refresh date=2026-06-20
```

`proactive_refresh` 设置了 `enable_serve: false`，不能通过 `reme proactive_refresh` 调用运行中的服务。
这条本地命令可能调用配置的 LLM，并更新 `_proactive.yaml`、`interests.yaml` 与 proactive catalog；它不会运行 Auto
Dream。完成后，通过上面的服务读取流程获取 topics。

## 与 auto_dream 的关系

Proactive refresh 和 Auto Dream 各自独立消费 daily notes：

```text
daily notes -> auto_dream -> digest
daily notes -> proactive_refresh_cron -> daily/<date>/interests.yaml -> proactive_read -> upper-level agent
```

Proactive 职责边界如下：

| 模块                     | 职责                                         |
|--------------------------|----------------------------------------------|
| `proactive_refresh`      | 从本地 CLI 单次运行 refresh writer 链路。    |
| `proactive_refresh_cron` | 每天 18:00 运行同一套 writer 链路。          |
| `proactive_step`         | 读取 `interests.yaml`，暴露给上层 Agent。    |

`proactive_read` 不修改任何文件，不更新 catalog，也不负责判断是否应该主动打扰用户。它只提供当天主题材料；是否推送、何时推送、用什么语气推送，应由调用方根据产品策略决定。

## 失败模式

| 场景                       | 行为                                          |
|----------------------------|-----------------------------------------------|
| `interests.yaml` 不存在    | `success=true`，`skipped=true`，`topics=[]`。 |
| YAML 无法读取或解析异常    | `success=false`，answer 返回错误摘要。        |
| YAML 存在但没有合法 topics | `success=true`，`topics=[]`。                 |

因此推荐调用方先检查 `success`，再检查 `skipped`，最后检查 `topics` 是否为空。
