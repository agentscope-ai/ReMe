# Proactive

`proactive_read` 是 ReMe 的主动记忆读取接口。它不重新分析 daily，也不调用 LLM，只读取独立 proactive refresh
流程写出的兴趣主题：

```text
daily/<date>/interests.yaml
```

上层 Agent 可以用它获取“今天值得主动关注什么”，再决定是否提醒、追问、推荐下一步或生成主动洞察。

每个 topic 和议程条目都带有 `kind`，取值固定为四类之一——`reminder`、`follow_up`、`new_idea`、`interest_extend`
（见下文“提议类型”）。四类都停留在“提议”：refresh 链路不会执行其中任何一件，只记录“值得对用户提起什么”。

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
      profile_max_chars: 1500
      capability_max_chars: 1200
      extends_enabled: true
      reminders_enabled: true
      new_ideas_enabled: true
      max_follow_ups: 5
      max_extends: 5
      max_new_ideas: 3
      reminder_lead_days: 3
      resources_enabled: true
      resource_max_chars: 40000
      max_notice_reminders: 3
      refine_enabled: true
      refine_timeout_seconds: 150
    - backend: proactive_topics_step
      known_threshold: 0.85
      known_threshold_calibrated_for: text-embedding-v4@1024
      min_push_confidence: 0.5
      min_reminder_confidence: 0.3
      max_topics: 10
      schedule_retention_days: 180
      max_schedules: 80
      max_schedule_history: 8
    - backend: proactive_plan_step
      max_reminder_cards: 8
    - backend: proactive_agenda_step
      min_discovery_slots: 2
      redelivery_cooldown_days: 3
    - backend: proactive_finish_step
      file_catalog: proactive

proactive_refresh_cron:
  backend: cron
  cron: "0 18 * * *"
  steps: *proactive_refresh_steps
```

上面的 anchor 只是为了紧凑展示；`default.yaml` 中显式写出了两组 steps。读取 job 为：

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
      include_delivered:
        type: boolean
        default: false
      delivery_cooldown_days:
        type: integer
        default: 3
  steps:
    - backend: proactive_step
      min_confidence: 0.4
      delivery_cooldown_days: 3
```

参数含义：

| 参数              | 作用                                                                        |
|-------------------|-----------------------------------------------------------------------------|
| `date`            | 要读取的日期，格式为 `YYYY-MM-DD`。为空时使用应用时区中的今天。              |
| `include_content` | 是否在 answer 和 metadata 中返回 YAML 原文，默认 `true`。                    |
| `horizon_days`    | 读取单日曝光文件，或在更宽时间窗下读取 truth source；默认 `1`。              |
| `min_confidence`  | 返回 topic 的最低置信度，默认 `0.4`；旧版 topic 使用 `0.5`。                 |
| `include_delivered`    | 是否连宿主已上报送达的议程条目一起返回；默认 `false`（扣留）。      |
| `delivery_cooldown_days` | 已送达条目的扣留天数，`0` 表示只扣留当天的送达；默认 `3`。       |

### Refresh 成本、文件与关闭方式

没有 daily Markdown 变化、没有新的外部通知、且没有到期日程时，refresh 会在调用 LLM 前结束，也不会生成新的曝光
文件。只有时钟算出到期提醒时，链路仍会零 LLM 调用地走完（`reminder_only=true`），保证当日议程拿到提醒。有变化
素材时，阶段 1 召回调用一次 LLM（回复不可用时最多重试一次），阶段 2 再对每个真正产出了候选的类型各调用一次小请求
（至多四次）；存在 push candidates 时 plan 会再调用一次，候选数大于一个时 agenda 会再调用一次。因此默认链路单轮
最多九次 LLM 调用，其中只有阶段 1 携带素材 blob。

Refresh 会维护可重建的 truth source `daily/_proactive.yaml`（open topics、resolved 墓碑，以及长生命周期的
`schedules` 日程注册表与 `deliveries` 投递记录），写入 `daily/<date>/interests.yaml`，并推进独立的
`proactive` file catalog。Auto Dream 不读取或写入这些 proactive 产物。

如需关闭自动 refresh，请使用不包含 `proactive_refresh_cron` job 的显式应用配置。若仍需按需维护，可保留
本地 one-shot 的 `proactive_refresh`；它设置了 `enable_serve: false`，不会暴露到 HTTP 或 MCP。

## 提议类型（Proposal Kinds）

extract 只发起一次召回调用，要求模型返回四个分节的提议。每类职责不同，且都不是必需项：单轮可以产出其中任意子集，
也可以一条都没有。

| 类型              | 职责                                                            | 对应场景卡                       | 单轮上限                                             |
|-------------------|-----------------------------------------------------------------|----------------------------------|------------------------------------------------------|
| `reminder`        | 用户自己的计划与日程，以及外部信源中必须告知用户的通知与关键事件 | `prepare_upcoming`               | 不设上限；无日期通知受 `max_notice_reminders`（3）约束 |
| `follow_up`       | 用户指派但未闭环、或持续关心事项的最近进展                       | `resume_task` / `answer_pending` | `max_follow_ups`（5）                                |
| `new_idea`        | 基于 agent 自身资源（数据、skill、工具与 API）可委托给用户的工作  | `propose_delegation`             | `max_new_ideas`（3）                                 |
| `interest_extend` | 用户关心主题的细化与延伸内容（扩展 wiki 网络节点）               | `explore_interest`               | `max_extends`（5）                                   |

`reminder` 优先级高于其余三类：agenda 步骤把到期日程**固定**在当日议程最前（按到期时间由近到远），既不会被压制也不会
被交换，发现类条目则始终保有 `min_discovery_slots`（2）个名额，因此日程排满的一天也不会退化成一张纯日历。提醒候选在
topics 步骤走更低门槛（`min_reminder_confidence` 0.3），并在 `proactive_read` 中豁免 `min_confidence`。

提醒是对照真实时钟（应用配置时区）算出来的，而不是只看当天的 daily note：extract 在任何 LLM 决策之前先扫描
`schedules` 注册表，确定性地预置 `overdue`、`due_today`、`conflict` 与 `due_soon`（`reminder_lead_days` 之内）四类
触发。`resource/<date>/` 下的外部通知走同一条时钟通道，因此接入新的外部信源不需要改 proactive 链路。

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
| `delivered_ids` | 本轮因送达冷却被扣留的议程 id（`include_delivered=false` 时）。 |

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

## 送达回执（Delivery Receipts）

refresh 只决定“什么值得提”，它无法知道宿主究竟提了什么。`proactive_feedback` 用来闭合这个环，避免消费方已经展示
过的条目明天又被提议一遍：

```bash
reme proactive_feedback date=2026-06-22 items='[{"topic_id":"<id>","status":"shown","channel":"inbox","consumer":"<host label>"}]'
```

每条回执接受 `topic_id`（唯一必填）、`status`、`channel`、`consumer`、`outcome`、`external_ref` 与 `agenda_date`；
`status` 取 `shown`（默认）、`executed`、`dismissed`、`failed`，`channel` 取 `in_conversation`、`inbox`、
`notification`、`other`（默认）。无法识别的取值回落到默认值而不是让调用失败。回执写入 truth source
`daily/_proactive.yaml` 的 `deliveries` 段，并按 `retention_days`（180）与 `max_deliveries`（500）裁剪。

- **身份**：宿主给了 `external_ref` 时以它为准，否则用 `(topic_id, agenda_date, channel)`；同一次送达重复上报会原地
  更新，因此可以先报 `shown`、之后再报 `executed`，不会产生第二条。
- **状态只前进**：已终结的记录（`executed`、`dismissed`）不会被迟到的 `shown` 降级回去；`failed` 不阻止下一轮重新投递。
- **两个读取方**：`proactive_step` 扣留仍在 `delivery_cooldown_days`（3）内的议程条目，`proactive_agenda_step` 则
  确定性压制这些候选并把可审计的理由写进 `suppressed`。`kind == "reminder"` 在两侧都豁免冷却——它的 `topic_id` 由
  稳定的日程标题派生，照常压制会让一条周期性提醒在投递一次之后就在整个窗口内静默。
- **可选契约**：从不上报的宿主会让 `deliveries` 保持为空，两个读取方的行为与之前完全一致。该步骤不调用 LLM，也不改
  `open_topics`/`resolved`/`schedules`。

读取方把自己扣留的条目以 `delivered_ids` 返回（`answer` 与 `metadata` 里都有），消费方用它与自己的本地台账对账。

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
