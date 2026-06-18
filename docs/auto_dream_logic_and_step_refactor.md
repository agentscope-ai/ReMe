# auto_dream 逻辑解读与 Step 拆分方案

## 1. 配置入口

`/Users/yuli/workspace/ReMe/reme/config/default.yaml` 里和 auto dream 相关的是四个 job:

| job | 用途 | 当前 steps |
|---|---|---|
| `dream` | 对单个文件做完整 dream,并顺手写一次 daily topics | `dream_step` -> `daily_topics_step` |
| `dream_extract` | auto_dream 内部使用的单文件 dream,只抽取和整合 digest,不写 daily topics | `dream_step` |
| `auto_dream` | 扫描某一天的 daily index 和 session notes,只处理新增/修改文件,最后聚合写当天兴趣主题 | `auto_dream_step` |
| `daily_topics` | 从 dream 产生的 topic candidates 中选最终兴趣主题,写 `daily/<date>/interests.md` | `daily_topics_step` |

关键点:

- `dream` 和 `auto_dream` 不是同一条链路。
- `dream` 是单文件命令,执行 `dream_step` 后立刻执行 `daily_topics_step`。
- `auto_dream` 默认 `dispatch_job: dream_extract`,所以它对每个文件只跑 `dream_step`,把所有文件产生的 `topic_candidates` 收集起来,最后只调用一次 `daily_topics`。
- `auto_dream` 默认写 3 个 topic,回看 7 天去重,输出 session id 是 `interests`。

配置片段的实际语义:

```yaml
auto_dream:
  steps:
    - backend: auto_dream_step
      dispatch_job: dream_extract
      emit_topics: true
      topic_dispatch_job: daily_topics
      topic_count: 3
      topic_diversity_days: 7
      topic_session_id: interests
```

也就是说,`auto_dream` 本身是一个调度器和增量扫描器,真正的 LLM dream 逻辑在 `DreamStep`,topic 写入在 `DailyTopicsStep`。

## 2. auto_dream_step 执行链路

实现位置:

- `reme/steps/evolve/auto_dream.py`
- `reme/steps/evolve/dream.py`
- `reme/steps/evolve/daily_topics.py`
- `reme/steps/file_io/_daily_index.py`

### 2.1 读取输入和默认日期

`AutoDreamStep.execute()` 从 runtime context 读取:

| 参数 | 语义 |
|---|---|
| `date` | 要扫描的日期,空字符串时用配置 timezone 下的今天 |
| `hint` | 透传给每个单文件 dream 的提示 |

日期默认逻辑:

```text
date_input 非空 -> 使用 date_input
date_input 为空 -> now(app_config.timezone).strftime("%Y-%m-%d")
```

`daily_dir` 不来自用户参数,而是来自 app config,默认是 `daily`。

### 2.2 先刷新 day-index

正式扫描前,auto_dream 会先调用:

```python
await refresh_day_index(self.file_store, today, daily_dir)
```

它会重建:

```text
daily/<date>.md
```

这个 day-index 文件包含 `daily/<date>/` 下每个 session note 的链接和 frontmatter 摘要。这样 auto_dream 后续处理的第一个文件就是当天总览。

隐含结果:

- 如果 session notes 有新增/删除/frontmatter 变化,day-index 的内容可能变化。
- day-index 被放在扫描列表第一位,所以当天总览先于具体 session note 被 dream。

### 2.3 扫描当天文件范围

扫描范围由 `_scan_today_files(vault, today, daily_dir)` 决定:

```text
1. daily/<date>.md
2. daily/<date>/**/*.md
```

处理顺序:

```text
daily/<date>.md first
daily/<date>/**/*.md sorted by path
```

但 auto_dream 会排除:

```text
daily/<date>/interests.md
```

也就是 `topic_session_id` 对应的 daily topics 文件。原因是 `interests.md` 是 auto_dream 自己产出的兴趣主题,不能再作为 dream 输入,否则容易自我循环。

### 2.4 用 file_catalog 做增量判断

auto_dream 构造两张表:

| 名称 | 来源 | 内容 |
|---|---|---|
| `existing` | 当前磁盘 | `{vault_relative_path: st_mtime}` |
| `indexed` | `file_catalog.get_nodes()` | `{vault_relative_path: st_mtime}` |

`indexed` 只保留当天范围:

```text
daily/<date>.md
daily/<date>/*
```

并且同样排除:

```text
daily/<date>/interests.md
```

然后做 diff:

| 条件 | 分类 | 行为 |
|---|---|---|
| `rel in existing`,但 catalog 没有 | added | 需要 dream |
| `rel in existing`,但 mtime 不同 | modified | 需要 dream |
| `rel in existing`,且 mtime 相同 | unchanged | 跳过 |
| catalog 有,但磁盘没有 | deleted | 从 catalog 删除 |

注意这里的 `file_catalog` 更像是 auto_dream 的“已处理 mtime 水位线”,不是语义索引本身。默认配置没有给 `auto_dream_step` 显式传 `file_catalog`,所以按 `BaseStep.Ref` 规则解析到 `file_catalog.default`。

### 2.5 先删除 catalog 中的缺失文件

如果某些当天文件已经不存在:

```python
await self.file_catalog.delete(to_delete)
```

这一步不需要 LLM,也不会阻塞后续 dream。删除失败只记录日志,当前实现不会把它计入 `result.files_failed`。

### 2.6 对新增/修改文件逐个 dispatch dream_extract

对每个 `to_dream` 文件,auto_dream 调:

```python
resp = await self.run_job("dream_extract", path=rel_path, hint=hint)
```

`dream_extract` 只有一个 step:

```yaml
steps:
  - backend: dream_step
```

`AutoDreamStep._dispatch_dream()` 会把 `resp.metadata` 重新校验成 `DreamResult`。如果 job 抛异常、metadata 不是 `DreamResult`、或 response `success=False`,都会转成带 `error` 的 `DreamResult`。

### 2.7 单文件 DreamStep 内部逻辑

`DreamStep.dream_one(path, hint)` 是真正的 per-file create_or_update。

它的主流程:

```text
1. path 为空 -> skipped
2. 没有 LLM -> error
3. _pack_material() 读取 vault-relative 文件内容
4. Phase 1: _extract()
5. 如果 Phase 1 没有 units -> skipped,但保留 topic_candidates
6. Phase 2: 对每个 unit 调 _integrate_unit()
7. 返回 DreamResult
```

#### Phase 1: extract

工具:

```python
_EXTRACT_TOOLS = ("read",)
```

输出 schema 是 `ExtractedUnits`:

```text
units: list[MemoryUnit]
topic_candidates: list[TopicCandidate]
```

每个 memory unit 包含:

| 字段 | 语义 |
|---|---|
| `name` | agent 内部短名 |
| `bucket` | `procedure` / `personal` / `wiki` 三选一 |
| `summary` | 这个抽象是什么,证据在哪 |

每个 topic candidate 包含:

| 字段 | 语义 |
|---|---|
| `title` | 兴趣主题标题 |
| `reason` | 为什么用户可能关心 |
| `evidence` | 证据指针 |
| `keywords` | 去重关键词 |

如果 LLM 输出了未知 bucket,当前代码会警告并改成 `wiki`。

#### Phase 2: integrate

每个 unit 单独发起一次 ReAct:

```text
system prompt = integrate_system_prompt_<bucket>
```

工具:

```python
_INTEGRATE_TOOLS = (
  "node_search",
  "read",
  "frontmatter_read",
  "write",
  "edit",
  "frontmatter_update",
)
```

输出 schema 是 `IntegrateOutcome`:

| 字段 | 语义 |
|---|---|
| `action` | `CREATE` / `CORROBORATE` / `REFINE` / `CORRECT` |
| `target_path` | 实际写入或更新的 digest path |
| `note` | 简短说明 |

`DreamStep` 根据 action 统计:

```text
CREATE -> nodes_created
其他 action -> nodes_updated
```

当前实现里,某个 unit 的 integrate 失败不会让整个 DreamResult 变成 error,只会在 summary 里记录 `FAILED`。这意味着文件级别仍会被 auto_dream 当作成功并写入 catalog mtime。

### 2.8 汇总 per-file 结果并更新 catalog

auto_dream 对每个文件的 `DreamResult` 做三件事:

| 情况 | 行为 |
|---|---|
| `dr.error` 非空 | `files_failed += 1`,不更新这个文件的 catalog mtime,下次会重试 |
| `dr.skipped` 为 true | `files_skipped += 1`,仍然 upsert mtime,避免每次重复跑 Phase 1 |
| 正常 dream | `files_dreamed += 1`,upsert mtime |

同时收集:

```python
topic_candidates.extend(dr.topic_candidates or [])
```

最后批量:

```python
await self.file_catalog.upsert(upsert_nodes)
```

### 2.9 聚合写 daily topics

如果:

```text
emit_topics == true
topic_candidates 非空
```

auto_dream 会调用:

```python
await self.run_job(
  "daily_topics",
  date=today,
  candidates=topic_candidates,
  topic_count=3,
  diversity_days=7,
  session_id="interests",
)
```

`DailyTopicsStep` 做:

```text
1. 清洗 candidates
2. 读取过去 diversity_days 天的 interests.md
3. 有 LLM 时用 select prompt 选最终 topics
4. 没有 LLM 时 fallback: 简单标题去重
5. 写 daily/<date>/interests.md
6. refresh_day_index()
```

写出的文件形态:

```text
daily/<date>/interests.md
```

frontmatter 包含:

```yaml
name: interests
description: "<n> interest topic(s) inferred for <date>."
date: <date>
topic_count: 3
diversity_days: 7
```

body 是 `# Interested Topics` 加编号列表。

auto_dream 收到 daily_topics 成功响应后,还会把这些文件的最新 mtime 写入 catalog:

```text
daily/<date>/interests.md
daily/<date>.md
```

这里有一个隐含行为:day-index 在 per-file dream 之后又因为 `interests.md` 被写入而刷新,auto_dream 会把刷新后的 `daily/<date>.md` mtime 标记为已处理。也就是说,仅由 interests 写入引发的 day-index 变化不会在下一轮再次触发 dream。

### 2.10 持久化与响应

如果:

```text
persist == true
并且有 upsert 或 delete
```

则:

```python
await self.file_catalog.dump()
```

最终 response:

```text
success = files_failed == 0 and not topics_error
answer = AutoDreamResult.summary
metadata = AutoDreamResult.model_dump()
```

summary 格式大致是:

```text
[AutoDreamStep] date=2026-06-18 scanned=... unchanged=... dreamed=... skipped=... failed=... deleted=...
  - daily/2026-06-18.md: OK (+1 created, ~2 updated)
  - daily/2026-06-18/session.md: SKIP
  - topics: OK (3 written to daily/2026-06-18/interests.md)
```

## 3. 当前逻辑的边界和风险

### 3.1 AutoDreamStep 职责过重

`AutoDreamStep` 同时负责:

- 日期解析
- day-index 刷新
- 文件扫描
- catalog diff
- 删除 catalog
- per-file job dispatch
- DreamResult 校验
- topic candidates 汇总
- daily_topics job dispatch
- topic 输出后的 catalog upsert
- catalog dump
- summary 渲染

这些职责可以拆成明确 step,提高可测试性和可替换性。

### 3.2 file_catalog 的语义不够显式

这里的 catalog 不是“今天有哪些文件”的普通目录索引,而是“哪些文件已经被 auto_dream 处理到某个 mtime”。建议在拆分后把它显式命名为 dream catalog / processed catalog,至少在 step 名和文档中说清楚。

### 3.3 integrate unit 失败不会触发文件重试

`DreamStep` 当前捕获单个 unit integrate 异常,写进 summary 后继续,但不设置 `DreamResult.error`。auto_dream 因此会把这个文件 mtime upsert,下次不会自动重试失败 unit。

这可能是有意的“尽量前进”,但如果要做严格一致性,应改成:

```text
任一 unit integrate 失败 -> DreamResult.error 非空 -> auto_dream 不更新 mtime
```

### 3.4 topic 写入导致的 day-index 变化被标记为已处理

auto_dream 写 `interests.md` 后刷新 day-index,并把 day-index 最新 mtime upsert 到 catalog。这样可以避免自生成内容触发循环,但也意味着 `daily/<date>.md` 中新增的 `interests.md` 链接不会被 dream。

这通常是合理的,因为 `interests.md` 本身被排除在 dream 输入之外。

### 3.5 删除 catalog 失败不影响 success

删除 catalog entry 失败只打日志,不会让 response failure。拆分后可以明确这个策略:

- catalog delete 是 best-effort,不影响 dream 主流程
- 或者 catalog delete 失败应导致整个 job failure

## 4. 拆分目标

重新拆分时,不按“每个小动作一个 step”拆,而按执行边界拆:

```text
非 LLM 准备/扫描/diff
  -> LLM: per-file dream
  -> LLM: daily topics
  -> 非 LLM response 汇总
```

核心原则:

- 使用 LLM 的阶段单独成 step,便于限流、重试、观测和替换模型。
- 不使用 LLM 的准备、扫描、diff 可以合并,避免 step 过碎。
- catalog 只是 auto_dream 的内部进度水位线,不提升为独立阶段。
- prompt 重新写,但可以复用旧 prompt 的核心内容和约束。
- 不做旧接口/旧格式兼容;按新 4-step pipeline 重新定义最干净的输入输出。

## 5. 新方案:拆成 4 个 Step

新方案不再是“逐文件 extract + 逐文件 integrate”。核心变化是:

```text
本轮 changed files
  -> 1 个 agent 一次性阅读所有 changed paths
  -> 输出全局去重/合并后的 unit list
  -> Python for 循环逐 unit integrate
```

这样一个抽象可能来自多个文件,Phase 1 就能合并为同一个 unit,避免同一天多个 session note 反复提出同一概念。

目标代码位置:

```text
reme/steps/evolve/dream/
  __init__.py
  models.py
  plan.py
  extract.py
  integrate.py
  topics.py
  finish.py
  prompts.yaml 或 dream.yaml
```

重构完成后删除旧文件,不保留兼容 alias:

```text
reme/steps/evolve/dream.py
reme/steps/evolve/auto_dream.py
reme/steps/evolve/daily_topics.py
```

### 5.0 Step 输入输出总表

| Step | 是否 LLM | 核心能力 | 输入 | 输出 / 写入 context | 副作用 |
|---|---:|---|---|---|---|
| `dream_extract_step` | 是 | 根据 dream catalog 找出本轮新增/修改/删除文件,把所有 changed paths 交给一个 agent,一次性输出跨文件合并后的 `unit_list` 和 `topic_list` | `context.date`; `context.hint`; `app_config.daily_dir`; `app_config.timezone`; `file_store.vault_path`; `file_catalog.dream`; step 参数 `topic_session_id=interests` | `dream.date`; `dream.hint`; `dream.daily_dir`; `dream.vault`; `dream.existing`; `dream.indexed`; `dream.changed_paths`; `dream.deleted_paths`; `dream.units`; `dream.topics`; `dream.extract_summary`; `dream.result.files_scanned/files_changed/files_deleted`; `dream.errors` | 刷新 `daily/<date>.md`; 删除 catalog 中缺失文件 entry; 读取所有 changed files; 本 step 不写 digest |
| `dream_integrate_step` | 是 | `for unit in units` 逐个执行原 Phase 2 integrate 逻辑,保持 node_search/read/write/edit/frontmatter_update 工具和 bucket prompt 不变 | `dream.units`; `dream.hint`; `dream.vault`; `app_config.digest_dir`; agent tools: `node_search/read/frontmatter_read/write/edit/frontmatter_update` | `dream.integrate_results`; `dream.nodes_created`; `dream.nodes_updated`; `dream.result.units_integrated/units_failed`; `dream.errors` | 写/更新 `digest/<bucket>/*.md`; 不更新 dream catalog |
| `dream_topics_step` | 是 | 根据 `topic_list` 更新 `daily/<date>/interests.yaml`; 读取当天已有 topics 和最近 N 天 topics 做去重 | `dream.date`; `dream.daily_dir`; `dream.topics`; `file_store.vault_path`; step 参数 `topic_count`; `topic_diversity_days`; `topic_session_id=interests` | `dream.topics_path=daily/<date>/interests.yaml`; `dream.topics_written`; `dream.topics_merged`; `dream.topics_skipped_duplicates`; `dream.errors` | 新建或更新 `daily/<date>/interests.yaml`; 刷新 `daily/<date>.md` |
| `dream_finish_step` | 否 | 统一收口:按 path checkpoint 成功处理的文件,持久化 catalog,渲染 summary 和 response metadata | `dream.changed_paths`; `dream.deleted_paths`; `dream.failed_paths`; `dream.integrate_results`; `dream.topics_path`; `dream.errors`; `dream.result`; `file_catalog.dream`; step 参数 `persist=true` | `context.response.success`; `context.response.answer`; `context.response.metadata` | upsert successful paths 的 mtime 到 `file_catalog.dream`; upsert `interests.yaml` 和 day-index mtime; `file_catalog.dump()` |

### Step 1: `dream_extract_step`

这是新的全局 Phase 1。它合并了当前 `auto_dream_step` 的扫描/diff 和当前 `DreamStep._extract()` 的抽取能力。

职责:

- 解析 `date` / `hint`。
- 刷新 day-index: `daily/<date>.md`。
- 扫描输入文件并统一交给 extract agent:
  - `daily/<date>.md`
  - `daily/<date>/<session_id>.md`
  - `daily/<date>/<resource_stem>.md`
  - 以及 `daily/<date>/**/*.md` 下其它当天 note
- 排除自生成文件:
  - `daily/<date>/interests.yaml`
- 读取 `file_catalog.dream`,按 mtime diff 出:
  - `changed_paths`
  - `unchanged_paths`
  - `deleted_paths`
- 删除 catalog 中 `deleted_paths`。
- 打包所有 `changed_paths` 的文件内容。
- 调用一次 extract agent,让它看见所有 changed paths。
- 输出跨文件合并后的 `unit_list` 和 `topic_list`。

新的 unit schema:

```python
class DreamUnit(BaseModel):
    name: str
    bucket: Literal["procedure", "personal", "wiki"]
    summary: str
    paths: list[str]
```

`paths` 是这个 unit 的证据来源列表。多个文件讲的是同一抽象时,Phase 1 必须合并成一个 unit:

```json
{
  "name": "jwt-session-expiry-policy",
  "bucket": "procedure",
  "summary": "How the project decides session expiry from compliance and product constraints.",
  "paths": [
    "daily/2026-06-18/auth.md",
    "daily/2026-06-18/api-review.md"
  ]
}
```

topic schema 可以沿用当前 `TopicCandidate`,但建议把来源改成 `paths`:

```python
class DreamTopicCandidate(BaseModel):
    title: str
    reason: str
    evidence: str
    keywords: list[str] = []
    paths: list[str] = []
```

输出到 context:

```python
{
  "dream": {
    "date": "YYYY-MM-DD",
    "changed_paths": [
      {"path": "daily/YYYY-MM-DD/a.md", "mtime": 1710000000.0}
    ],
    "deleted_paths": [],
    "units": [
      {
        "name": "jwt-session-expiry-policy",
        "bucket": "procedure",
        "summary": "...",
        "paths": ["daily/YYYY-MM-DD/a.md", "daily/YYYY-MM-DD/b.md"]
      }
    ],
    "topics": [
      {
        "title": "...",
        "reason": "...",
        "evidence": "...",
        "keywords": ["..."],
        "paths": ["daily/YYYY-MM-DD/a.md"]
      }
    ]
  }
}
```

LLM 调用数量:

```text
1 个 agent 任务
```

注意:

- 这个 step 不再为每个文件分别调用 `dream_extract`。
- 如果 `changed_paths` 为空,它不调用 LLM,直接输出空 `units/topics`。
- 只有 `dream_finish_step` 才把 changed file mtime 标为已处理。这样 integrate/topics 失败时不会误跳过。

### Step 2: `dream_integrate_step`

这是新的全局 Phase 2。它对 Step 1 输出的 `units` 做 Python for 循环,每个 unit 的 integrate 逻辑保持当前 `DreamStep._integrate_unit()` 不变。

职责:

- 遍历 `dream.units`。
- 每个 unit 根据 `unit.bucket` 选择:
  - `integrate_system_prompt_procedure`
  - `integrate_system_prompt_personal`
  - `integrate_system_prompt_wiki`
- material 不再是单文件 blob,而是这个 unit 对应 `paths` 的证据包。
- 调用当前相同工具:
  - `node_search`
  - `read`
  - `frontmatter_read`
  - `write`
  - `edit`
  - `frontmatter_update`
- 输出 `IntegrateOutcome`。

`integrate_user_message` 需要从单 `material_blob` 改成多路径 evidence blob:

```text
unit_name: ...
unit_bucket: ...
unit_summary: ...
source_paths:
  - daily/...
  - daily/...

# Evidence materials
### daily/.../a.md
...

### daily/.../b.md
...
```

输出:

```python
{
  "dream": {
    "integrate_results": [
      {
        "unit_name": "jwt-session-expiry-policy",
        "bucket": "procedure",
        "action": "CREATE",
        "target_path": "digest/procedure/jwt-session-expiry-policy.md",
        "source_paths": ["daily/YYYY-MM-DD/a.md", "daily/YYYY-MM-DD/b.md"],
        "note": "..."
      }
    ],
    "nodes_created": ["digest/procedure/jwt-session-expiry-policy.md"],
    "nodes_updated": []
  }
}
```

LLM 调用数量:

```text
N 个 agent 任务
N = len(dream.units)
```

失败策略建议:

- 任一 unit integrate 失败,记录到 `dream.errors`。
- 因为每个 unit 都有明确的 `paths`,失败 unit 对应的 paths 进入 `dream.failed_paths`。
- `dream_finish_step` 不 checkpoint `failed_paths`。
- 不在任何失败 unit `paths` 里的 changed paths 可以 checkpoint。
- 如果同一个 path 同时出现在成功 unit 和失败 unit 中,以失败为准,该 path 不 checkpoint。

### Step 3: `dream_topics_step`

这个 step 取代当前 `daily_topics_step`。目标文件固定为:

```text
daily/<date>/interests.yaml
```

职责:

- 读取 `dream.topics`。
- 如果 `daily/<date>/interests.yaml` 已存在,读取旧 topics。
- 读取最近 `topic_diversity_days` 天的 `daily/<previous-date>/interests.yaml` 作为历史去重上下文。
- 合并当天旧 topics + 新 topics。
- 去重:
  - 标题 normalize 后相同视为重复。
  - keywords 高重叠视为可能重复。
  - evidence/paths 完全相同视为重复。
  - 与最近 N 天历史 topics 重复时跳过。
- 可选使用 LLM 对候选 topic 做最终选择和改写。
- 写回 YAML。
- 刷新 day-index。

建议 YAML 格式:

```yaml
date: "2026-06-18"
updated_at: "2026-06-18T22:00:00+08:00"
topic_count: 3
diversity_days: 7
topics:
  - title: "JWT session expiry policy"
    reason: "The user repeatedly worked through compliance-driven auth expiry tradeoffs."
    evidence: "Mentioned in auth review and API notes."
    keywords: ["auth", "jwt", "session", "compliance"]
    paths:
      - "daily/2026-06-18/auth.md"
      - "daily/2026-06-18/api-review.md"
```

输入:

```text
dream.date
dream.daily_dir
dream.topics
topic_count
topic_diversity_days
topic_session_id
```

输出:

```python
{
  "dream": {
    "topics_path": "daily/YYYY-MM-DD/interests.yaml",
    "topics_written": 3,
    "topics_merged": 5,
    "topics_skipped_duplicates": 2
  }
}
```

LLM 调用数量:

```text
0 或 1 个 agent 任务
```

建议:

- 如果只是 append/去重,不必 LLM。
- 如果需要从很多 candidates 中挑 `topic_count` 个,才调用 LLM。
- 不读取也不写 `interests.md`;全新格式只认 `interests.yaml`。

### Step 4: `dream_finish_step`

这是非 LLM 收尾 step。

职责:

- 根据前面步骤结果决定 success。
- 计算 `failed_paths`:
  - 每个失败 unit 的 `unit.paths` 都进入 failed set。
  - 如果某个 path 同时属于成功 unit 和失败 unit,以失败为准。
- 计算 `checkpoint_paths`:
  - `changed_paths - failed_paths`
  - extract 成功但没有任何 unit/topics 的 changed paths 也可以 checkpoint,避免重复空跑。
- 把 `checkpoint_paths` 的当前 mtime upsert 到 `file_catalog.dream`。
- 把 `daily/<date>/interests.yaml` 的 mtime upsert 到 `file_catalog.dream`。
- 把刷新后的 `daily/<date>.md` 的 mtime upsert 到 `file_catalog.dream`。
- `deleted_paths` 的 catalog 删除在 extract step 已完成,finish 只负责 dump。
- `file_catalog.dump()`。
- 渲染 summary。
- 写 `context.response.metadata`。

输出 metadata 建议:

```python
{
  "date": "YYYY-MM-DD",
  "files_scanned": 10,
  "files_changed": 3,
  "files_deleted": 1,
  "paths_checkpointed": ["daily/2026-06-18/a.md"],
  "paths_failed": ["daily/2026-06-18/b.md"],
  "units_extracted": 4,
  "units_integrated": 4,
  "units_failed": 0,
  "topics_written": 3,
  "nodes_created": [...],
  "nodes_updated": [...],
  "errors": []
}
```

## 6. 拆分后的 YAML 形态

建议把 `auto_dream` 改成新的 dream pipeline:

```yaml
auto_dream:
  backend: base
  description: "Auto-dream: scan daily changes, extract cross-file units, integrate digest nodes, update daily interests."
  parameters:
    type: object
    properties:
      date:
        type: string
        description: "YYYY-MM-DD to scan; defaults to today in the dreamer's timezone"
        default: ""
      hint:
        type: string
        description: "caller guidance passed through to the dreamer LLM"
        default: ""
  steps:
    - backend: dream_extract_step
      file_catalog: dream
      topic_session_id: interests
    - backend: dream_integrate_step
    - backend: dream_topics_step
      topic_count: 3
      topic_diversity_days: 7
      topic_session_id: interests
    - backend: dream_finish_step
      file_catalog: dream
      persist: true
```

旧 job 删除,不做兼容 wrapper:

```yaml
dream:
  # 删除

dream_extract:
  # 删除

daily_topics:
  # 删除
```

## 7. 数据结构建议

建议所有跨 step 状态都放在 `context["dream"]`。

核心模型:

```python
class DreamUnit(BaseModel):
    name: str
    bucket: Literal["procedure", "personal", "wiki"]
    summary: str
    paths: list[str] = Field(default_factory=list)


class DreamTopic(BaseModel):
    title: str
    reason: str
    evidence: str = ""
    keywords: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)


class DreamState(BaseModel):
    date: str = ""
    hint: str = ""
    daily_dir: str = "daily"
    vault: str = ""
    changed_paths: list[dict] = Field(default_factory=list)
    unchanged_paths: list[str] = Field(default_factory=list)
    deleted_paths: list[str] = Field(default_factory=list)
    failed_paths: list[str] = Field(default_factory=list)
    checkpoint_paths: list[str] = Field(default_factory=list)
    units: list[DreamUnit] = Field(default_factory=list)
    topics: list[DreamTopic] = Field(default_factory=list)
    integrate_results: list[dict] = Field(default_factory=list)
    nodes_created: list[str] = Field(default_factory=list)
    nodes_updated: list[str] = Field(default_factory=list)
    topics_path: str = ""
    topics_written: int = 0
    errors: list[str] = Field(default_factory=list)
    result: dict = Field(default_factory=dict)
```

## 8. 实现任务

这是一次 breaking rewrite,不存在旧接口/旧格式迁移任务。剩余工作就是按新设计实现 4 个 step。

必须实现:

1. 在 `reme/steps/evolve/dream/` 下新增 `models.py`、helper 和 4 个 step 文件。
2. 重写 prompts:
   - `extract_system_prompt`
   - `extract_user_message`
   - `integrate_system_prompt_procedure`
   - `integrate_system_prompt_personal`
   - `integrate_system_prompt_wiki`
   - `integrate_user_message`
   - 可参考旧 prompt 的内容,但不保持旧 prompt 接口。
3. 实现 `dream_extract_step`:
   - 扫描 `daily/<date>.md` 与 `daily/<date>/**/*.md`。
   - 排除 `daily/<date>/interests.yaml`。
   - 根据 `file_catalog.dream` 计算 changed/unchanged/deleted。
   - 一个 agent 统一读取所有 changed paths,输出全局 units/topics。
   - 清洗 units: unknown bucket fallback 到 `wiki`; paths 去重; paths 必须来自 changed paths。
4. 实现 `dream_integrate_step`:
   - 按 `unit.paths` 打包 evidence。
   - for 循环逐 unit integrate。
   - 保留工具集合和 action 语义。
   - unit 失败时记录 `failed_paths += unit.paths`。
5. 实现 `dream_topics_step`:
   - 只读写 `daily/<date>/interests.yaml`。
   - 读取当天已有 YAML topics。
   - 读取最近 `topic_diversity_days` 天的 `interests.yaml` 做历史去重。
   - 写回去重后的 YAML。
   - 刷新 day-index。
6. 实现 `dream_finish_step`:
   - `checkpoint_paths = changed_paths - failed_paths`。
   - checkpoint 成功 paths 的 mtime。
   - checkpoint `interests.yaml` 和 `daily/<date>.md`。
   - dump `file_catalog.dream`。
   - 输出全新 response metadata。
7. 更新 `default.yaml`:
   - `auto_dream` 改为 4-step pipeline。
   - 删除 `dream`、`dream_extract`、`daily_topics` job。
   - 保留 `file_catalog.dream`。
8. 删除旧代码:
   - `reme/steps/evolve/auto_dream.py`
   - `reme/steps/evolve/dream.py`
   - `reme/steps/evolve/daily_topics.py`
   - 更新 `reme/steps/evolve/__init__.py`。

现在没有保留的迁移项:

- 不兼容 `interests.md`。
- 不保留单文件 `dream path=...`。
- 不保留 `dream_extract` job。
- 不保留 `daily_topics` job。
- 不要求 response metadata 兼容 `AutoDreamResult`。
- 不要求 prompt 入参兼容旧 `dream.yaml`。

## 9. 推荐测试用例

最低测试集:

| 场景 | 期望 |
|---|---|
| 当天没有任何文件 | scanned=0,success=true,no topics |
| 只有 day-index 新增 | `dream_extract_step` 调用一次全局 extract,finish checkpoint day-index |
| session note 新增 | extract evidence 中 day-index first,session notes sorted |
| session note mtime 未变 | unchanged+1,不进入 changed evidence |
| session note 删除 | catalog delete |
| `interests.yaml` 存在 | 不进入 scan/diff/changed evidence |
| 多个文件产出同一抽象 | extract 输出 1 个 unit,`paths` 包含多个 source path |
| extract 输出空 units/topics | finish 仍 checkpoint changed files,避免重复空跑 |
| 某个 unit integrate 失败 | 该 unit 的 `paths` 不 checkpoint,response failure |
| 同一 path 同时属于成功和失败 unit | 失败优先,该 path 不 checkpoint |
| 其它 path 的 units 都成功 | 这些 path 可以 checkpoint |
| 有 topic candidates | 写/更新 `daily/<date>/interests.yaml`,记录 topics_path/topics_written |
| 已存在 `interests.yaml` | 合并新旧 topics,不重复 |
| 最近 N 天已有相同 topic | 当前日 topics 去重跳过 |
| extract 输出 unknown bucket | 清洗后 bucket=`wiki` |
| prompt 输出 path 不在 changed paths | 该 unit 被丢弃或修正,不能 checkpoint 不明来源 |
