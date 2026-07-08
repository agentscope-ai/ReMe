# LongMemEval (LME) Benchmark

ReMe 内置两个可以直接拼成 [LongMemEval](https://github.com/xiaowu0162/longmemeval) 评测流程的
Step：一个**从会话上下文直接作答**的 Step，和一个用 LLM 给该答案**打 yes/no 评分**的 Step。配合 ReMe
已有的 `auto_memory`、`search` 等能力，它们可以像其他支持工具的服务一样接入官方 LME harness。

```{note}
下面的示例假定服务以 `reme start config=reme/config/jinli_lme.yaml` 启动。benchmark 循环本身不在
`jinli_lme.yaml` 中，它由 LME harness 仓库负责；本页面只讲清两个 Step 自身的输入输出约定。
```

## 两个 Step 一览

| Step 名称 | 读入 | 写出 |
|---|---|---|
| `context_answer_step` | `query`、`session_context`、`current_date` | `context_answer` |
| `answer_judge_step` | `query`、`agent_answer`、`golden_answer`、`question_type` | `answer_judgement`（`yes` / `no`） |

两个 Step 都依赖 `agent_wrapper`（用于实际回答 / 判分的 LLM）。它们是对 `reply` 的薄封装，不做任何额外
的检索，只负责 prompt 拼装。

## `context_answer_step`

直接根据传入的会话上下文（即一次对话的全部历史）回答问题，对应 LongMemEval 论文中的 "direct" baseline：
模型只看到历史，不接触 agent 的记忆库。

必填字段（来自 `RuntimeContext`）：

```yaml
query: string                # 当前问题
session_context: string      # 一次会话拼接好的历史
current_date: string         # `YYYY-MM-DD` 形式的"今天"，用于解析相对时间
```

产出：

```yaml
context_answer: string       # 模型自由形式的回答
metadata:
  query: ...
  session_context: ...
  current_date: ...
  context_answer: ...
```

默认的 user 消息模板（`user_message`）会要求模型先抽取相关事实再推理，这与公开的 LME prompt 一致。

## `answer_judge_step`

用四套 LLM 判分 prompt 中的一套，给 `agent_answer` 与 `golden_answer` 打分。判分 prompt 由 `question_type`
自动选择：

| `question_type`（归一化后） | 使用的判分 prompt |
|---|---|
| `temporal_reasoning` | `temporal_reasoning_system_prompt` |
| `knowledge_update` | `knowledge_update_system_prompt` |
| `single_session_preference` | `single_session_preference_system_prompt`（使用 *preference* rubric user 模板） |
| 其它 | `other_question_types_system_prompt` |

归一化规则是 lowercase + 把 `-` 和空格替换成 `_`，因此 `Temporal-Reasoning`、`temporal reasoning`、
`TEMPORAL_REASONING` 都会命中同一个判分 prompt。未知 category 会落到通用
`other_question_types_system_prompt`，这正好对应 LME 中上述三类之外的题目。

判分模型的原始输出会被用大小写不敏感的 `^\s*(yes|no)\b` 正则归一化：命中则 `answer_judgement` 设为
`yes` 或 `no`；否则保留原始输出的首个 token 并 lowercase（这样 harness 可以把意外输出当作失误处理，
而不是被误算成正确）。

必填字段：

```yaml
query: string
agent_answer: string        # 一般是上一步的 `context_answer`
golden_answer: string       # 来自 LME 数据集
question_type: string       # 上述 LME category 之一
```

产出：

```yaml
answer_judgement: string    # `yes` / `no`（或未规整时的原 token）
metadata:
  query: ...
  agent_answer: ...
  golden_answer: ...
  question_type: ...
  answer_judgement: ...
  raw_answer_judgement: ... # 模型的原始回答，便于调试判分异常
```

## 接入 Job 的样例

随包发布的 `jinli_lme.yaml` 只配置了常规的 Job（`update_index`、`auto_memory`、`version`、`search`）以及
Tokenizer / Embedding / LLM / Agent 组件栈。要让两个 benchmark Step 能被 LME harness 调用，需要再加一个
Job 把它们串起来。下面这段可以单独放在一个 YAML 里，也可以追加到 `jinli_lme.yaml`：

```yaml
jobs:
  lme_one_question:
    backend: base
    description: "对单条 LME 题目执行 context_answer_step + answer_judge_step。"
    parameters:
      type: object
      properties:
        query:           {type: string}
        session_context: {type: string}
        current_date:    {type: string}
        agent_answer:    {type: string}
        golden_answer:   {type: string}
        question_type:   {type: string}
      required: [query, session_context, agent_answer, golden_answer, question_type]
    steps:
      - backend: context_answer_step
      - backend: answer_judge_step
```

之后从 LME harness 触发就只是一次 POST：

```bash
reme app config=jinli_lme.yaml lme_one_question \
    query="When did Jon lose his job?" \
    session_context="2023-01-19: ..." \
    current_date="2024-03-01" \
    agent_answer="..." \
    golden_answer="..." \
    question_type="temporal_reasoning"
```

## 写 benchmark 时的注意点

- 缺少必填字段时两个 Step 都会抛 `ValueError`，harness 应该把它当作单题失败（带上 `question_id`）处理，
  而不是当作服务级错误。
- LME 数据集如果给出的是 `temporal-reasoning` 这种带连字符的 category，harness 侧应自行规一化；Step 内部
  只做安全回退处理。
- 评分时应使用 `answer_judgement`，`raw_answer_judgement` 仅用于诊断判分模型输出非 yes/no token 的情形。
- benchmark 循环建议同时保留两个 Step 写出的题目级 metadata（尤其是 `session_context` 长度与
  `current_date`），方便后续时间处理逻辑重构后用同一批输入重跑对比。
