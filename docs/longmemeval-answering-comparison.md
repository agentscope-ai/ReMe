# LongMemEval 回答方案对比

本文对比本地 500 条 LongMemEval 查询上的三条回答链路，以及对应的
LLM judge 结果。

## 总览

| 方案 | 回答文件 | Judge summary | 正确 | 错误 | 准确率 |
| --- | --- | --- | ---: | ---: | ---: |
| Agent 检索上下文后回答 | `agent_context_answer.json` | `agent_context_answer_judge_summary.json` | 351 | 149 | 70.2% |
| Gold session 上下文回答 | `context_answer.json` | `answer_judge_summary.json` | 392 | 108 | 78.4% |
| 直接搜索 session 的 Agent | `longmemeval_session_answer.json` | `longmemeval_session_answer_judge_summary.json` | 452 | 48 | 90.4% |

三个 summary 都覆盖 500 条查询，judge 调用失败数都是 0。

当前效果最好的是“直接搜索 session 的 Agent”。但它和常规检索链路不完全可比：
它允许 agent 访问该 query 同目录下完整的 `session/`，并进行多轮文件搜索。
另外两个 `context_answer` 方案更像固定上下文后的纯回答阶段：先选定 session，
把 session 格式化成一个 `session_context` 字符串，再让模型严格基于该上下文回答。

## 三个方案

### 1. Agent 检索上下文后回答

输出文件：

- `longmemeval/{index}/agent_context_answer.json`
- 对应 judge：`longmemeval/{index}/agent_context_answer_judge.json`

生成链路：

1. `scripts/cache_longmemeval_context_answers.py`
2. `--session-id-source=agent`
3. `--agent-key=memory_time_range_soft_boost`
4. `reme context_answer`
5. `reme/steps/common/context_answer.py`

这个方案从 `agent.json` 读取预测出来的 session ids，只加载这些 session 文件，
格式化为 `session_context`，然后调用 `context_answer` 严格根据上下文回答。

理解方式：

- 这是最接近“检索 + 回答”的端到端方案。
- 错误可能来自检索没召回、召回上下文不足，也可能来自最终回答模型。
- 因此 70.2% 是整条 pipeline 的分数，不只是回答模型的分数。

### 2. Gold session 上下文回答

输出文件：

- `longmemeval/{index}/context_answer.json`
- 对应 judge：`longmemeval/{index}/answer_judge.json`

生成链路：

1. `scripts/cache_longmemeval_context_answers.py`
2. `--session-id-source=answer`
3. `answer.json.answer_session_ids`
4. `reme context_answer`
5. `reme/steps/common/context_answer.py`

这个方案使用 `answer.json` 中的 gold `answer_session_ids`，然后调用同一个
`context_answer` step 从这些 gold sessions 中回答。

理解方式：

- 这个方案基本消除了检索误差，更能衡量“已给正确 session 后，模型能否抽取出正确答案”。
- 但它仍然不会 100% 正确：模型可能漏看证据、过度回答，或者答案和 golden answer
  在语义等价判断上不通过。
- 78.4% 说明当前 `context_answer` 的 prompt、结构化输出或上下文格式仍有优化空间。

### 3. 直接搜索 session 的 Agent

输出文件：

- `longmemeval/{index}/longmemeval_session_answer.json`
- 对应 judge：`longmemeval/{index}/longmemeval_session_answer_judge.json`

生成链路：

1. `scripts/cache_longmemeval_session_answers.py`
2. `reme longmemeval_session_answer`
3. `reme/steps/common/longmemeval_session_answer.py`
4. `agent_wrapper: claude_code`

这个方案只传入 `query_path`。step 会解析同目录下的 `session/`，然后让
Claude Code 风格的 agent 读取 `query.json`，并且只在该 `session/` 目录内搜索。
agent 可以 `ls`、`grep`、`Read` 具体 session 文件，并进行多轮关键词搜索。

理解方式：

- 当前准确率最高，为 90.4%。
- 它受益于工具调用和多轮搜索，可以在完整 session 目录里主动定位证据。
- 它适合作为 tool-search upper baseline，但不等同于“先检索固定上下文，再回答”的常规 RAG 实验。

## 结果重叠

三个方案正确集合的两两对比：

| 对比 | 两者都对 | 只有前者对 | 只有后者对 | 两者都错 |
| --- | ---: | ---: | ---: | ---: |
| Agent context vs Gold context | 331 | 20 | 61 | 88 |
| Agent context vs Session search | 332 | 19 | 120 | 29 |
| Gold context vs Session search | 370 | 22 | 82 | 26 |

额外统计：

- 三个方案都正确：316
- 三个方案都错误：22

`longmemeval_session_answer` 比两个 context-answer 方案多修复了不少 case：
相比 agent 检索上下文方案多对 120 条；相比 gold session 上下文方案多对 82 条。
这说明收益不只是最终答案措辞更好，而是“多轮搜索 + 选择性读取证据”的能力很关键。

## 相关 Step 代码行为

### `python_execute.py`

`PythonExecuteStep` 用 `sys.executable -I -c` 执行隔离的 Python 代码。

行为：

- 必须输入 `code`。
- 在临时目录中执行。
- 环境变量只保留 `PATH` 和 `PYTHONIOENCODING`。
- 成功时把 stdout 作为 answer 返回。
- 日志会打印执行代码、stdout、stderr。
- metadata 中记录 code、timeout、returncode、stdout、stderr。

这个 step 主要作为确定性时间计算工具使用。

### `memory_time_range.py`

`MemoryTimeRangeStep` 根据 `question` 和 `question_date` 抽取宽松但有用的记忆检索时间范围。

关键行为：

- 输出紧凑 JSON，可包含 `thinking`、`start_dt`、`end_dt`。
- 没有明确时间约束时返回 `{}`。
- 不从 `recent`、`lately` 这类模糊词强行推断时间范围。
- 如果需要相对日期、月份边界、星期计算等日期运算，必须调用 `python_execute`。
- 倾向宽松范围或空范围，避免过窄过滤导致答案 session 被排除。

这个 step 通过写入 `query.json.memory_time_range` 影响检索实验，并进一步产生
`agent.json` 中类似 `memory_time_range_soft_boost` 的检索结果。

### `context_answer.py`

`ContextAnswerStep` 从输入的 `session_context` 字符串中回答问题。它使用 Pydantic
结构化输出：

- `thinking`：简短说明上下文中的证据。
- `answer`：最终答案；证据不足时必须是 `unknown`。

它不负责搜索文件，也不知道上下文来自哪里。`agent_context_answer` 和
`context_answer` 的差别只在于脚本调用它之前选了哪些 session 文件。

### `longmemeval_session_answer.py`

`LongMemEvalSessionAnswerStep` 是文件搜索型 agent step。

行为：

- 输入 `longmemeval/{index}/query.json` 路径。
- 解析同目录下的 `session/`。
- prompt 要求 agent 只能读 `query.json` 和该 `session/` 下的文件。
- 明确禁止读其他目录或使用外部知识。
- 要求不要一次搜索失败就放弃，要多轮尝试关键词、实体、日期、偏好等。
- 最终只返回 answer 文本；空结果兜底为 `unknown`。

在 `demo.yaml` 中，这个 job 默认使用 `agent_wrapper: claude_code`。

### `answer_judge.py`

`AnswerJudgeStep` 根据 `query`、`agent_answer`、`golden_answer` 判断答案是否正确。

结构化输出：

- `thinking`：简短比较理由。
- `answer`：布尔值。

judge 把 `golden_answer` 作为真值。轻微措辞差异可以接受；缺少关键信息、矛盾、
改变答案含义的额外声明，或者 golden 有答案但 agent 拒答，都会判为错误。

## 复现命令

评估 agent 检索上下文后的回答：

```bash
python scripts/judge_longmemeval_context_answers.py \
  --context-answer-name agent_context_answer.json \
  --output-name agent_context_answer_judge.json \
  --summary-name agent_context_answer_judge_summary.json \
  --workers 8 \
  --timeout 180 \
  --refresh
```

评估 gold session 上下文回答：

```bash
python scripts/judge_longmemeval_context_answers.py \
  --context-answer-name context_answer.json \
  --output-name answer_judge.json \
  --summary-name answer_judge_summary.json \
  --workers 8 \
  --timeout 180 \
  --refresh
```

评估直接搜索 session 的回答：

```bash
python scripts/judge_longmemeval_context_answers.py \
  --context-answer-name longmemeval_session_answer.json \
  --output-name longmemeval_session_answer_judge.json \
  --summary-name longmemeval_session_answer_judge_summary.json \
  --workers 8 \
  --timeout 180 \
  --refresh
```

生成直接搜索 session 的回答：

```bash
python scripts/cache_longmemeval_session_answers.py --workers 8
```

从 gold sessions 生成 context answer：

```bash
python scripts/cache_longmemeval_context_answers.py \
  --session-id-source answer \
  --output-name context_answer.json \
  --workers 8
```

从 agent 预测 sessions 生成 context answer：

```bash
python scripts/cache_longmemeval_context_answers.py \
  --session-id-source agent \
  --agent-key memory_time_range_soft_boost \
  --output-name agent_context_answer.json \
  --workers 8
```

## 建议

三个指标适合看不同问题：

- `agent_context_answer_judge_summary.json`：看端到端检索 + 回答效果。
- `answer_judge_summary.json`：看 oracle sessions 下的答案抽取能力。
- `longmemeval_session_answer_judge_summary.json`：看允许 agent 直接 inspect 全部 session 文件时的 tool-search upper baseline。

改进常规 pipeline 时，优先对比方案 1 和方案 2 的失败 case：

- 如果方案 1 错、方案 2 对，瓶颈大概率在检索。
- 如果方案 1 和方案 2 都错，应优先改 `context_answer` prompt、上下文格式，或人工复查 judge 标准。
- 如果方案 3 对、方案 2 错，说明问题可能不只是给了正确 session，而是需要更好的 session 内搜索、证据定位和上下文压缩方式。
