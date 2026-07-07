# LongMemEval Commute Query Failure Analysis

## 背景

本次对比两个 LongMemEval agent run：

- 成功 run: `797f5e23-2a1a-48d9-a3ed-26842e347534`
- 失败 run: `25884d25-b9ec-485f-bd13-94e754593310`

问题相同：

```text
Question Date: 2023/05/30 (Tue) 20:36
Question: How long is my daily commute to work?
```

正确证据来自历史会话 `answer_40a90d51`：

```text
I've been listening to audiobooks during my daily commute, which takes 45 minutes each way.
```

因此答案应为：`45 minutes each way`，往返共 `90 minutes`。

## 分析脚本

新增脚本：

```text
scripts/compare_lme_agent_runs.py
```

复现命令：

```bash
python scripts/compare_lme_agent_runs.py \
  datasets/longmemeval/1/history_session/agentscope/797f5e23-2a1a-48d9-a3ed-26842e347534.jsonl \
  datasets/longmemeval/1/history_session/agentscope/25884d25-b9ec-485f-bd13-94e754593310.jsonl \
  --json-out /tmp/lme_compare_commute.json
```

脚本会解析：

- agent history JSONL
- offloaded tool results
- 每次 `vector_search` / `bm25_search` 的 query
- 返回文档数
- 命中的 haystack session
- 答案证据所在的返回排名 `answer_ranks`
- draft 和 final answer

## 对比结果

### 成功 run

`797f5e23-2a1a-48d9-a3ed-26842e347534`

最终答案：

```text
Based on the retrieved history, on 2023/05/22 you mentioned that your daily commute to work takes 45 minutes each way (90 minutes total round trip).
```

关键工具调用：

```text
bm25_search query='daily commute work time'
answer_ranks=[1]
sessions=answer_40a90d51, ...
```

成功点：

- 答案 session `answer_40a90d51` 排在 BM25 返回第 1 条。
- agent 识别了 `45 minutes each way`。
- agent 调用了 `add_draft` 保存证据。
- 最终基于 draft 给出正确答案。

### 失败 run

`25884d25-b9ec-485f-bd13-94e754593310`

最终答案：

```text
not provided in history
```

但关键工具调用已经返回答案：

```text
vector_search query='commute time how long'
answer_ranks=[5]
sessions=ultrachat_20736, ultrachat_368854, d600c646, 1f88b2b3, answer_40a90d51
```

失败 run 中，答案证据位于返回结果第 5 条：

```text
2023/05/22 (Mon) 21:18 answer_40a90d51 user:
I've been listening to audiobooks during my daily commute, which takes 45 minutes each way.
```

## 失败后的 thinking

失败 run 在检索到答案之后，立刻产生了错误判断：

```text
The vector search results don't contain specific information about the user's daily commute duration. Let me search more specifically with bm25_search for commute-related information.
```

之后 agent 持续沿用这个错误判断：

```text
The search results don't contain information about the user's daily commute to work.
The search results don't contain any information about the user's daily commute to work.
The search results don't seem to contain information about the user's daily commute to work.
The search results so far don't seem to contain information about the user's daily commute duration.
The searches are no longer returning new useful information.
```

最终 draft 为空：

```text
read_all_draft -> ""
```

因此输出：

```text
not provided in history
```

## 根因判断

这不是简单的“一个检索到了、一个没检索到”。

更准确地说：

```text
两个 run 的工具结果都召回了答案证据；
失败 run 是 agent 没有利用已经返回的证据。
```

主要差异：

1. 答案排名不同

   成功 run 中，答案在 BM25 返回第 1 条：`answer_ranks=[1]`。

   失败 run 中，答案在 vector search 返回第 5 条：`answer_ranks=[5]`。

2. 失败 run 的结果更容易被漏读

   失败 run 的前 4 条结果是长会话或弱相关内容，答案埋在第 5 条的较长文本里。agent 在阅读工具结果后直接判断“没有具体通勤时长”，说明它没有抽取到第 5 条里的 `45 minutes each way`。

3. 去重机制放大了漏读影响

   指令中说明 search results 会去重。答案 session 已经在第 5 次工具调用中返回过，后续搜索类似 query 时，同一内容可能不会再次出现。

4. 后续 query 没有切到关键上下文词

   真实证据的上下文是 `audiobooks during my daily commute`。失败 run 后续主要搜索：

   ```text
   commute / work / office / morning / drive / minutes
   ```

   没有有效利用 `audiobook` / `audiobooks` 这个上下文线索。

## 结论

失败原因是 retrieval 已经召回答案，但 agent 对低排名长结果漏读或误判，随后在去重机制下无法再次看到同一证据，最终错误地认为历史中没有提供答案。

一句话总结：

```text
召回成功，证据利用失败。
```

## 改进建议

1. 对工具结果做结构化 evidence scan

   在模型生成 thinking 前，可以先对返回结果做轻量规则检查，例如：

   - 是否包含数字 + 时间单位：`45 minutes`
   - 是否包含问题关键词：`commute`
   - 是否包含答案标记：`has_answer: true`

2. 对 `has_answer: true` 提升优先级

   当前工具结果里答案消息带有：

   ```json
   "has_answer": true
   ```

   agent 或 wrapper 可以显式将这类消息提取成短 evidence，而不是只把完整长会话交给模型阅读。

3. 对低排名命中做摘要提示

   如果 top-k 中任意文档包含强答案模式，可以在 tool result 前增加 summary，例如：

   ```text
   Potential answer evidence found in result #5:
   "daily commute ... takes 45 minutes each way"
   ```

4. 在 dedup 前保存候选证据

   一旦返回结果里包含部分匹配，应调用 `add_draft` 或内部 evidence buffer 保存。否则后续搜索去重后，agent 可能再也看不到同一证据。

