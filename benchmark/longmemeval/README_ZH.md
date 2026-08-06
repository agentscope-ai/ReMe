# LongMemEval 评测

[English version](./README.md)

LongMemEval 是一个面向**多轮多会话历史的长期记忆能力**的评测基准。每个条目提供一组按时间
顺序排列的用户与助手之间的会话，以及一个只能通过推理用户自有记忆才能回答的探测问题。ReMe
将会话摄入按条目隔离的工作区，以 agentic（ReAct）模式回答问题，最后由 LLM-as-judge 打分。

题型包括单会话（user / assistant / preference）、多会话推理、知识更新与时间推理等。

> 公共设置（依赖、凭据、日志约定）见[总评测说明](../README_ZH.md)。

## 1. 获取数据集

ReMe 仅使用 **cleaned-S** 版本，数据托管在 HuggingFace：
[agentscope-ai/ReMe_longmemeval_clean_s_v2](https://huggingface.co/datasets/agentscope-ai/ReMe_longmemeval_clean_s_v2)。
下载脚本经 hf-mirror.com 镜像源获取，如需更换源请修改 [`download.py`](./download.py) 中的
`BASE_URL`。

```bash
cd benchmark/longmemeval
python download.py            # 保存为 dataset/longmemeval_s_reme_cleaned.json，已存在则自动跳过
```

ground truth 已内嵌在数据文件中。

## 2. 运行

在仓库根目录执行：

```bash
python benchmark/longmemeval/run.py
python benchmark/longmemeval/run.py --config benchmark/longmemeval/config.yaml
python benchmark/longmemeval/run.py -q                        # 安静模式：仅评测级日志
python benchmark/longmemeval/run.py --log-level WARNING       # 降低评测 runner 日志
python benchmark/longmemeval/run.py --reme-log-level WARNING  # 降低 reme 内部日志
python benchmark/longmemeval/run.py --eval_only               # 复用已有工作区，仅执行查询 + 评判
```

## 3. 流程

1. 加载数据集（ground truth 已内嵌在数据文件中）。
2. 为每个条目创建独立工作区，按时间顺序摄入会话。
3. 当相邻会话跨越配置的时刻（默认 23:00）时触发 `auto_dream`。
4. 以 agentic（ReAct）模式回答每个问题。
5. 通过 `answer_judge` 任务对答案做二元（yes/no）评判，并输出各类型准确率。

## 4. 关键配置 —— `benchmark/longmemeval/config.yaml`

| 配置项 | 含义 |
| --- | --- |
| `dataset.path` | 待评测的数据集文件（如 `longmemeval_s_reme_cleaned.json`），已包含 ground truth。 |
| `dataset.start_index` / `num_items` | 评测条目的切片范围。 |
| `dataset.question_types` | 按问题类型过滤，空表示全部。 |
| `dataset.workspace_root` | 条目工作区根目录（`benchmark/longmemeval/workspaces/longmemeval-s`）。 |
| `evaluation.num_workers` | `0` = 自动（cpu-2），`1` = 串行，`>1` = 并行。 |
| `evaluation.filter_future_sessions` | 仅摄入时间戳 ≤ `question_date` 的会话。 |
| `reme.config` | 使用的 ReMe 配置（`lme.yaml`）。 |
| `reme.dream_trigger_hour` / `dream_scan_days` / `dream_max_units` | dream 触发行为。 |
| `output.dir` | 结果目录（`benchmark/longmemeval/results`）。 |

## 5. 输出

结果以 JSON 文件写入 `output.dir`，文件名为 `results_<timestamp>.json`，
同时控制台会打印含各类型准确率的汇总。日志约定在各基准间通用，见
[总说明](../README_ZH.md#输出与日志)。

## 6. 参考结果

### cleaned-s

**基础设置**

1. 使用修改后的 auto-memory prompt，关闭 auto-dream 机制
2. reme-memory 中的全部 session 的时间一定早于 question 的时间

**结果**

1. Agentic answer 框架回答，每次最多调用 5 次 search

| Category | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| single-session-user | 70 | 66 | 4 | 94.3% |
| single-session-assistant | 56 | 52 | 4 | 92.9% |
| knowledge-update | 78 | 60 | 18 | 76.9% |
| multi-session | 133 | 93 | 40 | 69.9% |
| temporal-reasoning | 133 | 78 | 55 | 58.6% |
| single-session-preference | 30 | 8 | 22 | 26.7% |
| **Overall** | **500** | **357** | **143** | **71.4%** |

2. prompted-based answer，每次固定使用原始 query 召回 10 个 fileChunk

| Category | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| single-session-assistant | 56 | 56 | 0 | 100.0% |
| single-session-user | 70 | 67 | 3 | 95.7% |
| knowledge-update | 78 | 69 | 9 | 88.5% |
| multi-session | 133 | 99 | 34 | 74.4% |
| temporal-reasoning | 133 | 83 | 50 | 62.4% |
| single-session-preference | 30 | 16 | 14 | 53.3% |
| **Overall** | **500** | **390** | **110** | **78.0%** |

3. golden session。使用与 prompt-based answer 相似的方法，唯一区别是，输入的 chunk 是 longMemEval 提供的 golden session。

| Category | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| single-session-assistant | 56 | 56 | 0 | 100.0% |
| single-session-user | 70 | 69 | 1 | 98.6% |
| knowledge-update | 78 | 74 | 4 | 94.9% |
| temporal-reasoning | 133 | 124 | 9 | 93.2% |
| multi-session | 133 | 117 | 16 | 88.0% |
| single-session-preference | 30 | 17 | 13 | 56.7% |
| **Overall** | **500** | **457** | **43** | **91.4%** |

4. golden session + time filter。和上一个实验的区别是，输入的 golden 被过滤了一次，要求输入 session 的时间戳必须早于 question 的时间才行。

一共被过滤掉了 75 个 session，44 个 question 受到了影响。temporal-reasoning 类型受影响最大。有 20 个 case 不包含任何一个 groundtruth session。根据 golden session 回答正确并且 golden session 非空，一共有 424 个 case。

| Category | Total | Correct | Wrong | Accuracy |
|---|---|---|---|---|
| knowledge-update | 78 | 75 | 3 | 96.2% |
| single-session-user | 70 | 67 | 3 | 95.7% |
| multi-session | 133 | 122 | 11 | 91.7% |
| single-session-assistant | 56 | 55 | 1 | 98.2% |
| temporal-reasoning | 133 | 91 | 42 | 68.4% |
| single-session-preference | 30 | 16 | 14 | 53.3% |
| **Overall** | **500** | **426** | **74** | **85.2%** |

5. 关闭 auto-memory 机制，根据原始 query 一次性混合检索召回原始 session，计算 recall。

| Category | Total | yes-judge | recall@5 / yes | recall@10 / yes |
|---|---|---|---|---|
| knowledge-update | 78 | 75 | 99.3% | 100% |
| single-session-user | 70 | 67 | 100% | 100% |
| multi-session | 133 | 122 | 91.8% | 95.8% |
| single-session-assistant | 56 | 55 | 100% | 100% |
| temporal-reasoning | 133 | 91 | 87.6% | 94.2% |
| single-session-preference | 30 | 16 | 100% | 100% |
| **Overall** | **500** | **426** | **87.6%** | **94.2%** |

### 最终 ground truth

#### agentic + prompted（最终 GT，2026-07-16）

| Category | Total | Agentic | Prompted limit=15 |
|---|---|---|---|
| single-session-assistant | 56 | 56/56 (100.0%) | 54/56 (96.4%) |
| single-session-user | 70 | 66/70 (94.3%) | 62/70 (88.6%) |
| knowledge-update | 78 | 75/78 (96.2%) | 67/78 (85.9%) |
| temporal-reasoning | 133 | 122/133 (91.7%) | 117/133 (88.0%) |
| multi-session | 133 | 115/133 (86.5%) | 101/133 (75.9%) |
| single-session-preference | 30 | 21/30 (70.0%) | 10/30 (33.3%) |
| **Overall** | **500** | **455/500 (91.0%)** | **411/500 (82.2%)** |

Prompted token 消耗：总 input 13,111,421（平均 26,275/题），总 output 313,370（平均 628/题）。
平均 sessions_ingested: 44.8，dreams_triggered: 0。
