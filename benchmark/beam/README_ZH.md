# BEAM 评测

[English version](./README.md)

BEAM 是一个面向**长上下文对话场景**的记忆能力评测基准。每个 case 包含一段被切分为多个
batch 的超长对话；ReMe 将每个 batch 转换为一个会话，按时间顺序摄入后，以 agentic（ReAct）
模式回答探测问题。答案由 BEAM 基于 rubric 的 `answer_judge` 任务打分，同时给出分级分数与二元
判定，并输出各类型平均分。

BEAM 按对话规模提供多种数据变体 —— `100K` / `500K` / `1M` / `10M`，可在不同上下文长度下
压测记忆系统。题型包括 abstention（拒答）、contradiction resolution（矛盾消解）、event
ordering（事件排序）、information extraction（信息抽取）、instruction following（指令遵循）、
knowledge update（知识更新）、multi-session reasoning（多会话推理）、preference following
（偏好遵循）、summarization（摘要）与 temporal reasoning（时间推理）。

> 公共设置（依赖、凭据、日志约定）见[总评测说明](../README_ZH.md)。

## 1. 获取数据集

BEAM 是公开仓库，clone 到 `benchmark/beam/dataset/` 下：

```bash
mkdir -p benchmark/beam/dataset
cd benchmark/beam/dataset
git clone https://github.com/mohammadtavakoli78/BEAM.git
```

clone 完成后，`benchmark/beam/dataset/BEAM/` 目录下应包含 `chats/`、`src/`、`topics/` 等子目录。

## 2. 运行

在仓库根目录执行：

```bash
python benchmark/beam/run.py
python benchmark/beam/run.py --config benchmark/beam/config.yaml
python benchmark/beam/run.py -q                        # 安静模式
python benchmark/beam/run.py --eval_only               # 复用已有工作区，仅执行查询 + 评判
```

## 3. 流程

1. 为每个 case 加载 `chat.json`，将每个 batch 转换为一个 ReMe 会话。
2. 按时间顺序将会话摄入独立工作区，随后执行 `digest_update`。
3. 以 agentic（ReAct）模式回答每个探测问题。
4. 通过 BEAM 基于 rubric 的 `answer_judge` 任务打分，并输出各类型平均分。

## 4. 关键配置 —— `benchmark/beam/config.yaml`

| 配置项 | 含义 |
| --- | --- |
| `dataset.beam_root` | BEAM 数据集根目录（`benchmark/beam/dataset/BEAM`）。 |
| `dataset.chat_size` | 运行的变体：`100K` / `500K` / `1M` / `10M`。 |
| `dataset.case_ids` | 指定 case（如 `["1","2"]`），空表示全部。 |
| `dataset.start_index` / `num_items` | case 分页（`num_items` 为 `0` 表示全部）。 |
| `dataset.workspace_root` | case 工作区根目录（`benchmark/beam/workspaces/beam`）。 |
| `evaluation.num_workers` | `0` = 自动，`1` = 串行，`>1` = 并行。 |
| `reme.config` | 使用的 ReMe 配置（`beam.yaml`）。 |
| `output.dir` | 结果目录（`benchmark/beam/results`）。 |

## 5. 输出

结果以 JSON 文件写入 `output.dir`，文件名为 `results_<chat_size>_<timestamp>.json`，
同时控制台会打印含各类型分数的汇总。日志约定在各基准间通用，见
[总说明](../README_ZH.md#输出与日志)。

## 6. 参考结果

> 以下结果使用 longmemeval 版本的 prompt。

### 100K

agentscope==2.0.4.post1，conda reme 环境，20 并发，eval-only（复用已构建 memory）
（2026-08-05，20 cases / 400 Qs，总耗时 46.0 min）

| 题型 | Agentic | Binary | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|---|
| abstention | 0.550 | 0.550 | 96,031 | 1,070 | 97,101 | 4.58 |
| contradiction_resolution | 0.438 | 0.412 | 32,263 | 872 | 33,135 | 2.48 |
| event_ordering | 0.501 | 0.423 | 140,195 | 5,163 | 145,358 | 4.70 |
| information_extraction | 0.873 | 0.832 | 50,245 | 883 | 51,128 | 3.15 |
| instruction_following | 0.750 | 0.725 | 37,986 | 848 | 38,834 | 2.67 |
| knowledge_update | 0.688 | 0.675 | 31,198 | 651 | 31,849 | 2.27 |
| multi_session_reasoning | 0.626 | 0.584 | 85,038 | 4,563 | 89,601 | 4.28 |
| preference_following | 0.925 | 0.912 | 34,281 | 989 | 35,270 | 2.50 |
| summarization | 0.623 | 0.461 | 89,657 | 2,056 | 91,713 | 4.12 |
| temporal_reasoning | 0.637 | 0.625 | 34,563 | 1,049 | 35,612 | 2.52 |
| **OVERALL** | **0.661** | **0.620** | **63,146** | **1,814** | **64,960** | **3.33** |

Memory Construction 平均 token 消耗（default agent，20 cases 全量构建）：

| Agent | input tok/case | output tok/case | total tok/case |
|---|---|---|---|
| default | 2,172,316 | 136,697 | 2,309,013 |

### 1M

agentscope==2.0.4.post1，conda reme 环境，20 并发，全量构建 memory
（2026-08-05，35 cases / 700 Qs，总耗时 459.2 min）

| 题型 | Agentic | Binary | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|---|
| abstention | 0.429 | 0.429 | 118,707 | 1,178 | 119,886 | 4.20 |
| contradiction_resolution | 0.391 | 0.364 | 49,787 | 810 | 50,597 | 2.50 |
| event_ordering | 0.558 | 0.456 | 201,514 | 3,889 | 205,403 | 4.79 |
| information_extraction | 0.809 | 0.772 | 78,950 | 894 | 79,844 | 3.00 |
| instruction_following | 0.852 | 0.832 | 55,757 | 924 | 56,681 | 2.81 |
| knowledge_update | 0.779 | 0.771 | 45,981 | 665 | 46,646 | 2.37 |
| multi_session_reasoning | 0.658 | 0.612 | 138,133 | 2,873 | 141,006 | 4.40 |
| preference_following | 0.798 | 0.777 | 51,796 | 920 | 52,716 | 2.53 |
| summarization | 0.693 | 0.537 | 158,794 | 2,905 | 161,700 | 4.44 |
| temporal_reasoning | 0.536 | 0.536 | 100,176 | 3,148 | 103,324 | 3.90 |
| **OVERALL** | **0.650** | **0.609** | **99,959** | **1,821** | **101,780** | **3.49** |

Memory Construction 平均 token 消耗（default agent，35 cases 全量构建）：

| Agent | input tok/case | output tok/case | total tok/case |
|---|---|---|---|
| default | 31,943,817 | 1,417,061 | 33,360,878 |
