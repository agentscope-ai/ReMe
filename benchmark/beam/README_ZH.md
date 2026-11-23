# BEAM 评测

[English version](./README.md)

BEAM 是一个面向**长上下文对话场景**的记忆能力评测基准，每个 case 包含一段被切分为多个 batch 的超长对话。

## 1. 数据准备

BEAM 按对话规模提供多种数据变体 —— `100K` / `500K` / `1M` / `10M`，可在不同上下文长度下
压测记忆系统。题型包括 abstention（拒答）、contradiction resolution（矛盾消解）、event
ordering（事件排序）、information extraction（信息抽取）、instruction following（指令遵循）、
knowledge update（知识更新）、multi-session reasoning（多会话推理）、preference following
（偏好遵循）、summarization（摘要）与 temporal reasoning（时间推理）。

BEAM 是公开仓库，clone 到 `benchmark/beam/dataset/` 下：

```bash
mkdir -p benchmark/beam/dataset
cd benchmark/beam/dataset
git clone https://github.com/mohammadtavakoli78/BEAM.git
```

clone 完成后，`benchmark/beam/dataset/BEAM/` 目录下应包含 `chats/`、`src/`、`topics/` 等子目录。

## 2. 使用 BEAM 插件

### 2.1 简介

BEAM 插件为每个 case 加载 `chat.json`，将每个 batch 转换为一个 ReMe 会话，按时间顺序摄入独立工作区，随后执行 `digest_update`。插件以 agentic（ReAct）模式回答探测问题，再通过 BEAM 基于 rubric 的 `answer_judge` 任务给出分级分数与二元判定，并输出各类型平均分。

### 2.2 如何使用

#### 安装与环境配置

在仓库根目录以 editable 模式安装 ReMe 和 BEAM 插件：

```bash
python -m pip install -e ".[as]"
reme plugins install ./plugins/beam --editable
reme plugins install ./plugins/beam-judge --editable
reme plugins validate beam
```

runner 显式启用已安装的 `beam` 插件，并将插件默认配置与 ReMe 内置的 `benchmark` 配置组合。
editable 安装会让 [`plugins/beam`](../../plugins/beam/README_ZH.md) 下的源码修改直接生效，无需重复安装。
自定义完整应用配置路径可通过 `reme.config` 指定，并可使用 `extends: benchmark`。
模型凭据通过公共 benchmark 配置中声明的环境变量设置。

#### 关键配置 — `benchmark/beam/config.yaml`

| 配置项 | 含义 |
| --- | --- |
| `dataset.beam_root` | BEAM 数据集根目录（`benchmark/beam/dataset/BEAM`）。 |
| `dataset.chat_size` | 运行的变体：`100K` / `500K` / `1M` / `10M`。 |
| `dataset.case_ids` | 指定 case（如 `["1","2"]`），空表示全部。 |
| `dataset.start_index` / `num_items` | case 分页（`num_items` 为 `0` 表示全部）。 |
| `dataset.workspace_root` | case 工作区根目录（`benchmark/beam/workspaces/beam`）。 |
| `evaluation.num_workers` | `0` = 自动，`1` = 串行，`>1` = 并行。 |
| `reme.config` | 使用的 ReMe 配置（`benchmark`）。 |
| `output.dir` | 结果目录（`benchmark/beam/results`）。 |

#### 运行评测

在仓库根目录执行：

```bash
python benchmark/beam/run.py
python benchmark/beam/run.py --config benchmark/beam/config.yaml
python benchmark/beam/run.py -q                        # 安静模式
python benchmark/beam/run.py --eval_only               # 复用已有工作区，仅执行查询 + 评判
```

### 2.3 结果

结果以 JSON 文件写入 `output.dir`，文件名为 `results_<chat_size>_<timestamp>.json`，
同时控制台会打印含各类型分数的汇总。日志约定在各基准间通用，见
[总说明](../README_ZH.md#输出与日志)。

#### 100K

agentscope==2.0.4.post1，conda reme 环境，20 并发，eval-only（复用已构建 memory）
（2026-08-05，20 cases / 400 Qs，总耗时 46.0 min）

| 题型 | Agentic | Binary | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|---|
| **总体** | **0.661** | **0.620** | **63.1K** | **1.8K** | **65.0K** | **3.33** |
| abstention | 0.550 | 0.550 | 96.0K | 1.1K | 97.1K | 4.58 |
| contradiction_resolution | 0.438 | 0.412 | 32.3K | 0.9K | 33.1K | 2.48 |
| event_ordering | 0.501 | 0.423 | 140.2K | 5.2K | 145.4K | 4.70 |
| information_extraction | 0.873 | 0.832 | 50.2K | 0.9K | 51.1K | 3.15 |
| instruction_following | 0.750 | 0.725 | 38.0K | 0.8K | 38.8K | 2.67 |
| knowledge_update | 0.688 | 0.675 | 31.2K | 0.7K | 31.8K | 2.27 |
| multi_session_reasoning | 0.626 | 0.584 | 85.0K | 4.6K | 89.6K | 4.28 |
| preference_following | 0.925 | 0.912 | 34.3K | 1.0K | 35.3K | 2.50 |
| summarization | 0.623 | 0.461 | 89.7K | 2.1K | 91.7K | 4.12 |
| temporal_reasoning | 0.637 | 0.625 | 34.6K | 1.0K | 35.6K | 2.52 |

Memory Construction 平均 token 消耗（default agent，20 cases 全量构建）：

| Agent | input tok/case | output tok/case | total tok/case |
|---|---|---|---|
| default | 2.2M | 136.7K | 2.3M |

#### 1M

agentscope==2.0.4.post1，conda reme 环境，20 并发，全量构建 memory
（2026-08-05，35 cases / 700 Qs，总耗时 459.2 min）

| 题型 | Agentic | Binary | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|---|
| **总体** | **0.650** | **0.609** | **100.0K** | **1.8K** | **101.8K** | **3.49** |
| abstention | 0.429 | 0.429 | 118.7K | 1.2K | 119.9K | 4.20 |
| contradiction_resolution | 0.391 | 0.364 | 49.8K | 0.8K | 50.6K | 2.50 |
| event_ordering | 0.558 | 0.456 | 201.5K | 3.9K | 205.4K | 4.79 |
| information_extraction | 0.809 | 0.772 | 79.0K | 0.9K | 79.8K | 3.00 |
| instruction_following | 0.852 | 0.832 | 55.8K | 0.9K | 56.7K | 2.81 |
| knowledge_update | 0.779 | 0.771 | 46.0K | 0.7K | 46.6K | 2.37 |
| multi_session_reasoning | 0.658 | 0.612 | 138.1K | 2.9K | 141.0K | 4.40 |
| preference_following | 0.798 | 0.777 | 51.8K | 0.9K | 52.7K | 2.53 |
| summarization | 0.693 | 0.537 | 158.8K | 2.9K | 161.7K | 4.44 |
| temporal_reasoning | 0.536 | 0.536 | 100.2K | 3.1K | 103.3K | 3.90 |

Memory Construction 平均 token 消耗（default agent，35 cases 全量构建）：

| Agent | input tok/case | output tok/case | total tok/case |
|---|---|---|---|
| default | 31.9M | 1.4M | 33.4M |

## 3. 使用 HoMe 优化插件

### 3.1 简介

HoMe 从数据集和指定的 Git 版本准备实验工作区，评测基线后迭代优化 BEAM 插件代码，并返回候选版本的 `commit_sha`；随后可用该版本在配置的评测集上进行独立评测。

### 3.2 如何使用

#### 实验准备

按上文获取数据集，并参照 [HoMe 运行指南](../../plugins/home/README.md#4-run-a-benchmark-experiment) 完成安装。
[config_home.yaml](config_home.yaml) 默认在 `benchmark/beam/workspaces/home-100K` 使用 **100K 全部 case** 进行优化。
[config_home_eval.yaml](config_home_eval.yaml) 默认使用 `benchmark/beam/workspaces/home-test-100K`，
评测 **100K 全部 case**，可能与训练数据重叠。可修改 `variant` 或指定 `case_ids` 更换评测数据。

从仓库根目录准备优化工作区：

```bash
python benchmark/beam/run_home.py
```

#### 实验运行

运行本地一次性优化：

```bash
python benchmark/beam/run_home.py --optimize
```

将 `CANDIDATE_SHA` 设为优化器返回的 `commit_sha` 或其唯一十六进制前缀，再运行本地一次性评测：

```bash
python benchmark/beam/run_home_eval.py --commit "$CANDIDATE_SHA"
```

服务方式见 [HoMe 评测指南](../../plugins/home/README.md#evaluate-through-the-service)。

### 3.3 结果

HoMe 在 BEAM-100K 上优化，在 BEAM-1M 上评测（35 个 case，700 道题）。
Token 用量和 LLM 调用次数均为问答阶段的每题平均值。

| 题型 | 得分 | 总 Token | 输入 Token | 输出 Token | LLM 调用 |
|---|---:|---:|---:|---:|---:|
| **总体** | **0.704** | **60.4K** | **57.8K** | **2.6K** | **6.7** |
| Abstention | 0.636 | 43.4K | 41.9K | 1.5K | 6.6 |
| Contradiction Resolution | 0.705 | 22.4K | 21.0K | 1.4K | 4.6 |
| Event Ordering | 0.546 | 123.0K | 116.6K | 6.4K | 12.7 |
| Information Extraction | 0.735 | 32.5K | 31.2K | 1.2K | 5.2 |
| Instruction Following | 0.901 | 30.7K | 29.4K | 1.3K | 5.5 |
| Knowledge Update | 0.671 | 19.2K | 18.3K | 0.9K | 4.0 |
| Multi-session Reasoning | 0.682 | 93.5K | 88.1K | 5.4K | 8.5 |
| Preference Following | 0.833 | 25.0K | 23.8K | 1.2K | 4.8 |
| Summarization | 0.726 | 173.2K | 169.0K | 4.2K | 9.4 |
| Temporal Reasoning | 0.604 | 41.4K | 39.1K | 2.3K | 6.0 |
