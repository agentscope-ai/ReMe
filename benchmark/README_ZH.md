# ReMe 评测复现说明

ReMe 内置两个记忆能力评测基准的复现指南：

- **LongMemEval** —— 面向多轮多会话历史的长期记忆能力评测。
- **BEAM** —— 面向长上下文对话场景、基于评分细则（rubric）打分的记忆能力评测。

每个基准都会运行完整的端到端流程：将会话摄入独立的按条目隔离的工作区，
分别以 prompted（检索 + LLM）和 agentic（ReAct）两种模式回答探测问题，
最后由 LLM-as-judge 对答案进行打分。

## 1. 环境准备

安装 ReMe 及 dev + core 附加依赖（Python 3.11+）：

```bash
pip install -e ".[dev,core]"
```

在项目根目录配置 `.env`（可从 `example.env` 复制），运行脚本会自动从仓库根目录加载 `.env`。
通常需要以下变量：

```bash
LLM_API_KEY=...
LLM_BASE_URL=...
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=...
```

模型名称与组件装配位于各基准引用的 ReMe 配置中
（`reme/config/lme.yaml` 与 `reme/config/beam.yaml`）。

## 2. 下载数据集

完整说明见 [`datasets/README.md`](../datasets/README.md)。

**LongMemEval**（从 HuggingFace 镜像下载）：

```bash
cd datasets/longmemeval
python download.py            # 下载 cleaned-S 数据文件，已存在则自动跳过
```

**BEAM**（公开仓库，clone 到 `datasets/` 下）：

```bash
cd datasets
git clone https://github.com/mohammadtavakoli78/BEAM.git
```

## 3. 运行 LongMemEval

在仓库根目录执行：

```bash
python benchmark/longmemeval/run.py
python benchmark/longmemeval/run.py --config benchmark/longmemeval/config.yaml
python benchmark/longmemeval/run.py -q                        # 安静模式：仅评测级日志
python benchmark/longmemeval/run.py --log-level WARNING       # 降低评测 runner 日志
python benchmark/longmemeval/run.py --reme-log-level WARNING  # 降低 reme 内部日志
python benchmark/longmemeval/run.py --eval_only               # 复用已有工作区，仅执行查询 + 评判
```

### 流程

1. 加载数据集（ground truth 已内嵌在数据文件中）。
2. 为每个条目创建独立工作区，按时间顺序摄入会话。
3. 当相邻会话跨越配置的时刻（默认 23:00）时触发 `auto_dream`。
4. 以 agentic（ReAct）和 prompted（检索 + LLM）两种模式回答每个问题。
5. 通过 `answer_judge` 任务对两种答案做二元（yes/no）评判，并输出各类型准确率。

### 关键配置 —— `benchmark/longmemeval/config.yaml`

| 配置项 | 含义 |
| --- | --- |
| `dataset.path` | 待评测的数据集文件（如 `longmemeval_s_reme_cleaned.json`），已包含 ground truth。 |
| `dataset.start_index` / `num_items` | 评测条目的切片范围。 |
| `dataset.question_types` | 按问题类型过滤，空表示全部。 |
| `dataset.workspace_root` | 条目工作区根目录（`memory_workspaces/longmemeval-s`）。 |
| `evaluation.num_workers` | `0` = 自动（cpu-2），`1` = 串行，`>1` = 并行。 |
| `evaluation.filter_future_sessions` | 仅摄入时间戳 ≤ `question_date` 的会话。 |
| `reme.config` | 使用的 ReMe 配置（`lme.yaml`）。 |
| `reme.dream_trigger_hour` / `dream_scan_days` / `dream_max_units` | dream 触发行为。 |
| `output.dir` | 结果目录（`benchmark/results/longmemeval`）。 |

## 4. 运行 BEAM

在仓库根目录执行：

```bash
python benchmark/beam/run.py
python benchmark/beam/run.py --config benchmark/beam/config.yaml
python benchmark/beam/run.py -q                        # 安静模式
python benchmark/beam/run.py --eval_only               # 复用已有工作区，仅执行查询 + 评判
```

### 流程

1. 为每个 case 加载 `chat.json`，将每个 batch 转换为一个 ReMe 会话。
2. 按时间顺序将会话摄入独立工作区，随后执行 `digest_update`。
3. 以配置的 `modes`（prompted / agentic）回答每个探测问题。
4. 通过 BEAM 基于 rubric 的 `answer_judge` 任务打分，并输出各类型平均分。

### 关键配置 —— `benchmark/beam/config.yaml`

| 配置项 | 含义 |
| --- | --- |
| `dataset.beam_root` | BEAM 数据集根目录（`datasets/BEAM`）。 |
| `dataset.chat_size` | 运行的变体：`100K` / `500K` / `1M` / `10M`。 |
| `dataset.case_ids` | 指定 case（如 `["1","2"]`），空表示全部。 |
| `dataset.start_index` / `num_items` | case 分页（`num_items` 为 `0` 表示全部）。 |
| `dataset.workspace_root` | case 工作区根目录（`memory_workspaces/beam`）。 |
| `evaluation.num_workers` | `0` = 自动，`1` = 串行，`>1` = 并行。 |
| `evaluation.modes` | 运行的回答模式：`prompted` 和/或 `agentic`。 |
| `reme.config` | 使用的 ReMe 配置（`beam.yaml`）。 |
| `output.dir` | 结果目录（`benchmark/results/beam`）。 |

## 5. 输出与日志

- **结果**：JSON 文件写入 `output.dir`
  （LongMemEval 为 `results_<timestamp>.json`，
  BEAM 为 `results_<chat_size>_<timestamp>.json`）。同时控制台会打印含各类型
  准确率/分数的汇总。
- **日志**：当 `output.log_to_file` 开启时，每次运行的日志写入
  `logs/<log_prefix>_<timestamp>/`（包含一个 `runner.log` 及每个 worker 进程的
  `worker-<pid>.log`）。

## 6. 终止运行

并行运行会派生进程树。若要干净地终止某次运行及其全部 worker：

```bash
bash benchmark/kill.sh <PID>
```

该脚本会先向整个进程树发送 `SIGTERM` 优雅终止，对 5 秒内未退出的进程再升级为 `SIGKILL`。

## 7. 参考结果

已记录的评测结果见：

- [`result-longmemeval.md`](./result-longmemeval.md)
- [`result-beam.md`](./result-beam.md)
