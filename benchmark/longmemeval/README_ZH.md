# LongMemEval 评测

[English version](./README.md)

LongMemEval 是一个面向**多轮多会话历史的长期记忆能力**的评测基准。每个条目提供一组按时间顺序排列的用户与助手之间的会话，以及一个只能通过推理用户自有记忆才能回答的探测问题。

## 1. 数据准备

题型包括单会话（user / assistant / preference）、多会话推理、知识更新与时间推理等。

ReMe 仅使用 **cleaned-S** 版本，数据托管在 HuggingFace：
[agentscope-ai/ReMe_longmemeval_clean_s_v2](https://huggingface.co/datasets/agentscope-ai/ReMe_longmemeval_clean_s_v2)。
下载脚本经 hf-mirror.com 镜像源获取，如需更换源请修改 [`download.py`](./download.py) 中的
`BASE_URL`。

```bash
cd benchmark/longmemeval
python download.py            # 保存为 dataset/longmemeval_s_reme_cleaned.json，已存在则自动跳过
```

ground truth 已内嵌在数据文件中。

## 2. 使用 LME 插件

### 2.1 简介

LME 插件加载内嵌 ground truth 的数据集，为每个条目创建独立工作区并按时间顺序摄入会话，再以 agentic（ReAct）模式回答探测问题，通过 LLM-as-judge 的 `answer_judge` 任务做二元（yes/no）评判并输出各类型准确率。若自定义应用配置启用了 `auto_dream`，会在相邻会话跨越配置时刻（默认 23:00）时触发；插件预设保持关闭。

### 2.2 如何使用

#### 安装与环境配置

在仓库根目录以 editable 模式安装 ReMe 和 LongMemEval 插件：

```bash
python -m pip install -e ".[as]"
reme plugins install ./plugins/lme --editable
reme plugins install ./plugins/lme-judge --editable
reme plugins validate lme
```

runner 显式启用已安装的 `lme` 插件，并将插件默认配置与 ReMe 内置的 `benchmark` 配置组合。
editable 安装会让 [`plugins/lme`](../../plugins/lme/README_ZH.md) 下的源码修改直接生效，无需重复安装。
自定义完整应用配置路径可通过 `reme.config` 指定，并可使用 `extends: benchmark`。
模型凭据通过公共 benchmark 配置中声明的环境变量设置。

#### 关键配置 — `benchmark/longmemeval/config.yaml`

| 配置项 | 含义 |
| --- | --- |
| `dataset.path` | 待评测的数据集文件（如 `longmemeval_s_reme_cleaned.json`），已包含 ground truth。 |
| `dataset.start_index` / `num_items` | 评测条目的切片范围。 |
| `dataset.question_types` | 按问题类型过滤，空表示全部。 |
| `dataset.workspace_root` | 条目工作区根目录（`benchmark/longmemeval/workspaces/longmemeval-s`）。 |
| `evaluation.num_workers` | `0` = 自动（cpu-2），`1` = 串行，`>1` = 并行。 |
| `evaluation.filter_future_sessions` | 仅摄入时间戳 ≤ `question_date` 的会话。 |
| `reme.config` | 使用的 ReMe 配置（`benchmark`）。 |
| `reme.dream_trigger_hour` / `dream_scan_days` / `dream_max_units` | dream 触发行为。 |
| `output.dir` | 结果目录（`benchmark/longmemeval/results`）。 |

#### 运行评测

在仓库根目录执行：

```bash
python benchmark/longmemeval/run.py
python benchmark/longmemeval/run.py --config benchmark/longmemeval/config.yaml
python benchmark/longmemeval/run.py -q                        # 安静模式：仅评测级日志
python benchmark/longmemeval/run.py --log-level WARNING       # 降低评测 runner 日志
python benchmark/longmemeval/run.py --reme-log-level WARNING  # 降低 reme 内部日志
python benchmark/longmemeval/run.py --eval_only               # 复用已有工作区，仅执行查询 + 评判
```

### 2.3 结果

结果以 JSON 文件写入 `output.dir`，文件名为 `results_<timestamp>.json`，
同时控制台会打印含各类型准确率的汇总。日志约定在各基准间通用，见
[总说明](../README_ZH.md#输出与日志)。

**参考结果**

#### cleaned-s

**结果**

agentscope==2.0.4.post1, conda reme env, 32 workers, eval-only（复用预构建记忆）
（2026-08-06，500 题，总计 10.0 min）

| 类型 | Agentic | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|
| **总体** | **0.894** | **43.4K** | **0.9K** | **44.3K** | **3.69** |
| knowledge-update | 0.910 | 31.6K | 0.6K | 32.2K | 2.90 |
| multi-session | 0.842 | 52.8K | 1.5K | 54.3K | 4.21 |
| single-session-assistant | 1.000 | 15.6K | 0.3K | 15.9K | 1.89 |
| single-session-preference | 0.633 | 36.8K | 0.8K | 37.6K | 3.60 |
| single-session-user | 0.986 | 27.4K | 0.4K | 27.8K | 2.60 |
| temporal-reasoning | 0.902 | 62.7K | 1.0K | 63.7K | 4.97 |

## 3. 使用 HoMe 优化插件

### 3.1 简介

HoMe 从数据集和指定的 Git 版本准备实验工作区，评测基线后迭代优化 LME 插件代码，并返回候选版本的 `commit_sha`；随后可用该版本在配置的评测集上进行独立评测。

### 3.2 如何使用

#### 实验准备

按上文获取数据集，并参照 [HoMe 运行指南](../../plugins/home/README.md#4-run-a-benchmark-experiment) 完成安装。
[config_home.yaml](config_home.yaml) 默认在 `benchmark/longmemeval/workspaces/home-groups-1-5` 使用
**group_1–group_5 的 250 个 case** 进行优化。训练 ID 和[独立审阅的证据标注](home_evidence_sessions.jsonl)来自
`dev_meta` 提交 `b53048e44833937df80a52fccceea78d054790e8`。原始划分使用 seed 42，按问题类型分层抽样。

[config_home_eval.yaml](config_home_eval.yaml) 默认使用 `benchmark/longmemeval/workspaces/home-test-held-out`，
排除来源实验实际使用的训练 ID。使用默认训练集时，评测 **group_6–group_10 的 250 个 case**。
若要显式选择 case，将 `exclude_training_cases` 设为 `false`，并把 `all_cases: true` 替换为 `case_ids: [...]`。

从仓库根目录准备优化工作区：

```bash
python benchmark/longmemeval/run_home.py
```

#### 实验运行

运行本地一次性优化：

```bash
python benchmark/longmemeval/run_home.py --optimize
```

将 `CANDIDATE_SHA` 设为优化器返回的 `commit_sha` 或其唯一十六进制前缀，再运行本地一次性评测：

```bash
python benchmark/longmemeval/run_home_eval.py --commit "$CANDIDATE_SHA"
```

服务方式见 [HoMe 评测指南](../../plugins/home/README.md#evaluate-through-the-service)。

### 3.3 结果

HoMe 在 LongMemEval-S 的 250 个训练 case 上优化。ReMe 基线和 HoMe 均在同一组 250 个留出 case 上评测。
Token 用量和 LLM 调用次数均为问答阶段的每题平均值。

#### ReMe 基线

| 题型 | 得分 | 总 Token | 输入 Token | 输出 Token | LLM 调用 |
|---|---:|---:|---:|---:|---:|
| **总体** | **0.888** | **43.9K** | **43.2K** | **0.8K** | **4.7** |
| Knowledge Update | 0.921 | 29.0K | 28.5K | 0.6K | 3.8 |
| Multi-session | 0.836 | 54.5K | 53.2K | 1.3K | 5.2 |
| Single-session Assistant | 1.000 | 15.8K | 15.5K | 0.3K | 2.9 |
| Single-session Preference | 0.667 | 31.4K | 30.6K | 0.7K | 4.1 |
| Single-session User | 0.971 | 26.6K | 26.3K | 0.3K | 3.5 |
| Temporal Reasoning | 0.882 | 65.7K | 64.8K | 0.9K | 6.2 |

#### HoMe

| 题型 | 得分 | 总 Token | 输入 Token | 输出 Token | LLM 调用 |
|---|---:|---:|---:|---:|---:|
| **总体** | **0.916** | **17.9K** | **17.2K** | **0.6K** | **3.0** |
| Knowledge Update | 0.974 | 9.7K | 9.3K | 0.5K | 2.2 |
| Multi-session | 0.806 | 19.3K | 18.3K | 1.0K | 3.3 |
| Single-session Assistant | 1.000 | 9.8K | 9.5K | 0.3K | 2.2 |
| Single-session Preference | 0.933 | 11.8K | 11.0K | 0.9K | 2.7 |
| Single-session User | 0.971 | 11.2K | 11.0K | 0.2K | 2.3 |
| Temporal Reasoning | 0.940 | 29.2K | 28.5K | 0.7K | 4.0 |
