# LongMemEval 评测

[English version](./README.md)

LongMemEval 是一个面向**多轮多会话历史的长期记忆能力**的评测基准。每个条目提供一组按时间
顺序排列的用户与助手之间的会话，以及一个只能通过推理用户自有记忆才能回答的探测问题。ReMe
将会话摄入按条目隔离的工作区，以 agentic（ReAct）模式回答问题，最后由 LLM-as-judge 打分。

题型包括单会话（user / assistant / preference）、多会话推理、知识更新与时间推理等。

在仓库根目录安装 ReMe 及运行依赖即可，无需执行 `reme plugins install`：

```bash
python -m pip install -e ".[as]"
```

runner 在每个 worker 中组合 ReMe 内置的 `benchmark` 配置与仓库内的
[插件源码](../../plugins/lme/README_ZH.md)，不安装插件、不修改全局 registry。
本目录继续保留评测参数、数据集及输出。自定义完整应用配置路径仍可通过 `reme.config` 指定，
并可使用 `extends: benchmark`。
模型凭据通过公共 benchmark 配置中声明的环境变量设置。

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
3. 若自定义应用配置启用了 `auto_dream`，在相邻会话跨越配置时刻（默认 23:00）时触发；插件预设保持关闭。
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
| `reme.config` | 使用的 ReMe 配置（`benchmark`）。 |
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

agentscope==2.0.4.post1, conda reme env, 32 workers, eval-only（复用预构建记忆）
（2026-08-06，500 题，总计 10.0 min）

| 类型 | Agentic | input tok/q | output tok/q | total tok/q | tool calls/q |
|---|---|---|---|---|---|
| knowledge-update | 0.910 | 31,581 | 589 | 32,169 | 2.90 |
| multi-session | 0.842 | 52,837 | 1,474 | 54,311 | 4.21 |
| single-session-assistant | 1.000 | 15,596 | 279 | 15,875 | 1.89 |
| single-session-preference | 0.633 | 36,802 | 818 | 37,620 | 3.60 |
| single-session-user | 0.986 | 27,433 | 359 | 27,792 | 2.60 |
| temporal-reasoning | 0.902 | 62,674 | 985 | 63,659 | 4.97 |
| **OVERALL** | **0.894** | **43,448** | **876** | **44,324** | **3.69** |
