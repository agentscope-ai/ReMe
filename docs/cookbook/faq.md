# Frequently Asked Questions
This document provides answers to frequently asked questions about our paper "[Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution](https://arxiv.org/pdf/2512.10696)".

## Reproduction Questions
### 1. experimental configuration

**Example:** Qwen3-8B + AppWorld
**Launch the ReMe service:**
```bash
reme2 \
  backend=http \
  http.port=8002 \
  llms.default.model_name=qwen3-8b \
  embedding_models.default.model_name=text-embedding-v4 \
  vector_stores.default.backend=es \
  vector_stores.default.hosts=http://xx.yy.zz.mm:nn
```
**Evaluation Code:** [run_appworld.py](https://github.com/agentscope-ai/ReMe/blob/main/benchmark/appworld/run_appworld.py) with the following parameters
|Experimental Settings|No Memory |ReMe (fixed) |ReMe (dynamic)|
|---|---|---|---|
|max_workers|16|16|16|
|batch_size|8|8|8|
|num_runs|4|4|1|
|num_trials|1|1|3|
|model_name|"qwen3-8b"|"qwen3-8b"|"qwen3-8b"|
|use_memory| False| True|True|
|use_memory_addition|False|False|True|
|use_memory_deletion|False|False|True|
|memory_base_url|""|"http://0.0.0.0:8002/"|"http://0.0.0.0:8002/"|
|load_file_path|""|[appworld_qwen3_8b.jsonl](https://github.com/agentscope-ai/ReMe/tree/main/docs/library/paper_data/task/appworld_qwen3_8b.jsonl)|[appworld_qwen3_8b.jsonl](https://github.com/agentscope-ai/ReMe/tree/main/docs/library/paper_data/task/appworld_qwen3_8b.jsonl)|

For parameter meanings, you can refer to [docs/cookbook/appworld](https://github.com/zouyingcao/ReMe/blob/main/docs/cookbook/appworld/quickstart.md) .

> [!NOTE]
> - Qwen3 thinking mode is activated for BFCL-V3 tasks and disabled for AppWorld tasks.
> - In ReMe(fixed) setting, there is no need to restart the ReMe service at each run since the experience pool is fixed. However, in ReMe(dynamic) setting, we need run separately to ensure consistent initial state. That is to say, to calculate Pass@4, you need 4 independent runs with restarting ReMe service and setting `num_runs=1` in each run.

### 2. about experience pool initialization
Taking Appworld as an example, you can refer to issues [#55](https://github.com/agentscope-ai/ReMe/issues/55), [#58](https://github.com/agentscope-ai/ReMe/issues/58). To reproduce the results in our paper, you can use our constructed memory data in [docs/library/paper_data](https://github.com/agentscope-ai/ReMe/tree/main/docs/library/paper_data/task).

### 3. evaluation metrics
- In our AppWorld experiments, we report Task Goal Completion (TGC) metric (claimed in Appendix A of our [paper](https://arxiv.org/pdf/2512.10696)), which measures percentage of tasks for which the agent passes all evaluation tests. [`after_score`](https://github.com/agentscope-ai/ReMe/blob/main/benchmark/appworld/appworld_react_agent.py#L218) is the percentage of tests passed for per task. To calculate TGC, only `after_score=1` means task completion. Therefore, we use threshold=1 in [run_exp_statistic.py](https://github.com/agentscope-ai/ReMe/blob/main/benchmark/appworld/run_exp_statistic.py#L43) to get Pass@k.
- In our paper, `Avg@4` is the `Pass@1` performance averaged over 4 independent runs. For simplicity, we organize the total collected 4 trajectories in a single file to calculate Pass@1 and Pass@4 together. Then, the results of Pass@1 and Avg@4 are equivalent.


### 4. reproduce baselines
- For Qwen3-series No-Memory performance on AppWorld, you can refer to issue [#49](https://github.com/agentscope-ai/ReMe/issues/49).
- About A-mem and LangMem code, please see [#67](https://github.com/agentscope-ai/ReMe/issues/67).

## Environment Setup
### 1. BFCL-V3 code version
We use the BFCL GitHub repository with commit_id=[ea13468](https://github.com/ShishirPatil/gorilla/commit/ea13468e4423454d0c213704fb87cf7cb3990433) in our experiments.

### 2. preprocess BFCL-V3 multi_turn_base data
Before running the experiments, you need to preprocess the BFCL-V3 data using this [script](https://github.com/agentscope-ai/ReMe/blob/main/benchmark/bfcl/preprocess.py) to get the suitable data format. Then, we randomly split the multi-turn-base data into train (50) and test (150) sets using [split_into_trainval.py](https://github.com/agentscope-ai/ReMe/blob/main/benchmark/bfcl/split_into_trainval.py) (our used split is [here](https://github.com/agentscope-ai/ReMe/issues/45#issuecomment-3890215360)). The training set is used to construct the initial experience pool and the remaining 150 testing tasks serve as the evaluation set.

### 3. pydantic version issue when running Appworld
AppWorld depends on an older version of pydantic, which is why a separate environment is needed. If you encounter issues running the experiments, try `pip install appworld` to override the dependencies.

### 4. AppWorld data not found
Ensure `appworld download data` completed successfully.

## Technical Questions
### 1. about memory growth
See [#44](https://github.com/agentscope-ai/ReMe/issues/44).
### 2. code for Experience Refinement
See [#52](https://github.com/agentscope-ai/ReMe/issues/52).
### 3. context length issue with AppWorld
See [#81](https://github.com/agentscope-ai/ReMe/issues/81).
