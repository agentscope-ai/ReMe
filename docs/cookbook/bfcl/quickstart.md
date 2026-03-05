# BFCL
Experiment Quick Start Guide

This guide helps you quickly set up and run BFCL experiments with ReMe integration.

## Env Setup

### 1. BFCL installation

#### Clone the repository
```bash
cd ReMe/benchmark/bfcl
git clone https://github.com/ShishirPatil/gorilla.git
cd gorilla
git checkout ea13468
```

#### Change directory to the `berkeley-function-call-leaderboard`
```bash
cd berkeley-function-call-leaderboard
```

#### Install the package in editable mode
```bash
pip install -e .
cd ../..
pip install -r requirements.txt
```

#### Move the dataset to the data folder under bfcl
```bash
cp -r gorilla/berkeley-function-call-leaderboard/bfcl_eval/data ./
```

#### Preprocess the data to get the suitable data format
```bash
python preprocess.py
```

**Note**: The original BFCL data is designed as a benchmark dataset and does not have a train/validation split, you can use ``split_into_trainval.py`` to split data into train and validation sets.

```bash
python split_into_trainval.py --input ./data/multiturn_data_base.jsonl  --train ./data/multiturn_data_base_train.jsonl --val ./data/multiturn_data_base_val.jsonl
```

### 2. Start ReMe Service

After collecting trajectories, Launch the ReMe service (make sure you have installed ReMe environment, if not please follow the steps in the [ReMe Installation Guide](https://github.com/agentscope-ai/ReMe/blob/main/doc/README.md) to install):

```bash
reme2 \
  backend=http \
  http.port=8002 \
  llms.default.model_name=qwen3-8b \
  embedding_models.default.model_name=text-embedding-v4 \
  vector_stores.default.backend=local \
  vector_stores.default.collection_name=bfcl
```

<details>
<summary>Option: init the task memory pool from scratch</summary>

- First, collect agent trajectories on training data set without task memory:

  ```bash
  # important: num_runs = 8, use_memory = False, experiment_suffix="wo-memory", data_path="data/multiturn_data_base_train.jsonl"
  python run_bfcl.py
  ```

- Second, using ReMe to construct the initial task memory pool:
  ```bash
  python init_task_memory_pool.py --jsonl_file ./exp_result/qwen3-8b/with_think/bfcl-multi-turn-base_wo-memory.jsonl
  ```

  > Parameters:
  > `jsonl_file`: Path to the collloaded trajectories
  > `service_url`: ReMe service URL (default: `http://localhost:8002`)
  > `n_threads`: Number of threads for processing
  > `output_file`: Output file to save results (optional)

  Now you have inited the task memory pool using `local` backend. Then, run the following `curl` command to dump the memory library:
  ```bash
  curl -X POST "http://0.0.0.0:8002/dump_memory" \
    -H "Content-Type: application/json" \
    -d '{
      "dump_file_path": "./library/bfcl.jsonl",
    }'
  ```

- Next time, you can import this previously exported task memory data to populate the new started workspace with existing knowledge:
  ```bash
  curl -X POST "http://0.0.0.0:8002/load_memory" \
    -H "Content-Type: application/json" \
    -d '{
      "load_file_path": "./library/bfcl.jsonl",
      "clear_existing": true
    }'
  ```
</details>

### 3. Run Experiments on Validation Set

Run you can compare agent performance on the validation set with task memory (`use_memory=True`) and without task memory:

```bash
# remember to change the configuration options, e.g., `data_path=./data/multiturn_data_base_val.jsonl`
python run_bfcl.py
```

**Note**:
- `max_workers`: Number of parallel workers
- `num_runs`: Number of times each task is repeated
- `model_name`: LLM model name
- `enable_thinking`: Control the model's thinking mode
- `data_path`: Path to the training dataset (default: `./data/multiturn_data_base_val.jsonl`)
- `answer_path`: Path to the possible answer, which are used to evaluate the model's output function (default: `./data/possible_answer`)
- Results are automatically saved to `./exp_result/{model_name}/{no_think/with_think}` directory

After running experiments, analyze the statistical results:

```bash
python run_exp_statistic.py
```

**What this script does:**
- Processes all result files in `./exp_result/`
- Calculates best@k&pass@k metrics for different k values
- Generates a summary table showing performance comparisons
- Saves results to `experiment_summary.csv`
