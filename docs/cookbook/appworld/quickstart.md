# AppWorld
Experiment Quick Start Guide

This guide helps you quickly set up and run AppWorld experiments with ReMe integration.

## Env Setup

### 1. Clone the Repository

```bash
git clone https://github.com/agentscope-ai/ReMe.git
cd ReMe/benchmark/appworld
```

### 2. Appworld Environment Setup

Create a new conda environment with Python 3.12:

```bash
conda create -p ./appworld-env python==3.12
conda activate ./appworld-env
```

Install required Python packages:

```bash
pip install -r requirements.txt
```

Install AppWorld and download the dataset:

```bash
pip install appworld
appworld install
appworld download data
```

**Note**: The AppWorld data will be saved in the current directory.

### 3. Start ReMe Service

Install ReMe (if not already installed)
If you haven't installed the ReMe environment yet, follow these steps:
```bash
# Go back to the project root
cd ../..

# Create ReMe environment
conda create -p ./reme-env python==3.12
conda activate ./reme-env

# Install ReMe
pip install .
```

Launch the ReMe service to enable memory library functionality:

```bash
reme2 \
  backend=http \
  http.port=8002 \
  llms.default.model_name=qwen3-8b \
  embedding_models.default.model_name=text-embedding-v4 \
  vector_stores.default.backend=es \
  vector_stores.default.collection_name=appworld \
  vector_stores.default.hosts=http://xx.yy.zz.mm:nn
```

### 4. Common Issues

**AppWorld data not found**: Ensure `appworld download data` completed successfully

**pydantic version issue**:  AppWorld depends on an older version of pydantic, which is why a separate environment is needed. If you encounter issues running the experiments, try `pip install appworld` to override the dependencies.



## Run Experiments

### 1. Test: With Memory vs Without Memory

Run the main experiment script to compare performance with and without memory:

```bash
python run_appworld.py
```

**What this does:**
- Runs AppWorld tasks on the test-normal set
- Compares agent performance with ReMe memory (`use_memory=True`) vs without memory
- Uses multiple workers for parallel processing
- Runs each task multiple times for statistical significance
- Results are automatically saved to `./exp_result/` directory

**Configuration options in `run_appworld.py`:**
- `max_workers`: Number of parallel workers (default: 16)
- `num_runs`: Number of times each task is repeated (default: 4)
- `batch_size`: Number of concurrent tasks per batch (default: 8)
- `num_trials`: Maximum number of self-reflections, failure-aware reflection mechanism is triggered when num_trials>1 (default: 1)
- `model_name`: Task execution model (default: "qwen3-8b")
- `use_memory`: Whether to use ReMe memory library (default: True)
- `use_memory_addition`: Whether to enable selective addition (default: False)
- `use_memory_deletion`: Whether to enable utility-based deletion (default: False)

### 2. View Experiment Results

After running experiments, analyze the statistical results:

```bash
python run_exp_statistic.py
```

**What this script does:**
- Processes all result files in `./exp_result/`
- Calculates best@k, pass@k metrics for different k values
- Generates a summary table showing performance comparisons
- Saves results to `experiment_summary.csv`

**Metrics explained:**
- `best@k`: Takes groups of k runs per task, finds the maximum score in each group, then averages these maximums
- `pass@k`: Takes groups of k runs per task, measures the probability that at least one out of k independent task runs is successful.
- Higher k values show potential performance, lower k values show consistency
- In our AppWorld experiments, we report Task Goal Completion (TGC) metric, which measures percentage of tasks for which the agent passes all evaluation tests.

**Output Files**

- `./exp_result/*.jsonl`: Raw experiment results for each configuration
- `./exp_result/experiment_summary.csv`: Statistical summary table
- Console output: Real-time progress and summary statistics
