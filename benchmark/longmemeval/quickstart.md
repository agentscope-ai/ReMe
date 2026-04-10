# Longmemeval
Experiment Quick Start Guide
This guide helps you quickly set up and run Longmemeval experiments with ReMe integration.

### 1. Start ReMe Service
Install ReMe (if not already installed)
If you haven't installed the ReMe environment yet, follow these steps:
```bash
# Create ReMe environment
conda create -p ./reme-env python==3.12
conda activate ./reme-env

# Install ReMe
pip install .
```

### 2. Clone the Repository
```bash
cd ./benchmark/longmemeval
mkdir -p data/
cd data/
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json
cd ..
```

### 3. Run Experiments
Launch the ReMe service to enable memory library functionality:
```bash
clear && python benchmark/longmemeval/eval_longmemeval_reme.py \
    --data_path benchmark/longmemeval/data/longmemeval_s_cleaned.json \
    --reme_model_name qwen-flash \
    --reme_model_name retrieve_model_name \
    --eval_model_name gpt-4o-mini-2024-07-18 \
    --batch_size 20 \
    --algo_version default
```


### 4. Evaluate Results
Evaluate the results of the experiments:
```bash
python benchmark/longmememeval/compute_stats.py \
    --results_dir bench_results/longmemeval_reme \
    --output_file bench_results/longmemeval_reme/statistics.json
```
The `compute_stats.py` script computes various statistics from the evaluation results.