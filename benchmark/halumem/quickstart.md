# Halumem
Experiment Quick Start Guide
This guide helps you quickly set up and run Halumem experiments with ReMe integration.

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
cd ./benchmark/halumem
git clone https://github.com/MemTensor/HaluMem.git
```

### 3. Run Experiments
Launch the ReMe service to enable memory library functionality:
```bash
clear && python benchmark/halumem/eval_reme.py \
    --data_path benchmark/halumem/HaluMem/data/HaluMem-Medium.jsonl \
    --reme_model_name gpt-4o-mini-2024-07-18 \
    --eval_model_name gpt-4o-mini-2024-07-18 \
    --batch_size 40 \
    --algo_version default
```

