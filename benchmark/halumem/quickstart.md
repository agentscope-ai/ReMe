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

### 2. Download the Dataset
```bash
cd ./benchmark/halumem
mkdir -p data
curl -L "https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/main/HaluMem-Medium.jsonl?download=true" -o data/HaluMem-Medium.jsonl
curl -L "https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/main/HaluMem-Long.jsonl?download=true" -o data/HaluMem-Long.jsonl
```

Dataset page:
https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/tree/main

If the official source is slow or inaccessible in mainland China, you can use a mirror:
```bash
cd ./benchmark/halumem
mkdir -p data
curl -L "https://hf-mirror.com/datasets/IAAR-Shanghai/HaluMem/resolve/main/HaluMem-Medium.jsonl?download=true" -o data/HaluMem-Medium.jsonl
curl -L "https://hf-mirror.com/datasets/IAAR-Shanghai/HaluMem/resolve/main/HaluMem-Long.jsonl?download=true" -o data/HaluMem-Long.jsonl
```

### 3. Run Experiments
Launch the ReMe service to enable memory library functionality:
```bash
clear && python benchmark/halumem/eval_reme.py \
    --data_path benchmark/halumem/data/HaluMem-Medium.jsonl \
    --reme_model_name gpt-4o-mini-2024-07-18 \
    --eval_model_name gpt-4o-mini-2024-07-18 \
    --batch_size 40 \
    --algo_version default
```

