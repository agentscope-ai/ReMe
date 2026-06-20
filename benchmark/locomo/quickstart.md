# LoCoMo — ReMeLight / ReMe 评测快速开始

### 1. 安装 ReMe

```bash
pip install -e ".[light]"
```

### 2. 下载数据集

```bash
cd benchmark/locomo
mkdir -p data

# 克隆原始 LoCoMo 仓库（包含 locomo10.json）
git clone https://github.com/luyanhexay/locomo-dynamemory.git /tmp/locomo-dynamemory
cp /tmp/locomo-dynamemory/data/locomo10.json data/
```

数据集信息：
- 论文: [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753)
- 项目页: https://snap-research.github.io/locomo
- 原始仓库: https://github.com/luyanhexay/locomo-dynamemory

### 3. 运行向量版评测（ReMe）

```bash
python benchmark/locomo/eval_reme.py \
    --data_path benchmark/locomo/data/locomo10.json \
    --reme_model_name qwen-flash \
    --eval_model_name qwen3-max \
    --top_k 20 \
    --user_num 5 \
    --max_concurrency 2
```

### 4. 运行文件版评测（ReMeLight）

```bash
python benchmark/locomo/eval_reme_light.py \
    --data_path benchmark/locomo/data/locomo10.json \
    --reme_model_name qwen-flash \
    --eval_model_name qwen3-max \
    --top_k 20 \
    --user_num 5 \
    --max_concurrency 2
```

首次跑建议 `--user_num 1` 验证流程，确认没问题再加。

### 5. 查看结果

```bash
# 最终指标
cat bench_results/reme_light/eval_statistics.json

# 逐条 QA 详情
cat bench_results/reme_light/eval_results.jsonl

# 文件版特有的：直接看记忆写得好不好
ls bench_results/reme_light/working_dirs/<user_name>/memory/
```
