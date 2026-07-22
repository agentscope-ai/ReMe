# Dataset Download Guide

This directory contains datasets required for ReMe evaluation. Some datasets are large and excluded from Git version control — they must be downloaded manually.

## LongMemEval (cleaned-S)

ReMe uses only the **cleaned-S** split of LongMemEval, hosted on HuggingFace:
[agentscope-ai/ReMe_longmemeval_clean_s_v2](https://huggingface.co/datasets/agentscope-ai/ReMe_longmemeval_clean_s_v2)
(the script downloads via the hf-mirror.com mirror).

Download it with:

```bash
cd benchmark/datasets/longmemeval

# Download the cleaned-S data file (skipped automatically if it already exists)
python download.py
```

After downloading, the directory should contain:

| File | Description |
| --- | --- |
| `longmemeval_s_reme_cleaned.json` | cleaned-S dataset with ground truth fields included |
| `download.py` | Download script (included in repo) |

> **Note**: The download script uses hf-mirror.com by default. To use a different mirror, modify `BASE_URL` in `download.py`.

Once the download completes, follow [`benchmark/README.md`](../README.md) to run the LongMemEval evaluation.

## BEAM

BEAM is a public repository. Clone it directly into the `benchmark/datasets/` directory:

```bash
cd benchmark/datasets
git clone https://github.com/mohammadtavakoli78/BEAM.git
```

After cloning, `benchmark/datasets/BEAM/` should contain `chats/`, `src/`, `topics/` and other subdirectories.
