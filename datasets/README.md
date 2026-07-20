# 数据集下载说明

本目录包含 ReMe 评测所需的数据集。部分数据集体积较大，不纳入 Git 版本管理，需要手动下载。

## LongMemEval（cleaned-S）

ReMe 仅使用 LongMemEval 的 **cleaned-S** 版本，数据托管在 HuggingFace：
[agentscope-ai/ReMe_longmemeval_clean_s_v2](https://huggingface.co/datasets/agentscope-ai/ReMe_longmemeval_clean_s_v2)
（下载脚本经 hf-mirror.com 镜像源获取）。

按以下步骤下载：

```bash
cd datasets/longmemeval

# 下载 cleaned-S 数据文件（已存在则自动跳过）
python download.py
```

下载完成后，目录下应包含以下文件：

| 文件名 | 说明 |
| --- | --- |
| `longmemeval_s_reme_cleaned.json` | cleaned-S 数据集，已包含 ground truth 字段 |
| `download.py` | 下载脚本（已随仓库提供） |

> **注意**：下载脚本使用 hf-mirror.com 镜像源，如需更换源请修改 `download.py` 中的 `BASE_URL`。

下载完成后即可参照 [`benchmark/README_ZH.md`](../benchmark/README_ZH.md) 运行 LongMemEval 评测。

## BEAM

BEAM 数据集为公开仓库，直接 clone 到 `datasets/` 目录下即可：

```bash
cd datasets
git clone https://github.com/mohammadtavakoli78/BEAM.git
```

clone 完成后，`datasets/BEAM/` 目录下应包含 `chats/`、`src/`、`topics/` 等子目录。
