# ReMe 实验配置与结果

> 数据来源：远端分支 `origin/dev/0517` 的 `README.md` 及 `benchmark/locomo/eval_reme.py`

---

## 一、实验配置

### 1.1 数据集

本次实验共涉及四个基准测试，分属两篇论文：

| 数据集 | 所属论文 | 类型 | 评测指标 |
|--------|---------|------|---------|
| **LoCoMo** | 长期记忆论文 | 对话式问答 | Single Hop、Multi Hop、Temporal、Open Domain、Overall |
| **HaluMem** | 长期记忆论文 | 记忆幻觉测试 | Memory Integrity、Memory Accuracy、QA Accuracy |
| **Appworld** | 过程记忆论文 | 任务完成 | Avg@4、Pass@4 |
| **BFCL-V3** | 过程记忆论文 | 函数调用（multi-turn-base） | Avg@4、Pass@4 |

### 1.2 模型

| 角色 | 模型 | 说明 |
|------|------|------|
| **ReMe 主干模型**（记忆摘要与检索） | `qwen-flash` | 用于记忆提取、摘要与检索 |
| **答案生成模型**（问答响应） | `qwen3-max` | 根据检索到的记忆生成结构化回答 |
| **评判模型**（LLM-as-a-Judge） | `qwen3-max`（via `qwen3_max_instruct`） | 对生成的答案进行 CORRECT/WRONG 判定 |

> **注意：** README 中写明"每个答案由 **GPT-4o-mini** 评分"，但实际代码中使用的是 **qwen3-max** 进行评判，README 描述与实现存在差异。

**过程记忆论文（Appworld & BFCL-V3）使用的模型：**
- Appworld：`Qwen3-8B`（非思考模式）
- BFCL-V3：`Qwen3-8B`（思考模式），随机划分 50 训练 / 150 验证

### 1.3 参数配置（LoCoMo 基准测试）

来源：`benchmark/locomo/eval_reme.py` 中的 `EvalConfig` 数据类。

| 参数 | 值 | 说明 |
|------|-----|------|
| `top_k` | 20 | 检索返回的记忆数量（top-k） |
| `user_num` | 1 | 处理的用户数量 |
| `max_concurrency` | 2 | 最大并发数 |
| `batch_size` | 40 | 对话记忆提取的批大小 |
| `enable_thinking_params` | False | 是否启用思考模式 |
| `algo_version` | `"locomo"` | 算法版本标签 |
| `time_interval` | 60 | 对话消息间的时间间隔（秒，用于时间戳格式化） |
| `output_dir` | `"bench_results/reme"` | 结果输出目录 |

### 1.4 评测协议

1. **LLM-as-a-Judge**：遵循 MemOS 方法论
2. 评判模型将每个答案标记为 **CORRECT** 或 **WRONG**
3. 评判时对比：问题 + 标准答案 + 生成答案
4. 基线结果来自各基线方法的原始论文，在对齐设置下复现
5. LoCoMo 对话按 session 划分（Caroline_Melanie 为 19 个 session，其余为 `len(conversation)/2 - 1`）

---

## 二、实验结果

### 2.1 LoCoMo 基准测试

| 方法 | Single Hop | Multi Hop | Temporal | Open Domain | Overall |
|------|-----------|-----------|----------|-------------|---------|
| MemoryOS | 62.43 | 56.50 | 37.18 | 40.28 | 54.70 |
| Mem0 | 66.71 | 58.16 | 55.45 | 40.62 | 61.00 |
| MemU | 72.77 | 62.41 | 33.96 | 46.88 | 61.15 |
| MemOS | 81.45 | 69.15 | 72.27 | 60.42 | 75.87 |
| HiMem | 89.22 | 70.92 | 74.77 | 54.86 | 80.71 |
| Zep | 88.11 | 71.99 | 74.45 | 66.67 | 81.06 |
| TiMem | 81.43 | 62.20 | 77.63 | 52.08 | 75.30 |
| TSM | 84.30 | 66.67 | 71.03 | 58.33 | 76.69 |
| MemR3 | 89.44 | 71.39 | 76.22 | 61.11 | 81.55 |
| **ReMe** | **89.89** | **82.98** | **83.80** | **71.88** | **86.23** |

### 2.2 HaluMem 基准测试

| 方法 | Memory Integrity | Memory Accuracy | QA Accuracy |
|------|-----------------|----------------|-------------|
| MemoBase | 14.55 | 92.24 | 35.53 |
| Supermemory | 41.53 | 90.32 | 54.07 |
| Mem0 | 42.91 | 86.26 | 53.02 |
| ProMem | **73.80** | 89.47 | 62.26 |
| **ReMe** | 67.72 | **94.06** | **88.78** |

### 2.3 Appworld 基准测试

> 模型：Qwen3-8B（非思考模式）

| 方法 | Avg@4 | Pass@4 |
|------|-------|--------|
| w/o ReMe | 0.1497 | 0.3285 |
| w/ ReMe | 0.1706 **(+2.09%)** | 0.3631 **(+3.46%)** |

> Pass@K 衡量在 K 个生成候选中至少有一个成功完成任务（score=1）的概率。
>
> 注：当前实验使用内部 AppWorld 环境，可能与公开版本略有差异。

### 2.4 BFCL-V3 基准测试

> 模型：Qwen3-8B（思考模式），multi-turn-base 任务，随机划分 50 训练 / 150 验证

| 方法 | Avg@4 | Pass@4 |
|------|-------|--------|
| w/o ReMe | 0.4033 | 0.5955 |
| w/ ReMe | 0.4450 **(+4.17%)** | 0.6577 **(+6.22%)** |

---

## 三、总结

- **LoCoMo**：ReMe 在所有子任务上均取得最优成绩，Overall 达到 **86.23**，领先第二名 MemR3（81.55）约 4.68 个百分点。
- **HaluMem**：ReMe 在 Memory Accuracy（94.06）和 QA Accuracy（88.78）上最优，Memory Integrity 略低于 ProMem。
- **Appworld**：接入 ReMe 后 Avg@4 提升 2.09%，Pass@4 提升 3.46%。
- **BFCL-V3**：接入 ReMe 后 Avg@4 提升 4.17%，Pass@4 提升 6.22%。
