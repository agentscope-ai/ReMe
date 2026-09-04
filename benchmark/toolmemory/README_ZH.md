## Towards Robust Tool Use in Agents via Experience-Driven Adaptive Guidance

**语言**：中文 / [English](./README.md)

> 论文：[arXiv:2608.03403](https://arxiv.org/abs/2608.03403)
> 代码：[https://github.com/WangCan1178/ExpG](https://github.com/WangCan1178/ExpG)

<p align="center">
  <img src="./gitcha.png" alt="ExpG 挑战与概览" width="85%">
</p>

### 简介

本目录归档基于 [Agentscope ReMe](https://github.com/agentscope-ai/ReMe) 的工具使用增强工作 **ExpG**：在 ReMe 记忆框架之上，从历史工具调用中挖掘、提炼并复用经验，为智能体提供工具的 **能力边界** 与 **最佳实践指导**，从而：

- 在动态或有噪环境下更鲁棒地选择和调用工具；
- 让较小模型在带有经验指导时超越更大、但无记忆的基线；
- 在工具选择、工具调用和响应生成等多个阶段带来一致收益。

**如何使用 ReMe：** 启动 Tool Memory 服务后，历史工具调用经 `add_tool_call_result` 写入并评估，经 `summary_tool_memory` 蒸馏成工具级指导，再经 `retrieve_tool_memory` 取回并注入后续推理。向量存储与服务接口由 ReMe 提供，经验获取 / 蒸馏 / 复用策略由 ExpG 实现。完整实现与实验见 [WangCan1178/ExpG](https://github.com/WangCan1178/ExpG)。

---

### ExpG 机制概览

ExpG 将工具调用视为可学习经验，并通过三阶段流水线完成经验的获取、提炼与复用：

1. **经验获取（Experience Acquisition）**
   - 从历史工具调用轨迹中分析调用质量（成功/失败、代价、时间等）；
   - 针对不同工具构建结构化的经验单元，记录调用上下文、参数模式和结果。

2. **经验蒸馏（Experience Distillation）**
   - 过滤无效 / 噪声经验，保留具有代表性的调用模式；
   - 基于“等价类”视角对经验进行聚合，覆盖常见模式与稀有失败模式；
   - 使用 LLM 对经验进行总结，形成可泛化的文本化指导（guidance）。

3. **经验复用（Experience Reuse）**
   - 在未来任务中，根据当前工具调用上下文检索相关经验 / 指导；
   - 将经验引导融入到工具选择、参数生成和响应整理等环节；
   - 使得代理在面对动态环境和不完美反馈时仍能保持稳定表现。

---

### 主实验结果

MetaTool、API-Bank、BFCL-V3 上的性能对比（%）。**加粗**为各模型组内最优。

| Model | Method | MetaTool Pass@1 | MetaTool Avg@3 | MetaTool Pass@3 | API-Bank Pass@1 | API-Bank Avg@3 | API-Bank Pass@3 | BFCL-V3 Pass@1 | BFCL-V3 Avg@3 | BFCL-V3 Pass@3 | Total Pass@1 | Total Avg@3 | Total Pass@3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5 nano | No Method | 72.62 | 72.76 | 78.49 | 82.96 | 83.46 | 86.97 | 53.80 | 53.00 | 60.95 | 70.82 | 70.62 | 76.63 |
| GPT-5 nano | Few-shot | 74.12 | 75.11 | 82.32 | 83.71 | 83.96 | **87.22** | 56.18 | 55.24 | 61.39 | 72.36 | 72.65 | 79.28 |
| GPT-5 nano | DRAFT | 73.94 | 73.04 | 78.97 | 84.21 | 83.46 | **87.22** | 57.27 | 57.27 | 62.26 | 72.52 | 71.58 | 77.23 |
| GPT-5 nano | Mem0 | 74.96 | 76.13 | 82.92 | 84.96 | 85.21 | **87.22** | 60.95 | 61.61 | 65.08 | 73.98 | 74.67 | 80.35 |
| GPT-5 nano | **ExpG** | **81.67** | **82.07** | **84.60** | **86.72** | **86.55** | **87.22** | **64.43** | **63.99** | **66.38** | **79.32** | **79.22** | **81.69** |
| DeepSeek-V3 | No Method | 83.10 | 82.94 | 84.66 | 84.71 | 84.38 | 85.46 | 58.79 | 59.65 | 65.94 | 78.92 | 78.66 | 81.37 |
| DeepSeek-V3 | Few-shot | 82.74 | 83.90 | 86.28 | 85.21 | 84.63 | 86.22 | 60.52 | 60.30 | 67.90 | 79.08 | 79.45 | 82.92 |
| DeepSeek-V3 | DRAFT | 80.23 | 80.79 | 82.44 | 84.96 | 85.63 | 86.47 | 62.26 | 61.61 | 68.55 | 77.70 | 77.80 | 80.54 |
| DeepSeek-V3 | Mem0 | 83.88 | 84.56 | 86.40 | 85.46 | 85.55 | 86.47 | 65.08 | 65.15 | 68.33 | 80.70 | 80.91 | 83.12 |
| DeepSeek-V3 | **ExpG** | **85.26** | **85.38** | **86.52** | **87.72** | **87.39** | **87.97** | **69.41** | **69.92** | **72.02** | **82.76** | **82.61** | **84.11** |
| Qwen3-8B | No Method | 76.51 | 76.97 | 77.71 | 83.96 | 83.88 | 84.21 | 58.79 | 58.28 | 60.30 | 74.46 | 74.41 | 75.56 |
| Qwen3-8B | Few-shot | 79.93 | 79.83 | 82.92 | 83.71 | 82.62 | 84.96 | 60.09 | 59.29 | 61.39 | 76.91 | 76.27 | 79.32 |
| Qwen3-8B | DRAFT | 78.19 | 77.33 | 77.89 | 85.71 | 84.96 | 85.46 | 60.74 | 60.30 | 62.91 | 76.20 | 75.18 | 76.35 |
| Qwen3-8B | Mem0 | 75.07 | 75.47 | 82.38 | 86.22 | 86.05 | 86.47 | 63.34 | 64.93 | 66.16 | 74.69 | 74.98 | 80.07 |
| Qwen3-8B | **ExpG** | **83.52** | **84.88** | **85.08** | **86.47** | **87.89** | **87.97** | **67.46** | **66.96** | **68.33** | **81.06** | **81.82** | **82.48** |
| Qwen3-32B | No Method | 80.05 | 79.43 | 80.17 | 84.71 | 84.88 | 85.21 | 65.15 | 65.08 | 66.16 | 78.05 | 77.55 | 78.41 |
| Qwen3-32B | **ExpG** | **84.68** | **85.02** | **86.28** | **86.97** | **87.30** | **87.72** | **70.72** | **71.01** | **73.32** | **82.48** | **82.56** | **84.14** |
| Qwen3-235B | No Method | 78.25 | 79.23 | 80.29 | 85.46 | 85.46 | 85.71 | 71.37 | 71.15 | 73.54 | 78.13 | 78.49 | 79.91 |
| Qwen3-235B | **ExpG** | **86.34** | **86.70** | **86.94** | **87.47** | **86.97** | **88.22** | **79.61** | **78.52** | **80.04** | **85.29** | **84.98** | **85.69** |

---

### 参考代码

| 路径 | 作用 |
| --- | --- |
| [`tool_memory.py`](./tool_memory.py) | 官方风格 ReMe Tool Memory HTTP 客户端（`add_tool_call_result` / `summary_tool_memory` / `retrieve_tool_memory`） |
| [`parse_tool_call_result_prompt.yaml`](./parse_tool_call_result_prompt.yaml) | 单次工具调用多维评估用的 prompt |
| [`summary_tool_memory_prompt.yaml`](./summary_tool_memory_prompt.yaml) | 将工具调用历史总结为 guidance 的 prompt |
| [`tool_memory_flows.yaml`](./tool_memory_flows.yaml) | Tool Memory 相关的 flow / op 配置摘录 |

以上为参考片段。完整可运行代码见 [WangCan1178/ExpG](https://github.com/WangCan1178/ExpG)。

---

### 引用

```bibtex
@misc{wang2026expg,
  title         = {Towards Robust Tool Use in Agents via Experience-Driven Adaptive Guidance},
  author        = {Can Wang and Haoran Chen and Li Yu and Ding Hao and Bohai Zhao and Zhaoyang Liu and Zhiying Tu},
  year          = {2026},
  eprint        = {2608.03403},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2608.03403},
  howpublished  = {\url{https://github.com/WangCan1178/ExpG}}
}
```
