## Towards Robust Tool Use in Agents via Experience-Driven Adaptive Guidance

**Language**: English (default) / [中文](./README_ZH.md)

> Paper: [arXiv:2608.03403](https://arxiv.org/abs/2608.03403)

<p align="center">
  <img src="gitcha.png" alt="ExpG challenges and overview" width="85%">
</p>

### Overview

This folder archives **ExpG**, a tool-use enhancement built on [Agentscope ReMe](https://github.com/agentscope-ai/ReMe). ExpG mines, distills, and reuses experience from historical tool calls to provide **capability boundaries** and **best-practice guidance**, which helps agents:

- Select and invoke tools more robustly under dynamic or noisy environments;
- Let smaller models with guidance outperform larger, memoryless baselines;
- Improve consistently across tool selection, tool calling, and response generation.

This work targets **ReMe 0.1.x–0.3.x with Tool Memory**. Current ReMe `main` (0.4+, Memory-as-File) has removed these APIs and needs additional adaptation.
**How ReMe is used:** Start ReMe's Tool Memory service; historical tool calls are written and evaluated via `add_tool_call_result`, distilled into tool-level guidance via `summary_tool_memory`, then retrieved and injected into later reasoning via `retrieve_tool_memory`. ReMe provides the vector store and service APIs; the acquisition / distillation / reuse strategy is implemented by ExpG.

---

### ExpG Mechanism

ExpG treats tool invocations as learnable experience and runs a three-stage pipeline:

1. **Experience Acquisition**
   - Analyze invocation quality from historical trajectories (success/failure, cost, latency, etc.);
   - Build structured experience units per tool, recording context, parameter patterns, and outcomes.

2. **Experience Distillation**
   - Filter noisy or unhelpful experiences and keep representative patterns;
   - Aggregate by equivalence classes to cover common and rare failure modes;
   - Summarize with an LLM into generalizable textual guidance.

3. **Experience Reuse**
   - Retrieve relevant experience / guidance for future tasks;
   - Inject guidance into tool selection, argument generation, and response synthesis;
   - Improve stability under dynamic environments and imperfect feedback.

---

### Main Results

Performance comparison (%) across MetaTool, API-Bank, and BFCL-V3. **Bold** indicates the best results within each model.

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

### Reference Code

| Path | Role |
| --- | --- |
| [`tool_memory.py`](./tool_memory.py) | Experiment-side HTTP client (`ToolMemoryFetcher`) that calls ReMe Tool Memory APIs |
| [`parse_tool_call_result_prompt.yaml`](./parse_tool_call_result_prompt.yaml) | Prompt for multi-aspect evaluation of each tool call |
| [`summary_tool_memory_prompt.yaml`](./summary_tool_memory_prompt.yaml) | Prompt for summarizing tool call history into guidance |
| [`tool_memory_flows.yaml`](./tool_memory_flows.yaml) | ReMe flow / op config excerpt for Tool Memory |

These are reference snippets for ReMe 0.1.x–0.3.x and are not runnable under current ReMe `main` without adaptation.

---

### Citation

```bibtex
@misc{wang2026expg,
  title         = {Towards Robust Tool Use in Agents via Experience-Driven Adaptive Guidance},
  author        = {Can Wang and Haoran Chen and Li Yu and Ding Hao and Bohai Zhao and Zhaoyang Liu and Zhiying Tu},
  year          = {2026},
  eprint        = {2608.03403},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2608.03403}
}
```
