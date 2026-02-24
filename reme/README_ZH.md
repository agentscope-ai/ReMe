# ReMe Memory Agent

> **Agent 驱动的记忆管理** — 由 LLM Agent 自主决定*何时*、*存什么*、*如何*存储与检索记忆，使用工具而非固定流水线。

ReMe Memory Agent 是 ReMe 的一个独立版本：记忆的摘要、存储、检索均由**配备工具的 LLM Agent** 完成。区别于僵化的规则流水线，Agent 会动态决定要加入哪些记忆、如何组织，以及如何跨多层级（Profile、向量库、历史）进行检索。

---

## 🧠 核心思路：记忆即 Agent 任务

传统记忆系统把压缩与检索当成确定性流水线：

- **固定窗口摘要** 忽略信息密度的不均匀分布 — 在低熵内容上浪费空间，在高熵交互中丢失关键语义。
- **纯向量检索** 把检索当作距离计算 — 难以处理时序歧义（如「去年方案」vs「今年方案」），也无法做多跳推理。
- **扁平化存储** 缺乏溯源能力 — 要么因过度摘要丢失细节，要么因保留原文而噪声激增。

ReMe Memory Agent 将记忆管理视为 **ReAct 式 Agent 任务**：

| 维度 | 传统做法 | ReMe Memory Agent |
|------|----------|-------------------|
| **摘要** | 固定窗口或启发式阈值 | Agent 评估语义复杂度与任务价值，自主选择编码粒度 |
| **检索** | 单层向量相似度 | Agent 在 User Profile、短期窗口、长期历史间跨层导航 |
| **查询处理** | 直接嵌入查找 | Agent 可解构模糊查询，修正语义漂移 |
| **时间感知** | 时间无关的嵌入 | 可选时间过滤与混合时空索引 |

---

## 🏗️ 架构

### Agent 层次结构

```
ReMe (Application)
    │
    ├── summarize_memory() ──► ReMeSummarizer
    │       │
    │       ├── AddHistory (tool)
    │       └── DelegateTask (tool)
    │               │
    │               ├── PersonalSummarizer  ──► AddAndRetrieveSimilarMemory, UpdateMemoryV2,
    │               │                           AddDraftAndReadAllProfiles, UpdateProfile
    │               ├── ProceduralSummarizer
    │               └── ToolSummarizer
    │
    └── retrieve_memory() ──► ReMeRetriever
            └── DelegateTask (tool)
                    │
                    ├── PersonalRetriever  ──► ReadAllProfiles, RetrieveMemory, ReadHistory
                    ├── ProceduralRetriever
                    └── ToolRetriever
```

- **ReMeSummarizer** / **ReMeRetriever** 负责编排流程，并将任务分派给专业 Agent。
- **DelegateTask** 根据 `memory_target`（user / task / tool）将工作路由到对应 Agent。
- 各专业 Agent 使用 `RetrieveMemory`、`AddMemory`、`UpdateProfile`、`ReadHistory` 等工具。
- `BaseMemoryAgent` 继承 `BaseReact` — Agent 通过 **推理 + 行动** 循环选择工具并解释结果。

### 核心组件（对应代码）

| 组件 | 文件 | 职责 |
|------|------|------|
| `ReMe` | `reme.py` | 主入口：`summarize_memory()`、`retrieve_memory()`、`add_memory()` 等 |
| `ReMeSummarizer` | `agent/memory/reme_summarizer.py` | 摘要编排；使用 AddHistory、DelegateTask |
| `ReMeRetriever` | `agent/memory/reme_retriever.py` | 检索编排；分派给 Personal/Procedural/Tool Agent |
| `PersonalSummarizer` | `agent/memory/personal/personal_summarizer.py` | 两阶段：(1) 增/查记忆 (2) 更新 Profile |
| `PersonalRetriever` | `agent/memory/personal/personal_retriever.py` | 结合 Profile + 向量 + 历史进行检索 |
| `DelegateTask` | `tool/memory/delegate_task.py` | 将任务路由到对应记忆 Agent |
| `RetrieveMemory` | `tool/memory/vector/retrieve_memory.py` | 语义相似检索，支持可选时间过滤 |
| `ReadAllProfiles` | `tool/memory/profiles/read_all_profiles.py` | 加载 User Profile（短期状态） |
| `UpdateProfile` | `tool/memory/profiles/update_profile.py` | 根据交互更新 User Profile |

---

## ✨ Agent 能力

### 1. 层次化检索

Agent 在多个层级间导航，而非单一向量索引：

- **User Profile** — 高优先级、低延迟的工作记忆，承载即时偏好与状态。
- **短期窗口** — 近期消息或历史块。
- **长期历史** — 向量库中的持久化记忆。

Agent 决定何时查 Profile、何时搜向量、何时读历史，从而提升相关性并降低检索噪声。

### 2. 多粒度存储

不同抽象层级并存：

- 高层摘要用于快速语义定位
- 原始上下文指针用于事实校验

类似人类的「闪光灯记忆」与「语义记忆」，在长对话中提升连贯性、减少事实幻觉。

### 3. 时间感知检索

嵌入模型通常是时间无关的。ReMe Memory Agent 支持：

- 可选时间过滤（单日期或日期范围）。
- 混合时空索引，区分相似内容在不同时间点（如旧方案 vs 新方案）。

### 4. User Profile 作为动态状态

User Profile 不是静态画像：由 Agent 在多次交互中持续维护。Agent 提取并更新显式约束、偏好与短期目标，减少个性漂移，保持回复与当前状态对齐。

### 5. 模块化与可扩展

- 摘要、存储、检索解耦。
- 可切换向量后端与存储实现。
- 多版本 Agent 变体（`default`、`v1`、`v2`、`halumem`、`longmemeval`）适配不同基准与场景。

---

## 🚀 快速开始

### 安装

```bash
pip install reme-ai
```

通过环境变量（如 `.env`）配置 LLM 与嵌入模型：

```bash
REME_EMBEDDING_API_KEY=sk-xxxx
REME_EMBEDDING_BASE_URL=https://xxxx/v1
REME_LLM_API_KEY=sk-xxxx
REME_LLM_BASE_URL=https://xxxx/v1
```

### 基础用法

```python
import asyncio
from reme.reme import ReMe

async def main():
    reme = ReMe(
        default_llm_config={"model_name": "qwen3-30b-a3b-thinking-2507"},
        default_embedding_model_config={"model_name": "text-embedding-v4"},
        default_vector_store_config={"backend": "memory"},
        target_user_names=["alice"],  # 可选：预注册记忆目标
        target_task_names=["planning"],
        target_tool_names=["web_search"],
    )
    await reme.start()

    # 摘要：让 Agent 从对话中提取并存储记忆
    messages = [
        {"role": "user", "content": "我喜欢深色模式，上午工作效率最高。", "time_created": "2025-02-21T10:00:00"},
        {"role": "assistant", "content": "已记录。默认深色模式和上午工作偏好。", "time_created": "2025-02-21T10:00:30"},
    ]
    answer = await reme.summarize_memory(
        messages=messages,
        user_name="alice",
        version="default",  # 或 "v1", "v2", "halumem", "longmemeval"
    )

    # 检索：让 Agent 为查询获取相关记忆
    answer = await reme.retrieve_memory(
        query="用户的界面和效率偏好是什么？",
        user_name="alice",
        top_k=5,
    )

    await reme.close()

asyncio.run(main())
```

---

## 📂 项目结构（Memory Agent）

```
reme/
├── reme.py              # ReMe 应用与主 API
├── agent/
│   └── memory/          # 记忆 Agent
│       ├── base_memory_agent.py
│       ├── reme_summarizer.py
│       ├── reme_retriever.py
│       ├── personal/    # PersonalSummarizer, PersonalRetriever 及变体
│       ├── procedural/  # ProceduralSummarizer, ProceduralRetriever
│       └── tool/        # ToolSummarizer, ToolRetriever
├── tool/
│   └── memory/          # 记忆 Agent 使用的工具
│       ├── delegate_task.py
│       ├── history/     # AddHistory, ReadHistory, ReadHistoryV2
│       ├── profiles/    # ReadAllProfiles, UpdateProfile 等
│       └── vector/      # RetrieveMemory, AddMemory, UpdateMemoryV2 等
└── config/              # 配置与提示词
```


---

## 🧪 实验

本实验部分在 LoCoMo、LongMemEval、HaluMem 三个数据集上进行评测，实验设置如下：

1. **ReMe 使用模型**：如各表 backbone 列所示。
2. **评估使用模型**：采用 LLM-as-a-Judge 协议（参照 MemOS）——每条回答由 GPT-4o-mini 裁判模型打分。

实验设置尽量与各基线论文保持一致，以复用其公开结果。

### LoCoMo

| Method | Single Hop | Multi Hop | Temporal | Open Domain | Overall |
|--------|-----------|-----------|----------|-------------|---------|
| MemoryOS | 62.43 | 56.50 | 37.18 | 40.28 | 54.70   |
| Mem0 | 66.71 | 58.16 | 55.45 | 40.62 | 61.00   |
| MemU | 72.77 | 62.41 | 33.96 | 46.88 | 61.15   |
| MemOS | 81.45 | 69.15 | 72.27 | 60.42 | 75.87   |
| HiMem | 89.22 | 70.92 | 74.77 | 54.86 | 80.71   |
| Zep | 88.11 | 71.99 | 74.45 | 66.67 | 81.06   |
| EverMemOS | 91.08 | 86.17 | 81.93 | 66.67 | 86.76   |
| TiMem | 81.43 | 62.20 | 77.63 | 52.08 | 75.30   |
| TSM | 84.30 | 66.67 | 71.03 | 58.33 | 76.69   |
| MemR3 | 89.44 | 71.39 | 76.22 | 61.11 | 81.55   |
| **ReMe** | — | — | — | — | 83.76   |

### LongMemEval

| Method | SS-User | SS-Asst | SS-Pref | Multi-S | Know. Upd | Temp. Reas | Overall |
|--------|---------|---------|---------|---------|----------|-----------|---------|
| MemU | 67.14 | 19.64 | 76.67 | 42.10 | 41.02    | 17.29     | 38.40   |
| Zep | 92.90 | 75.00 | 53.30 | 47.40 | 74.40    | 54.10     | 63.80   |
| Mem0 | 82.86 | 26.78 | 90.00 | 63.15 | 66.67    | 72.18     | 66.40   |
| MemOS | 95.71 | 67.86 | 96.67 | 70.67 | 74.26    | 77.44     | 77.80   |
| EverMemOS | 97.14 | 85.71 | 93.33 | 73.68 | 89.74    | 77.44     | 83.00   |
| TiMem | 95.71 | 82.14 | 63.33 | 70.83 | 86.16    | 68.42     | 76.88   |
| Hindsight (OSS-20B) | 95.7 | 94.6 | 66.7 | 84.6 | 79.7     | 79.7      | 83.6    |
| **ReMe** | — | — | — | — | —         | —          |  70.91  |

### HaluMem

| Method      | Memory Integrity | Memory Accuracy | QA Accuracy |
|-------------|------------------|-----------------|-------------|
| MemoBase    | 14.55            | 92.24           | 35.53       |
| Supermemory | 41.53            | 90.32           | 54.07       |
| Mem0        | 42.91            | 86.26           | 53.02       |
| ProMem      | 73.80            | 89.47           | 62.26       |
| **ReMe**        | 67.80            | 84.31           | 87.02       |

---

## 🔗 相关

- **reme_ai** — 基于流水线算子的 HTTP/MCP 服务（`summary_task_memory`、`retrieve_personal_memory` 等）。参见主项目 [ReMe README](../README.md)。
- **Benchmark** — `halumem`、`longmemeval` 通过 `from reme.reme import ReMe` 使用本 Memory Agent。

---

## 📄 License

Apache 2.0 — 详见 [LICENSE](../LICENSE)。
