# ReMe Memory Agent

> **Agent-driven memory management** — LLM agents that reason about *when*, *what*, and *how* to store and retrieve memories, using tools instead of fixed pipelines.

ReMe Memory Agent is a distinct version of ReMe where memory operations (summary, storage, retrieval) are performed by **autonomous LLM agents** equipped with tools. Rather than rigid rule-based pipelines, agents dynamically decide which memories to add, how to structure them, and how to retrieve across multiple layers (profiles, vector store, history).

---

## 🧠 Core Idea: Memory as an Agentic Task

Traditional memory systems treat compression and retrieval as deterministic pipelines:

- **Fixed-window summarization** ignores uneven information density — wasting space on low-entropy content and losing key semantics in high-entropy interactions.
- **Vanilla vector search** treats retrieval as distance calculation — it struggles with temporal ambiguity (e.g., “last year’s plan” vs. “this year’s plan”) and cannot perform multi-hop reasoning.
- **Flat storage** lacks provenance — agents either lose detail through aggressive summarization or drown in noise when keeping raw context.

ReMe Memory Agent reframes memory management as a **ReAct-style agentic task**:

| Aspect | Traditional approach | ReMe Memory Agent |
|--------|----------------------|-------------------|
| **Summarization** | Fixed windows or heuristic thresholds | Agent evaluates semantic complexity & task value, chooses encoding granularity |
| **Retrieval** | Single-layer vector similarity | Agent navigates across User Profile, Short-term Window, Long-term History |
| **Query handling** | Direct embedding lookup | Agent can deconstruct ambiguous queries and correct semantic drift |
| **Time awareness** | Time-agnostic embeddings | Optional time filters and hybrid spatio-temporal indexing |

---

## 🏗️ Architecture

### Agent hierarchy

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

- **ReMeSummarizer** / **ReMeRetriever** orchestrate workflows and delegate to specialized agents.
- **DelegateTask** routes work to the right agent based on `memory_target` (user, task, or tool).
- Each specialist agent uses tools such as `RetrieveMemory`, `AddMemory`, `UpdateProfile`, `ReadHistory`, etc.
- `BaseMemoryAgent` extends `BaseReact` — agents use **reasoning + acting** loops to choose tools and interpret results.

### Key components (from code)

| Component | File | Role |
|-----------|------|------|
| `ReMe` | `reme.py` | Main application: `summarize_memory()`, `retrieve_memory()`, `add_memory()`, etc. |
| `ReMeSummarizer` | `agent/memory/reme_summarizer.py` | Orchestrates summarization; uses `AddHistory` and `DelegateTask` |
| `ReMeRetriever` | `agent/memory/reme_retriever.py` | Orchestrates retrieval; delegates to Personal/Procedural/Tool agents |
| `PersonalSummarizer` | `agent/memory/personal/personal_summarizer.py` | Two-phase: (1) add/retrieve memories (2) update profile |
| `PersonalRetriever` | `agent/memory/personal/personal_retriever.py` | Uses profiles + vector + history for retrieval |
| `DelegateTask` | `tool/memory/delegate_task.py` | Routes tasks to the appropriate memory agent |
| `RetrieveMemory` | `tool/memory/vector/retrieve_memory.py` | Semantic similarity search with optional time filter |
| `ReadAllProfiles` | `tool/memory/profiles/read_all_profiles.py` | Loads User Profile (short-term state) |
| `UpdateProfile` | `tool/memory/profiles/update_profile.py` | Updates User Profile from interaction |

---

## ✨ Agentic capabilities

### 1. Hierarchical retrieval

Agents navigate multiple layers instead of a single vector index:

- **User Profile** — High-priority, low-latency working memory for immediate preferences and state.
- **Short-term window** — Recent messages or history blocks.
- **Long-term history** — Vector store for durable memories.

The agent decides when to check profiles, when to search vectors, and when to read history, improving relevance and reducing retrieval noise.

### 2. Multi-granularity storage

Different levels of abstraction coexist:

- High-level summaries for fast semantic positioning
- Pointers to raw context for fact verification

This mirrors human-like “flashbulb memory” + “semantic memory”, improving coherence and reducing factual hallucination in long conversations.

### 3. Time-aware retrieval

Embeddings are time-agnostic. ReMe Memory Agent supports:

- Optional time filters (single date or date range).
- Hybrid spatio-temporal indexing to distinguish similar content at different times (e.g., old plan vs. new plan).

### 4. User Profile as dynamic state

User Profile is not static: it is maintained by agents across interactions. Agents extract and update explicit constraints, preferences, and short-term goals, reducing persona drift and keeping responses aligned with the current state.

### 5. Modular, pluggable design

- Summary, storage, and retrieval are decoupled.
- Swappable vector backends and storage implementations.
- Versioned agent variants (`default`, `v1`, `v2`, `halumem`, `longmemeval`) for different benchmarks and use cases.

---

## 🚀 Quick start

### Install

```bash
pip install reme-ai
```

Configure LLM and embedding via environment variables (e.g. `.env`):

```bash
FLOW_LLM_API_KEY=sk-xxxx
FLOW_LLM_BASE_URL=https://xxxx/v1
FLOW_EMBEDDING_API_KEY=sk-xxxx
FLOW_EMBEDDING_BASE_URL=https://xxxx/v1
```

### Basic usage

```python
import asyncio
from reme.reme import ReMe

async def main():
    reme = ReMe(
        default_llm_config={"model_name": "qwen3-30b-a3b-thinking-2507"},
        default_embedding_model_config={"model_name": "text-embedding-v4"},
        default_vector_store_config={"backend": "memory"},
        target_user_names=["alice"],  # Optional: pre-register memory targets
        target_task_names=["planning"],
        target_tool_names=["web_search"],
    )
    await reme.start()

    # Summarize: let the agent extract and store memories from a conversation
    messages = [
        {"role": "user", "content": "I prefer dark mode and work best in the morning.", "time_created": "2025-02-21T10:00:00"},
        {"role": "assistant", "content": "Noted. I'll assume dark mode and morning productivity.", "time_created": "2025-02-21T10:00:30"},
    ]
    answer = await reme.summarize_memory(
        messages=messages,
        user_name="alice",
        version="default",  # or "v1", "v2", "halumem", "longmemeval"
    )

    # Retrieve: let the agent fetch relevant memories for a query
    answer = await reme.retrieve_memory(
        query="What are the user's UI and productivity preferences?",
        user_name="alice",
        top_k=5,
    )

    await reme.close()

asyncio.run(main())
```

### Programmatic memory operations

```python
# Add a memory explicitly
await reme.add_memory(
    memory_content="User prefers Python over JavaScript for scripting.",
    user_name="alice",
    when_to_use="When suggesting programming languages or tools",
)

# List memories
memories = await reme.list_memory(user_name="alice", limit=10)

# Update or delete
await reme.update_memory(memory_id="...", memory_content="Updated content.", user_name="alice")
await reme.delete_memory(memory_id="...")
```

---

## 📂 Project layout (memory agent)

```
reme/
├── reme.py              # ReMe application and main API
├── agent/
│   └── memory/          # Memory agents
│       ├── base_memory_agent.py
│       ├── reme_summarizer.py
│       ├── reme_retriever.py
│       ├── personal/    # PersonalSummarizer, PersonalRetriever, variants
│       ├── procedural/  # ProceduralSummarizer, ProceduralRetriever
│       └── tool/        # ToolSummarizer, ToolRetriever
├── tool/
│   └── memory/          # Tools used by memory agents
│       ├── delegate_task.py
│       ├── history/     # AddHistory, ReadHistory, ReadHistoryV2
│       ├── profiles/    # ReadAllProfiles, UpdateProfile, etc.
│       └── vector/      # RetrieveMemory, AddMemory, UpdateMemoryV2, etc.
└── config/              # Configuration and prompts
```

---

## 🧪 Experiments

Evaluations are conducted on three benchmarks: **LoCoMo**, **LongMemEval**, and **HaluMem**. Experimental settings:

1. **ReMe backbone**: as specified in each table.
2. **Evaluation protocol**: LLM-as-a-Judge following MemOS — each answer is scored by GPT-4o-mini and two auxiliary judge models; scores are averaged across the three judgments in a blind setting.

Baseline results are reproduced from their respective papers under aligned settings where possible.

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

## 🔗 Related

- **reme_ai** — HTTP/MCP service with pipeline-based operators (`summary_task_memory`, `retrieve_personal_memory`, etc.). See the main [ReMe README](../README.md).
- **Benchmarks** — `halumem`, `longmemeval` use the `reme` Memory Agent via `from reme.reme import ReMe`.

---

## 📄 License

Apache 2.0 — see [LICENSE](../LICENSE).
