<p align="center">
 <img src="docs/_static/figure/reme_logo.png" alt="ReMe Logo" width="50%">
</p>

<p align="center">
  <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python Version"></a>
  <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/pypi/v/reme-ai.svg?logo=pypi" alt="PyPI Version"></a>
  <a href="https://pepy.tech/project/reme-ai/"><img src="https://img.shields.io/pypi/dm/reme-ai" alt="PyPI Downloads"></a>
  <a href="https://github.com/agentscope-ai/ReMe"><img src="https://img.shields.io/github/commit-activity/m/agentscope-ai/ReMe?style=flat-square" alt="GitHub commit activity"></a>
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-black" alt="License"></a>
  <a href="./README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="./README_ZH.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="https://github.com/agentscope-ai/ReMe"><img src="https://img.shields.io/github/stars/agentscope-ai/ReMe?style=social" alt="GitHub Stars"></a>
</p>

<p align="center">
  <strong>A memory management toolkit for AI agents — Remember Me, Refine Me.</strong><br>
</p>

> For legacy versions, see [0.2.x Documentation](docs/README_0_2_x.md)

---

🧠 ReMe is a **memory management framework** built for **AI agents**, offering both **file-based** and **vector-based**
memory systems.

It addresses two core problems of agent memory: **limited context windows** (early information gets truncated or lost
during long conversations) and **stateless sessions** (new conversations cannot inherit history and always start from
scratch).

ReMe gives agents **real memory** — old conversations are automatically condensed, important information is persisted,
and the next conversation can recall it automatically.


---

## 📁 File-Based Memory System (ReMeLight)

> Memory as files, files as memory

Treat **memory as files** — readable, editable, and portable.
[CoPaw](https://github.com/agentscope-ai/CoPaw) implements long-term memory and context management by inheriting
`ReMeLight`.

| Traditional Memory Systems | File-Based ReMe    |
|----------------------------|--------------------|
| 🗄️ Database storage       | 📝 Markdown files  |
| 🔒 Opaque                  | 👀 Read anytime    |
| ❌ Hard to modify           | ✏️ Edit directly   |
| 🚫 Hard to migrate         | 📦 Copy to migrate |

```
working_dir/
├── MEMORY.md              # Long-term memory: user preferences, project config, etc.
├── memory/
│   └── YYYY-MM-DD.md      # Daily summary logs: written automatically after conversation ends
└── tool_result/           # Cache for oversized tool outputs (auto-managed, auto-cleaned when expired)
    └── <uuid>.txt
```

### Core Capabilities

[ReMeLight](reme/reme_light.py) is the core class of this memory system, providing complete memory management
capabilities for AI Agents:

| Method                 | Function                           | Key Components                                                                                                                                                              |
|------------------------|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `check_context`        | 📊 Check context size              | [ContextChecker](reme/memory/file_based/component/context_checker.py) — Check if context exceeds threshold and split messages                                               |
| `compact_memory`       | 📦 Compact history to summary      | [Compactor](reme/memory/file_based/component/compactor.py) — ReActAgent generates structured context checkpoint                                                             |
| `summary_memory`       | 📝 Write important memory to files | [Summarizer](reme/memory/file_based/component/summarizer.py) — ReActAgent + file tools (read / write / edit)                                                                |
| `compact_tool_result`  | ✂️ Compact oversized tool output   | [ToolResultCompactor](reme/memory/file_based/component/tool_result_compactor.py) — Truncate and save to `tool_result/`, keep file reference in message                      |
| `memory_search`        | 🔍 Semantic memory search          | [MemorySearch](reme/memory/file_based/tools/memory_search.py) — Vector + BM25 hybrid retrieval                                                                              |
| `get_in_memory_memory` | 🗂️ Create in-memory instance      | [ReMeInMemoryMemory](reme/memory/file_based/reme_in_memory_memory.py) — Token-aware memory management, supports compression summary and state serialization (static method) |
| `pre_reasoning_hook`   | 🔄 Pre-reasoning hook              | compact_tool_result + check_context + compact_memory + summary_memory(async)                                                                                                |
| `start`                | 🚀 Start memory system             | Initialize file store, file watcher, Embedding cache; clean up expired tool result files                                                                                    |
| `close`                | 📕 Close and clean up              | Clean tool result files, stop file watcher, save Embedding cache                                                                                                            |

---

### 🚀 Quick Start

#### Installation

```bash
pip install -e ".[light]"
```

#### Environment Variables

`ReMeLight` environment variables configure Embedding and storage backend

| Variable             | Description                   | Example                                             |
|----------------------|-------------------------------|-----------------------------------------------------|
| `LLM_API_KEY`        | LLM API key                   | `sk-xxx`                                            |
| `LLM_BASE_URL`       | LLM base URL                  | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDING_API_KEY`  | Embedding API key (Optional)  | `sk-xxx`                                            |
| `EMBEDDING_BASE_URL` | Embedding base URL (Optional) | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

#### Python Usage

```python
import asyncio

from reme.reme_light import ReMeLight


async def main():
    # Initialize ReMeLight
    reme = ReMeLight(
        default_as_llm_config={"model_name": "qwen3.5-35b-a3b"},
        # default_embedding_model_config={"model_name": "text-embedding-v4"},
        default_file_store_config={"fts_enabled": True, "vector_enabled": False},
    )
    await reme.start()

    messages = [...]  # Conversation message list

    # 1. Compact oversized tool outputs (prevent tool results from overflowing context)
    messages = await reme.compact_tool_result(messages)

    # 2. Compact history to structured summary (can pass previous summary for incremental update)
    summary = await reme.compact_memory(
        messages=messages,
        previous_summary="",
        max_input_length=128000,  # Model context window (tokens)
        compact_ratio=0.7,  # Trigger compaction when reaching max_input_length * 0.7
        language="zh",  # Summary language (zh / "")
    )

    # 3. Submit async summary task in background (non-blocking, writes to memory/YYYY-MM-DD.md)
    reme.add_async_summary_task(messages=messages)

    # 4. Pre-reasoning hook (auto compact tool results + generate summary)
    processed_messages, compressed_summary = await reme.pre_reasoning_hook(
        messages=messages,
        system_prompt="You are a helpful AI assistant.",
        compressed_summary="",
        max_input_length=128000,
        compact_ratio=0.7,
        memory_compact_reserve=10000,
        enable_tool_result_compact=True,
        tool_result_compact_keep_n=3,
    )

    # 5. Semantic memory search (Vector + BM25 hybrid retrieval)
    result = await reme.memory_search(query="Python version preference", max_results=5)

    # 6. Get in-memory instance (static method, manages single conversation context)
    memory = ReMeLight.get_in_memory_memory()
    for msg in messages:
        await memory.add(msg)
    token_stats = await memory.estimate_tokens(max_input_length=128000)
    print(f"Current context usage: {token_stats['context_usage_ratio']:.1f}%")
    print(f"Message tokens: {token_stats['messages_tokens']}")
    print(f"Estimated total tokens: {token_stats['estimated_tokens']}")

    # 7. Wait for background tasks before closing
    summary_result = await reme.await_summary_tasks()

    # Close ReMeLight
    await reme.close()


if __name__ == "__main__":
    asyncio.run(main())
```

> 📂 Full example code: [test_reme_light.py](tests/light/test_reme_light.py)
> 📋 Example output: [test_reme_light_log.txt](tests/light/test_reme_light_log.txt) (223,838 tokens → 1,105 tokens, 99.5%
> compression ratio)

### File-Based ReMeLight Memory System Architecture

[CoPaw MemoryManager](https://github.com/agentscope-ai/CoPaw/blob/main/src/copaw/agents/memory/memory_manager.py)
inherits `ReMeLight` and integrates memory capabilities into the Agent reasoning flow:

```mermaid
graph LR
    Agent[Agent] -->|pre_reasoning hook| Hook[pre_reasoning_hook]
    Hook --> TC[compact_tool_result<br>Compact tool output]
    TC --> CC[check_context<br>Token counting]
    CC -->|exceeds threshold| CM[compact_memory<br>Generate summary]
    CC -->|exceeds threshold| SM[summary_memory<br>Async persistence]
    SM -->|ReAct + FileIO| Files[memory/*.md]
    Agent -->|direct call| Search[memory_search<br>Vector+BM25]
    Agent -->|static method| InMem[get_in_memory_memory<br>Token-aware memory]
    Files -.->|FileWatcher| Store[(FileStore<br>Vector+FTS index)]
    Search --> Store
```

---

#### 1. check_context — Context Check

[ContextChecker](reme/memory/file_based/component/context_checker.py) uses token counting to determine if context
exceeds threshold, automatically splitting into "to compact" and "to keep" message groups.

```mermaid
graph LR
    M[messages] --> H[AsMsgHandler<br>Token counting]
    H --> C{total > threshold?}
    C -->|No| K[Return all messages]
    C -->|Yes| S[Keep from tail<br>reserve tokens]
    S --> CP[messages_to_compact<br>Early messages]
    S --> KP[messages_to_keep<br>Recent messages]
    S --> V{is_valid<br>Tool call aligned?}
```

- **Core Logic**: Keep `reserve` tokens from the tail, mark excess as to-compact
- **Integrity Guarantee**: Never split user-assistant pairs, never split tool_use/tool_result pairs

---

#### 2. compact_memory — Conversation Compaction

[Compactor](reme/memory/file_based/component/compactor.py) uses ReActAgent to compact history into **structured context
checkpoints**.

```mermaid
graph LR
    M[messages] --> H[AsMsgHandler<br>format_msgs_to_str]
    H --> A[ReActAgent<br>reme_compactor]
    P[previous_summary] -->|incremental update| A
    A --> S[Structured summary<br>Goal/Progress/Decisions...]
```

**Summary Structure** (context checkpoint):

| Field                 | Description                                      |
|-----------------------|--------------------------------------------------|
| `## Goal`             | User objectives                                  |
| `## Constraints`      | Constraints and preferences                      |
| `## Progress`         | Task progress                                    |
| `## Key Decisions`    | Key decisions                                    |
| `## Next Steps`       | Next action plan                                 |
| `## Critical Context` | File paths, function names, error messages, etc. |

- **Incremental Update**: When `previous_summary` is passed, automatically merges new conversation with old summary

---

#### 3. summary_memory — Memory Persistence

[Summarizer](reme/memory/file_based/component/summarizer.py) uses the **ReAct + file tools** pattern, letting AI
autonomously decide what to write and where.

```mermaid
graph LR
    M[messages] --> A[ReActAgent<br>reme_summarizer]
    A -->|read| R[Read memory/YYYY-MM-DD.md]
    R --> T{Think: How to merge?}
    T -->|write| W[Overwrite file]
    T -->|edit| E[Exact replacement]
    W --> F[memory/YYYY-MM-DD.md]
    E --> F
```

**File Tools** ([FileIO](reme/memory/file_based/tools/file_io.py)):

| Tool    | Function                  |
|---------|---------------------------|
| `read`  | Read file content         |
| `write` | Overwrite file            |
| `edit`  | Replace after exact match |

---

#### 4. compact_tool_result — Tool Result Compaction

[ToolResultCompactor](reme/memory/file_based/component/tool_result_compactor.py) solves context overflow caused by
oversized tool outputs.

```mermaid
graph LR
    M[messages] --> L{Iterate tool_result<br>len > threshold?}
    L -->|No| K[Keep as-is]
    L -->|Yes| T[truncate_text<br>Truncate to threshold]
    T --> S[Write full content to<br>tool_result/uuid.txt]
    S --> R[Append file path reference to message]
    R --> C[cleanup_expired_files<br>Clean expired files]
```

- **Auto Cleanup**: Expired files (exceeding `retention_days`) are automatically deleted during `start`/`close`/
  `compact_tool_result`

---

#### 5. memory_search — Memory Retrieval

[MemorySearch](reme/memory/file_based/tools/memory_search.py) provides **vector + BM25 hybrid retrieval** capability.

```mermaid
graph LR
    Q[query] --> E[Embedding<br>Vectorize]
    E --> V[vector_search<br>Semantic similarity]
    Q --> B[BM25<br>Keyword matching]
    V -->|" weight: 0.7 "| M[Dedupe + weighted fusion]
    B -->|" weight: 0.3 "| M
    M --> F[min_score filter]
    F --> R[Top-N results]
```

- **Fusion Mechanism**: Vector weight 0.7 + BM25 weight 0.3, balancing semantic similarity and exact matching

---

#### 6. get_in_memory_memory — In-Memory Session

[ReMeInMemoryMemory](reme/memory/file_based/reme_in_memory_memory.py) extends AgentScope's `InMemoryMemory`, providing
token-aware memory management.

```mermaid
graph LR
    C[content] --> G[get_memory<br>exclude_mark=COMPRESSED]
    G --> F[Exclude compressed messages]
    F --> P{prepend_summary?}
    P -->|Yes| S[Prepend previous-summary]
    S --> O[Output messages]
    P -->|No| O
```

| Feature                          | Description                                               |
|----------------------------------|-----------------------------------------------------------|
| `get_memory`                     | Filter by mark, auto-prepend compression summary          |
| `estimate_tokens`                | Estimate context token usage                              |
| `state_dict` / `load_state_dict` | State serialization/deserialization (session persistence) |

---

#### 7. pre_reasoning_hook — Pre-reasoning Preprocessing

Unified entry point integrating the above components, automatically managing context before each reasoning step.

```mermaid
graph LR
    M[messages] --> TC[compact_tool_result<br>Compact oversized tool output]
    TC --> CC[check_context<br>Calculate remaining space]
    CC --> D{messages_to_compact<br>not empty?}
    D -->|No| K[Return original messages + summary]
    D -->|Yes| V{is_valid?}
    V -->|No| K
    V -->|Yes| CM[compact_memory<br>Sync generate summary]
    V -->|Yes| SM[add_async_summary_task<br>Async persistence]
    CM --> R[Return messages_to_keep + new summary]
```

**Execution Flow**:

1. `compact_tool_result` — Compact oversized tool outputs
2. `check_context` — Check if context exceeds threshold
3. `compact_memory` — Generate compression summary (sync)
4. `summary_memory` — Persist memory (async background)

---

## 🗃️ Vector-Based Memory System

[ReMe Vector Based](reme/reme.py) is the core class for the vector-based memory system, supporting unified management of
three memory types:

| Memory Type                  | Purpose                                             |
|------------------------------|-----------------------------------------------------|
| **Personal memory**          | User preferences, habits                            |
| **Task / procedural memory** | Task execution experience, success/failure patterns |
| **Tool memory**              | Tool usage experience, parameter tuning             |

### Core Capabilities

| Method             | Function            | Description                                               |
|--------------------|---------------------|-----------------------------------------------------------|
| `summarize_memory` | 🧠 Summarize memory | Automatically extract and store memory from conversations |
| `retrieve_memory`  | 🔍 Retrieve memory  | Retrieve relevant memory by query                         |
| `add_memory`       | ➕ Add memory        | Manually add memory to vector store                       |
| `get_memory`       | 📖 Get memory       | Fetch a single memory by ID                               |
| `update_memory`    | ✏️ Update memory    | Update content or metadata of existing memory             |
| `delete_memory`    | 🗑️ Delete memory   | Delete specified memory                                   |
| `list_memory`      | 📋 List memory      | List memories with filtering and sorting                  |

### Installation and Environment Variables

Installation and environment variable configuration are the same as [ReMeLight](#installation). Set API keys via
environment variables, which can be written in a `.env` file in the project root.

### Python Usage

```python
import asyncio

from reme import ReMe


async def main():
    # Initialize ReMe
    reme = ReMe(
        working_dir=".reme",
        default_llm_config={
            "backend": "openai",
            "model_name": "qwen3.5-plus",
        },
        default_embedding_model_config={
            "backend": "openai",
            "model_name": "text-embedding-v4",
            "dimensions": 1024,
        },
        default_vector_store_config={
            "backend": "local",  # Supports local/chroma/qdrant/elasticsearch
        },
    )
    await reme.start()

    messages = [
        {"role": "user", "content": "Help me write a Python script", "time_created": "2026-02-28 10:00:00"},
        {"role": "assistant", "content": "Sure, I'll help you write it", "time_created": "2026-02-28 10:00:05"},
    ]

    # 1. Summarize memory from conversation (auto-extract user preferences, task experience, etc.)
    result = await reme.summarize_memory(
        messages=messages,
        user_name="alice",  # Personal memory
        # task_name="code_writing",  # Task memory
    )
    print(f"Summarize result: {result}")

    # 2. Retrieve relevant memory
    memories = await reme.retrieve_memory(
        query="Python programming",
        user_name="alice",
        # task_name="code_writing",
    )
    print(f"Retrieve result: {memories}")

    # 3. Manually add memory
    memory_node = await reme.add_memory(
        memory_content="User prefers concise code style",
        user_name="alice",
    )
    print(f"Added memory: {memory_node}")
    memory_id = memory_node.memory_id

    # 4. Get single memory by ID
    fetched_memory = await reme.get_memory(memory_id=memory_id)
    print(f"Fetched memory: {fetched_memory}")

    # 5. Update memory content
    updated_memory = await reme.update_memory(
        memory_id=memory_id,
        user_name="alice",
        memory_content="User prefers concise, well-commented code style",
    )
    print(f"Updated memory: {updated_memory}")

    # 6. List all memories for user (with filtering and sorting)
    all_memories = await reme.list_memory(
        user_name="alice",
        limit=10,
        sort_key="time_created",
        reverse=True,
    )
    print(f"User memory list: {all_memories}")

    # 7. Delete specified memory
    await reme.delete_memory(memory_id=memory_id)
    print(f"Deleted memory: {memory_id}")

    # 8. Delete all memories (use with caution)
    # await reme.delete_all()

    await reme.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### Technical Architecture

```mermaid
graph LR
    User[User / Agent] --> ReMe[Vector Based ReMe]
    ReMe --> Summarize[Memory Summarize]
    ReMe --> Retrieve[Memory Retrieve]
    ReMe --> CRUD[CRUD]
    Summarize --> PersonalSum[PersonalSummarizer]
    Summarize --> ProceduralSum[ProceduralSummarizer]
    Summarize --> ToolSum[ToolSummarizer]
    Retrieve --> PersonalRet[PersonalRetriever]
    Retrieve --> ProceduralRet[ProceduralRetriever]
    Retrieve --> ToolRet[ToolRetriever]
    PersonalSum --> VectorStore[Vector DB]
    ProceduralSum --> VectorStore
    ToolSum --> VectorStore
    PersonalRet --> VectorStore
    ProceduralRet --> VectorStore
    ToolRet --> VectorStore
```

## ⭐ Community & Support

- **Star & Watch**: Star helps more agent developers discover ReMe; Watch keeps you updated on new releases and
  features.
- **Share your work**: In Issues or Discussions, share what ReMe unlocks for your agents — we're happy to highlight
  great community examples.
- **Need a new feature?** Open a Feature Request; we'll iterate with the community.
- **Code contributions**: All forms of code contribution are welcome. See
  the [Contribution Guide](docs/contribution.md).
- **Acknowledgments**: Thanks to OpenClaw, Mem0, MemU, CoPaw, and other open-source projects for inspiration and
  support.

---

## 📄 Citation

```bibtex
@software{AgentscopeReMe2025,
  title = {AgentscopeReMe: Memory Management Kit for Agents},
  author = {ReMe Team},
  url = {https://reme.agentscope.io},
  year = {2025}
}
```

---

## ⚖️ License

This project is open source under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

---

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=agentscope-ai/ReMe&type=Date)](https://www.star-history.com/#agentscope-ai/ReMe&Date)
