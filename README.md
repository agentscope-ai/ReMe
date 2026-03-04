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
during
long conversations) and **stateless sessions** (new conversations cannot inherit history and always start from scratch).

ReMe gives agents **real memory** — old conversations are automatically condensed, important information is persisted,
and the next conversation can recall it automatically.

---

## 📁 File-Based CoPaw Memory System

> Memory as files, files as memory

Treat **memory as files** — readable, editable, and portable. [CoPaw](https://github.com/agentscope-ai/CoPaw)
integrates this memory system
through [MemoryManager](https://github.com/agentscope-ai/CoPaw/blob/main/src/copaw/agents/memory/memory_manager.py),
which inherits `ReMeCopaw` and exposes memory management capabilities.

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

[ReMeCopaw](reme/reme_copaw.py) is the core class of this memory system, providing complete memory management
capabilities for AI Agents:

| Method                   | Function                           | Key Components                                                                                                                                                      |
|--------------------------|------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `start`                  | 🚀 Start memory system             | Initialize file store, file watcher, Embedding cache; clean up expired tool result files                                                                            |
| `close`                  | 📕 Close and clean up              | Clean tool result files, stop file watcher, save Embedding cache                                                                                                    |
| `compact_memory`         | 📦 Compact history to summary      | [Compactor](reme/memory/file_based_copaw/compactor.py) — ReActAgent generates structured context checkpoint                                                         |
| `summary_memory`         | 📝 Write important memory to files | [Summarizer](reme/memory/file_based_copaw/summarizer.py) — ReActAgent + file tools (read / write / edit)                                                            |
| `compact_tool_result`    | ✂️ Compact oversized tool output   | [ToolResultCompactor](reme/memory/file_based_copaw/tool_result_compactor.py) — Truncate and save to `tool_result/`, keep file reference in message                  |
| `add_async_summary_task` | ⚡ Submit background summary task   | `asyncio.create_task`, summary doesn't block main conversation flow                                                                                                 |
| `await_summary_tasks`    | ⏳ Wait for background tasks        | Collect results from all background summary tasks, call before closing to ensure writes complete                                                                    |
| `memory_search`          | 🔍 Semantic memory search          | [MemorySearch](reme/memory/tools/chunk/memory_search.py) — Vector + BM25 hybrid retrieval                                                                           |
| `get_in_memory_memory`   | 🗂️ Create in-memory instance      | [CoPawInMemoryMemory](reme/memory/file_based_copaw/copaw_in_memory_memory.py) — Token-aware memory management, supports compression summary and state serialization |
| `update_params`          | ⚙️ Update runtime parameters       | Adjust `max_input_length`, `memory_compact_ratio`, `language` at runtime                                                                                            |

---

## 🗃️ Vector-Based ReMe

[ReMe Vector Based](reme/reme.py) is the core class for the vector-based memory system, supporting unified management of
three memory types:

| Memory Type                  | Purpose                                             | Usage Context |
|------------------------------|-----------------------------------------------------|---------------|
| **Personal memory**          | User preferences, habits                            | `user_name`   |
| **Task / procedural memory** | Task execution experience, success/failure patterns | `task_name`   |
| **Tool memory**              | Tool usage experience, parameter tuning             | `tool_name`   |

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

---

## 💻 ReMeCli: Terminal Assistant with File-Based Memory

<table border="0" cellspacing="0" cellpadding="0" style="border: none;">
  <tr style="border: none;">
    <td width="10%" style="border: none; vertical-align: middle; text-align: center;">
      <strong>马<br>上<br>有<br>钱</strong>
    </td>
    <td width="80%" style="border: none;">
      <video src="https://github.com/user-attachments/assets/d731ae5c-80eb-498b-a22c-8ab2b9169f87" autoplay muted loop controls></video>
    </td>
    <td width="10%" style="border: none; vertical-align: middle; text-align: center;">
      <strong>马<br>到<br>成<br>功</strong>
    </td>
  </tr>
</table>

### When Is Memory Written?

| Scenario                                    | Written to             | Trigger                            |
|---------------------------------------------|------------------------|------------------------------------|
| Auto-compact when context is too long       | `memory/YYYY-MM-DD.md` | Automatic in background            |
| User runs `/compact`                        | `memory/YYYY-MM-DD.md` | Manual compact + background save   |
| User runs `/new`                            | `memory/YYYY-MM-DD.md` | New conversation + background save |
| User says "remember this"                   | `MEMORY.md` or log     | Agent writes via `write` tool      |
| Agent finds important decisions/preferences | `MEMORY.md`            | Agent writes proactively           |

### Memory Retrieval Tools

| Method          | Tool            | When to use                      | Example                               |
|-----------------|-----------------|----------------------------------|---------------------------------------|
| Semantic search | `memory_search` | Unsure where it is, fuzzy lookup | "Earlier discussion about deployment" |
| Direct read     | `read`          | Know the date or file            | Read `memory/2025-02-13.md`           |

Search uses **vector + BM25 hybrid retrieval** (vector weight 0.7, BM25 weight 0.3), so queries using both natural
language and exact
keywords can match.

### Built-in Tools

| Tool            | Function       | Details                                                    |
|-----------------|----------------|------------------------------------------------------------|
| `memory_search` | Search memory  | Vector + BM25 hybrid search over MEMORY.md and memory/*.md |
| `bash`          | Run commands   | Execute bash commands with timeout and output truncation   |
| `ls`            | List directory | Show directory structure                                   |
| `read`          | Read file      | Text and images supported, with segmented reading          |
| `edit`          | Edit file      | Replace after exact text match                             |
| `write`         | Write file     | Create or overwrite, auto-create directories               |
| `execute_code`  | Run Python     | Execute code snippets                                      |
| `web_search`    | Web search     | Search via Tavily                                          |

---

## 🚀 Quick Start

### Installation

```bash
pip install -U reme-ai
```

### Environment Variables

API keys are set via environment variables; you can put them in a `.env` file in the project root:

| Variable                  | Description                      | Example                                             |
|---------------------------|----------------------------------|-----------------------------------------------------|
| `REME_LLM_API_KEY`        | LLM API key                      | `sk-xxx`                                            |
| `REME_LLM_BASE_URL`       | LLM base URL                     | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `REME_EMBEDDING_API_KEY`  | Embedding API key                | `sk-xxx`                                            |
| `REME_EMBEDDING_BASE_URL` | Embedding base URL               | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `TAVILY_API_KEY`          | Tavily search API key (optional) | `tvly-xxx`                                          |

### Using ReMeCli

#### Start ReMeCli

```bash
remecli config=cli
```

#### ReMeCli System Commands

> Year of the Horse easter egg: `/horse` — fireworks, galloping animation, and random horse-year blessings.

Commands starting with `/` control session state:

| Command    | Description                                                        | Waits for response |
|------------|--------------------------------------------------------------------|--------------------|
| `/compact` | Manually compact current conversation and save to long-term memory | Yes                |
| `/new`     | Start new conversation; history saved to long-term memory          | No                 |
| `/clear`   | Clear everything, **without saving**                               | No                 |
| `/history` | View uncompressed messages in current conversation                 | No                 |
| `/help`    | Show command list                                                  | No                 |
| `/exit`    | Exit                                                               | No                 |

**Difference between the three commands**

| Command    | Compact summary | Long-term memory | Message history |
|------------|-----------------|------------------|-----------------|
| `/compact` | New summary     | Saved            | Keep recent     |
| `/new`     | Cleared         | Saved            | Cleared         |
| `/clear`   | Cleared         | Not saved        | Cleared         |

> `/clear` permanently deletes; nothing is persisted anywhere.

### Using the ReMe Package

#### File-Based ReMe (CoPaw Memory System)

`ReMeCopaw` receives AgentScope components like `ChatModelBase`, `Formatter`, `Toolkit`, and configures Embedding and
storage backend via environment variables:

| Environment Variable       | Description                                   | Default                                             |
|----------------------------|-----------------------------------------------|-----------------------------------------------------|
| `EMBEDDING_API_KEY`        | Embedding service API Key                     | `""` (vector search disabled if not configured)     |
| `EMBEDDING_BASE_URL`       | Embedding service Base URL                    | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDING_MODEL_NAME`     | Embedding model name                          | `""`                                                |
| `EMBEDDING_DIMENSIONS`     | Vector dimensions                             | `1024`                                              |
| `EMBEDDING_CACHE_ENABLED`  | Whether to enable Embedding cache             | `true`                                              |
| `EMBEDDING_MAX_CACHE_SIZE` | Maximum cache entries                         | `2000`                                              |
| `FTS_ENABLED`              | Whether to enable full-text search (BM25)     | `true`                                              |
| `MEMORY_STORE_BACKEND`     | Storage backend (`auto` / `chroma` / `local`) | `auto` (local on Windows, chroma on others)         |

```python
import asyncio

from agentscope.formatter import ClaudeFormatter
from agentscope.model import get_model
from agentscope.token import HuggingFaceTokenCounter
from agentscope.tool import Toolkit

from reme.reme_copaw import ReMeCopaw


async def main():
    # Prepare AgentScope core components
    chat_model = get_model(config={"backend": "openai", "model_name": "qwen3.5-plus"})
    formatter = ClaudeFormatter()
    token_counter = HuggingFaceTokenCounter()
    toolkit = Toolkit()  # Can register additional tools

    # Initialize ReMeCopaw
    reme = ReMeCopaw(
        working_dir=".reme",  # Memory file storage directory
        chat_model=chat_model,
        formatter=formatter,
        token_counter=token_counter,
        toolkit=toolkit,
        max_input_length=128000,  # Model context window (tokens)
        memory_compact_ratio=0.7,  # Trigger compaction when reaching max_input_length * 0.7
        language="zh",  # Summary language (zh / "")
        tool_result_threshold=1000,  # Auto-save tool outputs exceeding this character count
        retention_days=7,  # tool_result/ file retention days
    )
    await reme.start()

    messages = [...]  # list[Msg], conversation history

    # 1. Compact oversized tool outputs (prevent tool results from overflowing context)
    messages = await reme.compact_tool_result(messages)

    # 2. Compact history to structured summary (trigger: context approaching limit)
    summary = await reme.compact_memory(
        messages=messages,
        previous_summary="",  # Can pass previous summary for incremental update
    )
    print(f"Compact summary:\n{summary}")

    # 3. Submit async summary task in background (non-blocking, writes to memory/YYYY-MM-DD.md)
    reme.add_async_summary_task(messages=messages)

    # 4. Semantic memory search (Vector + BM25 hybrid retrieval)
    result = await reme.memory_search(query="Python version preference", max_results=5)
    print(f"Search results: {result}")

    # 5. Get in-memory instance (CoPawInMemoryMemory, manages single conversation context)
    memory = reme.get_in_memory_memory()
    token_stats = await memory.estimate_tokens()
    print(f"Current context usage: {token_stats['context_usage_ratio']:.1f}%")

    # 6. Wait for background tasks before closing
    await reme.await_summary_tasks()
    await reme.close()


if __name__ == "__main__":
    asyncio.run(main())
```

#### Vector-Based ReMe

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

---

## 🏛️ Technical Architecture

### File-Based CoPaw Memory System Architecture

[CoPaw MemoryManager](https://github.com/agentscope-ai/CoPaw/blob/main/src/copaw/agents/memory/memory_manager.py)
inherits
`ReMeCopaw` and integrates memory capabilities into the Agent reasoning flow:

```mermaid
graph TB
    CoPaw["CoPaw MemoryManager\n(inherits ReMeCopaw)"] -->|pre_reasoning hook| Hook[MemoryCompactionHook]
    CoPaw --> ReMeCopaw[ReMeCopaw]
    Hook -->|exceeds threshold| ReMeCopaw
    ReMeCopaw --> CompactMemory[compact_memory\nHistory compaction]
    ReMeCopaw --> SummaryMemory[summary_memory\nWrite memory to files]
    ReMeCopaw --> CompactToolResult[compact_tool_result\nOversized tool output compaction]
    ReMeCopaw --> MemSearch[memory_search\nSemantic search]
    ReMeCopaw --> InMemory[get_in_memory_memory\nCoPawInMemoryMemory]
    CompactMemory --> Compactor[Compactor\nReActAgent]
    SummaryMemory --> Summarizer[Summarizer\nReActAgent + file tools]
    CompactToolResult --> ToolResultCompactor[ToolResultCompactor\nTruncate + save to file]
    Summarizer --> FileIO[FileIO\nread / write / edit]
    FileIO --> MemoryFiles[memory/YYYY-MM-DD.md]
    ToolResultCompactor --> ToolResultFiles[tool_result/*.txt]
    MemoryFiles -.->|File change| FileWatcher[Async File Watcher]
    FileWatcher -->|Update index| FileStore[Local DB]
    MemSearch --> FileStore
```

#### Auto-Compaction Trigger Flow

`MemoryCompactionHook` checks context token usage before each reasoning step, automatically triggering compaction when
threshold is exceeded:

```mermaid
graph LR
    A[pre_reasoning] --> B{Token exceeds threshold?}
    B -->|No| Z[Continue reasoning]
    B -->|Yes| C[compact_tool_result\nCompact oversized tool outputs in recent messages]
    C --> D[compact_memory\nGenerate structured context checkpoint]
    D --> E[Mark old messages as COMPRESSED]
    E --> F[add_async_summary_task\nBackground write to memory file]
    F --> Z
```

#### Context Compaction Summary Format

[Compactor](reme/memory/file_based_copaw/compactor.py) uses ReActAgent to compact history into structured **context
checkpoints**:

| Field                 | Description                                      |
|-----------------------|--------------------------------------------------|
| `## Goal`             | 🎯 User's objectives (can be multiple)           |
| `## Constraints`      | ⚙️ Constraints and preferences mentioned by user |
| `## Progress`         | 📈 Completed / in progress / blocked tasks       |
| `## Key Decisions`    | 🔑 Decisions made with brief reasons             |
| `## Next Steps`       | 🗺️ Next action plan (ordered list)              |
| `## Critical Context` | 📌 File paths, function names, error messages    |

Supports **incremental updates**: when `previous_summary` is passed, automatically merges new conversation with old
summary, preserving historical progress.

#### Tool Result Compaction

[ToolResultCompactor](reme/memory/file_based_copaw/tool_result_compactor.py) solves context overflow caused by oversized
tool outputs:

```mermaid
graph LR
    A[tool_result message] --> B{Content length > threshold?}
    B -->|No| C[Keep as-is]
    B -->|Yes| D[Truncate to threshold characters]
    D --> E[Write full content to tool_result/uuid.txt]
    E --> F[Append file reference path to message]
```

Expired files (exceeding `retention_days`) are automatically cleaned up during `start` / `close` /
`compact_tool_result`.

#### Memory Summary: ReAct + File Tools

[Summarizer](reme/memory/file_based_copaw/summarizer.py) uses the **ReAct + file tools** pattern, letting AI
autonomously decide what to write and where:

```mermaid
graph LR
    A[Receive conversation] --> B{Think: What's worth recording?}
    B --> C[Act: read memory/YYYY-MM-DD.md]
    C --> D{Think: How to merge with existing content?}
    D --> E[Act: edit to update file]
    E --> F{Think: Anything missing?}
    F -->|Yes| B
    F -->|No| G[Done]
```

[FileIO](reme/memory/file_based_copaw/file_io.py) provides file operation tools:

| Tool    | Function                       | Use case                                |
|---------|--------------------------------|-----------------------------------------|
| `read`  | Read file content (line range) | View existing memory, avoid duplicates  |
| `write` | Overwrite file                 | Create new memory file or major rewrite |
| `edit`  | Replace after exact match      | Append or modify specific sections      |

#### In-Memory Session Management

[CoPawInMemoryMemory](reme/memory/file_based_copaw/copaw_in_memory_memory.py) extends AgentScope's `InMemoryMemory`:

| Feature                          | Description                                                         |
|----------------------------------|---------------------------------------------------------------------|
| `get_memory`                     | Filter messages by mark, auto-prepend compression summary           |
| `estimate_tokens`                | Precisely estimate current context token usage and ratio            |
| `get_history_str`                | Generate human-readable conversation summary (with token stats)     |
| `state_dict` / `load_state_dict` | Support state serialization / deserialization (session persistence) |

#### Memory Retrieval

[MemorySearch](reme/memory/tools/chunk/memory_search.py) provides **vector + BM25 hybrid retrieval**:

| Retrieval           | Strength                                        | Weakness                               |
|---------------------|-------------------------------------------------|----------------------------------------|
| **Vector semantic** | Captures similar meaning with different wording | Weaker on exact token match            |
| **BM25 full-text**  | Strong exact token match                        | No synonym or paraphrase understanding |

**Fusion**: Both retrieval paths are used; results are combined by weighted sum (vector 0.7 + BM25 0.3), so both
natural-language queries and exact lookups get reliable results.

```mermaid
graph LR
    Q[Search query] --> V[Vector search × 0.7]
Q --> B[BM25 × 0.3]
V --> M[Dedupe + weighted merge]
B --> M
M --> R[Top-N results]
```

---

### Vector-Based ReMe Core Architecture

```mermaid
graph TB
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

---

## ⭐ Community & Support

- **Star & Watch**: Star helps more agent developers discover ReMe; Watch keeps you updated on new releases and
  features.
- **Share your work**: In Issues or Discussions, share what ReMe unlocks for your agents — we’re happy to highlight
  great community examples.
- **Need a new feature?** Open a Feature Request; we’ll iterate with the community.
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
