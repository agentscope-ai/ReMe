# ReMe CLI Quick Start

## Memory Management: Why Does AI Need This?

Anyone who has used LLMs knows the context window is limited. As conversations grow longer:

- The conversation gets cut off and can't continue
- Response quality drops noticeably — it forgets what was said earlier
- Start a new conversation? Everything from before is gone, back to square one

Worse, **even if the context isn't full, a new conversation starts as a blank slate**. The technical decisions you made
last time, your personal preferences, work left half-done — all gone.

ReMe solves this with two capabilities:

| Capability             | Purpose                                                                                                                 |
|------------------------|-------------------------------------------------------------------------------------------------------------------------|
| **Context compaction** | When conversations get too long, old content is automatically condensed into summaries to free up space for new content |
| **Long-term memory**   | Important information is persisted to disk and automatically retrieved in future conversations                          |

---

## File-Based Memory Design

ReMe's long-term memory doesn't depend on an external database — **Markdown files are the memory itself**. You can open
and edit them at any time.

> Memory design inspired by the [OpenClaw](https://github.com/openclaw/openclaw) memory architecture.

### File Structure

```
.reme/
├── MEMORY.md
└── memory/
    ├── 2025-02-12.md
    ├── 2025-02-13.md
    └── ...
```

### MEMORY.md — Long-Term Memory

Stores key information that rarely changes — essentially your "profile":

- **Location**: `{working_dir}/MEMORY.md`
- **Example content**: Project uses Python 3.12, prefers pytest, database is PostgreSQL
- **Written by**: Agent maintains it automatically via `write` / `edit` tools

### memory/YYYY-MM-DD.md — Daily Logs

One file per day, append-only, recording what happened:

- **Location**: `{working_dir}/memory/YYYY-MM-DD.md`
- **Example content**: Fixed login bug, deployed v2.1, discussed caching strategy
- **Written by**: Agent tool writes + triggered automatically during compaction

---

## ReMeCli Demo

<video src="https://github.com/user-attachments/assets/d731ae5c-80eb-498b-a22c-8ab2b9169f87" width="80%" controls></video>

---

## Installation

### PyPI (Recommended)

```bash
pip install reme-ai==0.3.0.0b1
```

### From Source

```bash
git clone https://github.com/agentscope-ai/ReMe.git
cd ReMe
pip install -e .
```

> Python >= 3.10

---

## Configuration

### Environment Variables

In addition to the yaml config file, API keys are set via environment variables. You can put them in a `.env` file at
the project root:

| Variable                  | Description        | Example                                             |
|---------------------------|--------------------|-----------------------------------------------------|
| `REME_LLM_API_KEY`        | LLM API Key        | `sk-xxx`                                            |
| `REME_LLM_BASE_URL`       | LLM Base URL       | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `REME_EMBEDDING_API_KEY`  | Embedding API Key  | `sk-xxx`                                            |
| `REME_EMBEDDING_BASE_URL` | Embedding Base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

> If you don't have an embedding service, search quality will be reduced. Make sure to also set `vector_enabled=false`.

### Web Search (Optional)

| Variable            | Description                         |
|---------------------|-------------------------------------|
| `TAVILY_API_KEY`    | Tavily Search API Key               |
| `DASHSCOPE_API_KEY` | DashScope LLM (with search) API Key |

> Pick one. If Tavily is available, it takes priority.

---

### Config File: cli.yaml

`remecli` loads [cli.yaml](https://github.com/agentscope-ai/ReMe/blob/main/reme/config/cli.yaml) on startup (
`config_path="cli"`). All core parameters are managed in this single file.

#### Parameter Reference

**Basic Configuration**

| Parameter     | Value   | Description                                       |
|---------------|---------|---------------------------------------------------|
| `backend`     | `cmd`   | Runtime mode. CLI uses `cmd`                      |
| `working_dir` | `.reme` | Workspace directory where memory files are stored |

**metadata — Context Window and Retrieval Parameters**

Controls how context space is allocated and how memory is searched:

| Parameter               | Default  | Description                                                         |
|-------------------------|----------|---------------------------------------------------------------------|
| `context_window_tokens` | `100000` | Total context window size (tokens)                                  |
| `reserve_tokens`        | `30000`  | Space reserved for output and system overhead                       |
| `keep_recent_tokens`    | `10000`  | How many recent conversation tokens to keep after compaction        |
| `vector_weight`         | `0.7`    | Vector search weight (BM25 = 1 - 0.7 = 0.3)                         |
| `candidate_multiplier`  | `2`      | Retrieval candidate pool multiplier. Higher = better recall, slower |

> Auto-compaction triggers when total message tokens >= `context_window_tokens - reserve_tokens`, i.e. 70,000 tokens by
> default.

**llms — LLM Models**

| Parameter          | Description                                   |
|--------------------|-----------------------------------------------|
| `backend`          | Backend type, uses OpenAI-compatible API      |
| `model_name`       | Model name, defaults to Qwen                  |
| `request_interval` | Request interval (seconds), for rate limiting |

**embedding_models — Embedding Models**

| Parameter    | Description                                 |
|--------------|---------------------------------------------|
| `backend`    | Embedding backend type                      |
| `model_name` | Model name, defaults to `text-embedding-v4` |
| `dimensions` | Vector dimensions, `1024`                   |

**memory_stores — Memory Storage**

| Parameter         | Description                                      |
|-------------------|--------------------------------------------------|
| `backend`         | Storage backend, defaults to `chroma` (ChromaDB) |
| `db_name`         | Database file name                               |
| `store_name`      | Collection name                                  |
| `embedding_model` | Which embedding model to use                     |
| `fts_enabled`     | Whether to enable BM25 full-text search          |
| `vector_enabled`  | Whether to enable vector semantic search         |

> Recommended to enable both `fts_enabled` and `vector_enabled` for the best hybrid retrieval results.

**file_watchers — File Monitoring**

| Parameter        | Description                            |
|------------------|----------------------------------------|
| `backend`        | Monitoring mode, `full` = full scan    |
| `memory_store`   | Corresponding memory store config      |
| `watch_paths`    | Directories/files to monitor           |
| `suffix_filters` | Which file suffixes to watch (`.md`)   |
| `recursive`      | Whether to recurse into subdirectories |
| `scan_on_start`  | Whether to do a full scan on startup   |

**token_counters — Token Counter**

| Parameter | Description                           |
|-----------|---------------------------------------|
| `backend` | Counting method, `base` uses tiktoken |

## Launch

```bash
remecli config=cli
```

After launch, [cli.yaml](https://github.com/agentscope-ai/ReMe/blob/main/reme/config/cli.yaml) is loaded automatically
and you can start chatting with Remy. ReMe handles compaction and memory in the background.

---

## System Commands

Type `/`-prefixed commands during a conversation to control state:

| Command    | Description                                                                                 | Blocks |
|------------|---------------------------------------------------------------------------------------------|--------|
| `/compact` | Manually compact the current conversation; also saves to long-term memory in the background | Yes    |
| `/new`     | Start a new conversation; history is saved to long-term memory in the background            | No     |
| `/clear`   | Clear everything, **without saving**                                                        | No     |
| `/history` | View uncompacted messages in the current conversation                                       | No     |
| `/help`    | Show command list                                                                           | No     |
| `/exit`    | Exit                                                                                        | No     |

### Comparing the Three Commands

| Command    | Compaction Summary    | Long-Term Memory | Message History       |
|------------|-----------------------|------------------|-----------------------|
| `/compact` | Generates new summary | Saved            | Keeps recent messages |
| `/new`     | Cleared               | Saved            | Cleared               |
| `/clear`   | Cleared               | Not saved        | Cleared               |

> `/clear` is a hard delete — once cleared, it's gone and not saved anywhere.

---

## ReMeCli Capabilities

### When Does Memory Get Written?

| Scenario                                          | Written To               | Trigger                             |
|---------------------------------------------------|--------------------------|-------------------------------------|
| Auto-compaction when context is too long          | `memory/YYYY-MM-DD.md`   | Automatic in background             |
| User runs `/compact`                              | `memory/YYYY-MM-DD.md`   | Manual compaction + background save |
| User runs `/new`                                  | `memory/YYYY-MM-DD.md`   | New conversation + background save  |
| User says "remember this"                         | `MEMORY.md` or daily log | Agent writes via `write` tool       |
| Agent identifies an important decision/preference | `MEMORY.md`              | Agent writes proactively            |

### Memory Retrieval

Two ways to find previously stored information:

| Method          | Tool            | When to Use                                | Example                                |
|-----------------|-----------------|--------------------------------------------|----------------------------------------|
| Semantic search | `memory_search` | Don't know where it's stored, fuzzy lookup | "previous discussion about deployment" |
| Direct read     | `read`          | Know the date or file                      | Read `memory/2025-02-13.md`            |

Search uses **vector + BM25 hybrid retrieval** (vector weight 0.7, BM25 weight 0.3), so both natural language queries
and exact keywords work.

### Built-in Tools

| Tool            | Function       | Details                                                      |
|-----------------|----------------|--------------------------------------------------------------|
| `memory_search` | Search memory  | Hybrid vector + BM25 search across MEMORY.md and memory/*.md |
| `bash`          | Run commands   | Execute bash commands with timeout and output truncation     |
| `ls`            | List directory | Show directory structure                                     |
| `read`          | Read files     | Supports text and images, with partial reads                 |
| `edit`          | Edit files     | Exact text match and replace                                 |
| `write`         | Write files    | Create or overwrite, auto-creates directories                |
| `execute_code`  | Run Python     | Execute code snippets                                        |
| `web_search`    | Web search     | Search via Tavily or DashScope                               |

---

## How Context Compaction Works

In short, long conversations are condensed into summaries while recent messages stay intact. Two trigger modes:

### Auto-Compaction

Before each conversation turn, ReMe checks current token usage. If it exceeds the threshold (
`context_window_tokens - reserve_tokens`), old messages are automatically compacted:

```
Before compaction:                    After compaction:
+--------------------------+          +--------------------------+
| Message 1: Hello         |          | Summary: Previously      |
| Message 2: Write code    |  ──────> |   helped user write code |
| Message 3: Tool output   |          |   and make adjustments   |
|   (very long)            |          +--------------------------+
| Message 4: Make changes  |          | Message 5: New request   |
| Message 5: New request   |          +--------------------------+
+--------------------------+
```

### Manual Compaction

Type `/compact` at any time to force-compact all current messages, regardless of the threshold.

### What Gets Preserved in the Summary?

| Content                     | Description                        | Example                                                  |
|-----------------------------|------------------------------------|----------------------------------------------------------|
| Goal                        | What the user wants to do          | "Build a login system"                                   |
| Constraints and preferences | Requirements the user specified    | "Use TypeScript, no frameworks"                          |
| Progress                    | What's been done so far            | "Login endpoint is done, registration still in progress" |
| Key decisions               | What was decided and why           | "Chose JWT over sessions for statelessness"              |
| Next steps                  | What to do next                    | "Implement password reset"                               |
| Key context                 | File names, function names, errors | "Main file is src/auth.ts"                               |
