<p align="center">
 <img src="docs/figure/reme_logo.png" alt="ReMe Logo" width="50%">
</p>

<p align="center">
  <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python Version"></a>
  <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/pypi/v/reme-ai.svg?logo=pypi" alt="PyPI Version"></a>
  <a href="https://pepy.tech/project/reme-ai/"><img src="https://img.shields.io/pypi/dm/reme-ai" alt="PyPI Downloads"></a>
  <a href="https://github.com/agentscope-ai/ReMe"><img src="https://img.shields.io/github/commit-activity/m/agentscope-ai/ReMe?style=flat-square" alt="GitHub commit activity"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-black" alt="License"></a>
  <a href="https://reme.agentscope.io"><img src="https://img.shields.io/badge/docs-ReMe-blue" alt="Documentation"></a>
  <a href="./README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="./README_ZH.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="https://github.com/agentscope-ai/ReMe"><img src="https://img.shields.io/github/stars/agentscope-ai/ReMe?style=social" alt="GitHub Stars"></a>
  <a href="https://deepwiki.com/agentscope-ai/ReMe"><img src="https://img.shields.io/badge/DeepWiki-Ask_Devin-navy.svg" alt="DeepWiki"></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/20528" target="_blank"><img src="https://trendshift.io/api/badge/repositories/20528" alt="agentscope-ai%2FReMe | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <strong>A local-first, self-evolving personal knowledge base for AI agents.</strong><br>
</p>

> Previous versions: [0.3.x](https://github.com/agentscope-ai/ReMe/tree/reme_v3) ·
> [0.2.x](https://github.com/agentscope-ai/ReMe/tree/v0.2.0.6) ·
> [MemoryScope](https://github.com/agentscope-ai/ReMe/tree/memoryscope_branch)

🧠 ReMe turns conversations and resources into readable, editable, searchable, and interconnected Markdown memory. It
works alongside agents such as QwenPaw, OpenClaw, Hermes, and Claude Code, continuously organizing what they learn while
keeping the files under the user's control.

## ✨ Core Ideas

- **Memory as File, File as Memory**: Markdown files with frontmatter and wikilinks serve as memory nodes that both
  users and agents can inspect, edit, move, and back up directly.
- **Self-evolving knowledge base**: Auto Memory, Auto Resource, and Auto Dream progressively transform conversations and
  resources into daily notes and long-term knowledge, while Auto Link writes relationships and sources back into the
  files.
- **Progressive hybrid search**: ReMe combines wikilinks, BM25, and embeddings for hybrid retrieval across keyword
  matching, optional semantic recall, and relationship expansion without loading every neighboring file into context.
- **Agent-friendly integration**: SKILL.md + CLI integration makes it easy for different agents to read, write,
  maintain, and reuse the same local workspace. HTTP, MCP, and Python integrations are also available.

<p align="center">
  <img src="docs/figure/design-philosophy.svg" alt="ReMe Design Philosophy" width="92%">
</p>

## 🔭 Use Cases

- **Personal assistants**: Give personal assistants such as
  [QwenPaw](https://github.com/agentscope-ai/QwenPaw), [OpenClaw](https://github.com/openclaw/openclaw), and
  [Hermes](https://github.com/nousresearch/hermes-agent) a user-editable long-term memory layer.
- **Coding agents**: Preserve coding style, project background, repository decisions, and workflow experience across
  sessions when integrating with coding agents such as [Claude Code](plugins/claude_code/reme).
- **LLM Wiki**: Turn conversations, notes, and resources into a searchable, traceable, and linked Markdown knowledge
  base that both users and agents can maintain.
- **Self-evolving agents**: Support agents that learn from experience by saving successful paths, failed attempts,
  reusable procedures, and periodic reflections as memory.

## 📰 News

- [2026.08] - Published the [ReMe blog](https://agentscope-ai.github.io/ReMe/?doc=en-reme-blog), an end-to-end introduction to its local-first memory
  architecture, self-evolving workflows, hybrid search, proactive discovery, and benchmark results.
- [2026.08] - [Experience-driven enhancement method](https://reme.agentscope.io/?doc=toolmemory-en) of agent tool-use execution built
  on ReMe is available on [arXiv:2608.03403](https://arxiv.org/abs/2608.03403).
- [2026.07] - Introduced optional Cookbooks: [Daily Paper](https://reme.agentscope.io/?doc=daily-paper-en) for paper discovery and
  analysis, and [Auto Fin](https://reme.agentscope.io/?doc=auto-fin-en) for researching the latest 24 hours of topic-related CLS news
  with local-memory search and validated historical wikilinks.
- [2026.07] - Our
  paper [Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution](https://aclanthology.org/2026.findings-acl.829/)
  has been accepted to Findings of ACL 2026.

## 🚀 Quick Start

### Installation

ReMe requires Python 3.11+.

Install from pip:

```bash
pip install "reme-ai[core]"
```

Install from source:

```bash
git clone https://github.com/agentscope-ai/ReMe.git
cd ReMe
pip install -e packages/reme_ai_studio -e ".[core]"
cd website
npm ci
npm run build:static
cd ..
```

The static build requires Node.js 22.13 or newer and makes Studio available from the source tree.

### Environment Variables

Configure environment variables when you want LLM-powered memory evolution or embedding retrieval. Embeddings are
disabled by default, so the default setup does not start an embedding model or require an embedding API key.

```bash
cat > .env <<'EOF'
# Optional: used only after embedding components are explicitly enabled in the config.
# EMBEDDING_API_KEY=sk-xxx
# EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Required for auto_memory, auto_resource, and auto_dream.
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EOF
```

Basic file operations, BM25 search, wikilink traversal, and reading proactive topics can run without LLM credentials.

`LLM_BACKEND` selects the provider wrapper registered in `reme/components/as_llm` (default `openai`). You can also use
[OrcaRouter](https://www.orcarouter.ai), an OpenAI-compatible gateway, with `LLM_BACKEND=orcarouter`,
`LLM_MODEL_NAME=orcarouter/auto`, and an `LLM_API_KEY` from OrcaRouter; `LLM_BASE_URL` then defaults to
`https://api.orcarouter.ai/v1`.

> [!NOTE]
> To enable embedding-based semantic retrieval, uncomment `components.as_embedding` and
> `components.embedding_store` in [`reme/config/default.yaml`](reme/config/default.yaml), then change
> `components.file_store.default.embedding_store` from `""` to `default`. See the
> [memory search guide](docs/en/memory_search.md) for details.

### Start the Service

```bash
reme start
```

The default service address is `127.0.0.1:2333`. If the port is occupied, specify another port:

```bash
reme start service.port=8181
# reme start workspace_dir=/tmp/reme-demo service.port=8181
```

```bash
reme version
reme health_check
reme help
curl -s http://127.0.0.1:2333/version -H 'Content-Type: application/json' -d '{}'
```

### ReMe Studio (Optional)

The `core` installation above includes Studio. After starting ReMe, open <http://127.0.0.1:2333/> to browse, edit, and
search the workspace. To add Studio to a base installation, use `pip install "reme-ai[web]"`. See the
[ReMe Studio guide](https://reme.agentscope.io/?doc=studio-en) for source builds, configuration, and development.

### 5-Minute Memory Demo

With the service running, write a memory node, let ReMe index it, then retrieve it:

```bash
reme write \
  path=digest/wiki/quick-start-demo \
  name="Quick Start Demo" \
  description="A first ReMe memory node" \
  content="# Quick Start Demo

ReMe stores agent memory as readable Markdown.

Related: [[digest/wiki/memory-as-file.md]]"

reme search query="agent memory markdown" limit=5
reme read path=digest/wiki/quick-start-demo start_line=1 end_line=20
```

The generated file is ordinary Markdown with frontmatter:

```markdown
---
name: Quick Start Demo
description: A first ReMe memory node
---

# Quick Start Demo

ReMe stores agent memory as readable Markdown.

Related: [[digest/wiki/memory-as-file.md]]
```

## 📚 Usage Guides

These Markdown guides cover the main user workflows and the runtime contracts implemented by the current code.

| Guide | What you will learn |
|-------|---------------------|
| [Quick Start](docs/en/quick_start.md) | Install ReMe, start the service, and run the first file and memory operations. |
| [Memory as File](docs/en/memory_as_file.md) | Understand workspace layers, frontmatter, wikilinks, chunks, and the file-as-source-of-truth model. |
| [Auto Memory](docs/en/auto_memory.md) | Preserve source conversations and distill reusable daily memory cards. |
| [Auto Resource](docs/en/auto_resource.md) | Import supported text resources and turn them into source-linked daily cards. |
| [Auto Dream](docs/en/auto_dream.md) and [Auto Link](docs/en/auto_link.md) | Consolidate daily notes into evolving digest nodes and readable wikilink relationships. |
| [Memory Search](docs/en/memory_search.md) | Use BM25, optional vectors, RRF fusion, line-range recall, and progressive link expansion. |
| [Proactive](docs/en/proactive.md) | Read interest topics safely and integrate them into a host agent's decision flow. |
| [Agent Integration Scenarios](docs/en/reme_scene.md) | Choose among CLI/SKILL.md, HTTP, MCP, and embedded Python integration. |
| [Framework](docs/en/framework.md) | Understand Application, Job, Step, Component, service, configuration, and lifecycle boundaries. |
| [ReMe Blog](https://agentscope-ai.github.io/ReMe/?doc=en-reme-blog) | Read the product story, design rationale, examples, and benchmark summary. |

## 🔌 Plugins

Plugins are optional Python distributions that contribute Component, Step, or Job backends and configuration. They are
installed separately and enabled explicitly by configuration. Auto Fin is the complete external-plugin example; Daily
Paper remains an optional research workflow while it is migrated to the same packaging model.

| Plugin / workflow                             | Capability                                                                                                    |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [Daily Paper](https://reme.agentscope.io/?doc=daily-paper-en) | Discover and rank papers, analyze PDFs with an agent, and generate file-native notes and a five-minute brief. |
| [Auto Fin](https://reme.agentscope.io/?doc=auto-fin-en)       | Fetch topic-related CLS news, search ReMe history, and generate wikilink-backed Markdown reports.             |

## 📁 Memory System

> Memory as File, File as Memory.

ReMe treats **memory as files**, progressively processing filtered conversation source records and external resources
from `session/` and `resource/` into `daily/`, then `digest/`. The default workspace is `.reme/` under the current
directory; `workspace_dir=...` selects a different user-owned location.

### Directory Structure

```text
<workspace_dir>/
├── metadata/       # Rebuildable indexes, graphs, catalogs, and caches
├── session/        # Conversation source records and agent sessions
│   ├── dialog/
│   │   └── <session_id>.jsonl  # Source messages saved by auto_memory
│   └── claude_code/
│       └── <session_id>.jsonl  # ReMe copy used by auto_memory_cc
├── mem_session/    # Generated agent-wrapper sessions/config, not user memory
│   ├── agentscope/
│   ├── claude_config/
│   └── codex/
├── resource/            # External raw materials
│   ├── <resource>.<ext>  # Root-level files enter today's daily layer
│   └── YYYY-MM-DD/
│       └── <resource>.<ext>
├── daily/               # Lightly processed memory: daily facts, conversation summaries, resource readings
│   ├── YYYY-MM-DD.md
│   └── YYYY-MM-DD/
│       ├── <generated_name>.md  # Topic-named conversation or resource card
│       └── interests.yaml
└── digest/              # Long-term memory: personal facts, procedural experience, knowledge nodes
    ├── personal/
    │   └── {topic/event}.md
    ├── procedure/
    │   └── {topic/event}.md
    └── wiki/
        └── {topic/event}.md
```

<p align="center">
  <img src="docs/figure/reme-overview.svg" alt="ReMe file-based memory system overview" width="92%">
</p>

## 🧭 Memory Design Philosophy

ReMe follows a capture → index → consolidate → recall loop. Workspace files remain the durable source of truth;
everything under `metadata/` is rebuildable.

| Capability                                  | Entry point                                     | What it does                                                                                                                                                   | Output                                                        |
|---------------------------------------------|-------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [`auto_memory`](docs/en/auto_memory.md)     | Agent hook or `reme auto_memory`                | Distills useful conversation facts while preserving a filtered conversation source record.                                                                     | `session/dialog/*.jsonl`, `daily/<date>/<generated-name>.md`  |
| [`auto_resource`](docs/en/auto_resource.md) | Resource watcher or `reme auto_resource`        | Turns files under `resource/` into source-linked, content-named daily cards.                                                                                   | `daily/<date>/<resource-card>.md`                             |
| [`auto_index`](docs/en/memory_search.md)    | Background watcher or `reme reindex`            | Live-indexes Markdown in `daily/` and `digest/`; a full rebuild also scans `resource/` and JSONL.                                                              | Searchable chunks, BM25, wikilink graph, and optional vectors |
| [`auto_dream`](docs/en/auto_dream.md)       | `dream_cron` or `reme auto_dream`               | By default, extracts up to five reusable units from changed files in the latest two-day window, then creates, corroborates, refines, or corrects digest nodes. | `digest/**`, `daily/<date>/interests.yaml`                    |
| [`proactive`](docs/en/proactive.md)         | `reme proactive` before an agent decides to act | Reads topics generated by `auto_dream`; the host agent decides whether and how to mention them.                                                                | Structured topics from `daily/<date>/interests.yaml`          |

<table>
  <tr>
    <td align="center" width="50%">
      <img src="docs/figure/memory-as-file.svg" alt="Memory as File" width="92%">
    </td>
    <td align="center" width="50%">
      <img src="docs/figure/auto-memory-resource.svg" alt="Auto Memory and Resource" width="92%">
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="docs/figure/auto-dream-and-proactive.svg" alt="Auto Dream and Proactive" width="92%">
    </td>
    <td align="center" width="50%">
      <img src="docs/figure/auto-index-and-memory-search.svg" alt="Auto Index and Memory Search" width="92%">
    </td>
  </tr>
</table>

Search returns matching chunks with line ranges and bounded wikilink neighbors. Optional vector results are fused with
BM25 through reciprocal rank fusion (RRF).

> [!IMPORTANT]
> `proactive` only reads and exposes interest topics produced by Auto Dream. It does not independently browse the web,
> send notifications, or rewrite the knowledge base; the host agent decides whether and how to act on a topic.

## 📊 Performance

ReMe evaluates multi-session and long-context memory with agentic search-and-read workflows. The figures below are the
published reference runs in this repository; model, prompt, dataset, and judging details are documented with each
benchmark.

| Benchmark                                                    | Setting      |              Sample size | Agentic score | Focus                                                              |
|--------------------------------------------------------------|--------------|-------------------------:|--------------:|--------------------------------------------------------------------|
| **[LongMemEval cleaned-s](https://reme.agentscope.io/?doc=longmemeval-en)** | **Overall**  |        **500 questions** |     **89.4%** | Cross-session retrieval, knowledge updates, and temporal reasoning |
| [BEAM](https://reme.agentscope.io/?doc=beam-en)                             | 100K context | 20 cases / 400 questions |         66.1% | Ten types of long-context memory tasks                             |
| [BEAM](https://reme.agentscope.io/?doc=beam-en)                             | 1M context   | 35 cases / 700 questions |         65.0% | Ultra-long conversation settings                                   |

ReMe also achieved a **0.580 PROC score across five user personas** in the repository's
[π-Bench evaluation](https://reme.agentscope.io/?doc=pibench-en), 2.4% above NanoBot under the same test-model configuration. PROC
measures proactive handling of hidden intent, clarification, cross-session preferences and conventions, task
dependencies, and underspecified requests.

## 🤝 Agent-friendly Integration

ReMe can run as a local memory service accessed through the CLI, HTTP API, or MCP server, or it can be embedded in the
host process through its Python API.

| Agents                                        | Recommended path                                                                                        | Available after integration                                                                             |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **QwenPaw**                                   | Embed ReMe in-process through its Python API.                                                           | Reuse the host application's lifecycle and model config while keeping memory local and file-based.      |
| **Claude Code**                               | Start the streamable HTTP MCP service and install [plugins/claude_code/reme](plugins/claude_code/reme). | MCP recall tools, a `reme-memory` skill, and a Stop hook that records sessions automatically.           |
| **Hermes**                                    | Start the HTTP service and install [plugins/hermes_agent](plugins/hermes_agent).                        | Recall relevant memory before model calls and enqueue `auto_memory` after each completed turn.          |
| **Other CLI-capable agents (OpenClaw/Codex)** | Copy or install [skills/reme_memory/SKILL.md](skills/reme_memory/SKILL.md).                             | Search, read, and write memory via the CLI; automatic recording requires explicit host lifecycle hooks. |

<p align="center"><b>Integration demos</b></p>

<table>
  <tr>
    <td align="center"></td>
    <td width="45%" align="center"><b>Auto Memory</b></td>
    <td width="45%" align="center"><b>Auto Dream</b></td>
  </tr>
  <tr>
    <td align="center"><b>QwenPaw</b></td>
    <td width="45%">
      <img src="docs/figure/qwenpaw-auto-memory.gif" alt="QwenPaw Auto Memory demo" width="100%">
    </td>
    <td width="45%">
      <img src="docs/figure/qwenpaw-auto-dream.gif" alt="QwenPaw Auto Dream demo" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center"><b>Claude Code</b></td>
    <td width="45%">
      <img src="docs/figure/cc-auto-memory.gif" alt="Claude Code Auto Memory demo" width="100%">
    </td>
    <td width="45%">
      <img src="docs/figure/cc-auto-dream.gif" alt="Claude Code Auto Dream demo" width="100%">
    </td>
  </tr>
</table>

## 🛠️ ReMe Operations

Run `reme help` for the full job list. Common workspace and maintenance commands are:

| Command                                   | Purpose                                                                                |
|-------------------------------------------|----------------------------------------------------------------------------------------|
| `reme status`                             | Show stateful data-component memory estimates and process RSS.                         |
| [`reme search`](docs/en/memory_search.md) | Retrieve memory with BM25 and wikilinks by default, plus vectors when enabled.         |
| `reme read` / `reme write` / `reme edit`  | Inspect and maintain Markdown memory files.                                            |
| `reme traverse` / `reme graph_snapshot`   | Explore wikilink neighborhoods or the category-rooted digest graph.                    |
| `reme chat`                               | Stream a read-only, workspace-aware agent conversation. Requires LLM credentials.      |
| `reme reindex`                            | Rebuild search and wikilink indexes from existing files.                               |

## 🤝 Community and Support

- **Issues, requests, and help**: Check [Open Issues](https://github.com/agentscope-ai/ReMe/issues) first. If there is no
  related discussion, open one with the background, expected behavior, and impact scope.
- **Code contributions**: Before making changes, read
  the [contribution guide](https://docs.agentscope.io/reme/latest/en/contribution). Source, schemas, and tests are the
  authoritative architecture and extension guide.
- **Documentation contributions**: Submit user-facing documentation changes to the
  [unified documentation repository](https://github.com/agentscope-ai/docs) under `reme/<version>/{en,zh}/`.
- **Commit convention**: Conventional Commits are recommended, for example `feat(search): add link expansion option` or
  `docs(zh): update quick start`.
- **Pre-submit checks**: Before submitting a PR, try to run `pre-commit run --all-files` and `pytest`. If tests that
  depend on LLMs, embeddings, or external services cannot run, explain that in the PR.
- **Documentation**: Visit [reme.agentscope.io](https://reme.agentscope.io).

### Contributors

Thanks to everyone who has contributed to ReMe:

<a href="https://github.com/agentscope-ai/ReMe/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=agentscope-ai/ReMe" alt="Contributors" />
</a>

## 📄 Citation

```bibtex
@software{ReMe2026,
  title = {Remember me, Refine me: Memory Management Kit for Agents},
  author = {ReMe Team},
  url = {https://reme.agentscope.io},
  year = {2026}
}
```

## ⚖️ License

This project is open source under the Apache License 2.0. See [LICENSE](./LICENSE) for details.
