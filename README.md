<p align="center">
 <img src="docs/figure/reme_logo.png" alt="ReMe Logo" width="50%">
</p>

<p align="center">
  <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python Version"></a>
  <a href="https://pypi.org/project/reme-ai/"><img src="https://img.shields.io/pypi/v/reme-ai.svg?logo=pypi" alt="PyPI Version"></a>
  <a href="https://pepy.tech/project/reme-ai/"><img src="https://img.shields.io/pypi/dm/reme-ai" alt="PyPI Downloads"></a>
  <a href="https://github.com/agentscope-ai/ReMe"><img src="https://img.shields.io/github/commit-activity/m/agentscope-ai/ReMe?style=flat-square" alt="GitHub commit activity"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-black" alt="License"></a>
  <a href="./README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="./README_ZH.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="https://github.com/agentscope-ai/ReMe"><img src="https://img.shields.io/github/stars/agentscope-ai/ReMe?style=social" alt="GitHub Stars"></a>
  <a href="https://deepwiki.com/agentscope-ai/ReMe"><img src="https://img.shields.io/badge/DeepWiki-Ask_Devin-navy.svg" alt="DeepWiki"></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/20528" target="_blank"><img src="https://trendshift.io/api/badge/repositories/20528" alt="agentscope-ai%2FReMe | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <strong>An agent memory layer that turns conversations and resources into readable, editable, searchable Markdown memory.</strong><br>
</p>

> Previous versions: [0.3.x](https://github.com/agentscope-ai/ReMe/tree/reme_v3) ·
> [0.2.x](https://github.com/agentscope-ai/ReMe/tree/v0.2.0.6) ·
> [MemoryScope](https://github.com/agentscope-ai/ReMe/tree/memoryscope_branch)

🧠 ReMe is a local-first memory layer for **AI agents**. It turns conversations and resources into file-based long-term
memory, then continuously indexes, links, and consolidates that memory for future recall.

## ✨ Core Ideas

- **Memory as File**: Markdown files with frontmatter and wikilinks serve as memory nodes that both users and agents can
  read and write directly.
- **Self-evolving knowledge base**: Auto Memory, Auto Resource, and Auto Dream progressively transform conversations and
  resources into long-term memories, while automatically building wikilink relationships.
- **Progressive hybrid search**: ReMe combines wikilinks, BM25, and embeddings for hybrid retrieval across keyword
  matching, semantic recall, and relationship expansion.
- **Agent-friendly integration**: SKILL.md + CLI integration makes it easy for different agents to read, write,
  maintain, and reuse memory.

<p align="center">
  <img src="docs/figure/design-philosophy.svg" alt="ReMe Design Philosophy" width="92%">
</p>

## 🔭 Use Cases

- **Personal assistants**: Give personal assistants such as
  [QwenPaw](https://github.com/agentscope-ai/QwenPaw), [OpenClaw](https://github.com/openclaw/openclaw), and
  [Hermes](https://github.com/nousresearch/hermes-agent) a user-editable long-term memory layer.
- **Coding agents**: Preserve coding style, project background, repository decisions, and workflow
  experience across sessions when integrating with coding agents such as [Claude Code](plugins/reme).
- **LLM Wiki**: Turn conversations, notes, and resources into a searchable, traceable, and linked Markdown
  knowledge base that both users and agents can maintain.
- **Self-evolving agents**: Support agents that learn from experience by saving successful paths, failed attempts,
  reusable procedures, and periodic reflections as memory.

## 📰 News

- [2026.07] - Introduced optional Cookbook workflows, starting with
  [Daily Paper](cookbook/daily_paper/README.md) for scheduled paper discovery, agent-assisted PDF analysis, reusable
  Markdown notes, and five-minute briefs.
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
pip install -e ".[core]"
```

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

After startup, check the service status. If you use a custom port, replace `2333` in the URL below with that port.

```bash
reme version
curl -s http://127.0.0.1:2333/version -H 'Content-Type: application/json' -d '{}'
```

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

## 🧑‍🍳 Cookbooks

Cookbooks are optional, end-to-end workflows assembled from ReMe jobs and steps. They are not enabled by the default
configuration; select the cookbook's standalone configuration when starting ReMe. Each new cookbook will be added as
another row in this table.

| Cookbook                                           | Capability                                                                                                    |
|----------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [Daily Paper](cookbook/daily_paper/README.md)      | Discover and rank papers, analyze PDFs with an agent, and generate file-native notes and a five-minute brief. |

## 📁 Memory System

> Memory as File, File as Memory.

ReMe treats **memory as files**, progressively processing raw conversations and external resources from `session/` and
`resource/` into `daily/`, then consolidating them into reusable long-term memory nodes under `digest/`.

### Directory Structure

```text
<workspace_dir>/
├── metadata/       # Persistent system state such as indexes, graphs, and catalogs
├── session/        # Raw conversations and agent sessions
│   ├── dialog/
│   │   └── <session_id>.jsonl
│   ├── agentscope/
│   └── claude_code/
├── resource/            # External raw materials
│   └── YYYY-MM-DD/
│       └── <resource>.<ext>
├── daily/               # Lightly processed memory: daily facts, conversation summaries, resource readings
│   ├── YYYY-MM-DD.md
│   └── YYYY-MM-DD/
│       ├── <session_event>.md
│       ├── <resource_stem>.md
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

> Capture raw dialogs and resources, refine them into long-term preferences, reusable experience, and valuable
> knowledge,
> while keeping the result editable by humans and agents.

### Automatic Memory Flow

ReMe follows a capture → index → consolidate → recall loop. Conversations and resources first become daily memory cards;
background jobs keep files searchable; `auto_dream` distills stable knowledge into `digest/`; agents recall memory
through search, wikilinks, or proactive topics.

| Capability                                  | Entry point                                     | What it does                                                                                    | Output                                                  |
|---------------------------------------------|-------------------------------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| [`auto_memory`](docs/en/auto_memory.md)     | Agent hook or `reme auto_memory`                | Distills useful conversation facts while preserving the raw session.                            | `session/dialog/*.jsonl`, `daily/<date>/<session>.md`   |
| [`auto_resource`](docs/en/auto_resource.md) | Resource watcher or `reme auto_resource`        | Turns files under `resource/<date>/` into source-linked daily cards.                            | `daily/<date>/<resource-card>.md`                       |
| [`auto_index`](docs/en/memory_search.md)    | Background watcher or `reme reindex`            | Maintains chunks, the BM25 index, the wikilink graph, and the optional embedding index.         | Searchable `daily/`, `digest/`, and `resource/` content |
| [`auto_dream`](docs/en/auto_dream.md)       | `dream_cron` or `reme auto_dream`               | Consolidates changed daily cards into long-term personal, procedure, and wiki memory.           | `digest/**`, `daily/<date>/interests.yaml`              |
| [`proactive`](docs/en/proactive.md)         | `reme proactive` before an agent decides to act | Reads topics generated by `auto_dream`; the host agent decides whether and how to mention them. | Structured topics from `daily/<date>/interests.yaml`    |

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

## 🤝 Agent-friendly Integration

ReMe runs as a local memory service and offers multiple integration paths: CLI, HTTP API, MCP server, and SDK. Different
agents can choose the path that fits their runtime while sharing the same local memory workspace.

| Agents                                               | Recommended path                                                            | What works out of the box                                                                     |
|------------------------------------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **QwenPaw**                                          | Embed ReMe via the Python SDK.                                              | Reuse the app's own lifecycle and model config while keeping memory local and file-based.     |
| **Claude Code**                                      | Start ReMe as an MCP service and install [plugins/reme](plugins/reme).      | MCP recall tools, a `reme-memory` skill, and a Stop hook that records sessions automatically. |
| **Other CLI-capable agents (OpenClaw/Hermes/Codex)** | Copy or install [skills/reme_memory/SKILL.md](skills/reme_memory/SKILL.md). | Search/read/write memory and call `auto_memory`, `auto_dream`, and `proactive` via the CLI.   |

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

ReMe operates the workspace through a unified job interface exposed by the CLI. Agents usually only need retrieval,
reading, writing, editing, and automatic memory commands. Lower-level indexing, frontmatter, and file operation commands
are mainly for maintenance, debugging, or advanced integration. Run `reme help` for the full job list.

| Command                                   | Purpose                                                                                |
|-------------------------------------------|----------------------------------------------------------------------------------------|
| `reme start`                              | Start the local ReMe service.                                                          |
| `reme version` / `reme health_check`      | Check package and component status.                                                    |
| `reme status`                             | Show stateful data-component memory estimates and process RSS.                         |
| [`reme search`](docs/en/memory_search.md) | Retrieve memory with BM25 and wikilinks by default, plus vectors when enabled.         |
| `reme read` / `reme write` / `reme edit`  | Inspect and maintain Markdown memory files.                                            |
| `reme auto_memory`                        | Turn conversation messages into daily memory cards. Requires LLM credentials.          |
| `reme auto_resource`                      | Interpret files under `resource/` into daily resource cards. Requires LLM credentials. |
| `reme auto_dream` / `reme proactive`      | Consolidate daily memory into long-term digest and surface topics worth attention.     |
| `reme reindex`                            | Rebuild search and wikilink indexes from existing files.                               |

## 🤝 Community and Support

- **Issues and requests**: Check [Open Issues](https://github.com/agentscope-ai/ReMe/issues) first. If there is no
  related discussion, open a new issue with background, expected behavior, and impact scope.
- **Code contributions**: Before making changes, read
  the [contribution guide](https://docs.agentscope.io/reme/stable/en/contributing). Source,
  schemas, and tests are the authoritative architecture and extension guide.
- **Documentation contributions**: Submit user-facing documentation changes to the
  [unified documentation repository](https://github.com/agentscope-ai/docs) under `reme/<version>/{en,zh}/`.
- **Commit convention**: Conventional Commits are recommended, for example `feat(search): add link expansion option` or
  `docs(zh): update quick start`.
- **Pre-submit checks**: Before submitting a PR, try to run `pre-commit run --all-files` and `pytest`. If tests that
  depend on LLMs, embeddings, or external services cannot run, explain that in the PR.
- **Get help**: Use [GitHub Issues](https://github.com/agentscope-ai/ReMe/issues) for bugs and feature requests. Project
  documentation is available at [https://docs.agentscope.io/](https://docs.agentscope.io/reme/stable/en/).

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
