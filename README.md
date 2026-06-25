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
  <strong>A memory management toolkit for AI agents — Remember Me, Refine Me.</strong><br>
</p>

> Previous versions: [0.3.x](https://github.com/agentscope-ai/ReMe/tree/reme_v3) ·
> [0.2.x](https://github.com/agentscope-ai/ReMe/tree/v0.2.0.6) ·
> [MemoryScope](https://github.com/agentscope-ai/ReMe/tree/memoryscope_branch)

🧠 ReMe is a memory management toolkit for **AI agents**. It turns conversations and resources into readable, editable, and searchable file-based long-term memory.

## ✨ Core Ideas

- **Memory as File**: Markdown files with frontmatter and wikilinks serve as memory nodes that both users and agents can read and write directly.
- **Self-evolving knowledge base**: Auto Memory, Auto Resource, and Auto Dream progressively transform conversations and resources into long-term Markdown memories, while automatically building wikilink relationships.
- **Progressive hybrid search**: ReMe combines wikilinks, BM25, and embeddings for hybrid retrieval across keyword matching, semantic recall, and relationship expansion.
- **Agent-friendly integration**: SKILL.md + CLI integration makes it easy for different agents to read, write, maintain, and reuse memory.

<p align="center">
  <img src="docs/figure/design-philosophy.svg" alt="ReMe Design Philosophy" width="92%">
</p>

<details>
<summary><b>Use Cases</b></summary>

<br>

- **Personal assistants**: Provide long-term memory for agents such as [QwenPaw](https://github.com/agentscope-ai/QwenPaw).
- **Coding assistants**: Preserve coding style, project background, and workflow experience across sessions.
- **Knowledge QA**: Progressively transform resources and conversations into a searchable, traceable, and linked Markdown knowledge base.
- **Task automation**: Reuse successful paths, lessons from failures, and operation procedures from past tasks.
</details>

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

Configure environment variables:

```bash
cat > .env <<'EOF'
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EOF
```

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

### Agent Integration

ReMe integrates with supported agent frameworks through **SKILL.md + CLI + hooks (optional)**. A typical integration looks like this:

- Add the [memory skill](skills/reme_memory/SKILL.md) to the agent and grant the agent permission to call the CLI.
- Call `auto_memory` and `proactive` from agent hooks as needed, so conversations are automatically consolidated into daily memories and proactive reminders can be read at the right time.
- `auto_index` and `auto_resource` are triggered by file monitoring to maintain indexes and process resources.
- `auto_dream` is triggered by a scheduled task to further organize daily memories into reusable long-term digest memories.

QwenPaw 2.0 will integrate the new ReMe version. A Claude Code plugin will also be released later to reduce manual integration work.

For more details, see the [Quick Start](docs/zh/quick_start.md).

## 📁 Memory System

> Memory as File, File as Memory.

ReMe treats **memory as files**, progressively processing raw conversations and external resources from `session/` and `resource/` into `daily/`, then consolidating them into reusable long-term knowledge nodes under `digest/`.

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
│       ├── <session_id>.md
│       ├── <resource_stem>.md
│       └── interests.yaml
└── digest/              # Long-term memory: personal facts, procedural experience, knowledge nodes
    ├── personal/
    ├── procedure/
    └── wiki/
```

<p align="center">
  <img src="docs/figure/reme-overview.svg" alt="ReMe file-based memory system overview" width="92%">
</p>

### Automatic Memory Flow

ReMe's automatic memory flow gradually turns raw conversations and resources into searchable, traceable, and reusable file-based memory. During normal operation, background watchers maintain indexes and process resources, agent hooks trigger conversation memory, and long-term consolidation plus proactive reminders run through scheduled tasks or on-demand calls.

<details>
<summary><b>Automatic Memory Capabilities</b></summary>

<br>

| Capability                                    | How it runs                                      | Purpose                                                                                                                                  | Main parameters                                      |
|-----------------------------------------------|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| [`auto_index`](docs/zh/memory_search.md)      | Background maintenance via `index_update_loop`   | Scans on startup and continuously watches Markdown/JSONL changes in `daily/`, `digest/`, and `resource/`; updates chunk, BM25, embedding, and wikilink graph indexes. | Config: `watch_dirs`, `watch_suffixes`               |
| [`auto_memory`](docs/zh/auto_memory.md)       | Agent after-reply hook; also callable on demand  | Saves raw conversation text and turns long-term valuable information into `daily/<date>/<session_id>.md` memory cards.                   | Required: `messages`; optional: `session_id`, `memory_hint` |
| [`auto_resource`](docs/zh/auto_resource.md)   | Automatically triggered by resource watching; also callable on demand | Reads resource changes under `resource/<date>/` and creates or updates LLM-named daily resource cards linked by `source_resource`.       | Required: `changes`; each item may include `path`, `file_path`, `change` |
| [`auto_dream`](docs/zh/auto_dream.md)         | Scheduled by `dream_cron`; also callable on demand | Scans daily input for a given date, extracts long-term memory units, integrates them into `digest/`, and writes `daily/<date>/interests.yaml`. | `date`, `hint`, `topic_count`, `topic_diversity_days` |
| [`proactive`](docs/zh/proactive.md)           | Read on demand before agent proactive reminders  | Reads `interests.yaml` generated by `auto_dream` and exposes topics worth attention to the upper-level agent; the caller decides whether to remind the user. | `date`, `include_content`                            |

</details>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="docs/figure/memory-as-file.svg" alt="Memory as File" width="100%">
    </td>
    <td align="center" width="50%">
      <img src="docs/figure/auto-memory-resource.svg" alt="Auto Memory and Resource" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="docs/figure/auto-dream-and-proactive.svg" alt="Auto Dream and Proactive" width="100%">
    </td>
    <td align="center" width="50%">
      <img src="docs/figure/auto-index-and-memory-search.svg" alt="Auto Index and Memory Search" width="100%">
    </td>
  </tr>
</table>

### Workspace Operation Interface

ReMe operates the workspace through a unified CLI / Service Job interface. Agents usually only need retrieval, read, write, edit, and automatic memory commands. Lower-level indexing, frontmatter, and file operation commands are mainly for maintenance, debugging, or advanced integration.

<details>
<summary><b>Workspace Operation Interface</b></summary>

<br>

| Category       | name                                 | Description                                                                 | Parameters                                             |
|----------------|--------------------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------|
| System status  | `version`                            | Returns the ReMe package version.                                           | None                                                   |
| System status  | `health_check`                       | Returns a health-check summary for ReMe components.                         | None                                                   |
| System status  | `help`                               | Lists registered jobs and their metadata.                                   | None                                                   |
| Retrieval/read | [`search`](docs/zh/memory_search.md) | Performs hybrid retrieval in the workspace with vector recall, BM25, and RRF fusion. | Required: `query`; optional: `limit`, `min_score`      |
| Retrieval/read | `node_search`                        | Recalls similar digest nodes by candidate abstraction name and description, mainly for `auto_dream` deduplication or association. | Required: `query`; optional: `limit`                   |
| Retrieval/read | `traverse`                           | Traverses the wikilink graph from a specified path.                         | Required: `path`; optional: `depth`, `direction`       |
| Retrieval/read | `read`                               | Reads a Markdown file under the workspace.                                      | Required: `path`; optional: `start_line`, `end_line`   |
| Retrieval/read | `read_image`                         | Reads an image file under the workspace and returns base64.                     | Required: `path`                                      |
| Index          | `reindex`                            | Clears file-store indexes and rebuilds indexes from existing files.         | Config: `watch_dirs`, `watch_suffixes`                 |
| Daily          | `daily_list`                         | Lists notes for a day.                                                      | `date`                                                 |
| Daily          | `daily_reindex`                      | Rebuilds the day-index page `daily/<date>.md`.                              | `date`                                                 |
| Metadata       | `frontmatter_read`                   | Reads file frontmatter.                                                     | Required: `path`                                      |
| Metadata       | `frontmatter_update`                 | Merges key-values into file frontmatter.                                    | Required: `path`, `metadata`                           |
| Metadata       | `frontmatter_delete`                 | Deletes specified keys from file frontmatter.                               | Required: `path`, `keys`                               |
| File operation | `stat`                               | Gets workspace path status, including size, mtime, existence, and file/directory type. | Required: `path`                                      |
| File operation | `list`                               | Lists files under a workspace path.                                             | `path`, `recursive`, `limit`                           |
| File operation | `write`                              | Creates or overwrites a Markdown file and writes name/description frontmatter. | Required: `path`, `name`, `description`, `content`; optional: `metadata` |
| File operation | `edit`                               | Performs full-text find-and-replace on a Markdown file.                     | Required: `path`, `old`, `new`                         |
| File operation | `move`                               | Moves or renames a workspace file and rewrites inbound wikilinks by default.    | Required: `src_path`, `dst_path`; optional: `overwrite`, `retarget` |
| File operation | `delete`                             | Deletes a workspace file or folder and returns inbound wikilinks that still exist. | Required: `path`                                      |

</details>

## 🤝 Community and Support

- **Issues and requests**: Check [Open Issues](https://github.com/agentscope-ai/ReMe/issues) first. If there is no related discussion, open a new issue with background, expected behavior, and impact scope.
- **Code contributions**: Before making changes, read the [contribution guide](docs/zh/contributing.md) and [code framework](docs/zh/framework.md), and follow the CLI / Service / Application / Job / Step / Component layering.
- **Documentation contributions**: For user-visible installation, configuration, invocation, or behavior changes, update `docs/zh/` or `README.md` accordingly.
- **Commit convention**: Conventional Commits are recommended, for example `feat(search): add link expansion option` or `docs(zh): update quick start`.
- **Pre-submit checks**: Before submitting a PR, try to run `pre-commit run --all-files` and `pytest`. If tests depending on LLMs, embeddings, or external services cannot run, explain that in the PR.
- **Get help**: Use [GitHub Issues](https://github.com/agentscope-ai/ReMe/issues) for bugs and feature requests. Project documentation is available at [https://reme.agentscope.io/](https://reme.agentscope.io/).

### Contributors

Thanks to everyone who has contributed to ReMe:

<a href="https://github.com/agentscope-ai/ReMe/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=agentscope-ai/ReMe" alt="Contributors" />
</a>

## 📄 Citation

```bibtex
@software{AgentscopeReMe2026,
  title = {AgentscopeReMe: Memory Management Kit for Agents},
  author = {ReMe Team},
  url = {https://reme.agentscope.io},
  year = {2026}
}
```

## ⚖️ License

This project is open source under the Apache License 2.0. See [LICENSE](./LICENSE) for details.

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=agentscope-ai/ReMe&type=Date)](https://www.star-history.com/#agentscope-ai/ReMe&Date)
