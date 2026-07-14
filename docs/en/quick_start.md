# Quick Start

## Installation

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

Installing the `core` extra is recommended. The current code imports the AgentScope wrapper, and self-evolving memory also
depends on it.

To use agent workflows such as `auto_memory`, `auto_resource`, and `auto_dream`, configure an LLM:

```bash
cat > .env <<'EOF'
LLM_BACKEND=openai
LLM_MODEL_NAME=qwen3.7-plus
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EOF
```

You can initially omit the LLM configuration if you only need basic file operations and BM25 retrieval.

---

## Start the Service

```bash
reme start
```

The default service address is `127.0.0.1:2333`. If the port is already in use:

```bash
reme start service.port=8181
```

```bash
reme version
reme health_check
reme list
```

`reme list` lists server actions. Ordinary commands invoke server Jobs over HTTP.

---

## Workspace Layout

The default workspace is `.reme/` under the current directory. It is created automatically at startup:

```text
.reme/
├── metadata/   # persistent indexes, graph, catalogs, and related state
├── session/    # agent sessions and original conversations
├── resource/   # external resources
├── daily/      # daily notes
└── digest/     # long-term memory
```

For directory layers, Markdown frontmatter, and wikilink semantics, see
[Memory as File](./memory_as_file.md).

You can also specify the workspace at startup:

```bash
reme start workspace_dir=/tmp/reme-demo service.port=8181
```

---

## Write, Index, and Search

```bash
reme write \
  path=digest/wiki/quick-start-demo \
  name="Quick Start Demo" \
  description="Example memory for the quick start" \
  content="# Quick Start Demo

ReMe indexes Markdown under the daily, digest, and resource directories.

Related link: [[digest/wiki/search-demo.md]]"
```

`path` is relative to the workspace. A missing suffix is automatically completed with `.md`. For Markdown files, `name` and
`description` are written to frontmatter.

The background watcher builds the index automatically. You can also rebuild it manually:

```bash
reme reindex
```

Search:

```bash
reme search query="quick start example memory" limit=5
```

Read:

```bash
reme read path=digest/wiki/quick-start-demo start_line=1 end_line=20
```

With the default configuration, retrieval is primarily BM25 plus wikilink graph expansion. Vector retrieval is supported by
the code, but the embedding store is disabled by default. For the full retrieval flow, see
[Memory Search](./memory_search.md).

---

## Files and Daily Notes

```bash
reme stat path=digest/wiki/quick-start-demo
reme edit path=digest/wiki/quick-start-demo old="indexes" new="continuously indexes"
reme frontmatter_read path=digest/wiki/quick-start-demo
reme frontmatter_update path=digest/wiki/quick-start-demo metadata='{"tags":["demo"]}'
```

The name `list` is used by the CLI to list actions, so the file-listing Job must be called over HTTP:

```bash
curl -s http://127.0.0.1:2333/list \
  -H 'Content-Type: application/json' \
  -d '{"path":"digest","recursive":true,"limit":50}'
```

Daily notes:

```bash
reme write path=daily/2026-06-20/demo-session.md name=demo-session description="Demo session" content="Recorded content"
reme daily_list
reme daily_reindex
```

`write` can create a daily note directly. Run `daily_reindex` when the day's index needs to be refreshed.

---

## Automatic Memory

```bash
reme auto_memory \
  session_id=chat-demo \
  messages='[{"role":"user","content":"I prefer to preserve project experience as Markdown."},{"role":"assistant","content":"Recorded."}]' \
  memory_hint="Record the user's preference"
```

After placing external material under `resource/YYYY-MM-DD/`, the default background task watches
`md/txt/json/jsonl/csv/yaml/html`. You can also trigger processing manually:

```bash
reme auto_resource changes='[{"path":"resource/2026-06-20/report.md","change":"added"}]'
```

Distill daily notes into long-term digest memory:

```bash
reme auto_dream date=2026-06-20
reme proactive date=2026-06-20
```

These flows require a working LLM. Without an LLM configuration, start with basic capabilities such as `write`, `read`, and
`search`.

For more detail, see [Auto Memory](./auto_memory.md), [Auto Resource](./auto_resource.md),
[Auto Dream](./auto_dream.md), and [Proactive](./proactive.md).

---

## HTTP and Configuration

Every service-enabled Job is exposed as `POST /<job>`:

```bash
curl -s http://127.0.0.1:2333/version \
  -H 'Content-Type: application/json' \
  -d '{}'

curl -s http://127.0.0.1:2333/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"quick start","limit":5}'
```

The default configuration comes from `reme/config/default.yaml`. Override it at startup with dot notation:

```bash
reme start \
  workspace_dir=/tmp/reme-demo \
  service.host=127.0.0.1 \
  service.port=8181 \
  enable_logo=false
```

You can also specify a YAML or JSON configuration file:

```bash
reme start config=/path/to/custom.yaml
```
