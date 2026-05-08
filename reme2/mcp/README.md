# reme2.mcp

Agent-facing MCP interface layer. Exposes the markdown vault under
`reme2/` (file_store + watcher + memory services) as MCP tools for
`claude-code` and other MCP clients.

## Layout

```
reme2/mcp/
├── __init__.py
├── server.py          MCP server bootstrap; defaults to ../config/full.yaml
└── steps/             @R.register MCP step shells
    ├── memory_io.py        memory_create/update/get/list/links/...
    ├── memory_retriever.py memory_search + memory_graph_search
    ├── sync.py             hot-path event-folder upsert
    └── topic_create.py     typed topic creation w/ schema gates
```

The MCP layer's job is to **project** existing primitives as MCP tools
— it owns no business logic. Everything else lives outside the
transport boundary so memory services don't form an import cycle:

| Concern | Location |
|---|---|
| Memory File System primitives | `reme2/component/file_store/`, `file_watcher/`, `file_parser/` |
| Three memory services | `reme2/memory/` — Retriever / Ingestor / Maintainer |
| Pure write helpers + agent toolkit | `reme2/memory/memory_io.py` |
| Vault domain models (Event, Topic) | `reme2/schema/vault/` |
| Path templates + name disambiguation | `reme2/utils/vault_paths.py` |
| Step response serialization | `reme2/component/runtime_response.py` |

Dependency direction is strict:
```
reme2.mcp  →  reme2.memory  →  reme2.component / reme2.schema / reme2.utils
```

## Run

```bash
# stdio MCP server, full tool surface
python -m reme2.mcp.server

# override config or any field
python -m reme2.mcp.server config=reme2/config/curated.yaml
python -m reme2.mcp.server components.file_watcher.default.watch_path=/abs/vault
```

Config profiles (in `reme2/config/`):
- `full.yaml`    — every memory_* primitive + sync + topic_create + ingest
- `curated.yaml` — opinionated 3-tool surface (`query`, `sync`, `ingest`)

## Tools exposed (full profile)

| Tool | Path | Purpose |
|---|---|---|
| `sync` | steps/sync.py | Hot-path event-folder upsert (idempotent per `(date, name)`). |
| `ingest` | reme2/memory/ingestor.py | Cold-path LLM-driven distillation. |
| `topic_create` | steps/topic_create.py | Typed topic creation with schema gates. |
| `memory_search` / `memory_graph_search` | steps/memory_retriever.py | V+K hybrid + optional graph BFS. |
| `memory_get` / `memory_list` / `memory_links` / `memory_backlinks` / `memory_resolve_wikilink` | steps/memory_io.py | Read primitives. |
| `memory_create` / `memory_update` / `memory_property_update` / `memory_rename` / `memory_delete` / `memory_archive` | steps/memory_io.py | Raw write primitives (prefer `ingest` / `sync`). |
| `memory_count_tokens` | steps/memory_io.py | Token estimation. |

## Smoke test

```bash
python reme2/config/smoke_test.py
```

Boots both `full` and `curated` profiles in a temp vault, exercises a
representative subset of jobs end-to-end (in-process — no MCP
transport).
