# reme2.mcp

Agent-facing MCP interface layer. Exposes the markdown vault under
`reme2/` (file_store + watcher + memory services) as MCP tools for
`claude-code` and other MCP clients.

## Layout

```
reme2/mcp/
├── __init__.py
├── server.py          MCP server bootstrap; defaults to ../config/service.yaml
└── steps/             @R.register MCP step shells
    ├── memory_retriever.py memory_search + memory_graph_search
    ├── memory_lint.py      read-only Maintainer projection (lint findings)
    └── sync.py             hot-path event-folder upsert
```

The 12 `memory_*` primitives (create/update/property_update/rename/
delete/archive/get/list/links/backlinks/resolve_wikilink/count_tokens)
live one layer down in `reme2/memory/memory_toolkit.py` — each is a
single `BaseStep` subclass with two class methods: `execute()` for the
MCP path (this layer) and a same-named method for the agent toolkit
path that the Ingestor's ReActAgent consumes. Importing
`reme2.mcp.steps` triggers all `@R.register` registrations.

The MCP layer's job is to **project** existing primitives as MCP tools
— it owns no business logic. Everything else lives outside the
transport boundary so memory services don't form an import cycle:

| Concern | Location |
|---|---|
| Memory File System primitives | `reme2/component/file_store/`, `file_watcher/`, `file_parser/` |
| Three memory services | `reme2/memory/` — Retriever / Ingestor / Maintainer |
| Engine API (pure, schema-free) | `reme2/memory/memory_io.py` |
| Schema-bound tools (BaseStep + agent toolkit) | `reme2/memory/memory_toolkit.py` |
| Memory schema (4 axes + presets + parser) | `reme2/memory/schema/` |
| Path templates + name disambiguation | `reme2/utils/vault_paths.py` |
| Step response serialization | `reme2/component/runtime_response.py` |

Dependency direction is strict:
```
reme2.mcp  →  reme2.memory (incl. memory.schema)  →  reme2.component / reme2.utils
```

The Ingestor (`reme2/memory/ingestor.py`) self-registers its `ingest`
MCP face — there's no shell for it under `steps/`. Topic creation lives
inside the Ingestor's R-M-W loop; there is no separate `topic_create`
tool.

## Run

```bash
# stdio MCP server, service-tier surface (loads ../config/service.yaml)
python -m reme2.mcp.server

# pick a different profile
python -m reme2.mcp.server config=reme2/config/expert.yaml

# override any nested key (Hydra-style)
python -m reme2.mcp.server components.file_watcher.default.watch_path=/abs/vault
```

Env vars (read by `server.py` and applied as overrides):

| Var | Effect |
|---|---|
| `VAULT_PATH` | Sets `file_watcher.default.watch_path`, `file_store.default.db_path`, and `service.sidecar_info_path` in one shot. Wins over CLI args. |
| `VAULT_HTTP_PORT` | Override `service.sidecar_http_port` (default `8765`). |

Both shipped profiles register an `mcp` service with `transport: stdio`
plus a sidecar HTTP server on `127.0.0.1:8765`. The sidecar lets local
hooks (e.g. the plugin's `vault_recall.py`) reach the same in-process
FileGraph over HTTP without re-bootstrapping; on startup the server
writes `{host, port}` to `service.sidecar_info_path`
(default `./vault/.reme/sidecar.json`).

## Profiles

Configs live in `reme2/config/`:

- **`expert.yaml`** — every primitive surfaced (16 tools): `sync` + `ingest`
  + `memory_search` / `memory_graph_search` + 5 read primitives + 6 raw
  write primitives + `memory_count_tokens` + `memory_lint`. For agents
  that should manage the vault directly with full control.
- **`service.yaml`** — opinionated 3-tool surface: `retrieve` (graph-aware
  hybrid retrieval), `remember` (single write entry — `mode=log` /
  `mode=distill`), `maintain` (vault hygiene sweep). One tool per
  memory service; the minimum the agent needs to read, log, distill,
  and clean up.

### Tools exposed (expert profile)

| Tool | Path | Purpose |
|---|---|---|
| `sync` | steps/sync.py | Hot-path event-folder upsert (idempotent per `(date, name)`). |
| `ingest` | reme2/memory/ingestor.py | Cold-path LLM-driven distillation; owns topic creation. |
| `memory_search` / `memory_graph_search` | steps/memory_retriever.py | V+K hybrid + optional graph BFS. |
| `memory_lint` | steps/memory_lint.py | Read-only projection of Maintainer's lint findings. |
| `memory_get` / `memory_list` / `memory_links` / `memory_backlinks` / `memory_resolve_wikilink` | reme2/memory/memory_toolkit.py | Read primitives. |
| `memory_create` / `memory_update` / `memory_property_update` / `memory_rename` / `memory_delete` / `memory_archive` | reme2/memory/memory_toolkit.py | Raw write primitives (prefer `ingest` / `sync`). |
| `memory_count_tokens` | reme2/memory/memory_toolkit.py | Token estimation. |

### Tools exposed (service profile)

| Tool | Backend | Purpose |
|---|---|---|
| `retrieve` | `memory_graph_search` | Graph-aware hybrid retrieval (vector + keyword + 1-hop wikilink BFS). |
| `remember` | `ingestor` | Single write entry — `mode=log` (zero-LLM event-folder upsert) / `mode=distill` (LLM R-M-W into topic graph). |
| `maintain` | `maintainer` | Vault hygiene sweep — lint + decay (merge/split require LLM, off by default). |

## End-to-end tests

```bash
python -m reme2.mcp.test
```

Boots one Application per profile against a temp vault, exercises every
registered MCP job end-to-end (in-process — no MCP transport), and
prints a per-check status line plus a final pass/fail summary. Exit
code = number of failed checks.

Layout (`reme2/mcp/test/`):

- `_helpers.py`     — shared vault seed, app factory, response decoder
- `test_expert.py`  — expert profile (25 checks across all 16 tools incl. schema gates)
- `test_service.py` — service profile (10 checks across `retrieve` / `remember` / `maintain`)
- `__main__.py`     — CLI runner
