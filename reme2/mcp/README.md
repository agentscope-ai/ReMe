# reme2.mcp

MCP transport layer. Exposes the agent-facing tools registered in
`reme2.memory` over MCP for `claude-code` and other MCP clients.

## Layout

```
reme2/mcp/
├── __init__.py
├── server.py    MCP server bootstrap; defaults to ../config/service.yaml
└── test/        end-to-end profile smoke tests
```

That's it. Every `@R.register` step the agent invokes lives one layer
down in `reme2/memory/` — there are no MCP-specific shells, because
none of the steps depend on MCP transport details. Importing
`reme2.memory` triggers all `@R.register` registrations.

| Concern | Location |
|---|---|
| Memory File System primitives | `reme2/component/file_store/`, `file_watcher/`, `file_parser/` |
| Three memory services | `reme2/memory/{retriever,ingestor,maintainer}.py` |
| Engine API (pure, schema-free) | `reme2/memory/memory_io.py` |
| Schema-bound write/read tools | `reme2/memory/memory_toolkit.py` |
| Search step shells | `reme2/memory/memory_search.py` |
| Lint step shell | `reme2/memory/memory_lint.py` |
| Hot-path event upsert | `reme2/memory/sync.py` |
| Memory schema (4 axes + presets + parser) | `reme2/memory/schema/` |
| Path templates + name disambiguation | `reme2/utils/vault_paths.py` |
| Step response serialization | `reme2/component/runtime_response.py` |

Dependency direction is strict:
```
reme2.mcp  →  reme2.memory (incl. memory.schema)  →  reme2.component / reme2.utils
```

The Ingestor (`reme2/memory/ingestor.py`) self-registers its `ingest`
MCP face. Topic creation lives inside the Ingestor's R-M-W loop; there
is no separate `topic_create` tool.

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

| Tool | Source | Purpose |
|---|---|---|
| `sync` | reme2/memory/sync.py | Hot-path event-folder upsert (idempotent per `(date, name)`). |
| `ingest` | reme2/memory/ingestor.py | Cold-path LLM-driven distillation; owns topic creation. |
| `memory_search` / `memory_graph_search` | reme2/memory/memory_search.py | V+K hybrid + optional graph BFS. |
| `memory_lint` | reme2/memory/memory_lint.py | Read-only projection of Maintainer's lint findings. |
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
