# ReMe plugin for QwenPaw

Connect [QwenPaw](https://github.com/agentscope-ai/ReMe) (the Qwen-series assistant)
to [ReMe](https://github.com/agentscope-ai/ReMe) — file-native long-term memory for
AI agents. The plugin gives QwenPaw **recall** (read long-term memory before replies)
and **automatic durable recording** of useful turns after the fact via a non-blocking
background writer. Consolidation of daily notes into long-term `digest/` knowledge
runs server-side in ReMe.

> **Status:** Skeleton plugin. This package ships the QwenPaw installable plugin
> shape, the ReMe MCP server configuration, the memory-recall Skill with three
> recall modes (semantic / topological / state), and an auto-memory stop hook
> adapted to QwenPaw's session layout. The deeper QwenPaw SDK-level integration
> (embedding the ReMe Python SDK and wiring it directly into QwenPaw's internal
> scheduler and conversation manager) is intentionally **out of scope** for this
> PR and is tracked separately. Contributions and follow-up PRs are welcome.

## What you get

- **MCP tools** from the running `reme` HTTP server: `search`, `traverse`,
  `daily_list`, `frontmatter_read`, `read`, `auto_memory`, and more.
- **Stop hook** (`hooks/auto_memory.py`) — when a QwenPaw conversation ends it
  calls ReMe's server-side `auto_memory` tool in a detached background process,
  anchored on the QwenPaw conversation id. The server resolves the transcript on
  disk and records durable facts into today's daily note. Recording is automatic
  and async — stopping the assistant is never delayed. Best-effort: if the ReMe
  service is unreachable it logs and gives up silently.
- **Skill** `reme-memory` — recall long-term memory before replying (semantic
  `search`, topological `traverse`, state `daily_list`/`frontmatter_read`, then
  `read` with citations), plus a server-health check. Recording is handled by
  the stop hook.

## Prerequisites

1. Install ReMe (Python 3.11+):

   ```bash
   pip install "reme-ai[core]"
   ```

2. Configure model credentials in a `.env` (see `example.env`):

   ```bash
   EMBEDDING_API_KEY=sk-xxx
   EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   LLM_API_KEY=sk-xxx
   LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   ```

3. Start the ReMe HTTP/MCP service (leave it running):

   ```bash
   reme start service.backend=mcp service.transport=streamable-http
   ```

   It serves `http://127.0.0.1:2333/mcp`. To use a different port, start with
   `service.port=<port>` and update the `url` in `.mcp.json` to match.

## Install the plugin

QwenPaw supports plugin installation from a repository subdirectory. From the
ReMe repository root:

```bash
qwenpaw plugins add ./plugins/qwenpaw
qwenpaw plugins enable reme
```

(Or point `qwenpaw plugins add` at the upstream GitHub repo + subpath once
published.) Then restart QwenPaw and confirm the `reme` MCP server and its
`reme-memory` skill are connected; the assistant will then automatically recall
memory and report server health.

## Directory layout

```
plugins/qwenpaw/
├── README.md                    # this file
└── reme/
    ├── .mcp.json                # ReMe streamable-http MCP endpoint
    ├── hooks/
    │   ├── hooks.json           # Stop hook registration
    │   └── auto_memory.py       # Fire-and-forget auto-memory (detached)
    └── skills/
        └── reme-memory/
            └── SKILL.md         # Recall protocol + 3-mode retrieval SOP
```

## Notes

- The MCP server URL lives in `plugins/qwenpaw/reme/.mcp.json`. Keep it in sync
  with how you start ReMe (host/port). The stop hook reads this same file to
  locate the server — override via `REME_HOST` / `REME_PORT` if needed.
- The stop hook requires `python3` on `PATH`. It resolves QwenPaw conversation
  transcripts under the QwenPaw data directory (override via
  `QWENPAW_DATA_DIR`). Logs go to `plugins/qwenpaw/reme/logs/auto_memory_hook.log`.
- Multi-user scoping: attach stable `tags: [user:alice]` / `tags: [conv:xxx]`
  frontmatter to files written by each profile, then use the search-step
  `tags` filter to keep recall results isolated between QwenPaw profiles.
