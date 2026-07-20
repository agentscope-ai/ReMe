# ReMe plugin for Codex

Connect Codex to [ReMe](https://github.com/agentscope-ai/ReMe) — file-native long-term memory
for AI agents. The plugin gives the agent **recall** (read long-term memory) and **records every
session automatically** via a Stop hook. Consolidation of daily notes into long-term `digest/`
knowledge runs server-side in ReMe.

## What you get

- **MCP tools** from the `reme` server: `search`, `traverse`, `daily_list`, `frontmatter_read`,
  `read`, `auto_memory_codex`, and more.
- **Stop hook** (`hooks/auto_memory.py`) — when a session ends it calls ReMe's server-side
  `auto_memory_codex` tool in a detached background process, passing the session id and transcript
  path from Codex's hook payload. The server reads the transcript JSONL and records the durable
  facts into today's daily note. Recording is fully automatic and asynchronous — the agent never
  records by hand, and stopping is never delayed. Best-effort: if the server is down it logs and
  gives up silently.
- **Skill** `reme-memory` — recall long-term memory before answering (semantic `search`, topological
  `traverse`, state `daily_list`/`frontmatter_read`, then `read` with citations), plus a server
  status check. Recording is handled silently by the Stop hook.

## Deployment model

The plugin **connects to a shared HTTP MCP server you start once** — it does not spawn ReMe. One
server means one set of background watchers / dream cron across all your Codex windows.

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

3. Start the ReMe MCP server (one time, leave it running):

   ```bash
   reme start service.backend=mcp service.transport=streamable-http
   ```

   It serves `http://127.0.0.1:2333/mcp`. To use a different port, start with
   `service.port=<port>` and update the `url` in `.mcp.json` to match.

## Install the plugin

```bash
codex plugin marketplace add agentscope-ai/ReMe
```

Then start a Codex session, type `/plugins`, find "ReMe Memory" in the marketplace tab, and install
it. Restart Codex, then confirm the `reme` MCP server tools are available (e.g. `search`,
`traverse`). The `reme-memory` skill can then recall memory and report server health.

### Trust the plugin hook

Codex skips hooks from non-managed plugins until you explicitly trust them. After installing,
open `/hooks` in Codex, find the **ReMe** Stop hook, review it, and mark it as trusted. Without
this step the automatic background recording is silently disabled — the skill will still be able
to recall memory, but new sessions will not be recorded.

## Supported platforms

- **macOS / Linux**: full support. The Stop hook double-forks so recording never blocks shutdown.
- **Windows**: supported with Python 3 on `PATH`. The hook re-spawns itself as a detached
  subprocess (`CREATE_NEW_PROCESS_GROUP`) to avoid blocking.

## Notes

- The plugin's MCP server URL lives in `plugins/reme/.mcp.json`. Keep it in sync with how you start
  ReMe (host/port). The Stop hook reads this same file to find the server (override with `REME_HOST`
  / `REME_PORT` env vars).
- The Stop hook needs `python3` on `PATH` (use `python` on Windows). The hook receives
  `PLUGIN_ROOT` (and `CLAUDE_PLUGIN_ROOT` as a compat alias). It logs to
  `plugins/reme/logs/auto_memory_hook.log`.
- The MCP tool-name prefix (`mcp__reme__…`) may include the server segment depending on your Codex
  version; the skill uses the `mcp__reme__*` wildcard so it works either way.
