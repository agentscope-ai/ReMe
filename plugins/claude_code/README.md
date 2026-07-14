# ReMe plugin for Claude Code

Connect Claude Code to [ReMe](https://github.com/agentscope-ai/ReMe) — file-native long-term memory
for AI agents. The plugin gives the agent **recall** (read long-term memory) and **records every
session automatically** via a Stop hook. Consolidation of daily notes into long-term `digest/`
knowledge runs server-side in ReMe.

## What you get

- **MCP tools** from the `reme` server: `search`, `traverse`, `daily_list`, `frontmatter_read`,
  `read`, `auto_memory_cc`, and more.
- **Stop hook** (`hooks/auto_memory.py`) — when a session ends it calls ReMe's server-side
  `auto_memory_cc` tool in a detached background process, passing **only the session id**. The server
  resolves that session's transcript on disk and records the durable facts into today's daily note.
  Recording is fully automatic and asynchronous — the agent never records by hand, and stopping is
  never delayed. Best-effort: if the server is down it logs and gives up silently.
- **Skill** `reme-memory` — recall long-term memory before answering (semantic `search`, topological
  `traverse`, state `daily_list`/`frontmatter_read`, then `read` with citations), plus a server
  status check. Recording is handled silently by the Stop hook.

## Deployment model

The plugin **connects to a shared HTTP MCP server you start once** — it does not spawn ReMe. One
server means one set of background watchers / dream cron across all your Claude Code windows.

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

```
/plugin marketplace add ./plugins
/plugin install reme@reme-marketplace
```

(Or point `/plugin marketplace add` at the GitHub repo + subpath once published.) Restart Claude Code, then
run `/mcp` to confirm the `reme` server and its tools are connected; the `reme-memory` skill can then
recall memory and report server health.

## Notes

- The plugin's MCP server URL lives in `plugins/reme/.mcp.json`. Keep it in sync with how you start
  ReMe (host/port). The Stop hook reads this same file to find the server (override with `REME_HOST`
  / `REME_PORT` env vars).
- The Stop hook needs `python3` on `PATH` and resolves transcripts under `~/.claude/projects`
  (override the base with `CLAUDE_CONFIG_DIR`). It logs to `plugins/reme/logs/auto_memory_hook.log`.
- The MCP tool-name prefix (`mcp__reme__…`) may include the server segment depending on your Claude
  Code version; the skill uses the `mcp__reme__*` wildcard so it works either way.
