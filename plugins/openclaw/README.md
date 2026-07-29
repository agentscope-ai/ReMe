# ReMe Plugin for OpenClaw

Connect OpenClaw to [ReMe](https://github.com/agentscope-ai/ReMe) — file-native long-term memory
for AI agents. The plugin gives OpenClaw **recall** (read long-term memory) and **records every
session automatically** via a Stop hook. Consolidation of daily notes into long-term `digest/`
knowledge runs server-side in ReMe.

## What you get

- **MCP tools** from the `reme` server: `search`, `traverse`, `daily_list`, `frontmatter_read`,
  `read`, `auto_memory_cc`, and more.
- **Stop hook** (`hooks/auto_memory.py`) — when a session ends it calls ReMe's server-side
  `auto_memory_cc` tool in a detached background process, passing **only the session id**. The
  server resolves that session's transcript on disk and records the durable facts into today's
  daily note. Recording is fully automatic and asynchronous — the agent never records by hand, and
  stopping is never delayed. Best-effort: if the server is down it logs and gives up silently.
- **Skill** `reme-memory` — recall long-term memory before answering (semantic `search`,
  topological `traverse`, state `daily_list`/`frontmatter_read`, then `read` with citations),
  plus a server status check. Recording is handled silently by the Stop hook.

## Deployment model

The plugin **connects to a shared HTTP MCP server you start once** — it does not spawn ReMe. One
server means one set of background watchers / dream cron across all your OpenClaw windows.

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

OpenClaw manages MCP servers natively via its `openclaw mcp` command. The plugin ships a bundled
`.mcp.json` and a Stop-hook manifest for auto-recording.

1. Copy or symlink the plugin directory to your OpenClaw plugins location:

   ```bash
   cp -r plugins/openclaw/ ~/.openclaw/plugins/reme
   ```

2. Configure the MCP connection (OpenClaw natively handles MCP registration):

   ```bash
   # Option A — point OpenClaw at the bundled MCP config:
   openclaw mcp set reme file://~/.openclaw/plugins/reme/reme/.mcp.json

   # Option B — register the URL directly:
   openclaw mcp add reme http --url http://127.0.0.1:2333/mcp
   ```

3. Restart OpenClaw. Verify the `reme` server and its tools are connected, then the
   `reme-memory` skill can recall memory and report server health.

## Plugin structure

```
plugins/openclaw/
├── .openclaw-plugin/
│   └── plugin.json              # OpenClaw plugin manifest
└── reme/
    ├── hooks/
    │   ├── hooks.json           # Stop hook configuration
    │   └── auto_memory.py       # Fire-and-forget session recording hook
    ├── skills/
    │   └── reme-memory/
    │       └── SKILL.md         # Memory recall skill
    └── .mcp.json                # MCP server connection config
```

## Notes

- The plugin's MCP server URL lives in `plugins/openclaw/reme/.mcp.json`. Keep it in sync with how
  you start ReMe (host/port). The Stop hook reads this same file to find the server (override with
  `REME_HOST` / `REME_PORT` env vars).
- The Stop hook needs `python3` on `PATH`. It logs to
  `plugins/openclaw/reme/logs/auto_memory_hook.log`.
- Recording is best-effort: if the server is down, the hook logs and gives up silently without
  blocking the session.

## Troubleshooting

### Tools not showing up

1. Verify ReMe is running: `curl -s http://127.0.0.1:2333/health_check`
2. Check the MCP URL matches your ReMe port: `openclaw mcp list`
3. Re-register the MCP server and restart OpenClaw

### Stop hook not recording

1. Check the log: `cat ~/.openclaw/plugins/reme/reme/logs/auto_memory_hook.log`
2. Verify `python3` is on `PATH`
3. Ensure the ReMe server is running when you end a session

### Connection refused

The MCP server may not be running. Start it with:

```bash
reme start service.backend=mcp service.transport=streamable-http
```

Then retry.
