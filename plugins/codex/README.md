# ReMe Plugin for Codex

Connect Codex to [ReMe](https://github.com/agentscope-ai/ReMe) — file-native long-term memory for AI agents. The plugin gives Codex **recall** (read long-term memory) and **records every session automatically** via a Stop hook.

## What you get

- **MCP tools** from the `reme` server: `search`, `traverse`, `daily_list`, `frontmatter_read`, `read`, `auto_memory_cc`, and more.
- **Stop hook** (`hooks/auto_memory.py`) — when a session ends it calls ReMe's server-side `auto_memory_cc` tool in a detached background process. Recording is fully automatic and asynchronous.
- **Skill** `reme-memory` — recall long-term memory before answering (semantic `search`, topological `traverse`, state `daily_list`/`frontmatter_read`, then `read` with citations), plus a server status check.

## Deployment Model

The plugin **connects to a shared HTTP MCP server you start once** — it does not spawn ReMe. One server means one set of background watchers / dream cron across all your Codex sessions.

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

## Install the Plugin

Copy the `plugins/codex/` directory to your Codex plugins location and install:

```bash
# Clone or copy the plugin directory
cp -r plugins/codex/ ~/.codex/plugins/reme/

# Configure MCP connection
# The .mcp.json is already bundled with the plugin
```

Restart Codex, then verify the `reme` server and its tools are connected. The `reme-memory` skill can then recall memory and report server health.

## Plugin Structure

```
plugins/codex/
├── .codex-plugin/
│   └── plugin.json          # Codex plugin manifest
├── hooks/
│   ├── hooks.json           # Stop hook configuration
│   └── auto_memory.py       # Session recording hook
├── skills/
│   └── reme-memory/
│       └── SKILL.md         # Memory recall skill
├── .mcp.json                # MCP server connection
└── README.md                # This file
```

## Notes

- The plugin's MCP server URL lives in `plugins/codex/reme/.mcp.json`. Keep it in sync with how you start ReMe (host/port). Override with `REME_HOST` / `REME_PORT` env vars.
- The Stop hook needs `python3` on `PATH`. It logs to `plugins/codex/hooks/auto_memory_hook.log`.
- The MCP tool-name prefix may include the server segment depending on your Codex version; the skill uses the wildcard pattern so it works either way.
- Recording is best-effort: if the server is down, the hook logs and gives up silently without blocking the session.

## Troubleshooting

### Tools not showing up

1. Verify ReMe is running: `curl -s http://127.0.0.1:2333/health_check`
2. Check the MCP server URL in `.mcp.json` matches your ReMe port
3. Restart Codex after installing the plugin

### Stop hook not recording

1. Check the log: `cat plugins/codex/hooks/auto_memory_hook.log`
2. Verify `python3` is on PATH
3. Ensure the ReMe server is running when you end a session

### Connection refused

The MCP server may not be running. Start it with:

```bash
reme start service.backend=mcp service.transport=streamable-http
```

Then retry.
