# ReMe memory for Claude Code

[中文说明](./README_ZH.md)

Start a ReMe service, then install the ReMe plugin in Claude Code. This is the supported integration
for this host. The plugin bundles automatic recall, completed-turn recording, MCP tools, and the
`reme-memory` skill in one installation. ReMe manages durable memory and consolidation in your workspace.

The plugin and its marketplace are maintained in `integrations/claude_code/` and installed from
this checkout. Keep the ReMe service running while using the plugin.

```text
Start ReMe → Install the ReMe plugin → Use memory in Claude Code
```

The host integration gives existing conversations access to ReMe's long-term memory. The [Proactive Runtime](https://reme.agentscope.io/en/proactive_runtime) can also use this agent as an execution backend for its own tasks. Those runs use task-owned sessions.

## Requirements

- Python 3.11+ available as `python3` in the host's environment. Optional stdio mode also needs `fastmcp>=4.1.4,<5`.
- A current Claude Code version supporting plugin command hooks, `async`, and Stop `last_assistant_message`.
- A running ReMe HTTP service exposing `search`, `auto_memory`, and `health_check`, with MCP enabled.
- A working model configuration on the ReMe server for memory extraction. Default BM25 recall does not need embeddings.

## 1. Start ReMe

```bash
pip install "reme-ai[core]"
reme start config=default workspace_dir=/absolute/path/to/workspace
```

The default endpoint is `http://127.0.0.1:2333`, with MCP at `/mcp`. Keep the unauthenticated HTTP
service on loopback or behind a protected proxy. Different workspaces isolate different memory
scopes; sharing one workspace between hosts is intentional cross-agent recall.

## 2. Install the plugin

In another terminal, prepare a user-owned installation source. HTTP is the default and connects
directly to the shared ReMe service without a persistent Python MCP proxy:

```bash
python3 /absolute/path/to/ReMe/integrations/configure_mcp.py claude-code
```

The command prints the absolute marketplace path. In Claude Code, use that path:

```text
/plugin marketplace add /absolute/path/to/claude-config/reme/plugin-marketplace
/plugin install reme@reme-marketplace
```

Restart Claude Code or reload the installed plugin, then open a new conversation. The installation
loads the bundled hooks, MCP connection, and skill together. Reinstall or reload the plugin after updates.
When upgrading code, regenerate the source before reinstalling the plugin. To switch to the
existing Python bridge, regenerate with `--transport stdio` and reinstall the plugin. Both modes use
the same Hooks, skill, settings, and retry queue. Neither requires changing `default.yaml` or `proactive.yaml`.

If a marketplace with this name was already registered from the checkout, remove the old plugin
and `/plugin marketplace remove reme-marketplace` before registering the generated source.
The user settings, retry queue, and logs outside the installation source are preserved.

## 3. Use memory in Claude Code

In a new conversation, ask “Check whether ReMe is available.” The installed plugin's tools check
the running service; `/mcp` shows the plugin's connection status. Then use Claude Code normally:
the plugin recalls relevant memory before prompts and records completed turns in batches of five.
For an explicit query, ask “What did we decide about Project Juniper? Cite the memory sources.”

Automatic recording sends completed user/assistant text to the configured ReMe service.
Set `auto_memory` to `false` for recall-only use.

## Optional configuration

The default service address works immediately after plugin installation. Create a configuration
file only when you want to change the defaults below.

Edit the user configuration file directly to change addresses and behavior. Plugin upgrades preserve
this file, and only overrides need to be saved.

Settings live at `${CLAUDE_CONFIG_DIR:-$HOME/.claude}/reme/config.json`. For a new file, copy the bundled example; edit an existing file in place:

```bash
mkdir -p "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/reme"
cp integrations/claude_code/reme/config.example.json "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/reme/config.json"
```

| Option | Default | Meaning |
| --- | --- | --- |
| `mcp_url` | `http://127.0.0.1:2333/mcp` | MCP address; removing the trailing `/mcp` gives the default HTTP base for hooks |
| `api_url` | Empty string | Optional HTTP Job API base; required for custom MCP paths |
| `auto_recall` | `true` | Search before a user prompt |
| `auto_memory` | `true` | Queue and submit completed turns; false also disables retries |
| `recall_limit` | `5` | Search result limit |
| `recall_min_score` | `0` | Minimum recall score |
| `recall_timeout` | `5` | Foreground HTTP timeout in seconds, at most 10 |
| `request_timeout` | `600` | Memory-write and MCP request timeout in seconds, at most 600 |
| `memory_interval` | `5` | Completed turns per batch; use 1 for immediate per-turn writes |
| `shutdown_timeout` | `2` | Total best-effort exit drain budget in seconds, at most 2 |
| `context_max_chars` | `8000` | Maximum recalled payload characters |
| `timezone` | `Asia/Shanghai` | Daily batch timezone; match the ReMe workspace |

For example, use another port and disable automatic recording:

```json
{
  "mcp_url": "http://127.0.0.1:2444/mcp",
  "auto_memory": false,
  "auto_recall": true,
  "memory_interval": 1
}
```

The configuration command reads the MCP address from this file. Hooks pick up edits on their next
invocation. In HTTP mode, rerun the command after an address change, reinstall the
plugin from the generated source, and restart the host. Alternatively, use
`python3 /absolute/path/to/ReMe/integrations/configure_mcp.py claude-code --mcp-url http://127.0.0.1:2444/mcp`
to update the JSON and generated connection together. In stdio mode, reconnect MCP after editing
the address or `request_timeout`. That timeout configures the stdio bridge; native HTTP tool-call
timeouts are controlled by Claude Code. No repository source or installed cache is edited.
For a custom MCP path such as `https://memory.example.com/tools`, also set `api_url` to the HTTP Job
API base, for example `https://memory.example.com/reme`.
The configuration command also accepts `--api-url` for this case. If the JSON already contains an
explicit `api_url`, changing the address with `--mcp-url` requires `--api-url` too, so Hooks cannot
silently keep sending conversations to an old HTTP service. If an HTTP installation no longer
matches the saved MCP address, Hooks skip work with `hook_failed` until it is regenerated and reinstalled.

Unknown fields and invalid values fail validation; hooks skip work and log `hook_failed` instead of
falling back to another service. Disabling `auto_memory` stops automatic capture and retries while
preserving pending batches; explicit MCP tools remain available. Changing the service address never
sends old pending conversations to the new service. Remove a field to restore its default.

Optional stdio mode uses a host-managed process to forward MCP. Start the host with a `python3`
environment containing FastMCP. If ReMe runs on another machine or in another Python environment,
run `python3 -m pip install "fastmcp>=4.1.4,<5"` in the host environment. Hooks use only the standard library.

## Lifecycle and failure behavior

- `UserPromptSubmit` calls `search` synchronously with a short timeout. Results are bounded and
  wrapped in `<reme-context>` as untrusted historical evidence. A failure does not reject the prompt.
- A synchronous `Stop` hook captures and queues the completed user/final-assistant text before the
  next turn can start. A separate native asynchronous `Stop` hook submits those captured batches
  through the JSON Job API without rereading the transcript. Tool output, reasoning, and injected
  ReMe context are excluded. The server does not need access to host files.
- Session and message IDs are stable and host-scoped. Repeated Stops are deduplicated. Per-session
  process locks serialize writes; batches never mix dates. Subagent events and hook continuations
  are excluded. No extra background daemon, package download, or ReMe process is started.
- Queues and acknowledgement receipts live under the host's `reme/queue/`, separate from the
  plugin cache and namespaced by endpoint. A failed or timed-out write stays queued. `SessionStart`
  retries pending work in the background; `SessionEnd` attempts a short, synchronous drain.
- Shutdown and delivery are best-effort. A host killed before the Stop hook captures a turn can
  lose that turn. A lost HTTP acknowledgement can cause a retry; stable message IDs prevent duplicate
  source messages, but memory extraction is **at least once**, not exactly once. Short residual
  batches survive for the next session when shutdown has insufficient time.
- Queue files contain source conversation text; retain them until delivery completes. Logs in
  `reme/hooks.log` contain event/error types only. Disabling `auto_memory` retains pending files.
- Daily `auto_dream` consolidation remains owned by the ReMe service. Unlike OpenClaw's long-lived
  Gateway adapter, these short-lived hook processes do not run their own scheduler.

## Verify and troubleshoot

1. In Claude Code, ask the installed ReMe plugin to check the connection; confirm its status in `/mcp`.
2. For a quick test, set `memory_interval` to `1`, then start a new conversation and ask the agent to
   remember a synthetic fact, such as “Project Juniper reviews are on Thursday.”
3. Wait for the write and inspect `reme/hooks.log` for `memory_saved` and the ReMe workspace's `daily/`.
4. Start a separate conversation and ask for that fact. Confirm recall is from ReMe, with source paths.

The installed plugin provides both automatic memory and explicit search/read and health checks.
If its tools are missing, check that the plugin is installed and enabled, the service is running,
and `service.jobs` exposes the required jobs. If its hooks do not load, use a host version supporting
the plugin requirements and reinstall or reload the plugin.

Capture currently reads Claude JSONL user/assistant records with UUIDs. The hook requires
`transcript_path` and `last_assistant_message`. Missing or unsupported transcripts are skipped rather
than guessed. Check `capture_skipped` or
`hook_failed` events when recall works but recording does not.

## Migration from the previous Claude Code plugin

Version 0.2 replaces the detached Stop-only `auto_memory_cc` client with automatic recall and
client-side completed-turn capture via `auto_memory`. The existing server job `auto_memory_cc`
remains available. Existing `session/claude_code/` transcripts and notes are preserved; new source
messages use ReMe's standard `{session_dir}/dialog/claude-code-<hash>.jsonl` files and namespaced IDs.
The plugin does not replay older conversations. Remove any manually installed duplicate Stop hook.
Logs move from the plugin's `logs/` directory to the host's `reme/hooks.log`.

For end-to-end checks, keep the session open until `memory_saved` appears. A short `claude -p`
process can exit before its asynchronous hook receives the write acknowledgement; the next session
retries captured work. A Markdown note alone does not prove that the queue was acknowledged.

## Source and validation

Run `pytest tests/unit/test_coding_agent_plugins.py -v` from the ReMe checkout. Tests isolate host
state and mock service calls. Native hook behavior is described in the [Claude Code hook reference](https://code.claude.com/docs/en/hooks).
ReMe is developed at [agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe) under Apache-2.0.
