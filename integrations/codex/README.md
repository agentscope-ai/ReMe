# ReMe memory for Codex

[中文说明](./README_ZH.md)

Start a ReMe service, then install the ReMe plugin in Codex. This is the supported integration
for this host. The plugin bundles automatic recall, completed-turn recording, MCP tools, and the
`reme-memory` skill in one installation. ReMe manages durable memory and consolidation in your workspace.

The plugin and its marketplace are maintained in `integrations/codex/` and installed from
this checkout. Keep the ReMe service running while using the plugin.

```text
Start ReMe → Install the ReMe plugin → Use memory in Codex
```

## Requirements

- Python 3.11+ available as `python3` in the host's environment, with FastMCP >= 3.4.2 (included with ReMe).
- Codex CLI 0.145.0 or a compatible client supporting local plugins, command hooks, and hook trust review.
- A graphical host supporting the `openai/settings` MCP extension for the native settings UI. CLI 0.145.0 supports hooks and tools but does not render this UI.
- A running ReMe HTTP service exposing `search`, `auto_memory`, and `health_check`, with MCP enabled.
- A working model configuration on the ReMe server for memory extraction. Default BM25 recall does not need embeddings.

## 1. Start ReMe

```bash
pip install "reme-ai[core]"
reme start workspace_dir=/absolute/path/to/workspace service.backend=http
```

The default endpoint is `http://127.0.0.1:2333`, with MCP at `/mcp`. Keep the unauthenticated HTTP
service on loopback or behind a protected proxy. Different workspaces isolate different memory
scopes; sharing one workspace between hosts is intentional cross-agent recall.

## 2. Install the plugin

Register this repository-local marketplace, then install its plugin:

```bash
codex plugin marketplace add /absolute/path/to/ReMe/integrations/codex
codex plugin add reme@reme-codex
```

Start Codex and review and trust the installed plugin's hooks in `/hooks`, then open a new conversation.
The installation loads the bundled hooks, MCP connection, and skill together. After plugin updates,
reinstall it and review any changed hook definitions.

## 3. Use memory in Codex

In a new conversation, ask “Check whether ReMe is available.” The installed plugin's tools check
the running service; `/mcp` shows the plugin's connection status. Then use Codex normally:
the plugin recalls relevant memory before prompts and records completed turns in batches of five.
For an explicit query, ask “What did we decide about Project Juniper? Cite the memory sources.”

Automatic recording sends completed user/assistant text to the configured ReMe service.
Set `auto_memory` to `false` for recall-only use.

## Configure through MCP settings

Open the installed ReMe MCP server's settings in a host that supports `openai/settings`.
The native form exposes the service address, automatic recall and memory switches, batch size,
recall limits, timeouts, and timezone. Save changes, then use **Check connection** to call
`health_check` on the saved address without sending conversation content. The settings remain
available when ReMe is offline, so a wrong address can be corrected there.

Settings are atomically saved to `${CODEX_HOME:-$HOME/.codex}/reme/config.json`, outside the plugin
cache. The next MCP tool call and hook invocation use the new values without a reconnect or reinstall.
An in-flight call keeps its original endpoint and settings. Disabling `auto_memory` pauses capture
and queued retries; explicit memory tools remain available.

| Option | Default | Meaning |
| --- | --- | --- |
| `mcp_url` | `http://127.0.0.1:2333/mcp` | MCP address for both explicit tools and automatic hooks; custom paths are supported |
| `auto_recall` | `true` | Search before a user prompt |
| `auto_memory` | `true` | Queue and submit completed turns; false also disables retries |
| `recall_limit` | `5` | Search result limit |
| `recall_min_score` | `0` | Minimum recall score |
| `recall_timeout` | `5` | Foreground MCP timeout in seconds, at most 10 |
| `request_timeout` | `600` | Memory-write and MCP request timeout in seconds, at most 600 |
| `memory_interval` | `5` | Completed turns per batch; use 1 for immediate per-turn writes |
| `shutdown_timeout` | `2` | Total best-effort exit drain budget in seconds, at most 2 |
| `context_max_chars` | `8000` | Maximum recalled payload characters |
| `timezone` | `Asia/Shanghai` | Daily batch timezone; match the ReMe workspace |

The JSON file remains user-owned and can also be edited directly; only overrides are required.
See [config.example.json](plugins/reme/config.example.json) for a complete example. Remove a field
from the file to restore its default. Unknown fields and invalid values fail validation; hooks skip
work and log `hook_failed` instead of falling back to another service. UI updates reject invalid
values without replacing the saved file.

Existing configurations keep working. The deprecated `api_url` field is accepted on read but is
no longer used and is removed on the next UI save. All operations use `mcp_url`. Standard `/mcp`
endpoints retain existing queue and receipt identities. Pending queues from a custom legacy
`api_url` remain untouched: they are not automatically retargeted. Address changes never forward
old pending conversations to the new service; switch back to the original endpoint to retry its queue.

Both the host-managed MCP server and short-lived hooks require FastMCP in the host's `python3`
environment. If ReMe runs elsewhere, install it with `python3 -m pip install "fastmcp>=3.4.2"`.
The plugin does not launch ReMe. Its native form follows the pinned
[OpenAI MCP Extensions settings contract](https://github.com/openai/mcp-extensions/blob/node-v0.1.0/docs/spec.md#structured-settings).
Clients without that extension can still use tools and hooks, but have no native settings form.

## Lifecycle and failure behavior

- `UserPromptSubmit` calls `search` synchronously with a short timeout. Results are bounded and
  wrapped in `<reme-context>` as untrusted historical evidence. A failure does not reject the prompt.
- Synchronous `Stop` hooks extract only the completed user/final-assistant text from the
  hook's local transcript, then submit `auto_memory` batches through MCP. Tool output,
  reasoning, and injected ReMe context are excluded. The server does not need access to host files.
- Session and message IDs are stable and host-scoped. Repeated Stops are deduplicated. Per-session
  process locks serialize writes; batches never mix dates. Subagent events and hook continuations
  are excluded. No extra background daemon, package download, or ReMe process is started.
- Queues and acknowledgement receipts live under the host's `reme/queue/`, separate from the
  plugin cache and namespaced by endpoint. A failed or timed-out write stays queued. `SessionStart`
  retries pending work synchronously; `SessionEnd` attempts a short drain.
- Codex 0.145.0 skips hooks marked `async`, so this adapter deliberately uses synchronous hooks.
  A due memory batch or a startup retry can delay completion by up to `request_timeout`.
  Claude Code uses its own native asynchronous adapter.
- Shutdown and delivery are best-effort. A host killed before the Stop hook captures a turn can
  lose that turn. A lost MCP acknowledgement can cause a retry; stable message IDs prevent duplicate
  source messages, but memory extraction is **at least once**, not exactly once. Short residual
  batches survive for the next session when shutdown has insufficient time.
- Queue files contain source conversation text; retain them until delivery completes. Logs in
  `reme/hooks.log` contain event/error types only. Disabling `auto_memory` retains pending files.
- Daily `auto_dream` consolidation remains owned by the ReMe service. Unlike OpenClaw's long-lived
  Gateway adapter, these short-lived hook processes do not run their own scheduler.

## Verify and troubleshoot

1. In Codex, ask the installed ReMe plugin to check the connection; confirm its status in `/mcp`.
2. For a quick test, set `memory_interval` to `1`, then start a new conversation and ask the agent to
   remember a synthetic fact, such as “Project Juniper reviews are on Thursday.”
3. Wait for the write and inspect `reme/hooks.log` for `memory_saved` and the ReMe workspace's `daily/`.
4. Start a separate conversation and ask for that fact. Confirm recall is from ReMe, with source paths.

The installed plugin provides both automatic memory and explicit search/read and health checks.
If its tools are missing, check that the plugin is installed and enabled, the service is running,
and `service.jobs` exposes the required jobs. If its hooks do not load, use a host version supporting
the plugin requirements and reinstall or reload the plugin.

Capture currently reads Codex rollout JSONL `event_msg` records (`user_message` and final `agent_message`). The hook requires
`transcript_path` and `last_assistant_message`. Missing or unsupported transcripts are skipped rather
than guessed. Codex transcript formats are not a stable public interface. Check `capture_skipped` or
`hook_failed` events when recall works but recording does not.

In Codex 0.145.0, an unattended `codex exec` MCP call can return `user cancelled MCP tool call`
when approval cannot be collected. Verify explicit tools in an interactive session and approve the
specific call. Trusted automatic hooks act as MCP clients themselves; they do not request per-call model tool approval.

## Source and validation

Run `pytest tests/unit/test_coding_agent_plugins.py tests/unit/test_codex_plugin_mcp.py -v` from the ReMe checkout. Tests isolate host
state and mock service calls. Native hook behavior is described in the [Codex hook reference](https://learn.chatgpt.com/docs/hooks).
ReMe is developed at [agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe) under Apache-2.0.
