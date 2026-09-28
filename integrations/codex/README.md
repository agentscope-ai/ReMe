# ReMe memory for Codex

[中文说明](./README_ZH.md)

This repository-local plugin gives Codex automatic recall before user prompts, automatic
recording of completed turns, and explicit ReMe MCP tools. Durable memory stays in a user-owned
ReMe workspace. Its plugin files and installation guide are maintained under `integrations/codex/` and can be
installed directly from this checkout.

```text
UserPromptSubmit → search → untrusted memory context → agent reply
Stop → completed user/assistant text → local retry queue → auto_memory → daily Markdown
```

## Requirements

- Python 3.11+ available as `python3` in the host's environment; the plugin uses only the standard library.
- Codex CLI 0.145.0 or a compatible client supporting local plugins, command hooks, and hook trust review.
- A running ReMe HTTP service exposing `search`, `auto_memory`, and `health_check`, with MCP enabled.
- A working model configuration on the ReMe server for memory extraction. Default BM25 recall does not need embeddings.

## Start ReMe

```bash
pip install "reme-ai[core]"
reme start workspace_dir=/absolute/path/to/workspace service.backend=http
```

The default endpoint is `http://127.0.0.1:2333`, with MCP at `/mcp`. Keep the unauthenticated HTTP
service on loopback or behind a protected proxy. Different workspaces isolate different memory
scopes; sharing one workspace between hosts is intentional cross-agent recall.

## Install the plugin

Register this repository-local marketplace, then install its plugin:

```bash
codex plugin marketplace add /absolute/path/to/ReMe/integrations/codex
codex plugin add reme@reme-codex
```

In the Codex app, the same local marketplace can be selected in the plugin directory. Start a new
thread after installing. Open `/hooks` in the CLI to review and trust the bundled hooks; enabling
a plugin alone does not trust its hooks. Open `/mcp` to verify ReMe. For updates, refresh the local
marketplace and reinstall the plugin, then review any changed hook definitions.

Automatic recording sends completed user/assistant text to the configured ReMe service.
Set `auto_memory` to `false` for recall-only use.

## Configuration

Optional settings live at `${CODEX_HOME:-$HOME/.codex}/reme/config.json`. Start from the bundled example:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/reme"
cp integrations/codex/plugins/reme/config.example.json "${CODEX_HOME:-$HOME/.codex}/reme/config.json"
```

| Option | Default | Meaning |
| --- | --- | --- |
| `auto_recall` | `true` | Search before a user prompt |
| `auto_memory` | `true` | Queue and submit completed turns; false also disables retries |
| `recall_limit` | `5` | Search result limit |
| `recall_min_score` | `0` | Minimum recall score |
| `recall_timeout` | `5` | Foreground HTTP timeout in seconds, at most 10 |
| `request_timeout` | `600` | Memory-write HTTP timeout in seconds, at most 600 |
| `memory_interval` | `5` | Completed turns per batch; use 1 for immediate per-turn writes |
| `shutdown_timeout` | `2` | Total best-effort exit drain budget in seconds, at most 2 |
| `context_max_chars` | `8000` | Maximum recalled payload characters |
| `timezone` | `Asia/Shanghai` | Daily batch timezone; match the ReMe workspace |

Unknown fields and invalid values fail validation. Changes apply on the next hook invocation.
The plugin's `.mcp.json` is the single endpoint setting: both MCP and hooks use its `reme.url`,
which must end in `/mcp`. For a different port or protected proxy, edit that source file **before
installation**, then refresh/reinstall the plugin. Do not edit a versioned installation cache.
There is no separate `REME_HOST`/`REME_PORT` hook override.

## Lifecycle and failure behavior

- `UserPromptSubmit` calls `search` synchronously with a short timeout. Results are bounded and
  wrapped in `<reme-context>` as untrusted historical evidence. A failure does not reject the prompt.
- Synchronous `Stop` hooks extract only the completed user/final-assistant text from the
  hook's local transcript, then submit `auto_memory` batches through the JSON Job API. Tool output,
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
  lose that turn. A lost HTTP acknowledgement can cause a retry; stable message IDs prevent duplicate
  source messages, but memory extraction is **at least once**, not exactly once. Short residual
  batches survive for the next session when shutdown has insufficient time.
- Queue files contain source conversation text; retain them until delivery completes. Logs in
  `reme/hooks.log` contain event/error types only. Disabling `auto_memory` retains pending files.
- Daily `auto_dream` consolidation remains owned by the ReMe service. Unlike OpenClaw's long-lived
  Gateway adapter, these short-lived hook processes do not run their own scheduler.

## Verify and troubleshoot

1. Call `reme health_check` against the configured service and check `/mcp` in Codex.
2. For a quick test, set `memory_interval` to `1`, then start a new conversation and ask the agent to
   remember a synthetic fact, such as “Project Juniper reviews are on Thursday.”
3. Wait for the write and inspect `reme/hooks.log` for `memory_saved` and the ReMe workspace's `daily/`.
4. Start a separate conversation and ask for that fact. Confirm recall is from ReMe, with source paths.

The `reme-memory` skill supports explicit search/read and health checks. Missing MCP tools can also
mean a disabled plugin or restricted `service.jobs`; it does not prove the server is stopped.
If hooks are unavailable in the installed host version, MCP and the skill remain the manual fallback.

Capture currently reads Codex rollout JSONL `event_msg` records (`user_message` and final `agent_message`). The hook requires
`transcript_path` and `last_assistant_message`. Missing or unsupported transcripts are skipped rather
than guessed. Codex transcript formats are not a stable public interface. Check `capture_skipped` or
`hook_failed` events when recall works but recording does not.

In Codex 0.145.0, an unattended `codex exec` MCP call can return `user cancelled MCP tool call`
when approval cannot be collected. Verify explicit tools in an interactive session and approve the
specific call. Automatic hooks use the Job API and do not depend on MCP tool approval.

## Source and validation

Run `pytest tests/unit/test_coding_agent_plugins.py -v` from the ReMe checkout. Tests isolate host
state and mock service calls. Native hook behavior is described in the [Codex hook reference](https://learn.chatgpt.com/docs/hooks).
ReMe is developed at [agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe) under Apache-2.0.
