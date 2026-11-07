# ReMe memory for ZCode

[中文说明](./README_ZH.md)

ZCode supports plugins that bundle skills, MCP servers, and lifecycle hooks. This directory packages
ReMe automatic recall, completed-turn recording, MCP memory tools, and the `reme-memory` skill as
one ZCode plugin.

## Requirements

- A current ZCode desktop build with plugin and Hook support.
- Python 3.11 or newer as `python3` in the environment that launches ZCode, with FastMCP 3.1 or newer.
- A running ReMe service exposing the MCP endpoint and the HTTP jobs `search` and `auto_memory`.

The default split development setup uses:

- MCP: `http://127.0.0.1:2333/mcp`
- HTTP Job API: `http://127.0.0.1:2334`

## Install

1. Open **Plugin Marketplace → Add → Add Plugin Marketplace**.
2. Add this absolute directory: `E:\path\to\ReMe\integrations\zcode`.
3. Open **Personal → ReMe Memory → Install**.
4. Open a new ZCode task after installation. Existing tasks can retain the old component catalog.

The marketplace source is `integrations/zcode/marketplace.json`; the installable plugin source is
`integrations/zcode/reme`.

## Configure

The plugin reads `%USERPROFILE%\.zcode\reme\config.json`, or
`${ZCODE_HOME}/reme/config.json` when `ZCODE_HOME` is set:

```powershell
New-Item -ItemType Directory -Force "$HOME\.zcode\reme"
Copy-Item integrations\zcode\reme\config.example.json "$HOME\.zcode\reme\config.json"
```

`mcp_url` controls the MCP tool connection. `api_url` controls the HTTP Job API used by lifecycle
hooks. Set `memory_interval` to `1` for immediate per-turn writes.

| Option | Default | Meaning |
| --- | --- | --- |
| `auto_recall` | `true` | Search before a user prompt |
| `auto_memory` | `true` | Capture and submit completed turns |
| `recall_limit` | `5` | Search result limit |
| `recall_min_score` | `0` | Minimum recall score |
| `recall_timeout` | `5` | Foreground timeout, at most 10 seconds |
| `request_timeout` | `600` | Memory write timeout, at most 600 seconds |
| `memory_interval` | `5` | Completed turns per batch |
| `context_max_chars` | `8000` | Maximum recalled context |
| `timezone` | `Asia/Shanghai` | Daily batch timezone |

Unknown fields and invalid values fail validation. Disabling `auto_memory` preserves pending queue
files and stops automatic Capture and retries. Changing the endpoint never sends old queued
conversations to the new service.

## Lifecycle

ZCode supports `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PermissionRequest`,
`PostToolUse`, `PostToolUseFailure`, and `Stop`. Its `async` Hook field currently has no runtime
effect, so this plugin does not use it.

- `UserPromptSubmit` caches the user text by `session_id` and `turn_id`, then performs bounded recall.
  Recall results are injected as untrusted historical evidence.
- `Stop` combines the cached user text with ZCode's final assistant response and writes one durable
  queue file. It then starts a detached process to submit queued batches, allowing the Hook to return
  without waiting for model extraction.
- `SessionStart` starts a detached force-flush of residual batches from earlier sessions.

ZCode's Stop transcript contains only the final assistant message, so the plugin does not depend on
that temporary file. Tool events, subagents, reasoning, and injected `<reme-context>` are not used
as source conversation text.

Queue files are namespaced by service endpoint and protected by per-session locks. A `.done` receipt
prevents repeated Stops from re-submitting an acknowledged turn. A failed or timed-out write stays
queued for the next Stop or SessionStart.

## Migration from manual hooks

If ReMe was previously added directly to `%USERPROFILE%\.zcode\cli\config.json`, remove the ReMe
entries from `hooks.events` and the manual `reme` MCP server after installing this plugin. Keep ZCode,
Orca, and other unrelated Hooks. Running both installations captures the same turn twice.

## Verify

Set `memory_interval` to `1`, open a new ZCode task, and ask it to remember a synthetic fact. Success
requires:

1. a `prompt_cached` row followed by `memory_saved` in `%USERPROFILE%\.zcode\reme\hooks.log`;
2. the corresponding queue `.done` receipt;
3. one new `session/dialog/zcode-*.jsonl` file with a user and final-assistant row;
4. a later task that recalls the saved fact with a source path.

Check **Settings → Plugins** for the installed plugin and **Settings → MCP** for the `reme` server.
If the plugin is enabled but its components are absent, start a new task before changing files.

`auto_dream` remains owned by the ReMe service; the plugin does not start another scheduler.
