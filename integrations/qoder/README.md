# ReMe memory for Qoder

[中文说明](./README_ZH.md)

Qoder supports installable plugins that bundle skills, MCP servers, hooks, and configuration. This
directory packages ReMe automatic recall, completed-turn recording, MCP memory tools, and the
`reme-memory` skill as one Qoder plugin.

## Requirements

- Qoder CLI 1.1.41 or newer with plugin support.
- Python 3.11 or newer as `python3` in the Qoder environment, with FastMCP 3.1 or newer.
- A running ReMe service exposing the MCP endpoint and the HTTP jobs `search` and `auto_memory`.

The default split development setup uses:

- MCP: `http://127.0.0.1:2333/mcp`
- HTTP Job API: `http://127.0.0.1:2334`

## Install

From this checkout:

```powershell
qodercli plugins marketplace add E:\path\to\ReMe\integrations\qoder
qodercli plugins install reme@reme-marketplace --scope user
```

Direct local installation is also supported:

```powershell
qodercli plugins install E:\path\to\ReMe\integrations\qoder\reme --scope user
```

Validate the package before installing:

```powershell
qodercli plugins validate E:\path\to\ReMe\integrations\qoder\reme --strict
```

## Configure

The plugin reads `%USERPROFILE%\.qoder\reme\config.json`, or
`${QODER_CONFIG_DIR}/reme/config.json` when `QODER_CONFIG_DIR` is set:

```powershell
New-Item -ItemType Directory -Force "$HOME\.qoder\reme"
Copy-Item integrations\qoder\reme\config.example.json "$HOME\.qoder\reme\config.json"
```

`mcp_url` controls the MCP tool connection. `api_url` controls the HTTP job API used by lifecycle
hooks. The two must not be conflated when the ReMe deployment uses separate ports.

Set `memory_interval` to `1` for immediate per-turn writes. The remaining keys and limits match the
Claude Code integration:

| Option | Default | Meaning |
| --- | --- | --- |
| `auto_recall` | `true` | Search before a user prompt |
| `auto_memory` | `true` | Capture and submit completed turns |
| `recall_limit` | `5` | Search result limit |
| `recall_min_score` | `0` | Minimum recall score |
| `recall_timeout` | `5` | Foreground timeout, at most 10 seconds |
| `request_timeout` | `600` | Memory write timeout, at most 600 seconds |
| `memory_interval` | `5` | Completed turns per batch |
| `shutdown_timeout` | `2` | Best-effort exit drain, at most 2 seconds |
| `context_max_chars` | `8000` | Maximum recalled context |
| `timezone` | `Asia/Shanghai` | Daily batch timezone |

## Lifecycle

`UserPromptSubmit` performs bounded recall and injects results as untrusted historical evidence.
At `Stop`, a synchronous hook captures the completed user/final-assistant pair into the local queue.
A parallel async hook submits captured batches to `POST {api_url}/auto_memory`. `SessionStart`
retries residual work and `SessionEnd` performs a short drain. Queue files are namespaced by service
endpoint and protected by per-session locks. Stable message IDs make repeated Stops idempotent.

Qoder transcript records use Claude-style `user` and `assistant` JSONL rows. Tool calls, tool
results, reasoning blocks, metadata messages, subagents, and injected `<reme-context>` are excluded.
The hook accepts complete final JSONL rows while Qoder finishes materializing the transcript.

## Migration from manual hooks

Remove any manually installed ReMe `UserPromptSubmit`, `Stop`, `SessionStart`, and `SessionEnd`
hooks from `settings.json`, and remove a manually configured `mcpServers.reme` entry. Keeping both
the plugin and manual installation active captures the same turn twice. Unrelated Qoder, Orca, and
memory-gate hooks should remain untouched.

## Verify

```powershell
qodercli plugins list
qodercli mcp list
```

Open a new Qoder conversation through CodexHost or directly. Success requires:

1. a `memory_saved` row in `%USERPROFILE%\.qoder\reme\hooks.log`;
2. the corresponding queue `.done` receipt;
3. one new `session/dialog/qoder-*.jsonl` file with a user and final-assistant row.

The service-owned `auto_dream` consolidation job remains enabled according to the ReMe service
configuration; the plugin does not start another scheduler.
