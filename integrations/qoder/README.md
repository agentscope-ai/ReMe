# ReMe memory for Qoder

[中文说明](./README_ZH.md)

Qoder supports installable plugins that bundle skills, MCP servers, hooks, and configuration. This
directory packages ReMe automatic recall, completed-turn recording, MCP memory tools, and the
`reme-memory` skill as one Qoder plugin.

## Requirements

- Qoder CLI 1.1.41 or newer with plugin support.
- Python 3.11 or newer available as `python3` in the environment Qoder uses for plugin hooks and
  MCP servers.
- FastMCP 3.1 or newer in that same Python environment. The `reme-ai[core]` install includes it.
- A separately running ReMe service exposing the MCP endpoint and the HTTP jobs `search` and
  `auto_memory`.

## 1. Prepare ReMe

Install ReMe and its Python dependencies from this repository. Use the Python interpreter that
Qoder can access as `python3`.

macOS or Linux:

```bash
cd /path/to/ReMe
python3 -m pip install -e ".[core]"
```

Windows PowerShell:

```powershell
cd E:\path\to\ReMe
python -m pip install -e ".[core]"
```

Automatic memory generation requires model credentials. Configure `LLM_API_KEY` and
`LLM_BASE_URL` as described in the [Quick Start](../../README.md#environment-variables).

Start ReMe in a separate terminal and leave it running:

```bash
reme start
```

The default HTTP service serves both the Job API and streamable HTTP MCP on
`http://127.0.0.1:2333`. The default Qoder configuration therefore uses:

- MCP: `http://127.0.0.1:2333/mcp`
- HTTP Job API: derived from `mcp_url` as `http://127.0.0.1:2333`

If the deployment intentionally uses separate ports, set `api_url` explicitly.

## 2. Install the plugin

The plugin is distributed from the repository-local marketplace. Keep the checkout available when
installing or updating it.

macOS or Linux:

```bash
cd /path/to/ReMe
reme_root="$(pwd)"
qodercli plugins validate "$reme_root/integrations/qoder/reme" --strict
qodercli plugins marketplace add "$reme_root/integrations/qoder"
qodercli plugins install reme@reme-marketplace --scope user
```

Windows PowerShell:

```powershell
cd E:\path\to\ReMe
$remeRoot = (Resolve-Path .).Path
qodercli plugins validate "$remeRoot\integrations\qoder\reme" --strict
qodercli plugins marketplace add "$remeRoot\integrations\qoder"
qodercli plugins install reme@reme-marketplace --scope user
```

Direct installation from the plugin directory is also supported:

```bash
qodercli plugins install /path/to/ReMe/integrations/qoder/reme --scope user
```

## 3. Configure

The plugin reads `~/.qoder/reme/config.json`, or `${QODER_CONFIG_DIR}/reme/config.json` when
`QODER_CONFIG_DIR` is set.

macOS or Linux:

```bash
mkdir -p "$HOME/.qoder/reme"
cp integrations/qoder/reme/config.example.json "$HOME/.qoder/reme/config.json"
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$HOME\.qoder\reme"
Copy-Item .\integrations\qoder\reme\config.example.json "$HOME\.qoder\reme\config.json"
```

The relevant defaults are:

```json
{
  "mcp_url": "http://127.0.0.1:2333/mcp",
  "api_url": "",
  "memory_interval": 1
}
```

An empty `api_url` is derived from `mcp_url` when that URL ends in `/mcp`. Set `api_url` only for a
split-port or reverse-proxy deployment. The shipped example sets `memory_interval` to `1` for
immediate per-turn writes; the code default is `5`.

| Option | Default | Meaning |
| --- | --- | --- |
| `mcp_url` | `http://127.0.0.1:2333/mcp` | ReMe MCP endpoint |
| `api_url` | `""` | Optional HTTP Job API override; derived from `mcp_url` when empty |
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

Restart Qoder after installation, or run `/plugins reload`. Reconnect MCP after changing
`mcp_url`; hook configuration changes apply on the next invocation.

## Lifecycle

`UserPromptSubmit` performs bounded recall and injects results as untrusted historical evidence.
At `Stop`, a synchronous hook captures the completed user/final-assistant pair into the local queue.
A parallel async hook submits captured batches to `POST {api_url}/auto_memory`. `SessionStart`
retries residual work and `SessionEnd` performs a short drain. Queue files are namespaced by service
endpoint and protected by per-session locks. Stable message IDs make repeated Stops idempotent.

Qoder transcript records use Claude-style `user` and `assistant` JSONL rows. Tool calls, tool
results, reasoning blocks, metadata messages, subagents, and injected `<reme-context>` are excluded.
The hook waits for the current final JSONL row instead of treating an older identical reply as the
new turn.

## Migration from manual hooks

Remove any manually installed ReMe `UserPromptSubmit`, `Stop`, `SessionStart`, and `SessionEnd`
hooks from `settings.json`, and remove a manually configured `mcpServers.reme` entry. Keeping both
the plugin and manual installation active captures the same turn twice. Unrelated Qoder, Orca, and
memory-gate hooks should remain untouched.

## Verify

First verify that Qoder discovered the plugin resources:

```bash
qodercli plugins list --json
```

The `reme` entry should be enabled and report one skill, one `reme` MCP server, and five hook
registrations. `qodercli mcp list` does not report servers bundled by plugins, so it is not a valid
connection test.

Open a fresh Qoder session and ask it to call the ReMe `health_check` and `version` MCP tools. Then
finish one memory-worthy turn with `memory_interval` set to `1` and verify all three signals:

1. a content-free `memory_saved` row in `~/.qoder/reme/hooks.log`;
2. the corresponding queue `.done` receipt under `~/.qoder/reme/queue/`;
3. a new `session/dialog/qoder-*.jsonl` file containing the user and final-assistant rows.

Open a second Qoder session and ask for the fact from the first session. Confirm that recall returns
it before considering the installation complete.

The service-owned `auto_dream` consolidation job remains enabled according to the ReMe service
configuration; the plugin does not start another scheduler.
