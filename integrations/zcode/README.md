# ReMe memory for ZCode

[中文说明](./README_ZH.md)

ZCode supports plugins that bundle skills, MCP servers, and lifecycle hooks. This directory packages
ReMe automatic recall, completed-turn recording, MCP memory tools, and the `reme-memory` skill as
one ZCode plugin.

## Requirements

- A ZCode desktop build with plugin and Hook support. The plugin accepts both payload spellings
  (`session_id`/`sessionId`, `turnId`/`turn_id`) and reads the final assistant text from
  `last_assistant_message` or `responseText`.
- Python 3.11 or newer in the environment that launches ZCode, with FastMCP 3.1 or newer.
- A running ReMe service exposing the MCP endpoint and the HTTP jobs `search` and `auto_memory`.

The plugin is supported on Windows, macOS and Linux. The Hook script uses only the Python standard
library; the only OS-specific step is detaching the delivery process (`creationflags` on Windows,
`start_new_session` elsewhere). Commands below are given for both Windows PowerShell and
macOS/Linux.

## 1. Prepare ReMe

Install ReMe and its Python dependencies with the interpreter ZCode can reach:

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

Automatic memory generation requires model credentials. Configure `LLM_API_KEY` and `LLM_BASE_URL`
as described in the [Quick Start](../../README.md#environment-variables).

Start ReMe in a separate terminal and leave it running. Pass an explicit workspace so memories do
not depend on the directory you happen to launch from:

macOS or Linux:

```bash
reme start workspace_dir="$HOME/reme-workspace"
```

Windows PowerShell:

```powershell
reme start workspace_dir="$HOME\reme-workspace"
```

The default `reme start` deployment serves both the Job API and streamable HTTP MCP on
`http://127.0.0.1:2333`. A split-port deployment is an explicit override: start the HTTP Job API on
another port and set `api_url` accordingly.

## 2. Install the plugin

1. Open **Plugin Marketplace → Add → Add Plugin Marketplace**.
2. Add this absolute directory: `E:\path\to\ReMe\integrations\zcode` (or
   `/path/to/ReMe/integrations/zcode`).
3. Open **Personal → ReMe Memory → Install**.
4. Open a new ZCode task after installation. Existing tasks can retain the old component catalog.

The marketplace source is `integrations/zcode/marketplace.json`; the installable plugin source is
`integrations/zcode/reme`.

## 3. Configure

The plugin reads `%USERPROFILE%\.zcode\reme\config.json`, or `${ZCODE_HOME}/reme/config.json` when
`ZCODE_HOME` is set.

macOS or Linux:

```bash
mkdir -p "$HOME/.zcode/reme"
cp integrations/zcode/reme/config.example.json "$HOME/.zcode/reme/config.json"
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$HOME\.zcode\reme"
Copy-Item integrations\zcode\reme\config.example.json "$HOME\.zcode\reme\config.json"
```

The shipped example is:

```json
{
  "mcp_url": "http://127.0.0.1:2333/mcp",
  "api_url": "",
  "memory_interval": 1
}
```

An empty `api_url` is derived from `mcp_url` when that URL ends in `/mcp`, which is how the default
single-port deployment is reached. Set `api_url` only for a split-port or reverse-proxy deployment.
The example sets `memory_interval` to `1` for immediate per-turn writes; the code default is `5`.

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
| `context_max_chars` | `8000` | Maximum recalled context |
| `timezone` | `Asia/Shanghai` | Daily batch timezone |

Unknown fields and invalid values fail validation. Disabling `auto_memory` preserves pending queue
files and stops automatic capture and retries, including delivery by an already-started worker.
Changing the endpoint never sends old queued conversations to the new service. After changing
`mcp_url`, reconnect MCP: ZCode reads the server settings when the plugin loads, so reload the
plugin or start a new task instead of reusing an open one.

## Lifecycle

ZCode supports `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PermissionRequest`,
`PostToolUse`, `PostToolUseFailure`, and `Stop`.

ZCode's hook reference documents an `async` field for hook commands. This plugin leaves it unset.
The `Stop` hook itself runs synchronously because it must read the final assistant message and
durably queue the turn; it then starts a detached delivery process and returns. Detaching is
independent of the hook executor: it keeps `Stop`'s foreground time bounded on every build,
including builds that support asynchronous command hooks, and the durable queue still covers a
process that exits or is killed later.

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
queued for the next Stop or SessionStart. A delivery request that arrives while another pass is
running marks that pass to scan again, so a session captured mid-delivery is not dropped.

## Migration from manual hooks

If ReMe was previously added directly to `%USERPROFILE%\.zcode\cli\config.json`, remove the ReMe
entries from `hooks.events` and the manual `reme` MCP server after installing this plugin. Keep ZCode,
Orca, and other unrelated Hooks. Running both installations captures the same turn twice.

## Verify

Start with the connection, before testing memory:

1. Confirm the plugin is installed in **Settings → Plugins** and the `reme` server appears in
   **Settings → MCP**.
2. In a new ZCode task, ask it to call the ReMe `health_check` and `version` MCP tools. Both must
   answer; if they fail, fix `mcp_url` or the service before continuing.

Then set `memory_interval` to `1`, open a new task, and ask it to remember a synthetic fact. Success
requires:

1. a `prompt_cached` row followed by `memory_saved` in `%USERPROFILE%\.zcode\reme\hooks.log`;
2. the corresponding queue `.done` receipt;
3. one new `session/dialog/zcode-*.jsonl` file with a user and final-assistant row;
4. a later task that recalls the saved fact with a source path.

If the plugin is enabled but its components are absent, start a new task before changing files.

`auto_dream` remains owned by the ReMe service; the plugin does not start another scheduler.
