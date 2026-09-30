# ReMe memory for Codex

[中文说明](./README_ZH.md)

Give Codex memory across conversations: start your own ReMe service, install the ReMe plugin, and
configure it through the native MCP settings form. The plugin recalls relevant history before a
prompt and records completed conversations. Your ReMe workspace owns the durable memory files.

## Before you start

- **Codex:** a client supporting local plugins, command hooks, and hook trust review. The native
  settings form also requires the `openai/settings` MCP extension. CLI 0.145.0 supports hooks and
  tools but does not render that form.
- **Plugin runtime:** Python 3.11+ available as `python3`, with FastMCP >= 3.4.2. Both the bundled
  MCP server and the hooks use this environment, even if ReMe runs elsewhere.
- **ReMe:** a running HTTP service with MCP enabled and `search`, `auto_memory`, and `health_check`
  exposed. Configure a model for memory extraction using the [ReMe configuration guide](../../docs/en/configuration.md).
  Default BM25 recall does not require an embedding model.
- **Source:** a local checkout of this repository. The plugin and its marketplace live in
  `integrations/codex/`; install from that checkout.

Check the Python environment from which you launch Codex:

```bash
python3 -c "import sys, fastmcp; print(sys.version); print(fastmcp.__version__)"
```

If the host environment does not include FastMCP, install it there with
`python3 -m pip install "fastmcp>=3.4.2"`.

## 1. Start ReMe

```bash
pip install "reme-ai[core]"
reme start \
  workspace_dir=/absolute/path/to/reme-workspace \
  service.backend=http \
  service.host=127.0.0.1 \
  service.port=2333
```

Keep this service running. Its MCP address is `http://127.0.0.1:2333/mcp`. If you choose a different
port, set the same address in the plugin in step 3. The address must be reachable **from the machine
running Codex**; `127.0.0.1` refers to that machine.

The plugin does not start ReMe. Keep the unauthenticated HTTP service on loopback or behind a
protected proxy. The service's `workspace_dir` determines which memory files it reads and writes.

## 2. Install the plugin and trust its hooks

Register the marketplace using the absolute path to your checkout, then install the plugin:

```bash
codex plugin marketplace add /absolute/path/to/ReMe/integrations/codex
codex plugin add reme@reme-codex
```

> **Screenshot placeholder — `plugin-installed.png`:** Capture the actual plugin detail page with the ReMe name, enabled state, and `reme` MCP server card.

<!--
![ReMe installed and enabled in Codex](./figures/plugin-installed.png)
-->

Restart the desktop client, or start a new Codex CLI session. Open the host's Hook settings
(`/hooks` in the CLI), find **ReMe / `reme@reme-codex`**, and review all four hooks:

| Hook | What it does | Required state |
| --- | --- | --- |
| `UserPromptSubmit` | Recalls relevant memory before your prompt | Enabled and trusted |
| `Stop` | Captures the completed turn and submits due memory batches | Enabled and trusted |
| `SessionStart` | Retries pending batches from earlier sessions | Enabled and trusted |
| `SessionEnd` | Attempts a short final flush | Enabled and trusted |

> **Screenshot placeholder — `hooks-trusted.png`:** Show all four ReMe hooks and their enabled/trusted state in the desktop settings or CLI `/hooks`.

<!--
![The four ReMe hooks enabled and trusted in Codex](./figures/hooks-trusted.png)
-->

Installing a plugin does not automatically trust its hooks. Until reviewed, the MCP connection can
work while automatic recall and recording remain inactive. See the [Codex hook trust guide](https://learn.chatgpt.com/docs/hooks#review-and-trust-hooks).

After updating the checkout, run `codex plugin add reme@reme-codex` again and restart the host.
Review changed hook definitions if prompted, then open a new conversation. User settings are stored
outside the plugin cache and survive upgrades.

## 3. Configure the ReMe MCP connection

In the desktop client's plugin directory, open the **installed ReMe plugin**. Find its `reme` MCP
server card and select the settings control labeled **Open MCP settings**. Open that connection's
ReMe settings form. Labels and placement can vary by host version; this is the MCP connection's
settings entry, separate from the plugin enable switch and Hook trust settings.

For a first test:

1. Set **ReMe MCP address** (`mcp_url`) to your service's full MCP URL, including `/mcp`.
2. Enable **Automatic recall** (`auto_recall`) and **Automatic memory** (`auto_memory`).
3. Set **Turns per memory batch** (`memory_interval`) to `1` so one completed turn triggers a write.
4. Save the form, then select **Check connection**. In the tested desktop client, the result appears
   in a tooltip on the button; hover over it to see the ReMe version and `healthy` status.

> **Screenshot placeholder — `mcp-settings.png`:** Show the MCP URL, automatic switches, batch size, and the Check connection result tooltip in the native form.

<!--
![Native ReMe MCP settings and a successful connection check](./figures/mcp-settings.png)
-->

**A healthy connection verifies service access. It does not verify Hook loading, trust, or memory
extraction.** Continue with the two-conversation test below. The connection check does not send
conversation content to ReMe.

If you cannot find the form, confirm that you opened the installed plugin's MCP connection and that
its process started successfully. A client without `openai/settings` support has no native form;
see [Configuration reference](#configuration-reference) for the user-owned settings file.

## 4. Verify recording and recall in separate conversations

Keep both automatic switches enabled and use `memory_interval=1`. Choose a unique test project
name so an older test cannot satisfy the recall check.

**Conversation A — record a fact:**

```text
Remember this confirmed decision for ReMe-Hook-Local-6502:
release reviews happen Friday at 16:45 UTC;
the verification phrase is amber-lynx-6502.
Do not use tools or skills or write files yourself; just acknowledge the facts.
```

Wait for the final reply and the Stop hook to finish. Check the local plugin log:

```bash
tail -n 10 "${CODEX_HOME:-$HOME/.codex}/reme/hooks.log"
```

A `memory_saved` event means ReMe acknowledged the batch. Also inspect the ReMe workspace's `daily/`
notes. A successful batch leaves `.done` receipts under the plugin's `reme/queue/` directory;
pending `.json` files remain until acknowledged. Do not delete pending files to clear an error.

> **Screenshot placeholder — `memory-recorded.png`:** Capture conversation A with its fact and final reply. This shows the conversation; verify delivery separately with `memory_saved` and a receipt as described above.

<!--
![The test fact and final acknowledgement in Codex conversation A](./figures/memory-recorded.png)
-->

**Conversation B — start a new conversation, then recall:**

```text
For ReMe-Hook-Local-6502, what is the review time and verification phrase?
Cite only supplied memory, including its source path.
Do not use tools or skills; if no memory is supplied, say you do not know.
```

The expected result includes **Friday at 16:45 UTC**, **amber-lynx-6502**, and a source path from
ReMe. Use a new conversation so the answer cannot come from the first conversation's context.

> **Screenshot placeholder — `memory-recalled.png`:** Capture conversation B with its question, recalled review time, verification phrase, and ReMe source path.

<!--
![Codex automatically recalls the test fact in a new conversation with a ReMe source](./figures/memory-recalled.png)
-->

After testing, keep `memory_interval=1` for per-turn writes or restore the default `5` to batch turns.
For explicit retrieval, ask “What did we decide about this project? Cite ReMe sources.” The bundled
`reme-memory` skill and MCP tools remain available alongside automatic hooks.

## Configuration reference

Change settings in the MCP form. The next hook or MCP tool call uses saved changes without a
reconnect; an in-flight call keeps its original settings. The form remains available when the
upstream ReMe service is offline, so you can correct its address.

| Setting | Default | Effect |
| --- | --- | --- |
| `mcp_url` | `http://127.0.0.1:2333/mcp` | Shared MCP URL for explicit tools and hooks; custom paths supported |
| `auto_recall` | `true` | Search before a user prompt |
| `auto_memory` | `true` | Capture and submit completed turns; false also pauses queued retries |
| `memory_interval` | `5` | Completed turns per batch; `1` submits each turn |
| `recall_limit` | `5` | Maximum search results |
| `recall_min_score` | `0` | Minimum recall score |
| `context_max_chars` | `8000` | Maximum recalled payload characters |
| `recall_timeout` | `5` | Recall timeout in seconds; greater than 0, at most 10 |
| `request_timeout` | `600` | Memory/tool timeout in seconds; greater than 0, at most 600 |
| `shutdown_timeout` | `2` | Exit flush budget in seconds; greater than 0, at most 2 |
| `timezone` | `Asia/Shanghai` | Daily batch timezone; match the ReMe workspace |

For recall-only use, turn off `auto_memory` and leave `auto_recall` on. To pause both automatic
behaviors, turn off both switches. Explicit MCP tools remain available.

Settings are saved atomically to `${CODEX_HOME:-$HOME/.codex}/reme/config.json`. You can also edit
this user-owned file directly; only overrides are needed. See [config.example.json](plugins/reme/config.example.json).
Remove a field to restore its default. Invalid values and unknown keys are rejected; failed UI
saves leave the previous file intact. Do not edit the installed plugin cache to configure ReMe.

Changing `mcp_url` never forwards old queued conversations to the new service. Switch back to the
original address to retry its queue. The legacy `api_url` key is accepted on read, no longer used,
and removed on the next UI save. Existing standard `/mcp` queue identities are preserved; custom
legacy HTTP queues remain untouched.

## Troubleshooting

Work through **connection → Hook state → recording → fresh-conversation recall** in that order.

| Symptom | Check or action |
| --- | --- |
| Cannot find ReMe settings | Open the installed plugin's `reme` MCP card → **Open MCP settings**. Confirm the host supports the native form and the MCP process starts. |
| Check connection shows a spinner but no chat reply | Hover over the button for the result tooltip. `healthy` and the ReMe version indicate a successful service check. |
| Connection is healthy, but nothing is recorded or recalled | Check that all four ReMe hooks are listed, enabled, and trusted. Installation and MCP health do not grant hook trust. |
| ReMe is absent from the Hook list | Update the checkout, reinstall the plugin, and restart the host. See the compatibility notes below. |
| Hook is listed but awaiting review | Review and trust its current definition, then start a new conversation. |
| `hook_failed` or MCP startup failure | Check the host's Python 3.11+/FastMCP environment, saved configuration, service reachability, and ReMe server logs. |
| No note after one turn | Set `memory_interval=1`, enable `auto_memory`, and wait for the final reply and Stop hook. ReMe also needs a working extraction model. |
| `capture_skipped` | Update to the current plugin. The hook needs a supported local transcript and final assistant message; current and legacy Codex event formats are supported. |
| `retry_failed` or pending queue files | Restore the original service/model connection and open a new session to retry. Keep the pending files. |
| Notes exist but recall misses them | Use a new conversation, enable `auto_recall`, check the service address/workspace, allow indexing to finish, and inspect recall limits and score filters. Try an explicit ReMe search to isolate retrieval from the hook. |

The log records status and exception types, not conversation text. Successful recall and idle
hooks do not necessarily create log entries; the absence of `hooks.log` alone does not prove
that no hook ran. Queue files do contain the source conversation text.

## Memory behavior and compatibility

- Recalled text is bounded and wrapped in `<reme-context>` as untrusted historical evidence.
  Only completed user/final-assistant text is recorded; tools, reasoning, subagents, commentary,
  and injected memory are excluded. ReMe does not need access to Codex transcript files.
- Stable session/message IDs, per-session locks, and receipts prevent repeated Stops from
  duplicating delivered source messages. Failed writes remain queued by service and session.
- Hooks are synchronous for compatibility with Codex 0.145.0, which skips asynchronous hooks.
  A due batch or startup retry can delay completion up to `request_timeout`. Session exit has a
  short, best-effort drain budget; remaining batches survive for the next session.
- Delivery is best-effort and extraction is at least once. Killing the host before capture can
  lose a turn; a lost acknowledgement can cause extraction to be retried.
- Durable memory and daily consolidation belong to the ReMe service. The plugin starts no
  separate daemon or scheduler.

This package uses `.codex-plugin/plugin.json` and `.mcp.json`. In native checks, Codex 0.159.2 and
desktop runtime 0.158.0-alpha.2.1 omitted these hooks when a root portable `plugin.json` was present,
even with explicit hook declarations. Retain the current Codex package layout. The parser supports
both legacy `user_message` / final `agent_message` and newer `item_completed` conversation events;
`response_item` records are excluded. See the [hook adapter](plugins/reme/hooks/auto_memory.py).

CLI 0.145.0 may return `user cancelled MCP tool call` for an explicit tool call when an unattended
session cannot collect approval. Verify such calls interactively. Automatic hooks use the trust
review described above.

## Maintainer checks

See the [screenshot checklist](./figures/README.md) for the five genuine screenshot slots and replacement steps.
Both languages share the same PNG files; text placeholders remain visible until screenshots are added.

```bash
pytest tests/unit/test_coding_agent_plugins.py tests/unit/test_codex_plugin_mcp.py -v
REME_CODEX_BIN=/path/to/codex pytest tests/integration/test_codex_plugin_install.py -v
```

The native installation test requires `hooks/list`, uses an isolated profile, and verifies all
four hooks before trust without model calls or hook execution. For documentation changes, run
`npm test` and `npm run build` in `github-pages/`.

References: [Codex hooks](https://learn.chatgpt.com/docs/hooks),
[OpenAI MCP settings contract](https://github.com/openai/mcp-extensions/blob/node-v0.1.0/docs/spec.md#structured-settings).
ReMe is developed at [agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe) under Apache-2.0.
