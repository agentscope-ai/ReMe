# Use ReMe in Codex

[中文说明](./README_ZH.md)

Connect Codex to your ReMe service to remember facts, preferences, and decisions across conversations.
Relevant memories are recalled before a prompt, completed conversations are recorded automatically,
and daily notes can be consolidated on a schedule. Your ReMe workspace owns the durable Markdown files.

## Before you start

- Use the latest Codex, with plugin installation and Hook review. Use the desktop app for native MCP settings.
- Make Python 3.11+ and `fastmcp>=3.4.2` available as `python3` in the environment that launches Codex,
  even when ReMe runs on another machine. Check with:

  ```bash
  python3 -c "import sys, fastmcp; print(sys.version); print(fastmcp.__version__)"
  ```

  If needed, install FastMCP there with `python3 -m pip install "fastmcp>=3.4.2"`.
- Configure ReMe's model for memory extraction using the [configuration guide](../../docs/en/configuration.md).
  Default BM25 search does not require an embedding model.
- Have a local checkout of this repository for installation.

## 1. Start your ReMe service

```bash
pip install "reme-ai[core]"
reme start \
  workspace_dir=/absolute/path/to/reme-workspace \
  service.backend=http \
  service.host=127.0.0.1 \
  service.port=2333 \
  service.tool_error_on_failure=true \
  jobs.dream_cron.backend=base \
  jobs.dream_cron.enable_serve=false
```

Keep the service running. Its MCP address is `http://127.0.0.1:2333/mcp`; it must be reachable from
**the machine running Codex**. Use your service's actual address when it runs elsewhere.
`workspace_dir` determines where your memory lives. Keep an unauthenticated service on loopback or behind a protected proxy.

Keep `service.tool_error_on_failure=true`: failed memory jobs must fail their MCP calls so unacknowledged
writes remain queued for retry. A successful health check does not verify this setting.

The last two options stop ReMe's built-in Dream timer so Codex can schedule consolidation. They use
existing service configuration and require no ReMe code changes. If your service already manages
consolidation and you want to keep that schedule, omit those options and turn off **Daily memory
consolidation** (`autoDreamEnabled`) in Codex. Choose one scheduler for a shared workspace.

## 2. Install ReMe and trust its Hooks

Use the absolute path to your checkout:

```bash
codex plugin marketplace add /absolute/path/to/ReMe/integrations/codex
codex plugin add reme@reme-codex
```

Restart the desktop app, or start a new CLI session. Open the installed ReMe entry and confirm it is enabled.

> **Screenshot placeholder — `plugin-installed.png`:** Actual Codex ReMe detail page, showing the enabled state and `reme` MCP card.

<!--
![ReMe installed and enabled in Codex](./figures/plugin-installed.png)
-->

In Codex's Hooks settings, review and trust **every ReMe entry**. In the CLI, use `/hooks`.
There are seven handlers across five events:

| Event | Purpose |
| --- | --- |
| `UserPromptSubmit` | Recall relevant memory before the prompt. |
| `Stop` | Save the completed turn locally, then deliver due batches in the background; two entries. |
| `SubagentStop` | The same two entries for subagents, used only when `rootAgentsOnly=false`. |
| `SessionStart` | Retry pending writes in the background. |
| `SessionEnd` | Attempt a short final delivery; preserve unfinished batches. |

Untrusted Hooks do not run. Installing or connecting the MCP server does not grant Hook trust.
After an update, review any changed entries again. See [Codex Hook review and trust](https://learn.chatgpt.com/docs/hooks#review-and-trust-hooks).

> **Screenshot placeholder — `hooks-trusted.png`:** All seven ReMe entries enabled and trusted in Codex Hooks settings or `/hooks`.

<!--
![ReMe Hooks enabled and trusted](./figures/hooks-trusted.png)
-->

## 3. Connect through MCP settings

Open the installed ReMe entry, find its **reme MCP server** card, and choose **Open MCP settings**.
This is the editable form for the service address and memory options.

For a first test, save these values:

| Field | Value |
| --- | --- |
| ReMe MCP address (`mcpUrl`) | `http://127.0.0.1:2333/mcp`, or your reachable service address |
| Automatic recall (`autoRecall`) | On |
| Automatic memory capture (`autoMemoryEnabled`) | On |
| Capture batch size (`autoMemoryInterval`) | `1` |
| Memory guidance language (`language`) | `en` or `zh` |

Save before using an action. Click **ReMe status** to open the status panel. Its Connection section checks the saved service
address and shows the ReMe version and `healthy` when reachable. This verifies service access;
recording and recall still need the conversation test below.

> **Screenshot placeholder — `mcp-settings.png`:** Actual native form with the address, capture switches, batch size, and the single ReMe status entry.

<!--
![ReMe MCP settings and connection result](./figures/mcp-settings.png)
-->

## 4. Verify memory across two conversations

Start a fresh conversation A. Use a unique project and phrase, for example:

```text
Remember this confirmed project decision: ReMe-Example-7319 reviews happen Friday
at 16:45 UTC. The verification phrase is amber-lynx-7319.
Do not use tools or write files yourself; just acknowledge the facts.
```

Keep the conversation open while recording finishes. In MCP settings, **ReMe status** should show
`memory_saved` and zero queued turns. In your ReMe workspace, confirm a `daily/` Markdown note
contains the fact. An acknowledgement by Codex alone is not evidence of a successful write.

> **Screenshot placeholder — `memory-recorded.png`:** Conversation A with the test fact and acknowledgement, without explicit tool calls.

<!--
![Codex conversation supplying a fact to remember](./figures/memory-recorded.png)
-->

Start a separate conversation B:

```text
For ReMe-Example-7319, when are the reviews and what is the verification phrase?
Use supplied ReMe memory and cite its source path. Do not call tools;
if no memory was supplied, say you do not know.
```

Expect the correct time, phrase, and a ReMe source path such as `daily/...md`. **ReMe status** should
include `recall_found`. This tests automatic recall independently of explicit search tools.
For everyday use, you can also ask Codex to search past decisions with `reme_search`.

> **Screenshot placeholder — `memory-recalled.png`:** Separate conversation B with the recalled facts and source path.

<!--
![Codex recalling memory in a separate conversation](./figures/memory-recalled.png)
-->

After verification, adjust the batch size for normal use; the default is five completed turns.
Short batches are also attempted at session boundaries. Failed or interrupted deliveries remain
queued locally for a later session. A short-lived `codex exec` can exit before background delivery
finishes; keep a session open when checking automatic writes.

## 5. Schedule or run memory consolidation

In **Auto Dream**, set **Daily memory consolidation**, **Auto Dream schedule**, **Workspace timezone**,
and optionally **Auto Dream hint**, then save. Defaults are `0 23 * * *` and `Asia/Shanghai`: daily at 23:00.
Only the daily form `minute hour * * *` is supported; for 02:30 use `30 2 * * *`.
The hint guides how ReMe consolidates existing daily notes into durable memory.

**ReMe status** shows the active schedule, timezone, next run, and latest result. The schedule runs
while Codex keeps the ReMe MCP connection alive. Closing Codex stops the timer; missed occurrences
are not replayed on startup. Multiple sessions sharing one `CODEX_HOME` use one scheduler. Separate
profiles or machines need their own coordination: enable only one scheduler for the same ReMe workspace.

For an immediate run, wait until pending writes are delivered, then click **Consolidate now (updates
memory files)** or explicitly ask Codex to use `reme_run_dream`. This updates existing memory files.
It remains available with the schedule disabled. Turning the schedule off does not cancel a request
already sent to ReMe; a timeout also does not prove that server-side processing has stopped.

## 6. Inspect status and adjust settings

Click **ReMe status** to open a dedicated panel from MCP settings. Connection checks and memory
status share this one entry. Four tabs organize the panel: **Overview**, **Auto Memory**,
**Memory Consolidation**, and **Components**. They show service health and the memory flow;
capture settings, queued turns/sessions, recall and recent activity; Dream scheduling and results;
and component memory estimates plus process RSS. Click **Refresh** to check again;
service details are collapsed in Components until needed. The initial view uses the opening call's result without
a second request. The panel follows the host theme and the saved `language` setting.

You can also ask Codex to call `reme_status` in a conversation. Hosts without an App view receive
the same status as text. Opening or refreshing the panel does not record or consolidate memory.
Component health and indexed-document counts are not exposed by the current ReMe MCP response;
the panel shows the available memory estimates without substituting other counts.
Queue and activity information remain available when the service is unreachable. Hook trust is
checked separately in Codex's Hooks settings.

> **Screenshot placeholder — `plugin-status.png`:** Real ReMe status panel or `reme_status` output with connection health, queued turns, recent activity, and the next Dream run.

<!--
![ReMe delivery and consolidation status in Codex](./figures/plugin-status.png)
-->

| Setting | Default | Meaning |
| --- | --- | --- |
| `mcpUrl` | `http://127.0.0.1:2333/mcp` | Full ReMe MCP URL. |
| `autoRecall` | `true` | Recall relevant memory before each prompt. |
| `autoMemoryEnabled` | `true` | Record completed user/assistant turns. |
| `autoMemoryInterval` | `5` | Completed turns per batch, from 1 to 1000. |
| `rootAgentsOnly` | `true` | Exclude subagent recall and recording. |
| `searchLimit` | `5` | Maximum search results, from 1 to 50. |
| `recallMinScore` | `0` | Minimum search score, nonnegative. |
| `language` | `en` | `en` or `zh` for memory guidance and local results; form labels stay in English. |
| `autoDreamEnabled` | `true` | Enable daily consolidation while the MCP connection is alive. |
| `dreamCron` | `0 23 * * *` | Daily consolidation time. |
| `dreamHint` | Empty | Guidance for scheduled and manual consolidation. |
| `timezone` | `Asia/Shanghai` | IANA timezone for memory dates and the daily schedule. |
| `requestTimeoutMs` | `10000` | Foreground request budget, from 1000 to 120000 ms. |
| `backgroundTimeoutMs` | `3600000` | Memory extraction and consolidation budget, from 1000 to 3600000 ms. |
| `shutdownTimeoutMs` | `5000` | Shutdown budget, from 100 to 60000 ms. Codex's `SessionEnd` limit restricts exit delivery to at most 2 seconds. |

Saved settings apply to subsequent calls without reinstalling. They live outside the plugin cache,
in `${CODEX_HOME:-~/.codex}/reme/config.json`. See [the complete configuration example](plugins/reme/config.example.json).
Changing `mcpUrl` does not send old queued conversations to the new service; switch back to the old
address to retry them. Settings changed during consolidation apply to subsequent runs.

## Troubleshooting and updates

| Symptom | Check |
| --- | --- |
| No MCP settings entry | Open the installed ReMe detail and its `reme` MCP card in the latest desktop app; confirm it is enabled and restart. |
| Blank status card followed by a plugin feature loading error | Check network access to the Codex sandbox page; see below. |
| Connection healthy, no recording or recall | Review every ReMe Hook; check the auto switches and recent activity. Health does not prove Hook execution. |
| Turns stay queued | Check the ReMe model and service logs, keep the session open, and confirm `service.tool_error_on_failure=true`. Retry happens at a later Hook. |
| Recall is empty | First confirm the fact exists in `daily/`, use its unique identifier, and inspect `searchLimit` and `recallMinScore`. |
| Dream did not run | Check the saved timezone, next run, schedule switch, and live MCP connection. Offline times are skipped. |
| Unknown configuration fields after upgrading | Back up `reme/config.json`, then replace obsolete fields using the current configuration example and reopen MCP settings. Old field names are not supported. |
| Python or FastMCP startup error | Check the `python3` environment inherited by Codex, including the desktop app. |

The panel serves its HTML and status data over local MCP, but Codex desktop also loads its own sandbox page.
If the card stays blank and the Codex client logs contain `guest_load_failed`, `ERR_UNEXPECTED`, or
`MCP sandbox RPC timed out`, check access from the machine running Codex:

```bash
curl -i --max-time 20 https://web-sandbox.oaiusercontent.com/mcp-app.html
```

If the response reports a corporate network or security software block, follow its approved process to allow
`web-sandbox.oaiusercontent.com`, including the page's static assets on that domain. Restart Codex and reopen the panel
after access is restored. This loading error alone does not establish whether ReMe or its Hooks are healthy and does
not call for changing `mcpUrl`, clearing memory, or reinstalling the plugin. Meanwhile, ask in a conversation:
“Call `reme_status` and show the returned status as text” to inspect the connection, queue, and consolidation schedule.

Content-free diagnostics are in `${CODEX_HOME:-~/.codex}/reme/hooks.log`; they include event names,
counts, and error classes, not conversation content. Pending conversation data is stored separately
in that directory until acknowledged. Delivery is best effort: forced termination before local
capture can lose a turn, and a lost acknowledgement can cause extraction to be retried.

To update, refresh your checkout and reinstall:

```bash
codex plugin remove reme@reme-codex
codex plugin add reme@reme-codex
```

Restart Codex and review changed Hooks. Your saved settings and pending writes remain outside the
installation cache. The [screenshot checklist](./figures/README.md) lists the six real Codex captures
reserved above; both language guides share the same images.
