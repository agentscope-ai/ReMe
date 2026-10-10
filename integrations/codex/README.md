# Use ReMe in Codex

[中文说明](./README_ZH.md)

Connect Codex to your ReMe service to remember project decisions, preferences, and unfinished work
across conversations. ReMe can automatically record and recall memories, and organize them each day.
Your memory files stay in your own ReMe workspace.

The screenshots below were captured in Codex desktop. Click an image to view it at full size.

## Before you start

- **Codex CLI >= 0.159.2** for installation and CLI use. Run `codex --version` to check.
- **Codex desktop >= 26.924.22138** for the MCP settings form and status panel. Check the version in **About**.
  These are the versions validated for this guide; older releases have not been verified.
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

Keep `service.tool_error_on_failure=true` so failed writes can be retried later.

This example lets Codex manage daily consolidation. If your ReMe service already has a schedule
you want to keep, omit the last two options and turn off **Daily memory consolidation** in Codex.
Enable daily consolidation in only one place for the same workspace.

## 2. Install ReMe and trust its Hooks

Use the absolute path to your checkout:

```bash
codex plugin marketplace add /absolute/path/to/ReMe/integrations/codex
codex plugin add reme@reme-codex
```

Restart the desktop app, or start a new CLI session. Open the installed ReMe entry and confirm it is enabled.

[![Installed ReMe details and MCP settings entry in Codex](./figures/plugin-installed.png)](./figures/plugin-installed.png)

In the desktop app, add and open a local project folder before entering **Settings → Hooks**.
Select **Local** if a host selector is shown, then review and trust **every ReMe entry** (currently seven).
In the CLI, start `codex` from your project folder and use `/hooks`.
This enables automatic recording and recall. A healthy service connection alone does not enable them.
After an update, review any changed entries again. See [Codex Hook review and trust](https://learn.chatgpt.com/docs/hooks#review-and-trust-hooks).

[![The seven ReMe Hook entries in Codex settings](./figures/hooks-trusted.png)](./figures/hooks-trusted.png)

## 3. Connect through MCP settings

Open the installed ReMe entry, find its **reme MCP server** card, and click the settings gear
(**Open MCP settings**).
Use this form to change the service address and memory options. You do not need to edit a configuration file.

For a first test, save these values:

| Field | Value |
| --- | --- |
| ReMe MCP address | `http://127.0.0.1:2333/mcp`, or your reachable service address |
| Automatic recall | On |
| Automatic memory capture | On |
| Capture batch size | `1` to record after each completed reply |
| Memory guidance language | `en` or `zh` |

Save, then click **ReMe status**. In **Overview**, check that the service is **Healthy** and the ReMe
version is shown. If it is unavailable, expand **Connection details** to check the address and confirm
that your service is running. Next, try recording and recalling a memory.

[![ReMe service address, automatic memory options, and Dream schedule in Codex settings](./figures/mcp-settings.png)](./figures/mcp-settings.png)

The screenshot uses `http://127.0.0.1:2444/mcp`; enter your own service address.

## 4. Try your first memory

Start a fresh conversation A. Use a unique project and phrase, for example:

```text
Remember this project decision: ReMe-Example-7319 reviews happen Friday
at 16:45 UTC. The project keyword is amber-lynx-7319.
```

After Codex replies, keep the conversation open while recording finishes. Open **ReMe status →
Auto Memory → Recent activity** and look for **Memory saved** (`memory_saved`). **Queued turns**
should return to zero. You can also read the saved note under `daily/` in your ReMe workspace.
Wait for the saved status before continuing; Codex's acknowledgement alone does not confirm recording.

[![Conversation A supplies the project decision and receives an acknowledgement](./figures/memory-recorded.png)](./figures/memory-recorded.png)

Start a separate conversation B:

```text
For ReMe-Example-7319, when are the reviews and what is the project keyword?
Use ReMe memory and include the source path in your answer.
```

Expect Friday at 16:45 UTC, `amber-lynx-7319`, and a source path such as `daily/...md`.
To confirm automatic recall, check **Auto Memory → Recent activity** for **Relevant memory recalled**
(`recall_found`). A correct answer without that event does not confirm that automatic recall is enabled.
Use **Show all … events** if it is outside the latest five activities.

[![Conversation B recalls the review time, keyword, and ReMe source path](./figures/memory-recalled.png)](./figures/memory-recalled.png)

For everyday use, ask naturally: “What did we decide about this project's release process?” or
“Find my earlier preferences for code reviews.” You can adjust **Capture batch size** after the first
check; the default is five completed replies. Use `1` if you want each reply recorded promptly.
Keep Codex open until queued turns are saved, especially during short CLI sessions.

## 5. Schedule or run memory consolidation

In **Auto Dream**, set **Daily memory consolidation**, **Auto Dream schedule**, **Workspace timezone**,
and optionally **Auto Dream hint**, then save. Defaults are `0 23 * * *` and `Asia/Shanghai`: daily at 23:00.
Only the daily form `minute hour * * *` is supported; for 02:30 use `30 2 * * *`.
Use the hint to emphasize what to retain, for example: “Prioritize project decisions and unresolved tasks.”

Open **ReMe status → Consolidation** to check the next run and latest result. Keep Codex and ReMe
running at the scheduled time; missed runs are not made up after restarting. If several machines
use the same ReMe workspace, enable daily consolidation on only one of them.

To organize your memories now, wait for **Queued turns** to reach zero, then click **Consolidate ReMe
memory now** in MCP settings. This updates your memory files and also works when the daily
schedule is off. Check **Consolidation** for the result. Turning off the schedule does not stop an
already-started run; after a timeout, check its status before trying again.

## 6. Check status and customize ReMe

Open **ReMe status** from MCP settings and choose the tab for what you want to check:

| Tab | What to check |
| --- | --- |
| Overview | Is ReMe healthy? Are any conversation turns waiting to be saved? Expand **Connection details** to see the service address. |
| Auto Memory | Are recording and recall enabled? Expand **Recent activity** to check the latest saves, recalls, or failures. |
| Consolidation | When will memories next be organized, and did the last run succeed? |
| Components | How much memory is ReMe using? Expand **Service details** when troubleshooting. |

Click **Refresh** for an updated view. Recent activity initially shows the latest five events.
Expand an event for its full timestamp and diagnostic details; use **Show all … events** to see the available history.

If you prefer a text answer, ask Codex:
“Call `reme_status` and summarize my connection, pending memories, and next consolidation.”

[![ReMe status Overview shows a healthy service and zero queued turns and sessions](./figures/plugin-status.png)](./figures/plugin-status.png)

Overview shows the service health and pending turns. Select **Auto Memory** to inspect recording and recall activity.

Change options in **Open MCP settings**, then save. New settings apply to subsequent activity; you do
not need to reinstall. Common adjustments are listed below using the names shown in the form.

| Field | Default | When to change it |
| --- | --- | --- |
| ReMe MCP address | `http://127.0.0.1:2333/mcp` | Connect to a different ReMe service. |
| Automatic memory capture | On | Turn off to pause automatic recording. |
| Capture batch size | `5` | Set to `1` to save after each completed reply; increase to group more replies per save. |
| Automatic recall | On | Turn off to stop adding past memories to new prompts. |
| Root agents only | On | Turn off if you also want subagent conversations recorded and recalled. |
| Memory guidance language | `en` | Choose `zh` for Chinese guidance and status text. Settings labels remain English. |
| Search result limit | `5` | Increase to retrieve more results, up to 50. |
| Minimum recall score | `0` | Increase to filter out lower-scoring matches. |
| Daily memory consolidation | On | Turn off if you prefer manual organization or already have a service schedule. |
| Auto Dream schedule | `0 23 * * *` | Set the daily time, for example `30 2 * * *` for 02:30. |
| Auto Dream hint | Empty | Describe which information to prioritize when organizing memory. |
| Workspace timezone | `Asia/Shanghai` | Set your timezone, such as `Europe/London`, for memory dates and the daily schedule. |

<details>
<summary>Timeout settings</summary>

These values are in milliseconds. Keep the defaults unless requests regularly time out.

| Field | Default | Use |
| --- | --- | --- |
| Request timeout (ms) | `10000` | Wait time for searches and status checks; maximum `120000`. |
| Background timeout (ms) | `3600000` | Wait time for recording and consolidation; maximum `3600000`. |
| Shutdown timeout (ms) | `5000` | Exit wait setting. Codex limits final memory delivery to at most two seconds, so increasing this cannot extend that wait. |

</details>

Before changing the service address, let pending turns finish saving. Any remaining turns stay associated
with the previous address; switch back to retry them. Changes made during consolidation apply to the next run.

## Troubleshooting and updates

| Symptom | Check |
| --- | --- |
| No MCP settings entry | Confirm your desktop version meets the requirement above, then open the installed ReMe detail and its `reme` MCP card; confirm it is enabled and restart. |
| Hooks settings keep loading or do not show ReMe | Open a local project folder and select the local host before reopening Hooks settings; see below. |
| Status cannot open after an update, with `No such file or directory` pointing to an old plugin version | Fully quit and reopen Codex, then reopen ReMe status from MCP settings. Closing the card or starting a new conversation does not reload an existing MCP connection. |
| Blank status card followed by a plugin feature loading error | Check network access to the Codex sandbox page; see below. |
| Connection healthy, no recording or recall | Review every ReMe Hook; check the auto switches and recent activity. Health does not prove Hook execution. |
| Turns stay queued | Keep Codex and ReMe running. Check the ReMe model configuration and service logs, and confirm the startup command includes `service.tool_error_on_failure=true`. |
| Recall is empty | Confirm the fact was saved, mention its project name in your question, and check **Search result limit** and **Minimum recall score**. |
| Dream did not run | Check the saved timezone, next run, schedule switch, and live MCP connection. Offline times are skipped. |
| Unknown configuration fields after upgrading | Back up `~/.codex/reme/config.json`, compare it with the [current configuration example](plugins/reme/config.example.json), and update obsolete fields before reopening settings. |
| Python or FastMCP startup error | Check the `python3` environment inherited by Codex, including the desktop app. |

<details>
<summary>Hooks settings keep loading</summary>

Codex lists Hooks in the context of project folders. First add and open a local folder in the desktop
app, select **Local** if a host selector is shown, and reopen **Settings → Hooks**. Use **Reload hooks**
when available. After installing or updating ReMe, restart Codex if the list remains stale.

If the desktop page still does not load, use the CLI from the same project folder:

```bash
cd /absolute/path/to/project
codex
```

Enter `/hooks`, then review the ReMe entries and their enabled and trusted states. Use the same
`CODEX_HOME` as the desktop app if you have customized it. A desktop page that keeps loading does not
by itself show that Hooks failed to run; check ReMe's recent activity to verify recording and recall.

</details>

<details>
<summary>The status card is blank or reports a loading error</summary>

Codex needs access to `web-sandbox.oaiusercontent.com` to display the panel. Check access from the
machine running Codex:

```bash
curl -i --max-time 20 https://web-sandbox.oaiusercontent.com/mcp-app.html
```

If the response reports a corporate network or security software block, follow its approved process to allow
`web-sandbox.oaiusercontent.com`. Restart Codex and reopen the panel after access is restored.
Meanwhile, ask in a conversation: “Call `reme_status` and show the returned status as text.”

</details>

For further diagnosis, check `~/.codex/reme/hooks.log`. If you set `CODEX_HOME`, use that directory
instead of `~/.codex`. Avoid deleting `reme/` while troubleshooting: it also contains your settings
and conversations still waiting to be saved.

To update, refresh your checkout and reinstall:

```bash
codex plugin remove reme@reme-codex
codex plugin add reme@reme-codex
```

Restart Codex and review changed Hooks. Your saved settings and pending memories are preserved.
