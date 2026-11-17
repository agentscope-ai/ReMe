# ReMe Desktop

Electron + React client for ReMe's AgentRuntime. The Python service owns Agents, native sessions and workspace files;
Desktop connects over JSON/SSE through its sandboxed preload bridge. Credentials remain in the backend environment.

The top bar contains **Remy** and **Knowledge**, with Settings at the right.

- Remy's sidebar lists the Chat Agent as **Remy** (`agent_id=chat`) and each named Cron or Proactive/event Agent.
- Selecting an Agent opens its current formal conversation. Temporary query forks never become the displayed current
  session. Browsing an Agent with no session leaves it empty until a message or explicit new-session action creates one.
- Each Agent has a conversation, Memory and a mailbox. Task Agents also expose Events. Task conversations show the latest
  native turn; previous turns remain in native history and model context. Related links open original workspace files.
- User conversations and automatic execution are serialized per Agent. An active child reply continues when switching
  Agents, with separate message histories and drafts. Chat Agent and task Agent conversations never share native session IDs.
- Remy's Updates tab retains reminder browsing, document details, replay, narration and explicit user-read receipts.
  Automatic Chat Agent replies retain their turn provenance across reconnects and restarts. AgentRuntime determines
  notification eligibility; the client applies quiet mode and its presentation budget.
  User reading and Agent inbox processing are independent; reading a reminder never consumes an event batch.
- Knowledge retains Workspace / Daily / Knowledge browsing, Markdown and YAML previews, wikilinks, tags and search.
  Tag-only browsing uses `list_tag_files`; selecting multiple tags uses OR semantics.
- The desktop character, voice settings, themes and native window behavior are migrated from ReMe3.

## Start

Use Python 3.11+, Node.js 22.13+ and npm. Install from the repository root:

```bash
python -m pip install -e reme_studio -e ".[dev,core]"
cd reme_desktop
npm ci
npx install-electron
```

Configure `LLM_API_KEY` and `LLM_BASE_URL` in the backend environment. The endpoint must support Responses API.
The optional voice functions also require `DASHSCOPE_API_KEY`. See
[`proactive.yaml`](../reme/config/proactive.yaml) for the model, collectors and initial Agents.

Terminal one, from the repository root:

```bash
reme start config=proactive workspace_dir=/absolute/path/to/your/workspace service.host=127.0.0.1 service.port=2333
```

Terminal two:

```bash
cd reme_desktop
npm run dev
```

Desktop defaults to `http://127.0.0.1:2333`; change it in Settings. The backend and Desktop are independent processes.
The workspace background is opaque by default. Settings → Appearance → Transparent background restores the macOS glass effect immediately and remembers your choice.
Closing the workspace hides the window; use the character or tray to reopen it. Quit through the tray to stop Desktop.

## Service contract

The client uses the following jobs from the same checkout:

| Job                                                     | Behavior                                                                             |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `agent_list`                                            | Remy and task identities, current formal session IDs and lightweight status          |
| `agent_session`                                         | Current session snapshot for `agent_id`, or a null snapshot before the first session |
| `agent_chat`                                            | Stream a user turn in that Agent's formal session; defaults to chat                  |
| `agent_new_session`                                     | Select a fresh empty formal session and retain previous native history               |
| `agent_events`                                          | Observe one Agent's current session plus global sidebar status                       |
| `agent_inbox`                                           | Page every Agent's pending, processing and processed input events                    |
| `agent_outbox`                                          | Page a task Agent's committed reminders                                              |
| `agent_reminder`                                        | Read canonical reminders and acknowledge user reading independently of processing    |
| `session_read` / `session_status`                       | Read retained native history / inspect native activity                               |
| `list`, `load`, `list_tags`, `list_tag_files`, `search` | Read-only Knowledge browsing                                                         |

Subscriptions start with an authoritative snapshot and coalesce live updates into replaceable message snapshots.
Text, tool activity, public reasoning summaries and attachments retain their order within each turn. Repeated updates
replace the same item; final text does not remove earlier progress. Every reconnect reconstructs the current view,
including unfinished output; requests are never resent to recover a display. Native transcripts remain the durable
source, while the runtime's bounded live projection bridges updates until native history catches up.
Polling also observes external workspace edits and works with session-capable wrappers. Stream errors and the terminal `DONE` retain ReMe's
normal service contract. Interrupted user requests are never automatically resent.

Tool details expand on demand. Markdown retains stable blocks while its active tail streams; each text item is an
independent Markdown document. The conversation follows new output only while the reader is at the bottom, offers a
return-to-latest action, and loads older messages in batches.

Normal progress messages remain in the conversation. Reasoning summaries stay inside expandable process details;
they do not separate adjacent tool calls. Native command actions identify reads, directory listings and searches,
including their paths and queries. Completed turns show a localized completion row with the retained process details.

The existing `manage_proactive_agent(action=query)` remains a separate temporary, read-only query. Desktop conversations
use `agent_chat`, so their history remains available to later automatic execution. See
[`AgentRuntime`](../reme/components/agent_runtime/base_agent_runtime.py) and
[service Steps](../reme/steps/proactive/agent/api.py) for the contracts.

The sidebar separates Remy, Proactive and Cron. Task groups collapse independently. Each conversation shows only the current formal session; memory, reminders, inboxes, event records and outboxes have dedicated views. Workspace links open the original file in Knowledge, including images; web links open the browser.

Inbox processing (`read_at`) and user reading (`user_read_at`) are independent. Reading a mailbox never claims events or consumes mail. Event history is bounded by the configured event-stream retention.

Local presentation receipts use `userRead` and `readSynced`; bubble presentation never marks a message as read.
The old `read` receipt format is unsupported. Close Desktop and remove obsolete receipts from
`desktop/presentation.json` in its application data directory before starting this version.

## Validate and package

```bash
npm run typecheck
npm test
npm run format:check
npm run pack
```

`npm run dist` builds the host platform's installer. Windows uses NSIS, macOS uses DMG/ZIP, and Linux uses AppImage.
Remy's artwork, source sheets, prompts and rebuild script live in [`assets/remy/`](assets/remy/README.md).
Packages contain the Electron client plus Remy's manifest and runtime frames; Python, credentials and workspace data
stay separately managed.
Native microphone access and system notification delivery additionally depend on operating-system permissions.
