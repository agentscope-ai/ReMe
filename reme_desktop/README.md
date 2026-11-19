# ReMe Desktop

Electron + React client for ReMe's AgentRuntime. The Python service owns Agents, native sessions and workspace files;
Desktop connects over JSON/SSE through its sandboxed preload bridge. Credentials remain in the backend environment.

The top bar contains **Remy**, **Tree Hollow** and **Knowledge**, with Settings at the right.

Tree Hollow manages one-sentence Todos and Ideas, including progress notes, deadlines, archives and links from
Ideas to Todos. It uses the `todo` and `idea` jobs in the `proactive` configuration; workspace JSON files own the records.

Task groups offer a **+** action to create tasks. Each task's **…** button and context menu share management,
pause/enable, folder and delete actions. Run details appear under Settings in the management panel, which saves execution settings and `TASK.md`
separately; description writes detect concurrent file edits. Pausing leaves the current round running, while
deleting stops it and retains history. Event-driven tasks still need an event source configured separately.

- Remy's sidebar lists the Chat Agent as **Remy** (`agent_name=Remy`) and each named Cron or Proactive/event Agent.
- Selecting an Agent opens its latest session snapshot. Remy continues the main Chat conversation; task questions
  run in temporary read-only forks and leave the task's current execution session unchanged. A task needs an existing
  session before it can answer questions. Only Remy offers **Clear conversation**, through its toolbar or `/clear`,
  while retaining personal memory and the backend's conversation history.
- Remy has a conversation, Memory and Updates. Task workspaces show the latest execution turn, the task description,
  input events and outgoing reminders. Query replies are displayed separately. Related links open original workspace files.
- Each task executes one automatic run at a time. A query can inspect the current task session while it runs on a backend
  that supports active forks. Switching Agents retains separate drafts and in-progress query replies.
- Remy's Updates tab retains reminder browsing, document details, replay, narration and explicit user-read receipts.
  Automatic Chat Agent replies retain their turn provenance across reconnects and restarts. AgentRuntime determines
  notification eligibility; the client applies quiet mode and its presentation budget.
  User reading and Agent inbox processing are independent; reading a reminder never consumes an event batch.
- Knowledge retains Workspace / Daily / Knowledge browsing, Markdown and YAML previews, wikilinks, tags and search.
  Tag-only browsing uses `list_tag_files`; selecting multiple tags uses OR semantics.
- The desktop includes the character, voice settings, themes and native window behavior.

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
[`demo.yaml`](../reme/config/demo.yaml) extends it with standalone examples, including the `speech_demo` round trip.
Codex dialogue and screenshots are collected independently every five minutes and both delivered to the `work` Agent.

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

| Job                                                                                            | Behavior                                                                              |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `list_agents`                                                                                  | Remy and task identities, current formal session IDs and lightweight status           |
| `read_agent_session`                                                                           | Agent view, messages and activities. Session ID is null before the first session.     |
| `ask_agent`                                                                                    | Continue main Chat, or stream a read-only fork query of the latest task session       |
| `create_chat_session`                                                                          | Start a fresh main Chat session; accepts no arguments                                 |
| `watch_agent_session`                                                                          | Observe current messages and state; Remy also carries the Agent roster                |
| `list_agent_inbox`                                                                             | Page task input events or Remy's derived reminder inbox                               |
| `list_agent_outbox`                                                                            | Page a task Agent's committed reminders                                               |
| `read_agent_reminder`                                                                          | Read canonical reminders and acknowledge user reading independently of processing     |
| `list_workspace`, `load_file`, `read_image`, `list_tags`, `list_tag_files`, `search_workspace` | Read-only Knowledge browsing                                                          |
| `asr` / `tts`                                                                                  | Voice input and playback                                                              |
| `get_client_info`                                                                              | Discover backend capabilities and workspace settings                                  |
| `manage_proactive_agent` / `agent_task`                                                        | Manage task settings and read/save the original task description with revision checks |
| `todo` / `idea`                                                                                | Manage Tree Hollow records and progress                                               |

Subscriptions use `snapshot`, `update` and `heartbeat`. Session changes are atomic snapshots; updates batch complete-message upserts.
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
they do not separate adjacent tool calls. The backend normalizes command actions into reads, directory listings and searches,
including their paths and queries. Completed turns show a localized completion row with the retained process details.

Task questions through `ask_agent` and `manage_proactive_agent(action=query)` share the same temporary read-only fork.
They use the latest task session during execution or while waiting for the next trigger, inherit its instructions and
context, and send the question directly. Query forks are retained for audit and never replace the task session.
Desktop displays each query answer separately from automatic progress; questions require an existing task session.
Only the main Chat Agent supports starting an empty conversation. See
[`AgentRuntime`](../reme/components/agent_runtime/base_agent_runtime.py) and
[service Steps](../reme/steps/proactive/agent/api.py) for the contracts.

The sidebar separates Remy, Proactive and Cron. Task groups collapse independently. Task workspaces show the latest execution turn and their query replies; memory, reminders, inboxes, event records and outboxes have dedicated views. Workspace links open the original file in Knowledge, including images; web links open the browser.

Inbox processing (`read_at`) and user reading (`user_read_at`) are independent. Reading a mailbox never claims events or consumes mail. Event history is bounded by the configured event-stream retention.

Local presentation receipts use `userRead` and `readSynced` in `desktop/presentation.json` under the application
data directory; bubble presentation never marks a message as read.

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

The Python schemas own the network contract. Run `python -m reme_desktop.scripts.client-contract` from the repository
root after changing it; `--check` detects drift. Generated wire types are separate from desktop IPC and presentation types.
The main connection publishes client capabilities and workspace settings with desktop state. Workspace pages derive
availability from that state and reset their drafts and pending reads when the endpoint or workspace changes.
