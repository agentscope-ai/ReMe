# ReMe Desktop

Electron + React desktop client for ReMe's proactive agent. Remy stays on the desktop, opens a conversation when clicked,
and shows incoming proactive messages in a small bubble. Memory and conversation history stay in the ReMe workspace.

## Run from source

Requires Node.js 22.13+ and a ReMe checkout with `pa_events.activities` support. Update the service and desktop together.
In one terminal, start the backend with an explicit
workspace outside this repository:

```bash
reme start config=proactive workspace_dir=/absolute/path/to/your/workspace
```

In another terminal:

```bash
cd reme_desktop
npm ci
npm run dev
```

The default endpoint is `http://127.0.0.1:2333`. Open chat and use Settings to change it. The client creates a PA session
only when the connected application has no selected session. Studio and Desktop follow the same application-owned PA
session; creating a new conversation switches both clients. Restarting Desktop reconnects to that session. Restoring the
selected session after restarting the ReMe service remains the backend's responsibility.

## Interaction

- Left-click Remy for a short happy response and chat; hold and drag to move it, including between displays. Its position
  is remembered. Transparent artwork margins pass clicks through to the desktop.
- Right-click the character for a native menu: chat, unread reminders, animation previews, quiet mode, size, settings,
  reset position, hide/show, and quit. The tray exposes the same controls; use it to restore a hidden character. Animation
  previews end automatically, and real conversation/reminder activity takes priority.
- Use **Command/Ctrl + Shift + Space**, or the tray/menu bar, to reopen chat. Closing chat keeps the subscription running.
- Enter sends a message; Shift + Enter inserts a newline. Stop explicitly cancels the current chat request.
- A completed proactive message shows a bubble. Expand opens the original conversation; close preserves its unread state;
  **10 minutes later** snoozes the same message. Hover pauses expiry. Ordinary bubbles have a ten-minute cooldown and a
  maximum of three appearances per hour; explicit snoozes bypass that cooldown. Backlogs coalesce to the latest candidate.
- Viewing a reminder's heading in the focused chat window marks it read. Historical messages on the first connection
  establish a quiet baseline instead of producing a burst of old reminders.
- Settings controls character size and a thirty-minute quiet period. Quiet mode affects desktop presentation, not the
  backend's collection or generation of memories.
- Optional system notifications are off by default. When enabled, they report long foreground replies (at least 30
  seconds), failures, and requests for attention while chat is unfocused. Hidden or unavailable characters can deliver
  reminders through native notifications, sharing the bubble cooldown and hourly budget. Clicking returns to the
  originating conversation/message; notifications from a superseded session cannot redirect the current conversation.

The main process owns HTTP/SSE, reconnection, the opaque subscription cursor, and presentation state. Subscription
`snapshot`, `reset`, `session_changed`, `message_upsert`, and `status_changed` follow the existing ReMe contract. Cursors
are kept with their in-memory snapshot, not restored alone after restarting the client. Interrupted replies are never
automatically resent. The renderer accesses desktop operations through a restricted, sandboxed preload bridge.

## Agent activity and animation

The service reduces portable agent events into `thinking`, `working`, `remembering`, `answering`, `waiting`, and terminal
phases. Subscription snapshots and `status_changed` carry `activities`, with independent `chat` and `proactive` entries.
Each entry identifies its selected PA session, native turn, source, phase, and optional tool. The desktop follows these
events for every client sharing that PA session, including conversations started in Studio. Activity is transient;
conversation files remain the source of truth.

The backend tracks unfinished tools by call ID for each turn. Out-of-order or duplicate results cannot finish another
tool, and the working phase remains until the last tool completes. Terminal outcomes clear that turn's tracking state.

`src/main/behavior.ts` owns animation priority: dragging, foreground work, visible reminders, completion/error feedback,
manual previews, attentive listening, background work, then ambient gestures or sleep. Working uses the existing thinking
clip; answering uses speaking. Foreground work interrupts previews, and dragging restores the latest activity when released.
Repeated text deltas do not restart an animation. Short thinking/recall transitions are smoothed over 350 ms.

Reminders raise the note, hold it while their bubble is visible, and lower it when the bubble closes. A playback ID rejects
stale completion callbacks. The renderer only plays the chosen clip or segment in `once`, `loop`, or `hold` mode.
Neutral idle/listening gestures vary every 15–45 seconds; semantic actions such as thinking and reminding are event-driven.
After five minutes without interaction Remy sleeps. Quiet mode suppresses ambient/background gestures and bubbles while
foreground interaction remains available. No model calls are made to choose ambient actions.

The desktop chooses one surface: a reminder bubble, an active task label, a request for attention, or brief outcome
feedback. Status updates do not consume reminder appearances. Hidden/preempted reminders retain their remaining visible
time. Display duration, cooldown, and hourly counts begin only after the renderer acknowledges a painted bubble or the
OS confirms a native notification; presentation never marks a message read.

Animation playback reports start, end, and failure with its playback ID. Missing acknowledgements and asset failures
recover to idle without changing the agent's real activity. The companion receives only presentation data; chat history
is excluded. Revisioned IPC patches update changed fields, and native bounds change only when the layout or position does.

## Files and assets

`src/main/` owns native windows, ReMe connectivity and local persistence; `src/preload/` defines the renderer bridge;
`src/renderer/` contains the character, animation player and chat UI; `src/shared/` contains transport and IPC contracts.

Character artwork has one source: `../assets/remy/anime/animations/v1/`. The player uses the manifest's PNG frames and
timed holds. Original sheets, prompts and timing are kept with the artwork so frames can be rebuilt. Packaging copies
only the manifest and frames into application resources.

Settings and scoped read/display/snooze receipts are JSON files under `app.getPath("userData")/desktop/`. The client does
not keep a separate conversation database. Malformed settings are preserved and reported for repair.

For isolated development, set `REME_DESKTOP_API_URL` to override the endpoint and `REME_DESKTOP_DATA_DIR` to an existing,
dedicated directory for Electron data. Tests supply their own temporary directory and mocked HTTP service.

## Validation and packaging

```bash
npm test                 # Streaming, replay/reset, bubble policy, placement and persistence
npm run typecheck
npm run format:check
npm run test:e2e          # Real Electron windows against a local mocked ReMe service
npm run test:dev          # Development renderer and sandboxed preload
npm run pack             # Local unpacked application for the host platform
npm run test:packaged    # macOS: validate the locally packaged application
npm run dist             # Installers for the host platform; never publishes
```

Build outputs (`out/`, `dist/`) and UI test captures (`reports/`) are ignored. Installer signing is a release concern;
local package generation does not establish Windows/Linux behavior, macOS Spaces/fullscreen behavior, or performance.
Real-provider integration is separate from the mocked desktop tests. Voice input/output is not included in this version.
