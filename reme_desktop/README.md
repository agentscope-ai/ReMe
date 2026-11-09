# ReMe Desktop

Electron + React desktop client for ReMe's proactive agent. Remy stays on the desktop, opens a conversation when clicked,
and mirrors replies and proactive messages in a small bubble. Memory and conversation history stay in the ReMe workspace.

The wide workspace has two persistent tabs: **Remy** pairs reminders and a scheduled-task result inbox with the
conversation; **Knowledge** pairs Studio's Workspace / Daily / Knowledge directory views with a document reader. The top-right **Settings** icon opens
connection, appearance, character, notifications, voice, and project links; clicking it again returns to the previous
page. Switching pages preserves the chat draft and reading position. Sidebars can be resized with a pointer or the
arrow keys on their separator.
The sidebar button at the left of the title bar hides or restores the current page's sidebar. Each page remembers its
collapsed state and width across restarts; drafts, documents, and selected settings remain mounted while it is hidden.

On macOS the workspace keeps native window controls and uses native Liquid Glass when the optional addon is available,
falling back to Electron vibrancy. Reading surfaces remain mostly opaque. System reduced-transparency and reduced-motion
preferences are respected. Light, dark, and system themes are available in Settings, alongside Chinese and English.
The initial window follows the display's aspect ratio; subsequent opens restore its saved bounds within a connected display.
Companion status and reminder bubbles share the selected theme and glass styling, fit their content, and wrap long text
within a bounded width. Resizing a bubble keeps the desktop character anchored in place.

Remy stays in its own desktop window. Opening, resizing, focusing, or hiding the workspace does not move or resize the
character. Drag it to reposition it; the selected size and position are saved.

Knowledge uses read-only `app_config`, `list`, `load`, `list_tags`, `list_tag_files`, and `search` jobs through main-process
IPC. File and directory paths come from the connected service; Markdown supports workspace wikilinks and YAML memory
tags. The reader displays every frontmatter property in a collapsible panel and uses its title/name or leading Markdown
heading as the document title. Invalid YAML remains visible for inspection. Selecting tags browses matching files
without a text query; multiple selected tags use OR semantics. Tag counts
cover the entire workspace, while results follow the selected directory. Tag browsing requires the matching ReMe backend
with `list_tag_files` and a healthy tag index. Missing knowledge capabilities do not disconnect Remy chat.

## 启动 / Startup

ReMe 后端与 Desktop 是两个独立进程。先启动后端，再启动桌面；开发模式和安装包都遵循这个顺序。
ReMe and Desktop are separate processes. Start the backend first, then the desktop, whether developing or using an installer.

### 1. 安装依赖 / Install dependencies

需要 Python 3.11+、Node.js 22.13+ 和 npm。先激活用于 ReMe 的 Python 环境，再从仓库根目录执行：
Use Python 3.11+, Node.js 22.13+, and npm. Activate your ReMe Python environment, then run from the repository root:

```bash
python -m pip install -e reme_studio -e ".[core]"
cd reme_desktop
npm ci
npx install-electron
```

后端和桌面端使用同一份 checkout，确保包含 `pa_events.activities`、`list_tag_files` 等当前接口。
Use the same checkout for both processes so the backend provides the current activity and knowledge APIs.

### 2. 配置凭证 / Configure credentials

在**仓库根目录**创建或编辑 `.env`，保留已有配置，并替换下面的示例值。ReMe 启动时会从当前目录及其父目录查找该文件。
Create or edit `.env` in the **repository root**, keeping existing settings and replacing the example values below.
ReMe loads this file from the current directory or its parents at startup.

```dotenv
LLM_API_KEY=your-chat-api-key
LLM_BASE_URL=https://your-responses-provider.example/v1

# Optional / 可选
# CODEX_MODEL_NAME=your-model-id
# DASHSCOPE_API_KEY=your-dashscope-api-key
```

当前 `proactive` 配置使用 API key 认证，聊天服务地址必须支持 Responses API。模型配置以
[`proactive.yaml`](../reme/config/proactive.yaml) 为准；覆盖 `CODEX_MODEL_NAME` 时也要检查对应的 model catalog 能力。
语音识别和播报需要额外的 `DASHSCOPE_API_KEY`，它与聊天模型的 key 分开配置。凭证只放在后端环境中，不提交到 Git。

The current `proactive` configuration uses API-key authentication, and the chat endpoint must support the Responses API.
See [`proactive.yaml`](../reme/config/proactive.yaml) for model settings; review the model catalog capabilities when
overriding `CODEX_MODEL_NAME`. Recognition and narration additionally need `DASHSCOPE_API_KEY`, configured separately from
the chat key. Keep credentials in the backend environment and out of Git.

### 3. 终端一：启动后端 / Terminal 1: start the backend

回到仓库根目录，保持该终端运行。将示例工作区替换为你自己的目录；已有数据时继续使用原来的 `workspace_dir`。
Return to the repository root and keep this terminal running. Choose your own workspace path; reuse the original
`workspace_dir` to retain existing memories and sessions.

macOS / Linux:

```bash
cd /absolute/path/to/ReMe
reme start config=proactive workspace_dir="$HOME/ReMeWorkspace" service.host=127.0.0.1 service.port=2333
```

Windows PowerShell:

```powershell
cd C:/path/to/ReMe
reme start config=proactive workspace_dir="C:/ReMeWorkspace" service.host=127.0.0.1 service.port=2333
```

后端启动后，可在 <http://127.0.0.1:2333/> 打开 Studio。Desktop 与 Studio 共享同一后端管理的 PA 会话。
Once the service is ready, open Studio at <http://127.0.0.1:2333/>. Desktop and Studio share the backend-selected PA session.

### 4. 终端二：启动桌面 / Terminal 2: start the desktop

另开终端，从仓库根目录进入 `reme_desktop`。以下命令也适用于 Windows PowerShell：
Open another terminal at the repository root and enter `reme_desktop`. These commands also work in Windows PowerShell:

```bash
cd reme_desktop
npm run dev
```

单击角色始终展示主窗口并置前，最小化时先恢复。
也可使用托盘菜单或 `Command/Ctrl + Shift + Space` 打开窗口。
默认连接 `http://127.0.0.1:2333`，右上角设置图标中可以修改服务地址、语言和主题。只有后端没有已选会话时，Desktop 才会创建会话。
主动提醒卡片右下角的“详细解读”会带上该条提醒，在当前会话中继续解读，并保留输入框草稿。

Clicking Remy always shows the main window and brings it to the front, restoring it first if minimized.
You can also open the window from the tray menu or with `Command/Ctrl + Shift + Space`.
The default endpoint is `http://127.0.0.1:2333`; the top-right Settings icon opens endpoint, language, and theme preferences.
Desktop creates a PA session only if the backend has no selected session.
Choose “Explain in detail” at the bottom-right of a proactive reminder to discuss it in the current conversation while keeping your draft.

### 构建后运行 / Run a production build

后端保持运行，在 `reme_desktop` 中执行以下命令。修改源码后需要重新构建，`npm start` 直接读取 `out/`。
Keep the backend running, then run these commands inside `reme_desktop`. Rebuild after source changes: `npm start` reads `out/`.

```bash
npm run build
npm start
```

### 打包与安装 / Package and install

在目标系统上的 `reme_desktop` 中选择对应命令。安装包包含 Electron 客户端和 Remy 素材；Python 后端、凭证及工作区独立管理。
Run the appropriate command in `reme_desktop` on the target OS. Packages contain the Electron client and Remy artwork;
the Python backend, credentials, and workspace are managed separately.

| 命令 / Command                | 用途 / Purpose                                                                                  |
| ----------------------------- | ----------------------------------------------------------------------------------------------- |
| `npm run pack`                | 在 `dist/` 生成本机可直接运行的解包应用 / Build an unpacked app for the host OS under `dist/`.  |
| `npm run dist`                | 生成本机安装包：macOS DMG/ZIP、Windows NSIS、Linux AppImage / Build installers for the host OS. |
| `npm run dist -- --win --x64` | 在 Windows 构建 x64 NSIS 安装包 / Build the Windows x64 NSIS installer on Windows.              |

macOS 打开生成的 DMG，将应用拖入 Applications 后启动；Windows 运行 `dist/` 下的 `*-x64-setup.exe`，安装后从开始菜单打开；
Linux 为生成的 AppImage 添加执行权限后运行。安装后每次使用仍需先启动后端；已有后端服务时直接打开桌面应用即可。

On macOS, open the DMG and drag the app into Applications. On Windows, run `*-x64-setup.exe` from `dist/` and launch from
the Start Menu. On Linux, make the generated AppImage executable and run it. Start the backend before launching the
installed desktop app; if the service is already running, just open Desktop.

### 停止、重启与排查 / Stop, restart, and troubleshoot

- 开发模式：在桌面终端按 `Ctrl+C` 停止 Desktop，在后端终端按 `Ctrl+C` 停止服务。重启时先后端、再桌面，沿用相同工作区和地址。
  Development: stop each process with `Ctrl+C` in its terminal. Restart the backend first, then Desktop, using the same workspace and endpoint.
- 安装版：在托盘菜单选择退出。关闭聊天窗口只会隐藏工作区，角色与后台连接仍继续运行；隐藏角色可从托盘恢复。
  Installed app: choose Quit from the tray menu. Closing chat hides the workspace while Remy and the connection keep running; restore a hidden character from the tray.
- 连接不上：先确认后端终端没有启动错误，再检查设置中的地址与 `service.host` / `service.port` 是否一致。配置或凭证修改后重启后端。
  Connection failures: check the backend terminal for startup errors and match the endpoint to its host and port. Restart the backend after changing its configuration or credentials.
- 麦克风、屏幕采集和通知权限在系统设置中配置。远程后端的屏幕采集等工具运行在后端所在机器。
  Configure microphone, screen-recording, and notification permissions in system settings. Backend tools such as screen capture run on the service's machine.

### Windows 平台说明 / Windows platform notes

Windows 目标为 Windows 10/11 x64，使用不透明工作区和自定义最小化、最大化/还原、隐藏按钮。macOS Liquid Glass 插件在 Windows 跳过。
原生通知需要安装版的开始菜单快捷方式；开发模式无法验证实际通知投递。录音还需要在 Windows 麦克风隐私设置中允许桌面应用。

The Windows target is Windows 10/11 x64. It uses an opaque workspace with custom minimize, maximize/restore, and hide
controls. The optional macOS Liquid Glass addon is skipped on Windows. Native notifications require the installed Start
Menu shortcut; development runs do not establish delivery. Recording also requires Windows microphone privacy settings
to allow desktop apps.

## Interaction

- Left-click Remy for a short happy response and chat; hold and drag to move it, including between displays. Its position
  is remembered. Transparent artwork margins pass clicks through to the desktop.
- Right-click the character for a native menu: chat, unread reminders, animation previews, quiet mode, size, settings,
  reset position, hide/show, and quit. The tray exposes the same controls; use it to restore a hidden character. Animation
  previews end automatically, and real conversation/reminder activity takes priority.
- Choose **Language → 简体中文 / English** from the character's right-click menu or tray menu. Chinese is the default,
  regardless of system language. The choice is saved in desktop settings and updates menus, chat, task labels, and
  notifications without reconnecting or restarting playback. Conversation content keeps its original language.
- Use **Command/Ctrl + Shift + Space**, or the tray/menu bar, to reopen chat. Closing chat keeps the subscription running.
- Enter sends a message; Shift + Enter inserts a newline. Stop explicitly cancels the current chat request.
- Chat and the character share the same message text and execution status. Reply bubbles stream the current answer,
  then remain for twelve visible seconds after completion. Hover and narration pause expiry. Expand locates the same
  message in chat; dismissing a reply does not cancel generation. Reply previews do not use the reminder budget.
- A completed proactive message shows a bubble. Expand opens the original conversation; close preserves its unread state;
  **10 minutes later** snoozes the same message. Hover pauses expiry. Ordinary bubbles have a ten-minute cooldown and a
  maximum of three appearances per hour; explicit snoozes bypass that cooldown. Backlogs coalesce to the latest candidate.
  Opening chat does not hide the bubble, and reading the message does not shorten an already visible preview.
- Viewing a reminder's heading in the focused Remy tab marks it read. Knowledge and Settings never mark reminders read.
  Selecting an inbox entry locates its original message in the conversation. Historical messages on the first connection
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

The additive `task_activities` field tracks active task runs by `run_id`, with `source="task"` and a `task_id`. These
entries describe independent native sessions, survive selected-chat changes, and never add task transcripts to chat.
A run remains busy between its two queries and during publication; finishing, cancellation, or an upstream disconnect
clears its live presentation. These fields require an updated service for task-state animation; older services continue
to drive chat and proactive animations.

The backend tracks unfinished tools by call ID for each turn. Out-of-order or duplicate results cannot finish another
tool, and the working phase remains until the last tool completes. Terminal outcomes clear that turn's tracking state.

`src/main/behavior.ts` owns animation priority: dragging, foreground work, visible reminders, completion/error feedback,
manual previews, attentive listening, background work, then ambient gestures or sleep. Working uses the existing thinking
clip; foreground answering uses speaking, while background answering uses remembering. Foreground work interrupts
previews, and dragging restores the latest activity when released. Active tasks take priority over proactive work.
Repeated text deltas and phase changes using the same looping clip preserve its playback clock. Short thinking/recall
transitions are smoothed over 350 ms. Clip changes blend briefly; idle blinks have varied pauses. The system's reduced
motion preference uses static poses while preserving playback completion and status updates.

Reminders raise the note, hold it while their bubble is visible, and lower it when the bubble closes. A playback ID rejects
stale completion callbacks. The renderer only plays the chosen clip or segment in `once`, `loop`, or `hold` mode.
Reminding and completion/error feedback remain event-driven. Idle gestures include petting a fish, looking around,
reading a note, and briefly daydreaming. They use the asset manifest's relative weights, avoid consecutive repeats, and
have individual cooldowns. Settings → Remy offers calm (40–80 seconds), natural (18–40 seconds, default), and lively
(8–18 seconds) rests between gestures; after two minutes without interaction, these rests become 50% longer.
Remy sleeps after fifteen minutes without interaction and with no agent work by default. Settings also offers five or
thirty minutes, or no automatic sleep. Quiet mode suppresses ambient/background
gestures and reminder bubbles while foreground interaction remains available. No model calls are made to choose ambient actions.

`src/main/presentation.ts` chooses one surface: the current reply or its status, a reminder, speech status, or background
work. Both windows render the main process's conversation projection with the same message-body component; animation
timing and asset failures cannot delay or replace its text. The character receives only the selected, shortened preview.
Status updates do not consume reminder appearances. Hidden/preempted reminders retain their remaining visible time.
Reminder duration, cooldown, and hourly counts begin only after the renderer acknowledges a painted bubble or the OS
confirms a native notification; presentation never marks a message read.

Animation playback reports start, end, and failure with its playback ID. Missing acknowledgements and asset failures
recover to idle without changing the agent's real activity. The companion receives only presentation data; chat history
is excluded. Revisioned IPC patches update changed fields, and native bounds change only when the layout or position does.

## Voice

Voice uses the connected ReMe service's optional `asr_demo` and `tts_demo` jobs. The built-in proactive configuration
provides both; set `DASHSCOPE_API_KEY` in the service environment. Voice remains optional: a service without these jobs
still supports text chat. Model, voice, and encoding are configured on the ReMe service. The proactive configuration
defaults to `longanhuan_v3.1` with a sweet, light, playful female voice instruction.

- Hold the microphone or **Command/Ctrl + R** in the focused Remy page and release to transcribe. Releasing either key
  finishes the recording; macOS accepts both Command + R and Control + R. **Command/Ctrl + Shift + M** also toggles recording.
  Voice controls wait for the audio host to initialize; a temporary readiness message does not require a new conversation.
  Recognition inserts text at the current draft selection and sends the composed message by default. Turn off
  **Send after transcription** in settings to keep an editable draft instead. **Escape** cancels.
  Recordings stop after 24 seconds. Switching pages or losing focus cancels recording. Very short recordings are
  discarded locally; a conservative energy check also rejects silence and isolated clicks. It does not detect speech.
  Empty recognition results show a brief inline notice and never send the draft. The DashScope ASR adapter maps the
  explicit `ASR_RESPONSE_HAVE_NO_WORDS` response to an empty transcription, preserving request diagnostics. Other provider
  failures remain errors.
- Assistant messages and reminder bubbles have a speaker button. Select it again, or use the playback stop control, to
  stop narration. Narration preserves the full text, omitting code blocks and link URLs. Paragraphs and sentences are
  grouped into roughly 200–300-character parts with a 400-character maximum; decimals and version numbers stay together.
  Synthesis is serial: two parts are prepared before playback, then at most two future parts are buffered while audio
  plays in order. A missing part waits rather than being skipped. Each synthesis failure is retried once; a remaining
  error pauses at that part. **Resume narration** retries the failed part without replaying the completed prefix.
- Settings provides automatic narration, separate reply/reminder switches, and volume. New replies are read aloud by
  default; reminder narration is off. Both automatic and manual narration read the complete message. Reconnection and history
  snapshots stay silent; quiet mode interrupts automatic narration. Explicit speaker playback remains available.
- One audio host serves both windows. Actual playback drives speaking animation; recording drives listening animation.
  Spoken reminders keep their bubble visible through synthesis and playback. Narration never marks a reminder read.
- Starting a recording, sending a new message, switching sessions/endpoints, or disconnecting cancels obsolete voice
  work. Hiding chat cancels recording and manual chat playback. Microphone denial and service/playback errors leave
  the draft and conversation intact, with a retryable explanation in chat.

Microphone access is restricted to the chat window during an explicit recording request. Allow microphone access in
system privacy settings if prompted. Recordings are sent to the configured ReMe service and its ASR provider; Desktop
does not save recordings or generated audio. Synthesized audio has a bounded five-minute in-memory cache.

## Files and assets

`src/main/` owns native windows, ReMe connectivity and local persistence; `src/preload/` defines the renderer bridge;
`src/renderer/` contains the character, animation player and chat UI; `src/shared/` contains transport and IPC contracts.
User-facing text is managed in paired Chinese/English entries in `src/shared/i18n.ts`, with typed keys and interpolation.

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
npm run test:packaged    # macOS/Windows: validate the locally packaged application
npm run dist             # Installers for the host platform; never publishes
npm run dist -- --win --x64         # Windows x64 NSIS installer; never publishes
```

Build Windows installers on Windows. The installer is written to `dist/ReMe Desktop-<version>-x64-setup.exe` and offers
a per-user installation with a selectable directory and a Start Menu shortcut. Uninstalling preserves desktop settings
and the separately managed ReMe workspace. The application icon is derived from the existing Remy artwork.

Desktop CI runs on macOS and Windows. On Windows it also installs into a directory containing spaces, checks the Start
Menu shortcut's application ID, runs the packaged tests against the installed executable, and uninstalls the app.
Successful Windows jobs upload the installer as the `reme-desktop-windows-x64` artifact. To test a different installed
copy, set `REME_DESKTOP_EXECUTABLE` to its full executable path before running `npm run test:packaged`.

Build outputs (`out/`, `dist/`) and UI test captures (`reports/`) are ignored. Installer signing is a release concern;
local package generation does not establish behavior on another OS. Windows 10, mixed-DPI multi-monitor setups, real
notification delivery, microphone hardware, macOS Spaces/fullscreen behavior, and performance need manual validation.
Real-provider integration is separate from the mocked desktop tests. Voice tests use a virtual microphone and locally
generated WAV audio against a mocked ReMe service; they do not establish recognition quality or live-provider availability.
