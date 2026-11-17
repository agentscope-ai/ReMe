export type SessionStatus = "active" | "idle" | "not_loaded" | "error";

export interface AgentActivity {
  session_id: string;
  execution_session_id?: string | null;
  turn_id: string;
  source: "chat" | "proactive" | "task";
  phase:
    | "thinking"
    | "working"
    | "remembering"
    | "answering"
    | "waiting"
    | "completed"
    | "failed"
    | "interrupted";
  tool?: string | null;
  task_id?: string | null;
  started_at?: string | null;
}

export type Activities = Partial<Record<"chat" | "proactive", AgentActivity>>;

export type TaskActivities = Record<string, AgentActivity>;

export interface AnimationIntent {
  id: number;
  clip: string;
  mode: "once" | "loop" | "hold";
  startedAt?: number;
  segment?: "enter" | "hold" | "exit";
}

export interface Behavior {
  animation: AnimationIntent;
  label: string;
  error: string;
}

export type PlaybackStatus = "started" | "ended" | "failed";

export interface StatusNotice {
  kind: "status";
  id: string;
  text: string;
  tone: "working" | "waiting" | "success" | "error";
  messageId?: string;
}

export interface DisplayMessage {
  id: string;
  turnId?: string;
  originSessionId?: string;
  createdAt?: string | null;
  role: "user" | "assistant";
  source: "chat" | "proactive";
  content: string;
  items?: TranscriptItem[];
  completed: boolean;
  streaming: boolean;
  status: StatusNotice | null;
  error?: string;
}

export type ItemStatus =
  "running" | "completed" | "failed" | "interrupted" | "waiting" | "unknown";

interface ItemBase {
  id: string;
  status: ItemStatus;
}

export interface TextItem extends ItemBase {
  type: "text";
  text: string;
  phase?: "commentary" | "final_answer";
}

export interface SummaryItem extends Omit<TextItem, "type"> {
  type: "summary";
  kind: "reasoning" | "plan";
}

export interface ToolAction {
  kind:
    | "read"
    | "list"
    | "search"
    | "command"
    | "edit"
    | "webSearch"
    | "browse"
    | "image"
    | "tool";
  target?: string;
  query?: string;
}

export interface ToolItem extends ItemBase {
  type: "tool";
  name: string;
  input: string;
  output: string;
  actions: ToolAction[];
}

export interface ArtifactItem extends ItemBase {
  type: "artifact";
  label: string;
  url: string;
  mediaType: string;
}

export type TranscriptItem = TextItem | SummaryItem | ToolItem | ArtifactItem;

export interface MessageBubble {
  kind: "message";
  message: DisplayMessage;
  target: ItemTarget;
  canSnooze: boolean;
  presentationId: number;
  paused: boolean;
}

export type CompanionSurface =
  MessageBubble | StatusNotice | { kind: "menu"; id: string; text: string };

export type ItemTarget =
  | { kind: "message"; id: string }
  | { kind: "mail"; id: string; agentId: string };

export type ItemAction = "open" | "remind" | "explain" | "copy";

export interface StreamChunk {
  chunk_type: string;
  event?: "start" | "delta" | "end";
  chunk: unknown;
  done?: boolean;
  status?:
    "running" | "completed" | "failed" | "interrupted" | "waiting" | null;
  session_id?: string | null;
  turn_id?: string | null;
  block_id?: string | null;
  message_id?: string | null;
  tool_call_id?: string | null;
  tool_call_name?: string | null;
  media_type?: string | null;
  metadata?: Record<string, unknown>;
}

export interface SessionMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  source: "chat" | "proactive";
  created_at?: string | null;
  expires_at?: string | null;
  turn_id?: string | null;
  completed: boolean;
  delivery: "idle" | "immediate";
  origin_session_id?: string | null;
  notification_eligible: boolean;
  chunks: StreamChunk[];
}

/** Renderer-safe messages, normalized once at the subscription boundary. */
export interface TranscriptMessage extends Omit<SessionMessage, "chunks"> {
  items: TranscriptItem[];
  outcome?: NonNullable<StreamChunk["status"]>;
  error?: string;
}

export interface SessionSnapshot {
  session_id: string;
  revision?: string;
  state: { session_id: string; status: SessionStatus; active_flags: string[] };
  messages: SessionMessage[];
}

export interface AgentEvent {
  agent_id?: string;
  agents?: AgentView[];
  type:
    | "snapshot"
    | "session_changed"
    | "message_upsert"
    | "status_changed"
    | "heartbeat";
  session_id?: string | null;
  status?: SessionStatus | null;
  activities: Activities;
  task_activities?: TaskActivities;
  message?: SessionMessage;
  snapshot?: SessionSnapshot;
  reason?: string;
  error?: string;
}

export interface Receipt {
  userRead: boolean;
  readSynced: boolean;
  presented: boolean;
  suppressed?: boolean;
  snoozeUntil?: number;
}

export interface Presentation {
  initialized: boolean;
  receipts: Record<string, Receipt>;
}

export const activityLevels = ["calm", "normal", "lively"] as const;
export const sleepDurations: readonly number[] = [5, 15, 30, 0];

export interface Settings {
  language: Language;
  theme: "system" | "light" | "dark";
  transparentBackground: boolean;
  endpoint: string;
  scale: number;
  activityLevel: (typeof activityLevels)[number];
  sleepMinutes: number;
  quietUntil: number;
  notifications: boolean;
  voiceAutoSend: boolean;
  voiceEnabled: boolean;
  voiceReplies: boolean;
  voiceReminders: boolean;
  voiceVolume: number;
  anchor?: { x: number; y: number };
  windowBounds?: { x: number; y: number; width: number; height: number };
}

export type SettingsUpdate = Partial<Omit<Settings, "anchor" | "windowBounds">>;
export type DesktopPage = "remy" | "knowledge" | "settings";

export interface PendingTurn {
  id: string;
  query: string;
  outcome?: NonNullable<StreamChunk["status"]>;
  sessionId: string | null;
  turnId?: string;
  streaming: boolean;
  error?: string;
}

export interface AgentView {
  agent_id: string;
  name: string;
  kind: "chat" | "cron" | "event";
  enabled: boolean;
  running: boolean;
  status: SessionStatus;
  session_id: string | null;
  memory_path: string;
  directory: string;
  pending_events: number;
  next_trigger_at: string | null;
  last_completed_at: string | null;
  error: string | null;
}

export interface DesktopState {
  revision: number;
  selectedAgentId?: string;
  agents?: AgentView[];
  connection: "connecting" | "connected" | "reconnecting" | "error";
  connectionError: string;
  settings: Settings;
  workspace: string;
  sessionId: string | null;
  status: SessionStatus | null;
  messages: DisplayMessage[];
  reminders: DisplayMessage[];
  mailboxRevision: number;
  pending: PendingTurn | null;
  surface: CompanionSurface | null;
  unread: number;
  readMessageIds: string[];
  activePage: DesktopPage;
  documentTarget: { path: string; request: number } | null;
  chatVisible: boolean;
  chatFocused: boolean;
  focusTarget: ItemTarget | null;
  focusRequest: number;
  behavior: Behavior;
  speech: SpeechState;
}

export interface SpeechState {
  asr: boolean;
  tts: boolean;
  ready: boolean;
  capture: {
    id: string;
    phase: "starting" | "recording" | "transcribing";
  } | null;
  playback: {
    id: number;
    messageId: string;
    kind: "manual" | "reply" | "reminder";
    phase: "loading" | "playing" | "error";
    part: number;
    total: number;
    preparing?: number;
  } | null;
  error: ErrorData | null;
}

export interface SpeechAudio {
  audio: string;
  format: "wav" | "mp3" | "pcm" | "opus";
  sample_rate: number;
}

export type DesktopPatch = Partial<DesktopState> & { revision: number };

export interface AnimationClip {
  id: string;
  ticks: number[];
  frame_files: string[];
  segments?: Record<"enter" | "hold" | "exit", number[]>;
}

export interface AnimationManifest {
  version: 2;
  ambient: string[];
  ambient_weights?: Record<string, number>;
  canvas: [number, number];
  output_fps: number;
  animations: AnimationClip[];
}

export interface WindowLayout {
  visible: boolean;
  body: { left: number; top: number; size: number };
  bubble: {
    left: number;
    top: number;
    width: number;
    height: number;
    side: "top" | "bottom" | "left" | "right";
    tip: number;
  };
}

export interface Bootstrap {
  state: DesktopState;
  manifest: AnimationManifest;
  layout: WindowLayout;
  platform: string;
  version: string;
}

export type DesktopAction =
  | { type: "speech.ready"; ready: boolean; token: string }
  | { type: "speech.capture"; id: string }
  | { type: "speech.recording"; id: string }
  | { type: "speech.cancel"; id: string }
  | { type: "speech.speak"; sessionId: string; messageId: string }
  | { type: "speech.stop" }
  | { type: "speech.retry" }
  | { type: "speech.clearError" }
  | { type: "speech.report"; id: number; status: PlaybackStatus }
  | { type: "playback.report"; id: number; status: PlaybackStatus }
  | {
      type: "bubble.shown";
      sessionId: string;
      id: string;
      presentationId: number;
    }
  | { type: "interaction" }
  | { type: "companion.menu" }
  | { type: "companion.menu.close" }
  | { type: "settings.open" }
  | { type: "animation.play"; clip: string }
  | { type: "app.quit" }
  | { type: "chat.open"; target?: ItemTarget }
  | { type: "item.action"; target: ItemTarget; action: ItemAction }
  | { type: "chat.close" }
  | { type: "chat.send"; query: string }
  | { type: "chat.stop" }
  | { type: "page.select"; page: DesktopPage }
  | { type: "window.minimize" }
  | { type: "window.maximize" }
  | { type: "companion.reset" }
  | { type: "companion.visibility"; visible: boolean }
  | { type: "agent.select"; agentId: string }
  | { type: "session.new" }
  | { type: "messages.read"; sessionId: string; ids: string[] }
  | { type: "items.read" }
  | { type: "bubble.dismiss"; id: string }
  | { type: "bubble.snooze"; id: string }
  | { type: "bubble.pause"; id: string; paused: boolean }
  | ({ type: "settings.update" } & SettingsUpdate)
  | { type: "document.open"; path: string }
  | { type: "link.open"; url: string; sourcePath?: string };

export type DesktopCommand =
  | { type: "settings.open" }
  | { type: "speech.play"; id: number; audio: SpeechAudio }
  | { type: "speech.stop"; id: number };

export interface DesktopApi {
  bootstrap(): Promise<Bootstrap>;
  action(action: DesktopAction): Promise<ErrorData | null>;
  workspace(
    request: WorkspaceRequest,
  ): Promise<JobResponse | { error: ErrorData }>;
  transcribe(request: {
    id: string;
    audio: Uint8Array;
  }): Promise<
    { text: string; error?: never } | { error: ErrorData; text?: never }
  >;
  onState(callback: (state: DesktopPatch) => void): () => void;
  onLayout(callback: (layout: WindowLayout) => void): () => void;
  onCommand(callback: (command: DesktopCommand) => void): () => void;
  pointer(interactive: boolean): void;
  drag(phase: "start" | "move" | "end"): void;
}

export const channels = {
  bootstrap: "reme:bootstrap",
  action: "reme:action",
  state: "reme:state",
  layout: "reme:layout",
  pointer: "reme:pointer",
  drag: "reme:drag",
  command: "reme:command",
  transcribe: "reme:transcribe",
  workspace: "reme:workspace",
} as const;

export interface JobResponse<T = unknown> {
  success: boolean;
  answer: T;
  metadata: Record<string, unknown>;
}

export interface WorkspaceConfig {
  workspace_dir: string;
  daily_dir: string;
  digest_dir: string;
  tag_key: string;
  jobs: string[];
}

export type WorkspaceRequest =
  | { type: "config" }
  | {
      type: "outbox";
      agentId: string;
      state?: MailState;
      offset?: number;
      limit?: number;
    }
  | { type: "reminder"; agentId: string; id: string; markRead?: boolean }
  | {
      type: "inbox";
      agentId: string;
      state?: EventState;
      offset?: number;
      limit?: number;
    }
  | { type: "list"; path: string }
  | { type: "load"; path: string }
  | { type: "image"; path: string }
  | { type: "tags"; page: number }
  | { type: "tagFiles"; tags: string[]; path: string; page: number }
  | { type: "search"; query: string; tags: string[]; path: string };

export interface TagFiles {
  total_files: number;
  total_pages: number;
  page: number;
  items: { path: string; tags: string[] }[];
}
import type { ErrorData, Language } from "./i18n.ts";

export type MailState = "all" | "pending" | "processed";
export type EventState = MailState | "processing";

export interface AgentMail {
  id: string;
  from_agent_id: string;
  priority: "P0" | "P1" | "P2";
  message: string;
  paths: string[];
  created_at: string;
  delivered_at: string | null;
  read_at: string | null;
  user_read_at: string | null;
}

export interface InputEvent {
  reminder?: AgentMail;
  event: {
    id: string;
    event_ts: string;
    source: string;
    description: string;
    paths: string[];
    metadata: Record<string, unknown>;
  };
  state: Exclude<EventState, "all">;
  consume_ts: string | null;
}

export interface ResourcePage<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}
