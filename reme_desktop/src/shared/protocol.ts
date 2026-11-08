export type SessionStatus = "active" | "idle" | "not_loaded" | "error";

export interface AgentActivity {
  session_id: string;
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
  run_id?: string | null;
  task_id?: string | null;
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
  createdAt?: string | null;
  role: "user" | "assistant";
  source: "chat" | "proactive";
  content: string;
  completed: boolean;
  streaming: boolean;
  status: StatusNotice | null;
  error?: string;
}

export interface MessageBubble {
  kind: "message";
  message: DisplayMessage;
  presentationId: number;
  paused: boolean;
}

export type CompanionSurface = MessageBubble | StatusNotice;

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
  metadata?: Record<string, unknown>;
}

export interface SessionMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  source: "chat" | "proactive";
  created_at?: string | null;
  turn_id?: string | null;
  completed: boolean;
  chunks: StreamChunk[];
}

export interface SessionSnapshot {
  session_id: string;
  state: { session_id: string; status: SessionStatus; active_flags: string[] };
  messages: SessionMessage[];
}

export interface PaEvent {
  type:
    | "snapshot"
    | "session_changed"
    | "message_upsert"
    | "status_changed"
    | "reset"
    | "heartbeat";
  cursor?: string;
  session_id?: string | null;
  status?: SessionStatus | null;
  activities: Activities;
  task_activities?: TaskActivities;
  notifications?: NotificationWindow | null;
  message?: SessionMessage;
  snapshot?: SessionSnapshot;
  reason?: string;
  error?: string;
}

export interface NotificationWindow {
  starts_at: string;
  ends_at: string;
  closing_message_id?: string | null;
}

export interface Receipt {
  read: boolean;
  presented: boolean;
  suppressed?: boolean;
  snoozeUntil?: number;
}

export interface Presentation {
  initialized: boolean;
  receipts: Record<string, Receipt>;
}

export interface Settings {
  language: Language;
  theme: "system" | "light" | "dark";
  endpoint: string;
  scale: number;
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
  text: string;
  sessionId: string | null;
  turnId?: string;
  streaming: boolean;
  error?: string;
}

export interface DesktopState {
  revision: number;
  connection: "connecting" | "connected" | "reconnecting" | "error";
  connectionError: string;
  settings: Settings;
  workspace: string;
  sessionId: string | null;
  status: SessionStatus | null;
  messages: DisplayMessage[];
  pending: PendingTurn | null;
  surface: CompanionSurface | null;
  unread: number;
  readMessageIds: string[];
  activePage: DesktopPage;
  chatVisible: boolean;
  chatFocused: boolean;
  focusMessageId: string | null;
  focusRequest: number;
  behavior: Behavior;
  speech: SpeechState;
}

export interface SpeechState {
  asr: boolean;
  tts: boolean;
  capture: {
    id: string;
    phase: "starting" | "recording" | "transcribing";
  } | null;
  playback: {
    id: number;
    messageId: string;
    kind: "manual" | "reply" | "reminder";
    phase: "loading" | "playing";
    part: number;
    total: number;
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
  body: { left: number; top: number; size: number };
  bubble: {
    left: number;
    top: number;
    width: number;
    maxWidth: number;
    height: number;
  };
  presence?: {
    host: "desktop" | "app";
    visible: boolean;
    opacity: number;
    motion?: { id: number; startedAt: number; direction: 1 | -1 };
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
  | { type: "speech.ready"; ready: boolean }
  | { type: "speech.capture"; id: string }
  | { type: "speech.recording"; id: string }
  | { type: "speech.cancel"; id: string }
  | { type: "speech.speak"; sessionId: string; messageId: string }
  | { type: "speech.stop" }
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
  | { type: "chat.open"; messageId?: string }
  | { type: "chat.close" }
  | { type: "chat.send"; query: string }
  | { type: "chat.stop" }
  | { type: "page.select"; page: DesktopPage }
  | { type: "window.minimize" }
  | { type: "window.maximize" }
  | { type: "companion.reset" }
  | { type: "companion.visibility"; visible: boolean }
  | { type: "session.new" }
  | { type: "messages.read"; sessionId: string; ids: string[] }
  | { type: "bubble.dismiss"; id: string }
  | { type: "bubble.snooze"; id: string }
  | { type: "bubble.pause"; id: string; paused: boolean }
  | ({ type: "settings.update" } & SettingsUpdate)
  | { type: "link.open"; url: string };

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
  measureSurface(size: import("./surface.ts").SurfaceSize): void;
  measureHabitat(value: {
    slot: { x: number; y: number; width: number; height: number } | null;
    reducedMotion: boolean;
  }): void;
  drag(phase: "start" | "move" | "end"): void;
}

export const channels = {
  bootstrap: "reme:bootstrap",
  action: "reme:action",
  state: "reme:state",
  layout: "reme:layout",
  pointer: "reme:pointer",
  surfaceSize: "reme:surface-size",
  habitat: "reme:habitat",
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
  | { type: "list"; path: string }
  | { type: "load"; path: string }
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
