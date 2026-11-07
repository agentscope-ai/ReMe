export type SessionStatus = "active" | "idle" | "not_loaded" | "error";

export interface AgentActivity {
  session_id: string;
  turn_id: string;
  source: "chat" | "proactive";
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
}

export type Activities = Partial<
  Record<AgentActivity["source"], AgentActivity>
>;

export interface AnimationIntent {
  id: number;
  clip: string;
  mode: "once" | "loop" | "hold";
  segment?: "enter" | "hold" | "exit";
}

export interface Behavior {
  animation: AnimationIntent;
  label: string;
  notice: StatusNotice | null;
  error: string;
}

export type PlaybackStatus = "started" | "ended" | "failed";

export interface StatusNotice {
  kind: "status";
  id: string;
  text: string;
  tone: "working" | "waiting" | "success" | "error";
}

export type CompanionSurface = Bubble | StatusNotice;

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
  message?: SessionMessage;
  snapshot?: SessionSnapshot;
  reason?: string;
  error?: string;
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
}

export type SettingsUpdate = Partial<Omit<Settings, "anchor">>;

export interface Bubble {
  kind: "reminder";
  id: string;
  presentationId: number;
  text: string;
  paused: boolean;
}

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
  messages: SessionMessage[];
  pending: PendingTurn | null;
  bubble: Bubble | null;
  surface: CompanionSurface | null;
  unread: number;
  chatVisible: boolean;
  chatFocused: boolean;
  focusMessageId: string | null;
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
    below: boolean;
    tail: number;
  };
}

export interface Bootstrap {
  state: DesktopState;
  manifest: AnimationManifest;
  layout: WindowLayout;
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
} as const;
import type { ErrorData, Language } from "./i18n.ts";
