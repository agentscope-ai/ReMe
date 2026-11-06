export type SessionStatus = "active" | "idle" | "not_loaded" | "error";

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
  endpoint: string;
  scale: number;
  quietUntil: number;
  anchor?: { x: number; y: number };
}

export interface Bubble {
  id: string;
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
  connection: "connecting" | "connected" | "reconnecting" | "error";
  connectionError: string;
  settings: Settings;
  workspace: string;
  sessionId: string | null;
  status: SessionStatus | null;
  messages: SessionMessage[];
  pending: PendingTurn | null;
  bubble: Bubble | null;
  unread: number;
  chatVisible: boolean;
  focusMessageId: string | null;
}

export interface AnimationClip {
  id: string;
  label: string;
  loop: boolean;
  ticks: number[];
  frame_files: string[];
  render_scale?: number;
}

export interface AnimationManifest {
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
  | {
      type: "settings.update";
      endpoint?: string;
      scale?: number;
      quietUntil?: number;
    }
  | { type: "link.open"; url: string };

export type DesktopCommand =
  { type: "animation.play"; id: string } | { type: "settings.open" };

export interface DesktopApi {
  bootstrap(): Promise<Bootstrap>;
  action(action: DesktopAction): Promise<void>;
  onState(callback: (state: DesktopState) => void): () => void;
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
} as const;
