import type {
  AgentMessage,
  TranscriptItem,
  SessionStatus,
  AgentView,
  ClientInfo,
} from "./wire.ts";
export type {
  AgentMessage,
  AgentEvent,
  AgentView,
  ClientInfo,
  ClientActivity,
  Activities,
  SessionStatus,
  TranscriptItem,
  ItemStatus,
  TextItem,
  SummaryItem,
  ToolAction,
  ToolItem,
  ArtifactItem,
} from "./wire.ts";

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
  taskInput?: boolean;
}

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
  | { kind: "mail"; id: string; agentName: string };

export type ItemAction = "open" | "remind" | "explain" | "copy";

export interface StreamChunk {
  chunk_type: string;
  chunk: unknown;
  done?: boolean;
}

export interface Receipt {
  userRead: boolean;
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
  companionVisible?: boolean;
  windowBounds?: { x: number; y: number; width: number; height: number };
}

export type SettingsUpdate = Partial<
  Omit<Settings, "anchor" | "windowBounds" | "companionVisible">
>;
export const WORKBENCH_PAGES = [
  "remy",
  "tasks",
  "notebook",
  "knowledge",
  "settings",
] as const;
export type StudioPage = (typeof WORKBENCH_PAGES)[number];

export interface PendingTurn {
  id: string;
  query: string;
  outcome?: NonNullable<AgentMessage["outcome"]>;
  sessionId: string | null;
  turnId?: string;
  streaming: boolean;
  error?: string;
}

export interface StudioState {
  revision: number;
  selectedAgentName?: string;
  agents?: AgentView[];
  connection: "connecting" | "connected" | "reconnecting" | "error";
  connectionError: string;
  settings: Settings;
  clientInfo: ClientInfo | null;
  sessionId: string | null;
  status: SessionStatus | null;
  messages: DisplayMessage[];
  taskRunStatus?: StatusNotice;
  reminders: DisplayMessage[];
  pending: PendingTurn | null;
  surface: CompanionSurface | null;
  unread: number;
  readMessageIds: string[];
  activePage: StudioPage;
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

export type StudioAction =
  | { type: "speech.ready"; ready: boolean; token: string }
  | { type: "speech.capture"; id: string }
  | { type: "speech.recording"; id: string }
  | { type: "speech.cancel"; id: string }
  | { type: "speech.speak"; sessionId: string; messageId: string }
  | { type: "speech.stop" }
  | { type: "speech.retry" }
  | { type: "speech.clearError" }
  | { type: "speech.report"; id: number; status: PlaybackStatus }
  | { type: "interaction" }
  | { type: "item.action"; target: ItemTarget; action: ItemAction }
  | { type: "chat.send"; query: string }
  | { type: "chat.stop" }
  | { type: "page.select"; page: StudioPage }
  | { type: "agent.select"; agentName: string }
  | { type: "agent.reveal"; agentName: string }
  | { type: "session.new" }
  | { type: "messages.read"; sessionId: string; ids: string[] }
  | { type: "items.read" }
  | ({ type: "settings.update" } & SettingsUpdate)
  | { type: "document.open"; path: string }
  | { type: "clipboard.copy"; text: string }
  | { type: "link.open"; url: string; sourcePath?: string };

export type StudioCommand =
  | { type: "settings.open" }
  | { type: "speech.play"; id: number; audio: SpeechAudio }
  | { type: "speech.stop"; id: number };

export interface StudioApi {
  action(action: StudioAction): Promise<ErrorData | null>;
  workspace(
    request: WorkspaceRequest,
  ): Promise<JobResponse | { error: ErrorData }>;
  transcribe(request: {
    id: string;
    audio: Uint8Array;
  }): Promise<
    { text: string; error?: never } | { error: ErrorData; text?: never }
  >;
  onCommand(callback: (command: StudioCommand) => void): () => void;
}

export interface JobResponse<T = unknown> {
  success: boolean;
  answer: T;
  metadata: Record<string, unknown>;
}

export type WorkspaceRequest =
  | import("./backend.ts").BackendRequest
  | import("./tasks.ts").TaskRequest
  | import("./notebook.ts").NotebookRequest
  | {
      type: "outbox";
      agentName: string;
      state?: ReminderState;
      offset?: number;
      limit?: number;
    }
  | { type: "reminder"; agentName: string; id: string }
  | {
      type: "inbox";
      agentName: string;
      state?: EventState;
      offset?: number;
      limit?: number;
    }
  | { type: "list"; path: string }
  | { type: "load"; path: string }
  | { type: "save"; path: string; content: string; expectedMtime: string }
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

export type ReminderState = "all" | "pending" | "processed";
export type EventState = ReminderState | "processing";

export interface TaskReminder {
  id: string;
  from_agent_name: string;
  priority: "P0" | "P1" | "P2";
  message: string;
  delivered_at: string;
  read_at: string | null;
}

export interface InputEvent {
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
