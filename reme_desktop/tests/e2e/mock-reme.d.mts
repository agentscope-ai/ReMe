export function toneWave(seconds?: number): Buffer;
export function mockReMe(options?: {
  speech?: boolean;
  workspace?: boolean;
  ttsSeconds?: number;
  taskResults?: import("../../src/shared/protocol.ts").ProactiveInboxItem[];
}): Promise<{
  url: string;
  push(id: string, content?: string, completed?: boolean): void;
  upsert(message: import("../../src/shared/protocol.ts").SessionMessage): void;
  setTaskIds(value: string[]): void;
  setReplyDelay(value: number): void;
  notificationWindow(value: {
    enabled: boolean;
    starts_at: string;
    ends_at: string;
    closing_message_id?: string | null;
  }): void;
  activity(phase: string, source?: "chat" | "proactive", turnId?: string): void;
  taskActivity(phase: string, taskId?: string, runId?: string): void;
  cursors: (string | undefined)[];
  ttsRequests: { text: string }[];
  asrRequests: { audio: string; audio_format: string; sample_rate: number }[];
  setSpeechError(value: boolean): void;
  setSpeechErrorText(value: string): void;
  setAsrText(value: string): void;
  setTags(value: [string, number][]): void;
  setDocument(path: string, content: string): void;
  setSpeechDelay(value: number): void;
  disconnect(): void;
  close(): Promise<void>;
}>;
