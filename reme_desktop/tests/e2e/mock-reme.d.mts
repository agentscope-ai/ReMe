export function toneWave(seconds?: number): Buffer;
export function mockReMe(options?: {
  speech?: boolean;
  workspace?: boolean;
}): Promise<{
  url: string;
  push(id: string, content?: string, completed?: boolean): void;
  notificationWindow(value: {
    starts_at: string;
    ends_at: string;
    closing_message_id?: string | null;
  }): void;
  activity(phase: string, source?: "chat" | "proactive", turnId?: string): void;
  cursors: (string | undefined)[];
  ttsRequests: { text: string }[];
  asrRequests: { audio: string; audio_format: string; sample_rate: number }[];
  setSpeechError(value: boolean): void;
  setAsrText(value: string): void;
  setTags(value: [string, number][]): void;
  setDocument(path: string, content: string): void;
  setSpeechDelay(value: number): void;
  disconnect(): void;
  close(): Promise<void>;
}>;
