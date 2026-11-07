export function toneWave(seconds?: number): Buffer;
export function mockReMe(options?: { speech?: boolean }): Promise<{
  url: string;
  push(id: string, content?: string, completed?: boolean): void;
  activity(phase: string, source?: "chat" | "proactive", turnId?: string): void;
  cursors: (string | undefined)[];
  ttsRequests: { text: string }[];
  asrRequests: { audio: string; audio_format: string; sample_rate: number }[];
  setSpeechError(value: boolean): void;
  setSpeechDelay(value: number): void;
  disconnect(): void;
  close(): Promise<void>;
}>;
