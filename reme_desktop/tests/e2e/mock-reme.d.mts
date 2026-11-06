export function mockReMe(): Promise<{
  url: string;
  push(id: string, content?: string, completed?: boolean): void;
  activity(phase: string, source?: "chat" | "proactive", turnId?: string): void;
  cursors: (string | undefined)[];
  disconnect(): void;
  close(): Promise<void>;
}>;
