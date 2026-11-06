export function mockReMe(): Promise<{
  url: string;
  push(id: string, content?: string, completed?: boolean): void;
  cursors: (string | undefined)[];
  disconnect(): void;
  close(): Promise<void>;
}>;
