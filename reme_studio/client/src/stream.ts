import { UiError } from "./i18n.ts";
import type { StreamChunk } from "./protocol.ts";

/** DONE terminates transport; only reply_end can settle a chat outcome. */
export async function readStream(
  stream: ReadableStream<Uint8Array>,
  onChunk: (chunk: StreamChunk) => void,
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const emit = (event: string) => {
    if (event.length > 8 * 1024 * 1024) throw new UiError("streamTooLarge");
    const data = event
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (!data) return false;
    const chunk: StreamChunk =
      data === "[DONE]"
        ? { chunk_type: "done", chunk: "", done: true }
        : JSON.parse(data);
    if (!chunk || typeof chunk.chunk_type !== "string")
      throw new UiError("invalidStream");
    onChunk(chunk);
    return chunk.chunk_type === "done" || chunk.done === true;
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop() ?? "";
      for (const event of events) if (emit(event)) return;
      if (buffer.length > 8 * 1024 * 1024) throw new UiError("streamTooLarge");
      if (done) {
        if (emit(buffer)) return;
        throw new UiError("streamInterrupted");
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
