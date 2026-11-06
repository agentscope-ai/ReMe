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

export interface ReplyProgress {
  text: string;
  turnId?: string;
  sessionId?: string;
  status?: StreamChunk["status"];
  error?: string | UiError;
  blocks: Map<string, string>;
}

export function reduceReply(
  previous: ReplyProgress,
  chunk: StreamChunk,
): ReplyProgress {
  const next = { ...previous, blocks: new Map(previous.blocks) };
  if (chunk.session_id) next.sessionId = chunk.session_id;
  if (chunk.turn_id) next.turnId = chunk.turn_id;
  if (chunk.chunk_type === "content" && typeof chunk.chunk === "string") {
    const key = chunk.block_id ?? "content";
    next.blocks.set(
      key,
      chunk.event && chunk.event !== "delta"
        ? chunk.chunk
        : (next.blocks.get(key) ?? "") + chunk.chunk,
    );
    next.text = [...next.blocks.values()].join("");
  } else if (chunk.chunk_type === "reply_end") {
    const reply =
      chunk.chunk && typeof chunk.chunk === "object"
        ? (chunk.chunk as Record<string, unknown>)
        : {};
    next.status = chunk.status ?? (reply.status as ReplyProgress["status"]);
    if (typeof reply.result === "string") next.text = reply.result;
    if (typeof reply.error === "string" && reply.error)
      next.error = reply.error;
  } else if (
    chunk.chunk_type === "error" &&
    chunk.metadata?.will_retry !== true
  ) {
    next.error =
      typeof chunk.chunk === "string" ? chunk.chunk : new UiError("replyError");
  }
  return next;
}
