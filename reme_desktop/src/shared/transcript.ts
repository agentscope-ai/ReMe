import type {
  ItemStatus,
  StreamChunk,
  TranscriptItem,
  ToolItem,
} from "./protocol.ts";
import { toolActions } from "./tool-activity.ts";

export interface ReplyProgress {
  text: string;
  items: TranscriptItem[];
  turnId?: string;
  sessionId?: string;
  status?: NonNullable<StreamChunk["status"]>;
  error?: string;
}

export const emptyReply = (): ReplyProgress => ({ text: "", items: [] });
const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
const printable = (value: unknown): string =>
  value == null
    ? ""
    : typeof value === "string"
      ? value
      : JSON.stringify(value, null, 2);
const settled = (status: ItemStatus) =>
  ["completed", "failed", "interrupted", "unknown"].includes(status);

/** Live deltas and restored snapshots use exactly the same ordered item reducer. */
export function reduceReply(
  previous: ReplyProgress,
  chunk: StreamChunk,
): ReplyProgress {
  const next = {
    ...previous,
    sessionId: chunk.session_id ?? previous.sessionId,
    turnId: chunk.turn_id ?? previous.turnId,
  };
  const payload = record(chunk.chunk);
  const metadata = chunk.metadata ?? {};
  const native = record(metadata.native);
  const id =
    chunk.tool_call_id ??
    chunk.block_id ??
    chunk.message_id ??
    chunk.chunk_type;
  const index = next.items.findIndex((item) => item.id === id);
  const current = next.items[index];
  if (
    current &&
    settled(current.status) &&
    chunk.event !== "end" &&
    ["content", "think", "tool_call", "tool_result", "data"].includes(
      chunk.chunk_type,
    )
  )
    return previous;
  const status =
    chunk.status ?? (chunk.event === "end" ? "completed" : "running");
  let item: TranscriptItem | undefined;

  if (chunk.chunk_type === "content" || chunk.chunk_type === "think") {
    if (
      metadata.visibility === "diagnostic" ||
      (chunk.chunk_type === "think" &&
        !["reasoning_summary", "plan"].includes(String(metadata.kind)))
    )
      return next;
    const type = chunk.chunk_type === "content" ? "text" : "summary";
    const text = current?.type === type ? current.text : "";
    const phase = metadata.phase ?? native.phase;
    const block = {
      id,
      status,
      text:
        chunk.chunk == null
          ? text
          : chunk.event && chunk.event !== "delta"
            ? printable(chunk.chunk)
            : text + printable(chunk.chunk),
      phase:
        phase === "commentary" || phase === "final_answer"
          ? (phase as "commentary" | "final_answer")
          : current?.type === type
            ? current.phase
            : undefined,
    };
    item =
      type === "text"
        ? { ...block, type }
        : {
            ...block,
            type,
            kind: metadata.kind === "plan" ? "plan" : "reasoning",
          };
  } else if (
    chunk.chunk_type === "tool_call" ||
    chunk.chunk_type === "tool_result"
  ) {
    const tool: ToolItem =
      current?.type === "tool"
        ? current
        : {
            id,
            type: "tool",
            name: chunk.tool_call_name ?? "tool",
            input: "",
            output: "",
            status: "running",
            actions: [],
          };
    const result = chunk.chunk_type === "tool_result";
    const failed =
      payload.is_error === true ||
      payload.success === false ||
      Boolean(payload.error) ||
      (typeof (payload.exitCode ?? payload.exit_code) === "number" &&
        (payload.exitCode ?? payload.exit_code) !== 0);
    const input =
      payload.type === "fileChange" && Array.isArray(payload.changes)
        ? payload.changes
            .map((change) => {
              const { path, diff } = record(change);
              return [path, diff]
                .filter((value) => typeof value === "string")
                .join("\n");
            })
            .join("\n\n")
        : (payload.arguments ?? payload.input ?? payload.command);
    const output =
      payload.type === "commandExecution" || payload.type === "fileChange"
        ? (payload.aggregatedOutput ?? payload.error ?? "")
        : payload.type === "mcpToolCall"
          ? (payload.error ?? payload.result ?? "")
          : (payload.result ?? payload.content ?? chunk.chunk);
    const content = result ? output : (input ?? chunk.chunk);
    const field = result ? "output" : "input";
    const name =
      chunk.tool_call_name ??
      (typeof payload.tool === "string"
        ? payload.tool
        : typeof payload.name === "string"
          ? payload.name
          : tool.name);
    item = {
      ...tool,
      name,
      ...(input !== undefined ? { input: printable(input) } : {}),
      actions:
        (!result && chunk.event !== "delta") ||
        !tool.actions.length ||
        payload.type === "commandExecution" ||
        payload.type === "fileChange" ||
        Array.isArray(payload.commandActions)
          ? toolActions(name, chunk.chunk)
          : tool.actions,
      status: failed
        ? "failed"
        : settled(tool.status) && (!result || chunk.event === "delta")
          ? tool.status
          : result
            ? status
            : "running",
      [field]:
        chunk.chunk == null
          ? tool[field]
          : chunk.event === "delta" && typeof content === "string"
            ? tool[field] + content
            : printable(content),
    };
  } else if (
    chunk.chunk_type === "data" &&
    metadata.visibility !== "diagnostic" &&
    (chunk.media_type || payload.url || payload.type === "imageView")
  ) {
    let url =
      typeof payload.url === "string"
        ? payload.url
        : typeof payload.path === "string"
          ? payload.path
          : typeof chunk.chunk === "string"
            ? chunk.chunk
            : "";
    if (
      chunk.media_type?.startsWith("image/") &&
      typeof chunk.chunk === "string" &&
      !/^(data:|https?:|file:|\/)/.test(url)
    )
      url = `data:${chunk.media_type};base64,${url}`;
    if (url)
      item = {
        id,
        type: "artifact",
        status,
        url,
        mediaType: chunk.media_type ?? "application/octet-stream",
        label:
          typeof payload.name === "string"
            ? payload.name
            : url.startsWith("data:")
              ? "Image"
              : url,
      };
  } else if (chunk.chunk_type === "reply_end") {
    next.status = chunk.status ?? (payload.status as ReplyProgress["status"]);
    if (typeof payload.error === "string" && payload.error)
      next.error = payload.error;
    // A reply outcome can correct the final text, while preserving earlier commentary.
    const final =
      next.items.findLast(
        (item) => item.type === "text" && item.phase === "final_answer",
      ) ?? next.items.findLast((item) => item.type === "text");
    if (typeof payload.result === "string" && (payload.result || !final)) {
      const finalId = final?.id ?? `${next.turnId ?? "reply"}:final`;
      next.items = final
        ? next.items.map((item) =>
            item === final
              ? {
                  ...final,
                  type: "text",
                  text: payload.result as string,
                  phase: "final_answer",
                  status: "completed",
                }
              : item,
          )
        : [
            ...next.items,
            {
              id: finalId,
              type: "text",
              text: payload.result,
              phase: "final_answer",
              status: "completed",
            },
          ];
    }
    next.items = next.items.map((item) =>
      item.status === "running" && item.type !== "tool"
        ? { ...item, status: next.status ?? "interrupted" }
        : item,
    );
  } else if (chunk.chunk_type === "error" && metadata.will_retry !== true) {
    next.error = typeof chunk.chunk === "string" ? chunk.chunk : "Reply failed";
  }
  if (item)
    next.items =
      index < 0
        ? [...next.items, item]
        : next.items.map((value, i) => (i === index ? item! : value));
  next.text = next.items
    .filter((item) => item.type === "text")
    .map((item) => item.text)
    .join("\n\n");
  return next;
}

export const readReply = (chunks: StreamChunk[]): ReplyProgress =>
  chunks.reduce(reduceReply, emptyReply());

export function finishItems(
  items: TranscriptItem[],
  status: ItemStatus,
): TranscriptItem[] {
  return items.map((item) =>
    settled(item.status)
      ? item
      : {
          ...item,
          status:
            status === "completed" && item.type === "tool" ? "unknown" : status,
        },
  );
}
