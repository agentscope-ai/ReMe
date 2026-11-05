import type {
  ChatBlock,
  ChatMessage,
  DetailBlock,
  StreamChunk,
  StreamPayload,
  ToolBlock,
  StreamStatus,
} from "./types";

const detailTypes = new Set(["think", "data", "approval", "usage"]);

export async function chatStreamError(
  operation: () => Promise<void>,
  signal: AbortSignal,
  fallback: string,
): Promise<string | undefined> {
  try {
    await operation();
    return undefined;
  } catch (error) {
    if (signal.aborted) return undefined;
    return error instanceof Error ? error.message : fallback;
  }
}

export function decodeSseEvent(raw: string): StreamChunk | null {
  const data = raw
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");
  if (!data) return null;
  if (data === "[DONE]") return { chunk_type: "done", chunk: "", done: true };
  return JSON.parse(data) as StreamChunk;
}

export async function readChatStream(
  stream: ReadableStream<Uint8Array>,
  onChunk: (chunk: StreamChunk) => void,
  interruptedMessage: string,
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const emit = (event: string): boolean => {
    const chunk = decodeSseEvent(event);
    if (!chunk) return false;
    onChunk(chunk);
    return chunk.chunk_type === "done" || chunk.done;
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop() || "";
      for (const event of events) {
        if (emit(event)) return;
      }
      if (done) {
        if (emit(buffer)) return;
        throw new Error(interruptedMessage);
      }
    }
  } finally {
    // Cleanup must not replace the original transport or parsing error.
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

const payloadText = (payload: StreamPayload): string =>
  payload === null
    ? ""
    : typeof payload === "string"
    ? payload
    : JSON.stringify(payload, null, 2);

const appendError = (
  blocks: ChatBlock[],
  text: string,
  metadata?: Record<string, unknown>,
): ChatBlock[] => {
  if (!text) return blocks;
  const last = blocks.at(-1);
  if (
    last?.type === "error" &&
    last.metadata?.will_retry === metadata?.will_retry
  ) {
    return [...blocks.slice(0, -1), { ...last, text: last.text + text }];
  }
  return [
    ...blocks,
    { id: `error:${blocks.length}`, type: "error", text, metadata },
  ];
};

const settled = (status: string): status is "done" | "error" | "interrupted" =>
  ["done", "error", "interrupted"].includes(status);
const blockStatus = (status?: StreamStatus | null) =>
  status === "failed"
    ? "error"
    : status === "interrupted"
    ? "interrupted"
    : status === "waiting"
    ? "waiting"
    : "done";

export function finishChatMessage(
  message: ChatMessage,
  error?: string,
  { finishTools = true }: { finishTools?: boolean } = {},
): ChatMessage {
  const failed =
    Boolean(error) ||
    message.status === "failed" ||
    message.blocks?.some(
      (block) => block.type === "error" && block.metadata?.will_retry !== true,
    );
  const status = failed
    ? "failed"
    : message.status === "running"
    ? "interrupted"
    : message.status === "interrupted" || message.status === "waiting"
    ? message.status
    : "completed";
  let blocks = (message.blocks || []).map((block): ChatBlock => {
    if (block.type === "content" || block.type === "error") return block;
    if (!finishTools && block.type === "tool" && !settled(block.status))
      return block;
    return {
      ...block,
      status: settled(block.status) ? block.status : blockStatus(status),
      expanded: false,
    };
  });
  if (error) blocks = appendError(blocks, error);
  return { ...message, status, blocks };
}

function applyContent(blocks: ChatBlock[], chunk: StreamChunk): ChatBlock[] {
  if (chunk.chunk === null) return blocks;
  const text = payloadText(chunk.chunk);
  if (!text && (!chunk.event || chunk.event === "delta")) return blocks;
  const id = chunk.block_id || "";
  const index = id
    ? blocks.findIndex(
        (block) => block.type === "content" && block.id === `content:${id}`,
      )
    : blocks.length - 1;
  const current = blocks[index];
  if (current?.type === "content" && (id || index === blocks.length - 1)) {
    const next = [...blocks];
    next[index] = {
      ...current,
      text: chunk.event && chunk.event !== "delta" ? text : current.text + text,
      metadata: { ...current.metadata, ...chunk.metadata },
    };
    return next;
  }
  return [
    ...blocks,
    {
      id: `content:${id || blocks.length}`,
      type: "content",
      text,
      metadata: chunk.metadata,
    },
  ];
}

function detailId(blocks: ChatBlock[], chunk: StreamChunk): string {
  if (chunk.block_id) return `${chunk.chunk_type}:${chunk.block_id}`;
  if (chunk.chunk_type === "approval" && chunk.metadata?.review_id) {
    return `approval:${String(chunk.metadata.review_id)}`;
  }
  if (chunk.chunk_type === "usage") {
    const active = [...blocks]
      .reverse()
      .find((block) => block.type === "usage" && block.status === "streaming");
    if (active) return active.id;
  }
  const last = blocks.at(-1);
  if (
    last &&
    last.type !== "content" &&
    last.type !== "tool" &&
    last.type !== "error" &&
    last.sourceType === chunk.chunk_type &&
    last.status === "streaming"
  )
    return last.id;
  return `${chunk.chunk_type}:${blocks.length}`;
}

function applyDetail(blocks: ChatBlock[], chunk: StreamChunk): ChatBlock[] {
  const id = detailId(blocks, chunk);
  const index = blocks.findIndex((block) => block.id === id);
  const payloads =
    chunk.chunk !== null && (payloadText(chunk.chunk) || chunk.event === "end")
      ? [chunk.chunk]
      : [];
  const done =
    chunk.event === "end" ||
    (chunk.chunk_type === "usage" &&
      (chunk.input_tokens != null || chunk.output_tokens != null));
  const type = detailTypes.has(chunk.chunk_type)
    ? (chunk.chunk_type as DetailBlock["type"])
    : "unknown";
  const current = blocks[index];
  const nextBlock: DetailBlock =
    current &&
    current.type !== "content" &&
    current.type !== "tool" &&
    current.type !== "error"
      ? {
          ...current,
          payloads:
            chunk.event && chunk.event !== "delta" && chunk.chunk !== null
              ? payloads
              : [...current.payloads, ...payloads],
          status: done ? blockStatus(chunk.status) : "streaming",
          expanded: current.expanded,
          mediaType: chunk.media_type || current.mediaType,
          inputTokens: chunk.input_tokens ?? current.inputTokens,
          outputTokens: chunk.output_tokens ?? current.outputTokens,
          metadata: { ...current.metadata, ...chunk.metadata },
        }
      : {
          id,
          type,
          sourceType: chunk.chunk_type,
          payloads,
          status: done ? blockStatus(chunk.status) : "streaming",
          expanded: true,
          mediaType: chunk.media_type || undefined,
          inputTokens: chunk.input_tokens ?? undefined,
          outputTokens: chunk.output_tokens ?? undefined,
          metadata: chunk.metadata,
        };
  if (index < 0) return [...blocks, nextBlock];
  const next = [...blocks];
  next[index] = nextBlock;
  return next;
}

function applyTool(blocks: ChatBlock[], chunk: StreamChunk): ChatBlock[] {
  const last = blocks.at(-1);
  const toolId =
    chunk.tool_call_id ||
    chunk.block_id ||
    (last?.type === "tool" && !settled(last.status)
      ? last.id.slice("tool:".length)
      : `${blocks.length}`);
  const id = `tool:${toolId}`;
  const index = blocks.findIndex(
    (block) => block.type === "tool" && block.id === id,
  );
  const current = index >= 0 ? (blocks[index] as ToolBlock) : undefined;
  const result = chunk.chunk_type === "tool_result";
  const previous =
    (result ? current?.resultPayloads : current?.callPayloads) || [];
  const payloads =
    chunk.chunk === null
      ? previous
      : chunk.event && chunk.event !== "delta"
      ? [chunk.chunk]
      : [...previous, ...(payloadText(chunk.chunk) ? [chunk.chunk] : [])];
  // TOOL_CALL end finishes argument streaming; only TOOL_RESULT end finishes execution.
  const completed =
    chunk.status === "completed" ||
    chunk.status === "failed" ||
    chunk.status === "interrupted" ||
    (result &&
      chunk.event === "end" &&
      chunk.status !== "running" &&
      chunk.status !== "waiting");
  const status =
    current && (current.status === "error" || current.status === "interrupted")
      ? current.status
      : completed
      ? blockStatus(chunk.status)
      : current && settled(current.status)
      ? current.status
      : chunk.status === "waiting"
      ? "waiting"
      : result
      ? "running"
      : "calling";
  const nextBlock: ToolBlock = {
    id,
    type: "tool",
    name: chunk.tool_call_name || current?.name || "ReMe tool",
    callPayloads: result ? current?.callPayloads || [] : payloads,
    resultPayloads: result ? payloads : current?.resultPayloads || [],
    status,
    expanded: settled(status) ? false : current?.expanded ?? true,
    mediaType: chunk.media_type || current?.mediaType,
    metadata: { ...current?.metadata, ...chunk.metadata },
  };
  if (index < 0) return [...blocks, nextBlock];
  const next = [...blocks];
  next[index] = nextBlock;
  return next;
}

function reduceStreamChunk(
  message: ChatMessage,
  chunk: StreamChunk,
): ChatMessage {
  if (chunk.chunk_type === "reply_end") {
    const outcome =
      chunk.chunk &&
      typeof chunk.chunk === "object" &&
      !Array.isArray(chunk.chunk)
        ? (chunk.chunk as ChatMessage["reply"])
        : undefined;
    const answer = outcome?.result || "";
    const hasContent = (message.blocks || [])
      .filter((block) => block.type === "content")
      .map((block) => block.text)
      .join("")
      .endsWith(answer);
    const completed =
      answer && !hasContent
        ? {
            ...message,
            blocks: [
              ...(message.blocks || []),
              {
                id: `content:final:${message.id}`,
                type: "content" as const,
                text: answer,
              },
            ],
          }
        : message;
    const failed =
      chunk.status === "failed" &&
      !(completed.blocks || []).some(
        (block) =>
          block.type === "error" && block.metadata?.will_retry !== true,
      );
    let blocks = failed
      ? appendError(
          completed.blocks || [],
          String(
            outcome?.error ||
              chunk.metadata?.error ||
              chunk.metadata?.finished_reason ||
              "failed",
          ),
        )
      : completed.blocks;
    if (outcome?.structured_output != null) {
      blocks = [
        ...(blocks || []),
        {
          id: `structured:${message.id}`,
          type: "data",
          sourceType: "structured_output",
          payloads: [outcome.structured_output as StreamPayload],
          status: "done",
          expanded: false,
        },
      ];
    }
    return finishChatMessage(
      {
        ...completed,
        blocks,
        status: outcome?.status || chunk.status || undefined,
        reply: outcome,
      },
      undefined,
      { finishTools: chunk.event === "end" },
    );
  }
  if (chunk.chunk_type === "done" || chunk.done) {
    return finishChatMessage(message);
  }
  if (chunk.chunk_type === "reply_start")
    return { ...message, status: "running" };
  if (chunk.metadata?.visibility === "diagnostic") return message;

  let blocks = message.blocks || [];
  if (
    chunk.chunk_type === "think" ||
    (chunk.chunk_type === "content" && payloadText(chunk.chunk)) ||
    chunk.chunk_type === "tool_call"
  ) {
    const thinkingId =
      chunk.chunk_type === "think" ? detailId(blocks, chunk) : undefined;
    blocks = blocks.map(
      (block): ChatBlock =>
        block.type === "think" &&
        block.status === "streaming" &&
        block.id !== thinkingId
          ? { ...block, status: "done", expanded: false }
          : block,
    );
  }
  if (chunk.chunk_type === "content")
    return { ...message, blocks: applyContent(blocks, chunk) };
  if (chunk.chunk_type === "tool_call" || chunk.chunk_type === "tool_result") {
    return { ...message, blocks: applyTool(blocks, chunk) };
  }
  if (chunk.chunk_type === "error")
    return {
      ...message,
      blocks: appendError(blocks, payloadText(chunk.chunk), chunk.metadata),
    };
  return { ...message, blocks: applyDetail(blocks, chunk) };
}

export function applyStreamChunk(
  message: ChatMessage,
  chunk: StreamChunk,
): ChatMessage {
  return {
    ...reduceStreamChunk(message, chunk),
    events: [...(message.events || []), chunk],
  };
}

export function toggleChatBlock(
  message: ChatMessage,
  blockId: string,
  expanded: boolean,
): ChatMessage {
  return {
    ...message,
    blocks: (message.blocks || []).map((block) =>
      block.id === blockId && block.type !== "content" && block.type !== "error"
        ? { ...block, expanded }
        : block,
    ),
  };
}

export function formatStreamPayloads(payloads: StreamPayload[]): string {
  return payloads
    .map((payload, index) => {
      const separator =
        index > 0 &&
        (typeof payload !== "string" || typeof payloads[index - 1] !== "string")
          ? "\n\n"
          : "";
      return separator + payloadText(payload);
    })
    .join("");
}
