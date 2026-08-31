import type {
  ChatBlock,
  ChatMessage,
  DetailBlock,
  StreamChunk,
  StreamPayload,
  ToolBlock,
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

const payloadText = (payload: StreamPayload): string =>
  typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);

const appendError = (blocks: ChatBlock[], text: string): ChatBlock[] => {
  if (!text) return blocks;
  const last = blocks.at(-1);
  if (last?.type === "error") {
    return [...blocks.slice(0, -1), { ...last, text: last.text + text }];
  }
  return [...blocks, { id: `error:${blocks.length}`, type: "error", text }];
};

export function finishChatMessage(
  message: ChatMessage,
  error?: string,
): ChatMessage {
  let blocks = (message.blocks || []).map((block): ChatBlock => {
    if (block.type === "content" || block.type === "error") return block;
    return {
      ...block,
      status: block.status === "error" ? "error" : "done",
      expanded: false,
    };
  });
  if (error) blocks = appendError(blocks, error);
  return { ...message, blocks };
}

function applyContent(blocks: ChatBlock[], chunk: StreamChunk): ChatBlock[] {
  const text = payloadText(chunk.chunk);
  if (!text) return blocks;
  const id = chunk.block_id || "";
  const index = id
    ? blocks.findIndex(
        (block) => block.type === "content" && block.id === `content:${id}`,
      )
    : blocks.length - 1;
  const current = blocks[index];
  if (current?.type === "content" && (id || index === blocks.length - 1)) {
    const next = [...blocks];
    next[index] = { ...current, text: current.text + text };
    return next;
  }
  return [
    ...blocks,
    { id: `content:${id || blocks.length}`, type: "content", text },
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
  const payloads = payloadText(chunk.chunk) ? [chunk.chunk] : [];
  const done =
    chunk.chunk_type === "usage" &&
    (chunk.input_tokens !== undefined || chunk.output_tokens !== undefined);
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
          payloads: [...current.payloads, ...payloads],
          status: done ? "done" : "streaming",
          expanded: true,
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
          status: done ? "done" : "streaming",
          expanded: true,
          mediaType: chunk.media_type,
          inputTokens: chunk.input_tokens,
          outputTokens: chunk.output_tokens,
          metadata: chunk.metadata,
        };
  if (index < 0) return [...blocks, nextBlock];
  const next = [...blocks];
  next[index] = nextBlock;
  return next;
}

function applyTool(blocks: ChatBlock[], chunk: StreamChunk): ChatBlock[] {
  const last = blocks.at(-1);
  const activeToolId =
    last?.type === "tool" && last.status !== "done" && last.status !== "error"
      ? last.id.slice("tool:".length)
      : undefined;
  const toolId =
    chunk.tool_call_id ||
    chunk.block_id ||
    activeToolId ||
    `${chunk.chunk_type}:${blocks.length}`;
  const id = `tool:${toolId}`;
  const index = blocks.findIndex(
    (block) => block.type === "tool" && block.id === id,
  );
  const current = index >= 0 ? (blocks[index] as ToolBlock) : undefined;
  const hasPayload = payloadText(chunk.chunk).length > 0;
  const result = chunk.chunk_type === "tool_result";
  const state = String(chunk.metadata?.state || "").toLowerCase();
  const failed = state.includes("error") || state.includes("fail");
  const nextBlock: ToolBlock = {
    id,
    type: "tool",
    name: chunk.tool_call_name || current?.name || "ReMe tool",
    callPayloads: result
      ? current?.callPayloads || []
      : [
          ...(current?.callPayloads || []),
          ...(hasPayload ? [chunk.chunk] : []),
        ],
    resultPayloads: result
      ? [
          ...(current?.resultPayloads || []),
          ...(hasPayload ? [chunk.chunk] : []),
        ]
      : current?.resultPayloads || [],
    status: failed
      ? "error"
      : result
      ? state
        ? "done"
        : "running"
      : "calling",
    expanded: true,
    mediaType: chunk.media_type || current?.mediaType,
    metadata: { ...current?.metadata, ...chunk.metadata },
  };
  if (index < 0) return [...blocks, nextBlock];
  const next = [...blocks];
  next[index] = nextBlock;
  return next;
}

export function applyStreamChunk(
  message: ChatMessage,
  chunk: StreamChunk,
): ChatMessage {
  if (chunk.chunk_type === "reply_end") {
    const answer =
      typeof chunk.metadata?.answer === "string" ? chunk.metadata.answer : "";
    const hasContent = (message.blocks || []).some(
      (block) => block.type === "content" && block.text.length > 0,
    );
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
    return finishChatMessage(completed);
  }
  if (chunk.chunk_type === "done" || chunk.done) {
    return finishChatMessage(message);
  }
  if (chunk.chunk_type === "reply_start") return message;

  const blocks = message.blocks || [];
  if (chunk.chunk_type === "content")
    return { ...message, blocks: applyContent(blocks, chunk) };
  if (chunk.chunk_type === "tool_call" || chunk.chunk_type === "tool_result") {
    return { ...message, blocks: applyTool(blocks, chunk) };
  }
  if (chunk.chunk_type === "error")
    return {
      ...message,
      blocks: appendError(blocks, payloadText(chunk.chunk)),
    };
  return { ...message, blocks: applyDetail(blocks, chunk) };
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
  return payloads.map(payloadText).join("");
}
