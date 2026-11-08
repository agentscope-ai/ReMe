import { applyStreamChunk, finishChatMessage } from "./chat-stream.ts";
import type { ChatMessage, SessionSnapshot, WorkspaceTab } from "./types";

export function mergeSessionTab(
  tab: Extract<WorkspaceTab, { type: "agent" }>,
  snapshot: SessionSnapshot,
  expectedLastId?: string,
) {
  if (
    tab.streaming ||
    tab.sessionId !== snapshot.session_id ||
    tab.messages.at(-1)?.id !== expectedLastId
  )
    return tab;
  const messages = mergeSessionMessages(tab.messages, snapshot);
  return messages === tab.messages ? tab : { ...tab, messages };
}

/** Server history owns ordering; streamed blocks retain their detail and expansion state. */
export function mergeSessionMessages(
  current: ChatMessage[],
  snapshot: SessionSnapshot,
): ChatMessage[] {
  const messages = snapshot.messages.map((entry): ChatMessage => {
    const previous = current.find(
      (message) =>
        message.id === entry.id ||
        (entry.source === "chat" &&
          entry.turn_id &&
          message.turnId === entry.turn_id &&
          message.role === entry.role),
    );
    let message: ChatMessage = {
      id: entry.id,
      role: entry.role,
      content: entry.content,
      source: entry.source,
      createdAt: entry.created_at,
      turnId: entry.turn_id || undefined,
      blocks: [],
    };
    if (entry.role === "assistant") {
      const streamed = previous?.blocks
        ?.filter((block) => block.type === "content")
        .map((block) => block.text)
        .join("");
      const settled = previous?.blocks?.every(
        (block) =>
          block.type === "content" ||
          (block.type !== "error" && block.status === "done"),
      );
      const chunks = entry.chunks.length
        ? entry.chunks
        : [
            {
              chunk_type: "content",
              chunk: entry.content,
              done: false,
            },
          ];
      for (const chunk of chunks) message = applyStreamChunk(message, chunk);
      if (previous?.reply) {
        message = applyStreamChunk(message, {
          chunk_type: "reply_end",
          event: "end",
          chunk: previous.reply,
          status: previous.reply.status,
          done: false,
        });
      }
      if (entry.completed) message = finishChatMessage(message);
      if (previous?.events?.length) {
        message.events = previous.events;
      }
      const failed = message.blocks?.some(
        (block) =>
          block.type === "error" ||
          (block.type !== "content" && block.status === "error"),
      );
      if (
        previous?.blocks?.length &&
        streamed === entry.content &&
        entry.completed &&
        settled &&
        !failed
      ) {
        message.blocks = previous.blocks;
      }
    }
    return message;
  });
  // A disconnect can hide reply_start even though Codex accepted the request.
  // Match pending prompts in order, after the last known native message, so an
  // older identical prompt cannot consume a new rejected request.
  const recovered = new Set<ChatMessage>();
  const knownTurns = new Set(current.map((message) => message.turnId));
  let after = -1;
  current.forEach((message, index) => {
    const nativeIndex = messages.findIndex(
      (entry) =>
        entry.id === message.id ||
        (entry.source === "chat" &&
          entry.turnId &&
          entry.turnId === message.turnId &&
          entry.role === message.role),
    );
    if (nativeIndex >= 0) {
      after = Math.max(after, nativeIndex);
    } else if (message.localOnly && message.role === "user") {
      const acceptedIndex = messages.findIndex(
        (entry, position) =>
          position > after &&
          entry.source === "chat" &&
          entry.role === "user" &&
          !knownTurns.has(entry.turnId) &&
          entry.content === message.content,
      );
      if (acceptedIndex < 0) return;
      after = acceptedIndex;
      recovered.add(message);
      const assistant = current[index + 1];
      if (assistant?.localOnly && assistant.role === "assistant")
        recovered.add(assistant);
    }
  });
  // Requests absent from native history still need their visible error.
  messages.push(
    ...current.filter(
      (message) => message.localOnly && !recovered.has(message),
    ),
  );
  return JSON.stringify(messages) === JSON.stringify(current)
    ? current
    : messages;
}
