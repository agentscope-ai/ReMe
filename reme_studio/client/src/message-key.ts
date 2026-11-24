import type { DisplayMessage } from "./protocol.ts";

export const messageKey = (message: DisplayMessage): string =>
  message.turnId && message.source === "chat"
    ? `turn:${message.turnId}:${message.role}`
    : `message:${message.id}`;
