import type { DesktopState } from "./protocol.ts";

export interface DisplayMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  proactive?: boolean;
  streaming?: boolean;
  error?: string;
}

export function displayMessages(state: DesktopState): DisplayMessage[] {
  const pending =
    state.pending?.sessionId === state.sessionId ? state.pending : null;
  const result: DisplayMessage[] = state.messages.map((message) => ({
    id: message.id,
    role: message.role,
    content: message.content,
    proactive: message.source === "proactive",
    ...(pending?.turnId &&
    message.role === "assistant" &&
    message.turn_id === pending.turnId
      ? {
          content: message.completed
            ? message.content
            : pending.text || message.content,
          streaming: pending.streaming && !message.completed,
          error: pending.error,
        }
      : {}),
  }));
  if (!pending) return result;
  const matches = (role: "user" | "assistant") =>
    pending.turnId &&
    state.messages.some(
      (message) => message.role === role && message.turn_id === pending.turnId,
    );
  if (!matches("user"))
    result.push({
      id: `local-user:${pending.id}`,
      role: "user",
      content: pending.query,
    });
  if (!matches("assistant"))
    result.push({
      id: `local-assistant:${pending.id}`,
      role: "assistant",
      content: pending.text,
      streaming: pending.streaming,
      error: pending.error,
    });
  return result;
}
