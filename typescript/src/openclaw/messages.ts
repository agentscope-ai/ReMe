import { createHash } from "node:crypto";

import type { ReMeMessage } from "../core/types.js";

interface MessageRecord {
  id?: unknown;
  role?: unknown;
  content?: unknown;
  created_at?: unknown;
  timestamp?: unknown;
}

/** Map an OpenClaw conversation id to a filename-safe ReMe session id. */
export function openClawSessionId(value: string): string {
  return `openclaw-${hash(value).slice(0, 24)}`;
}

/** Extract the last completed user/assistant pair from an OpenClaw history. */
export function captureLastTurn(
  messages: unknown[],
  sessionId: string,
  userPrompt?: string,
): ReMeMessage[] {
  let assistant: ReMeMessage | null = null;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const record = toRecord(messages[index]);
    if (!record) continue;
    if (!assistant && record.role === "assistant") {
      assistant = normalizeMessage(record, "assistant", sessionId, index);
      continue;
    }
    if (assistant && record.role === "user") {
      const user = normalizeMessage(
        record,
        "user",
        sessionId,
        index,
        userPrompt,
      );
      return user ? [user, assistant] : [];
    }
  }
  return [];
}

function normalizeMessage(
  value: MessageRecord,
  role: "user" | "assistant",
  sessionId: string,
  index: number,
  textOverride?: string,
): ReMeMessage | null {
  const text =
    textOverride?.trim() || stripAutoRecallContext(messageText(value.content));
  if (!text) return null;
  const nativeId =
    typeof value.id === "string" && value.id ? value.id : `${index}\n${text}`;
  const createdAt = timestamp(value.created_at ?? value.timestamp);
  return {
    id: `openclaw-${hash(`${sessionId}\n${nativeId}`).slice(0, 20)}`,
    name: role,
    role,
    content: [{ type: "text", text }],
    ...(createdAt ? { created_at: createdAt } : {}),
  };
}

function stripAutoRecallContext(value: string): string {
  const opening = '<reme-context source="auto-recall">';
  const start = value.indexOf(opening);
  if (start === -1) return value;
  const closing = "</reme-context>";
  const end = value.indexOf(closing, start + opening.length);
  if (end === -1) return value;
  return `${value.slice(0, start)}${value.slice(end + closing.length)}`.trim();
}

function messageText(content: unknown): string {
  if (typeof content === "string") return content.trim();
  if (!Array.isArray(content)) return "";
  return content
    .filter(
      (part): part is { type: "text"; text: string } =>
        isRecord(part) && part.type === "text" && typeof part.text === "string",
    )
    .map((part) => part.text.trim())
    .filter(Boolean)
    .join("\n\n")
    .trim();
}

function timestamp(value: unknown): string {
  if (typeof value === "string") {
    const time = Date.parse(value);
    return Number.isFinite(time) ? new Date(time).toISOString() : "";
  }
  if (typeof value === "number" && Number.isFinite(value))
    return new Date(value).toISOString();
  return "";
}

function toRecord(value: unknown): MessageRecord | null {
  return isRecord(value) ? value : null;
}

function hash(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
