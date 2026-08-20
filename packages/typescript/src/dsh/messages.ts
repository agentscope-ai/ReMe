import { createHash } from "node:crypto";

import type { ReMeMessage } from "../core/types.js";
import type { SessionEvent } from "./types.js";

interface MessageLike {
  id?: unknown;
  content?: unknown;
  source?: Record<string, unknown>;
}

export function remeSessionId(sessionId: string): string {
  const digest = createHash("sha256")
    .update(String(sessionId))
    .digest("hex")
    .slice(0, 24);
  return `dsh-${digest}`;
}

export function captureMessage(
  event: SessionEvent,
  sessionId: string,
): ReMeMessage | null {
  const message = eventMessage(event);
  if (!message || message.source?.kind === "plugin") return null;
  if (event.type === "user/message" && message.source?.kind !== "user")
    return null;

  const role = event.type === "assistant/message" ? "assistant" : "user";
  const text = messageText(message);
  if (!text) return null;
  const suffix = Number.isSafeInteger(event.seq)
    ? String(event.seq)
    : stableSuffix(message, text);
  const createdAt = eventTime(event);
  return {
    id: `dsh-${shortHash(sessionId)}-${suffix}`,
    name: role,
    role,
    content: [{ type: "text", text }],
    ...(createdAt ? { created_at: createdAt } : {}),
  };
}

export function messageText(message: MessageLike): string {
  if (typeof message.content === "string") return message.content.trim();
  if (!Array.isArray(message.content)) return "";
  return message.content
    .filter(
      (part): part is { type: "text"; text: string } =>
        isRecord(part) && part.type === "text" && typeof part.text === "string",
    )
    .map((part) => part.text.trim())
    .filter(Boolean)
    .join("\n\n")
    .trim();
}

export function messagesDay(messages: ReMeMessage[], timezone: string): string {
  const days = messages
    .map((message) => timestampDay(message.created_at, timezone))
    .filter(Boolean);
  return days.sort().at(-1) || "";
}

function timestampDay(value: string | undefined, timezone: string): string {
  if (!value) return "";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return value.slice(0, 10);
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: timezone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);
  const part = (type: Intl.DateTimeFormatPartTypes) =>
    parts.find((item) => item.type === type)?.value || "";
  return `${part("year")}-${part("month")}-${part("day")}`;
}

function eventMessage(event: SessionEvent): MessageLike | null {
  if (event.type === "user/message") return toMessage(event.data);
  if (event.type === "assistant/message" && isRecord(event.data))
    return toMessage(event.data.message);
  return null;
}

function toMessage(value: unknown): MessageLike | null {
  if (!isRecord(value)) return null;
  return {
    id: value.id,
    content: value.content,
    source: isRecord(value.source) ? value.source : undefined,
  };
}

function eventTime(event: SessionEvent): string {
  if (!Number.isFinite(event.time) || (event.time ?? -1) < 0) return "";
  return new Date(event.time as number).toISOString();
}

function shortHash(value: unknown): string {
  return createHash("sha256").update(String(value)).digest("hex").slice(0, 12);
}

function stableSuffix(message: MessageLike, text: string): string {
  return shortHash(`${String(message.id || "")}\n${text}`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
