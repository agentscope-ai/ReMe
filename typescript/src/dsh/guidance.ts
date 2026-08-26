import type { DshSession } from "./types.js";
export { memoryGuidance } from "../core/guidance.js";

export const REME_PLUGIN_SOURCE = "reme-memory";

export function hasGuidance(
  session: DshSession,
  pendingMessages: readonly unknown[] = [],
): boolean {
  return (
    (session.events || []).some(
      (event) => event.type === "user/message" && isGuidance(event.data),
    ) || pendingMessages.some(isGuidance)
  );
}

function isGuidance(value: unknown): boolean {
  const source = isRecord(value) ? value.source : undefined;
  return (
    isRecord(source) &&
    source.kind === "plugin" &&
    source.plugin === REME_PLUGIN_SOURCE &&
    source.form === "instructions"
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
