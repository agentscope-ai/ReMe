import type { DshSession } from "./types.js";
export { memoryGuidance } from "../core/guidance.js";

export const REME_PLUGIN_SOURCE = "reme-memory";

export function hasGuidance(session: DshSession): boolean {
  return (session.events || []).some((event) => {
    const source = isRecord(event.data) ? event.data.source : undefined;
    return (
      event.type === "user/message" &&
      isRecord(source) &&
      source.kind === "plugin" &&
      source.plugin === REME_PLUGIN_SOURCE &&
      source.form === "instructions"
    );
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
