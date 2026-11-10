import type { CompanionSurface } from "./protocol.ts";

export const previewContent = (content: string): string =>
  content.length > 400 ? `${content.slice(0, 400)}…` : content;

/** Identify visible content changes independently of the window's geometry. */
export function surfaceKey(surface: CompanionSurface | null): string {
  if (!surface) return "";
  return JSON.stringify([
    surface.kind,
    surface.kind === "message" ? surface.message.id : surface.id,
    surface.kind === "message" ? surface.presentationId : null,
    surface.kind === "message"
      ? previewContent(surface.message.content)
      : surface.text,
    surface.kind === "message" ? surface.message.status : null,
    surface.kind === "message" ? surface.message.streaming : null,
    surface.kind === "message" ? surface.paused : null,
  ]);
}
