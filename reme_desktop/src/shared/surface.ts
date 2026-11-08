import type { CompanionSurface } from "./protocol.ts";

export interface SurfaceSize {
  key: string;
  width: number;
  height: number;
}

export const previewContent = (content: string): string =>
  content.length > 400 ? `${content.slice(0, 400)}…` : content;

/** Identify the visible content so late measurements cannot resize a replacement. */
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
