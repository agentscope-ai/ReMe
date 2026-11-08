import type { CompanionSurface } from "./protocol.ts";

export interface SurfaceSize {
  key: string;
  width: number;
  height: number;
}

/** Identify the visible content so late measurements cannot resize a replacement. */
export function surfaceKey(surface: CompanionSurface | null): string {
  if (!surface) return "";
  return JSON.stringify([
    surface.kind,
    surface.id,
    surface.kind === "reminder" ? surface.presentationId : null,
    surface.text.slice(0, 400),
  ]);
}
