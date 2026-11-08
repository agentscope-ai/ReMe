export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export const clamp = (n: number, min: number, max: number) =>
  Math.max(min, Math.min(Math.max(min, max), n));

/** First-open proportions follow the display; restored bounds stay inside its work area. */
export function workspacePlacement(
  area: Rect,
  display: Rect,
  saved?: Rect,
): Rect {
  const ratio = display.width / display.height;
  const height = Math.round(
    Math.min(area.height * 0.82, (area.width * 0.82) / ratio, 920),
  );
  const width = Math.round(height * ratio);
  const bounds = saved ?? {
    width,
    height,
    x: area.x + (area.width - width) / 2,
    y: area.y + (area.height - height) / 2,
  };
  const w = Math.min(Math.max(500, bounds.width), area.width);
  const h = Math.min(Math.max(400, bounds.height), area.height);
  return {
    width: w,
    height: h,
    x: Math.round(clamp(bounds.x, area.x, area.x + area.width - w)),
    y: Math.round(clamp(bounds.y, area.y, area.y + area.height - h)),
  };
}

/** Anchor is the canvas's bottom centre; bubble expansion never moves the character. */
export function companionPlacement(
  anchor: { x: number; y: number },
  scale: number,
  surface: "status" | "message" | null,
  area: Rect,
  measured?: { width: number; height: number },
) {
  const size = Math.round(180 * scale);
  const x = clamp(
    anchor.x,
    area.x + size / 2 + 16,
    area.x + area.width - size / 2 - 16,
  );
  const y = clamp(anchor.y, area.y + size + 16, area.y + area.height - 16);
  const body = {
    x: Math.round(x - size / 2),
    y: Math.round(y - size),
    width: size,
    height: size,
  };
  const bubble = surface !== null;
  const maxWidth = Math.max(
    1,
    Math.min(surface === "status" ? 280 : 320, area.width - 24),
  );
  const width = Math.min(measured?.width ?? maxWidth, maxWidth);
  const height = Math.min(
    measured?.height ?? (surface === "status" ? 36 : 180),
    Math.max(1, area.height - 24),
  );
  const gap = 6;
  let bubbleY = body.y - height - gap;
  const below = bubbleY < area.y + 8;
  if (below) bubbleY = y + gap;
  bubbleY = clamp(bubbleY, area.y + 8, area.y + area.height - height - 8);
  const bubbleX = clamp(
    x - width / 2,
    area.x + 12,
    area.x + area.width - width - 12,
  );
  const left = Math.floor(
    Math.min(body.x - 12, bubble ? bubbleX - 8 : body.x - 12),
  );
  const top = Math.floor(
    Math.min(body.y - 12, bubble ? bubbleY - 8 : body.y - 12),
  );
  const right = Math.ceil(
    Math.max(body.x + size + 12, bubble ? bubbleX + width + 8 : 0),
  );
  const bottom = Math.ceil(Math.max(y + 12, bubble ? bubbleY + height + 8 : 0));
  return {
    anchor: { x: Math.round(x), y: Math.round(y) },
    bounds: { x: left, y: top, width: right - left, height: bottom - top },
    layout: {
      body: { left: body.x - left, top: body.y - top, size },
      bubble: {
        left: bubbleX - left,
        top: bubbleY - top,
        width,
        maxWidth,
        height,
      },
    },
  };
}
