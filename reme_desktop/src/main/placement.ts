export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export const clamp = (n: number, min: number, max: number) =>
  Math.max(min, Math.min(Math.max(min, max), n));

/** Anchor is the canvas's bottom centre; bubble expansion never moves the character. */
export function companionPlacement(
  anchor: { x: number; y: number },
  scale: number,
  bubble: boolean,
  area: Rect,
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
  const width = Math.min(320, area.width - 24);
  let bubbleY = body.y - 188;
  const below = bubbleY < area.y + 8;
  if (below) bubbleY = y + 10;
  bubbleY = clamp(bubbleY, area.y + 8, area.y + area.height - 180 - 8);
  const bubbleX = clamp(
    x - width + 48,
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
  const bottom = Math.ceil(Math.max(y + 12, bubble ? bubbleY + 180 + 8 : 0));
  return {
    anchor: { x: Math.round(x), y: Math.round(y) },
    bounds: { x: left, y: top, width: right - left, height: bottom - top },
    layout: {
      body: { left: body.x - left, top: body.y - top, size },
      bubble: {
        left: bubbleX - left,
        top: bubbleY - top,
        width,
        below,
        tail: clamp(x - bubbleX, 20, width - 20),
      },
    },
  };
}
