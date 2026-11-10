import type { WindowLayout } from "../shared/protocol.ts";

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

/** Geometry depends on the character and work area, never on streaming text. */
export function companionPlacement(
  anchor: { x: number; y: number },
  scale: number,
  surface: "status" | "message" | "menu" | null,
  area: Rect,
) {
  const margin = 16;
  const gap = 12;
  const padding = 10;
  const size = Math.max(
    1,
    Math.min(
      Math.round(180 * scale),
      area.width - margin * 2,
      area.height - margin * 2,
    ),
  );
  const x = Math.round(
    clamp(
      anchor.x,
      area.x + size / 2 + margin,
      area.x + area.width - size / 2 - margin,
    ),
  );
  const y = Math.round(
    clamp(anchor.y, area.y + size + margin, area.y + area.height - margin),
  );
  const body = {
    x: Math.round(x - size / 2),
    y: Math.round(y - size),
    width: size,
    height: size,
  };
  const inner = {
    x: area.x + margin,
    y: area.y + margin,
    width: area.width - margin * 2,
    height: area.height - margin * 2,
  };
  const status = surface === "status";
  const menu = surface === "menu";
  const preferredWidth = Math.round(
    menu
      ? 236
      : status
        ? clamp(size + 30, 180, 220)
        : clamp(size * 1.65, 260, 320),
  );
  const preferredHeight = menu
    ? 380
    : status
      ? 38
      : Math.round(clamp(size * 1.1, 180, 212));
  const spaces: (Rect & { side: WindowLayout["bubble"]["side"] })[] = [
    { ...inner, side: "top", height: Math.max(0, body.y - gap - inner.y) },
    {
      ...inner,
      side: "bottom",
      y: y + gap,
      height: Math.max(0, inner.y + inner.height - y - gap),
    },
    {
      ...inner,
      side: "right",
      x: body.x + size + gap,
      width: Math.max(0, inner.x + inner.width - body.x - size - gap),
    },
    { ...inner, side: "left", width: Math.max(0, body.x - gap - inner.x) },
  ];
  const fits = (width: number, height: number) => {
    const available = spaces.filter(
      (space) => space.width >= width && space.height >= height,
    );
    return (
      available.find(({ side }) => side === "top" || side === "bottom") ??
      available.sort((a, b) => b.width - a.width)[0]
    );
  };
  const space =
    fits(preferredWidth, preferredHeight) ??
    fits(status ? 140 : 240, status ? 38 : 160) ??
    spaces.reduce((best, next) =>
      next.width * next.height > best.width * best.height ? next : best,
    );
  const width = Math.max(1, Math.min(preferredWidth, space.width));
  const height = Math.max(1, Math.min(preferredHeight, space.height));
  const vertical = space.side === "top" || space.side === "bottom";
  const headY = body.y + size * 0.35;
  const bubbleX = vertical
    ? clamp(x - width / 2, space.x, space.x + space.width - width)
    : space.side === "left"
      ? body.x - gap - width
      : space.x;
  const bubbleY = vertical
    ? space.side === "top"
      ? body.y - gap - height
      : space.y
    : clamp(headY - height / 2, space.y, space.y + space.height - height);
  const tip = clamp(
    vertical ? x - bubbleX : headY - bubbleY,
    20,
    (vertical ? width : height) - 20,
  );
  const bubble = surface !== null;
  const left = Math.floor(
    Math.min(body.x, bubble ? bubbleX : body.x) - padding,
  );
  const top = Math.floor(Math.min(body.y, bubble ? bubbleY : body.y) - padding);
  const right = Math.ceil(
    Math.max(body.x + size, bubble ? bubbleX + width : body.x + size) + padding,
  );
  const bottom = Math.ceil(
    Math.max(y, bubble ? bubbleY + height : y) + padding,
  );
  return {
    anchor: { x, y },
    bounds: { x: left, y: top, width: right - left, height: bottom - top },
    layout: {
      body: { left: body.x - left, top: body.y - top, size },
      bubble: {
        left: bubbleX - left,
        top: bubbleY - top,
        width,
        height,
        side: space.side,
        tip,
      },
    },
  };
}
