import { clamp, companionPlacement, type Rect } from "./placement.ts";

type Point = { x: number; y: number };
export interface HabitatDisplay {
  id: number;
  bounds: Rect;
  workArea: Rect;
}
export interface HabitatInput {
  displays: HabitatDisplay[];
  slot: Rect | null;
  window: Rect | null;
  scale: number;
  reducedMotion: boolean;
  surfaceHeight?: number;
}
export interface HabitatFrame {
  host: "desktop" | "app";
  anchor: Point;
  opacity: number;
  motion?: { id: number; startedAt: number; direction: 1 | -1 };
}

export function intersection(a: Rect, b: Rect): Rect {
  const x = Math.max(a.x, b.x),
    y = Math.max(a.y, b.y);
  return {
    x,
    y,
    width: Math.max(0, Math.min(a.x + a.width, b.x + b.width) - x),
    height: Math.max(0, Math.min(a.y + a.height, b.y + b.height) - y),
  };
}

const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);
const nearest = (point: Point, displays: HabitatDisplay[]) =>
  displays.reduce((best, display) => {
    const gap = (area: Rect) =>
      distance(point, {
        x: clamp(point.x, area.x, area.x + area.width),
        y: clamp(point.y - 1, area.y, area.y + area.height),
      });
    return gap(display.bounds) < gap(best.bounds) ? display : best;
  });
const anchored = (point: Point, scale: number, area: Rect) =>
  companionPlacement(point, scale, null, area).anchor;

/** Location owns travel only. Conversation and animation remain owned by the controller. */
export class CompanionHabitat {
  private host: HabitatFrame["host"] = "desktop";
  private anchor: Point;
  private home: Point;
  private pinned = false;
  private candidate = "";
  private since = 0;
  private sequence = 0;
  private route: {
    from: Point;
    entry: Point;
    to: Point;
    host: HabitatFrame["host"];
    displayId: number;
    startedAt: number;
    duration: number;
    teleport: boolean;
    id: number;
  } | null = null;

  constructor(anchor: Point) {
    this.anchor = this.home = { ...anchor };
  }

  get pending(): boolean {
    return Boolean(this.route || this.candidate);
  }
  get travelling(): boolean {
    return Boolean(this.route);
  }
  unpin(): void {
    this.pinned = false;
    this.candidate = "";
  }
  place(point: Point, pin = true): void {
    this.host = "desktop";
    this.anchor = this.home = { ...point };
    this.route = null;
    this.candidate = "";
    this.pinned = pin;
  }

  private frame(now: number): HabitatFrame {
    const route = this.route;
    if (!route) return { host: this.host, anchor: this.anchor, opacity: 1 };
    const elapsed = now - route.startedAt;
    if (route.teleport && elapsed < 150)
      return {
        host: "desktop",
        anchor: route.from,
        opacity: 1 - elapsed / 150,
      };
    const walking = Math.max(0, elapsed - (route.teleport ? 150 : 0));
    const progress = Math.min(1, walking / route.duration);
    const eased = progress * progress * (3 - 2 * progress);
    this.anchor = {
      x: route.entry.x + (route.to.x - route.entry.x) * eased,
      y: route.entry.y + (route.to.y - route.entry.y) * eased,
    };
    if (progress >= 1) {
      this.host = route.host;
      if (this.host === "desktop") this.home = { ...this.anchor };
      this.route = null;
      return { host: this.host, anchor: this.anchor, opacity: 1 };
    }
    return {
      host: "desktop",
      anchor: this.anchor,
      opacity: route.teleport ? Math.min(1, walking / 150) : 1,
      motion: {
        id: route.id,
        startedAt: route.startedAt + (route.teleport ? 150 : 0),
        direction: route.to.x < route.entry.x ? -1 : 1,
      },
    };
  }

  private travel(
    to: Point,
    host: HabitatFrame["host"],
    display: HabitatDisplay,
    input: HabitatInput,
    now: number,
  ): void {
    const source = nearest(this.anchor, input.displays);
    const teleport = source.id !== display.id;
    const size = 180 * input.scale;
    const entry = teleport
      ? anchored(
          {
            x: input.window
              ? input.window.x + input.window.width + size / 2
              : to.x + size,
            y: to.y,
          },
          input.scale,
          display.workArea,
        )
      : this.anchor;
    if (input.reducedMotion || (distance(entry, to) < 2 && !teleport)) {
      this.anchor = to;
      this.host = host;
      this.route = null;
      return;
    }
    this.route = {
      from: this.anchor,
      entry,
      to,
      host,
      displayId: display.id,
      teleport,
      startedAt: now,
      duration: clamp(distance(entry, to) / 0.32, 450, 1400),
      id: ++this.sequence,
    };
    this.host = "desktop";
  }

  update(input: HabitatInput, now = Date.now()): HabitatFrame {
    if (input.reducedMotion && this.route) {
      this.anchor = this.route.to;
      this.host = this.route.host;
      this.route = null;
    }
    let current = this.frame(now);
    const display = nearest(current.anchor, input.displays);
    // A removed monitor or a changed work area must never strand the character.
    if (
      (this.route &&
        !input.displays.some((item) => item.id === this.route!.displayId)) ||
      current.anchor.x < display.bounds.x ||
      current.anchor.x > display.bounds.x + display.bounds.width ||
      current.anchor.y < display.bounds.y ||
      current.anchor.y > display.bounds.y + display.bounds.height
    ) {
      this.place(
        anchored(current.anchor, input.scale, display.workArea),
        false,
      );
      current = this.frame(now);
    }
    if (!this.route && this.host === "desktop")
      this.anchor = anchored(this.anchor, input.scale, display.workArea);

    const slot = input.slot;
    const resident = this.host === "app" || this.route?.host === "app";
    const slotDisplay = slot
      ? nearest(
          { x: slot.x + slot.width / 2, y: slot.y + slot.height / 2 },
          input.displays,
        )
      : null;
    const area =
      slot && slotDisplay ? intersection(slot, slotDisplay.workArea) : null;
    const margin = resident ? 32 : 40;
    const available =
      !this.pinned &&
      area &&
      area.width >= 180 * input.scale + margin &&
      area.height >=
        180 * input.scale + Math.max(80, (input.surfaceHeight ?? 0) + 40);
    if (available && slotDisplay) {
      const target = anchored(
        { x: area.x + area.width / 2, y: area.y + area.height - 16 },
        input.scale,
        area,
      );
      if (this.host === "app") {
        this.anchor = target;
        return this.frame(now);
      }
      if (this.route?.host === "app") {
        // Track a moving/resizing window without restarting the walk clock.
        if (slotDisplay.id === this.route.displayId) this.route.to = target;
        else {
          this.route = null;
          this.candidate = "";
        }
      }
      if (!this.route) {
        const key = `${slotDisplay.id}:${Math.round(target.x)}:${Math.round(target.y)}`;
        if (key !== this.candidate) {
          this.candidate = key;
          this.since = now;
        }
        if (now - this.since >= 300) {
          this.candidate = "";
          this.travel(target, "app", slotDisplay, input, now);
        }
      }
    } else {
      this.candidate = "";
      if (resident) {
        const size = 180 * input.scale;
        const home =
          nearest(this.home, input.displays).id === display.id
            ? this.home
            : {
                x: display.workArea.x + display.workArea.width - size / 2 - 16,
                y: display.workArea.y + display.workArea.height - 24,
              };
        const candidates = [
          home,
          ...[0, 1].map((side) => ({
            x:
              display.workArea.x +
              (side ? display.workArea.width - size / 2 - 16 : size / 2 + 16),
            y: display.workArea.y + display.workArea.height - 24,
          })),
        ].map((point) => anchored(point, input.scale, display.workArea));
        const overlap = (point: Point) => {
          const rect = input.window
            ? intersection(
                {
                  x: point.x - size / 2,
                  y: point.y - size,
                  width: size,
                  height: size,
                },
                input.window,
              )
            : null;
          return rect ? rect.width * rect.height : 0;
        };
        candidates.sort((a, b) => overlap(a) - overlap(b));
        this.travel(candidates[0], "desktop", display, input, now);
      }
    }
    return this.frame(now);
  }
}
