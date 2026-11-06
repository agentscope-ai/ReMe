import type { Bubble } from "../shared/protocol.ts";

interface Candidate {
  id: string;
  text: string;
  snoozed?: boolean;
}

/** One reminder, with time counted only while visible and confirmed by the renderer. */
export class BubbleQueue {
  active: Bubble | null = null;
  private waiting: Candidate | null = null;
  private remaining = 0;
  private confirmation = 0;
  private confirmed = false;
  private visible = false;
  private previous = 0;
  private pausedAt = 0;
  private sequence = 0;
  private retryAt = 0;
  private shown: number[] = [];
  private lastShown = -Infinity;

  constructor(
    readonly cooldownMs = 10 * 60_000,
    private readonly onSuperseded: (id: string) => void = () => {},
  ) {}

  offer(candidate: Candidate): void {
    if (candidate.id === this.active?.id || candidate.id === this.waiting?.id)
      return;
    if (this.waiting?.snoozed && !candidate.snoozed) {
      this.onSuperseded(candidate.id);
      return;
    }
    if (this.waiting) this.onSuperseded(this.waiting.id);
    this.waiting = candidate;
  }

  private advance(now: number): void {
    const elapsed = Math.max(0, now - this.previous);
    if (this.visible && this.active) {
      if (!this.confirmed) this.confirmation -= elapsed;
      else if (!this.active.paused) this.remaining -= elapsed;
    }
    this.previous = now;
  }

  presented(id: string, presentationId: number, now: number): boolean {
    if (
      this.active?.id !== id ||
      this.active.presentationId !== presentationId ||
      this.confirmed
    )
      return false;
    this.advance(now);
    this.confirmed = true;
    this.lastShown = now;
    this.shown.push(now);
    return true;
  }

  pause(id: string, paused: boolean, now: number): void {
    if (this.active?.id !== id || this.active.paused === paused) return;
    this.advance(now);
    if (paused) this.pausedAt = now;
    this.active = { ...this.active, paused };
  }

  dismiss(id: string): void {
    if (this.active?.id === id) this.active = null;
    if (this.waiting?.id === id) this.waiting = null;
  }

  clear(): void {
    this.active = null;
    this.waiting = null;
    this.visible = false;
    this.retryAt = 0;
  }

  tick(now: number, blocked: boolean): Bubble | null {
    this.advance(now);
    if (this.active && !this.confirmed && this.confirmation <= 0) {
      this.waiting ??= { id: this.active.id, text: this.active.text };
      this.active = null;
      this.retryAt = now + 30_000;
    }
    if (
      this.active &&
      this.confirmed &&
      (this.remaining <= 0 ||
        (this.active.paused && now - this.pausedAt > 120_000))
    )
      this.active = null;
    this.shown = this.shown.filter((time) => now - time < 3_600_000);
    if (
      !blocked &&
      !this.active &&
      this.waiting &&
      now >= this.retryAt &&
      (this.waiting.snoozed ||
        (now - this.lastShown >= this.cooldownMs && this.shown.length < 3))
    ) {
      const candidate = this.waiting;
      this.waiting = null;
      this.remaining = Math.max(
        8_000,
        Math.min(15_000, 4_000 + [...candidate.text].length * 100),
      );
      this.confirmation = 5000;
      this.confirmed = false;
      this.active = {
        kind: "reminder",
        id: candidate.id,
        presentationId: ++this.sequence,
        text: candidate.text,
        paused: false,
      };
    }
    this.visible = !blocked && this.active !== null;
    return this.visible ? this.active : null;
  }
}
