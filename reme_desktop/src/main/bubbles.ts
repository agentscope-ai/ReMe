import type { Bubble } from "../shared/protocol.ts";

interface Candidate {
  id: string;
  text: string;
  snoozed?: boolean;
}

/** One visible bubble; an offline backlog remains unread instead of playing in a burst. */
export class BubbleQueue {
  active: Bubble | null = null;
  private waiting: Candidate | null = null;
  private deadline = 0;
  private remaining = 0;
  private shown: number[] = [];
  private lastShown = -Infinity;
  private pausedAt = 0;

  constructor(
    private readonly onPresented: (id: string) => void,
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

  pause(id: string, paused: boolean, now: number): void {
    if (this.active?.id !== id || this.active.paused === paused) return;
    if (paused) {
      this.remaining = Math.max(0, this.deadline - now);
      this.pausedAt = now;
    } else this.deadline = now + this.remaining;
    this.active = { ...this.active, paused };
  }

  dismiss(id: string): void {
    if (this.active?.id === id) this.active = null;
    if (this.waiting?.id === id) this.waiting = null;
  }

  clear(): void {
    this.active = null;
    this.waiting = null;
  }

  tick(now: number, blocked: boolean): Bubble | null {
    if (blocked) {
      this.active = null;
      return null;
    }
    if (
      this.active &&
      ((!this.active.paused && now >= this.deadline) ||
        (this.active.paused && now - this.pausedAt > 120_000))
    )
      this.active = null;
    this.shown = this.shown.filter((time) => now - time < 3_600_000);
    if (
      !this.active &&
      this.waiting &&
      (this.waiting.snoozed ||
        (now - this.lastShown >= this.cooldownMs && this.shown.length < 3))
    ) {
      const candidate = this.waiting;
      this.waiting = null;
      this.remaining = Math.max(
        8_000,
        Math.min(15_000, 4_000 + [...candidate.text].length * 100),
      );
      this.deadline = now + this.remaining;
      this.lastShown = now;
      this.shown.push(now);
      this.active = { id: candidate.id, text: candidate.text, paused: false };
      this.onPresented(candidate.id);
    }
    return this.active;
  }
}
