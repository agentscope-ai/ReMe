export interface Reminder {
  id: string;
  text: string;
  presentationId: number;
  paused: boolean;
}

interface Candidate {
  id: string;
  text: string;
  snoozed?: boolean;
}

/** One reminder, with time counted only while visible and confirmed by the renderer. */
export class BubbleQueue {
  active: Reminder | null = null;
  private waiting: Candidate[] = [];
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
  private speaking = false;

  constructor(
    readonly cooldownMs = 10 * 60_000,
    private readonly onSuperseded: (id: string) => void = () => {},
    private readonly maxPerHour = 3,
    private readonly coalesce = true,
  ) {}

  offer(candidate: Candidate): void {
    if (
      candidate.id === this.active?.id ||
      this.waiting.some((item) => item.id === candidate.id)
    )
      return;
    if (this.coalesce) {
      if (this.waiting[0]?.snoozed && !candidate.snoozed) {
        this.onSuperseded(candidate.id);
        return;
      }
      for (const item of this.waiting) this.onSuperseded(item.id);
      this.waiting = [];
    }
    this.waiting.push(candidate);
  }

  private advance(now: number): void {
    const elapsed = Math.max(0, now - this.previous);
    if (this.visible && this.active) {
      if (!this.confirmed) this.confirmation -= elapsed;
      else if (!this.active.paused && !this.speaking) this.remaining -= elapsed;
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

  holdForSpeech(speaking: boolean, now: number): void {
    if (speaking === this.speaking) return;
    this.advance(now);
    this.speaking = speaking;
  }

  dismiss(id: string): void {
    if (this.active?.id === id) this.active = null;
    this.waiting = this.waiting.filter((item) => item.id !== id);
  }

  retain(ids: Set<string>): void {
    if (this.active && !ids.has(this.active.id)) this.active = null;
    this.waiting = this.waiting.filter((item) => ids.has(item.id));
  }

  clear(): void {
    this.active = null;
    this.waiting = [];
    this.visible = false;
    this.retryAt = 0;
    this.speaking = false;
  }

  tick(now: number, blocked: boolean): Reminder | null {
    this.advance(now);
    if (this.active && !this.confirmed && this.confirmation <= 0) {
      if (!this.coalesce || !this.waiting.length)
        this.waiting.unshift({ id: this.active.id, text: this.active.text });
      this.active = null;
      this.retryAt = now + 30_000;
    }
    if (
      this.active &&
      this.confirmed &&
      (this.remaining <= 0 ||
        (this.active.paused && !this.speaking && now - this.pausedAt > 120_000))
    )
      this.active = null;
    this.shown = this.shown.filter((time) => now - time < 3_600_000);
    if (
      !blocked &&
      !this.active &&
      this.waiting.length > 0 &&
      now >= this.retryAt &&
      (this.waiting[0].snoozed ||
        (now - this.lastShown >= this.cooldownMs &&
          this.shown.length < this.maxPerHour))
    ) {
      const candidate = this.waiting.shift()!;
      this.remaining = Math.max(
        8_000,
        Math.min(15_000, 4_000 + [...candidate.text].length * 100),
      );
      this.confirmation = 5000;
      this.confirmed = false;
      this.active = {
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
