import { translator } from "../shared/i18n.ts";
import { activityNotice, terminal } from "../shared/messages.ts";
import type {
  AgentActivity,
  CompanionSurface,
  DesktopState,
  StatusNotice,
  TaskActivities,
} from "../shared/protocol.ts";
import type { Reminder } from "./bubbles.ts";

/** Presentation follows conversation facts; animation playback never owns text. */
export class PresentationController {
  private turnId = "";
  private running = false;
  private sequence = 0;
  private remaining = 0;
  private previous = 0;
  private visible = false;
  private held = false;
  private pausedAt = 0;
  private dismissed = false;
  private messageId = "";

  get foregroundActive(): boolean {
    return Boolean(this.messageId && !this.dismissed && this.remaining > 0);
  }

  reset(): void {
    this.turnId = "";
    this.running = false;
    this.remaining = 0;
    this.visible = false;
    this.held = false;
    this.pausedAt = 0;
    this.dismissed = false;
    this.messageId = "";
  }

  observe(
    activity: AgentActivity | undefined,
    live: boolean,
    now = Date.now(),
  ): void {
    this.advance(now);
    if (!activity || (terminal(activity.phase) && !live)) {
      this.reset();
      this.turnId = activity?.turn_id ?? "";
      return;
    }
    this.follow(activity.turn_id, !terminal(activity.phase));
  }

  private follow(turnId: string, running: boolean): void {
    if (turnId !== this.turnId) {
      this.turnId = turnId;
      this.sequence++;
      this.dismissed = false;
      this.pausedAt = 0;
      this.remaining = 12_000;
    }
    if (running !== this.running) {
      this.running = running;
      this.remaining = 12_000;
    }
  }

  private advance(now: number): void {
    if (this.visible && !this.held && !this.pausedAt && !this.running)
      this.remaining = Math.max(
        0,
        this.remaining - Math.max(0, now - this.previous),
      );
    this.previous = now;
  }

  action(
    id: string,
    action: "dismiss" | "pause",
    paused: boolean,
    now = Date.now(),
  ): boolean {
    if (id !== this.messageId) return false;
    this.advance(now);
    if (action === "dismiss") this.dismissed = true;
    else this.pausedAt = paused ? now : 0;
    return true;
  }

  resolve(
    state: DesktopState,
    activity: AgentActivity | undefined,
    tasks: TaskActivities,
    proactive: AgentActivity | undefined,
    reminder: Reminder | null,
    visible: boolean,
    dragging: boolean,
    now = Date.now(),
  ): CompanionSurface | null {
    this.advance(now);
    const pending =
      state.pending?.sessionId === state.sessionId ? state.pending : null;
    if (pending) {
      const ended =
        activity &&
        activity.turn_id === pending.turnId &&
        terminal(activity.phase);
      this.follow(
        pending.turnId ?? pending.id,
        pending.streaming && !pending.error && !ended,
      );
    }
    const current = state.messages.findLast(
      (message) =>
        message.role === "assistant" &&
        message.source === "chat" &&
        (message.turnId === this.turnId ||
          message.id === `local-assistant:${this.turnId}`),
    );
    this.messageId = current?.id ?? "";
    this.held = Boolean(
      current &&
      state.speech.playback?.messageId === current.id &&
      state.speech.playback.phase !== "error",
    );
    if (this.pausedAt && now - this.pausedAt > 120_000) this.dismissed = true;
    const reply = !this.dismissed && this.remaining > 0 ? current : undefined;
    this.visible = visible && !dragging && state.connection === "connected";
    if (!this.visible) return null;
    if (reply) {
      if (reply.content)
        return {
          kind: "message",
          message: reply,
          presentationId: this.sequence,
          paused: Boolean(this.pausedAt),
        };
      if (reply.status) return reply.status;
    }
    if (reminder) {
      const message = state.messages.find(
        (message) => message.id === reminder.id,
      );
      if (message)
        return {
          kind: "message",
          message,
          presentationId: reminder.presentationId,
          paused: reminder.paused,
        };
    }
    const t = translator(state.settings.language);
    const notice = (id: string, text: string): StatusNotice => ({
      kind: "status",
      id,
      text,
      tone: "working",
    });
    const capture = state.speech.capture;
    if (capture || state.speech.playback?.phase === "playing")
      return notice(
        "speech",
        t(
          capture
            ? capture.phase === "starting"
              ? "speechOpeningMic"
              : capture.phase === "transcribing"
                ? "speechTranscribing"
                : "speechListening"
            : "speechPlaying",
        ),
      );
    if (state.chatFocused || state.settings.quietUntil > now) return null;
    const task = Object.values(tasks).find((item) => !terminal(item.phase));
    const background =
      task ?? (proactive && !terminal(proactive.phase) ? proactive : undefined);
    return background
      ? {
          ...activityNotice(background, state.settings.language),
          text: task
            ? t("taskBackground", { task: task.task_id! })
            : t("background"),
        }
      : null;
  }
}
