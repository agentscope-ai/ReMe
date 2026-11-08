import type { AgentActivity } from "../shared/protocol.ts";
import { defaultLanguage, translate } from "../shared/i18n.ts";
import type { Language, TranslationKey } from "../shared/i18n.ts";

export interface SystemNotice {
  id: string;
  scope: string;
  sessionId: string;
  messageId?: string;
  expiresAt?: number;
  title: string;
  body: string;
}

/** Observe every phase, including quiet snapshots, so old outcomes never alert later. */
export class NotificationPolicy {
  private turn = "";
  private phase = "";
  private started = 0;

  reset(): void {
    this.turn = "";
    this.phase = "";
    this.started = 0;
  }

  observe(
    activity: AgentActivity | undefined,
    live: boolean,
    eligible: boolean,
    now = Date.now(),
    language: Language = defaultLanguage,
  ): Pick<SystemNotice, "id" | "title" | "body"> | null {
    if (!activity || activity.source !== "chat") return null;
    const turn = `${activity.session_id}:${activity.turn_id}`;
    if (turn !== this.turn) {
      this.turn = turn;
      this.phase = "";
      this.started = now;
    }
    const changed = this.phase !== activity.phase;
    this.phase = activity.phase;
    if (!live || !eligible || !changed) return null;
    const title: TranslationKey | null =
      activity.phase === "waiting"
        ? "notifyWaiting"
        : activity.phase === "failed"
          ? "notifyFailed"
          : activity.phase === "completed" && now - this.started >= 30_000
            ? "notifyCompleted"
            : null;
    return title
      ? {
          id: `turn:${turn}:${activity.phase}`,
          title: translate(language, title),
          body: translate(language, "notifyOpen"),
        }
      : null;
  }
}
