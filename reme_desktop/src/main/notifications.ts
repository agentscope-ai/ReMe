import type { AgentActivity } from "../shared/protocol.ts";

export interface SystemNotice {
  id: string;
  scope: string;
  sessionId: string;
  messageId?: string;
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
    const title =
      activity.phase === "waiting"
        ? "Remy 需要你处理一下"
        : activity.phase === "failed"
          ? "Remy 的回复未完成"
          : activity.phase === "completed" && now - this.started >= 30_000
            ? "Remy 已完成回复"
            : null;
    return title
      ? {
          id: `turn:${turn}:${activity.phase}`,
          title,
          body: "点击返回对应对话查看",
        }
      : null;
  }
}
