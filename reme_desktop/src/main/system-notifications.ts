import { Notification } from "electron";
import type { DesktopState } from "../shared/protocol.ts";
import type { SystemNotice } from "./notifications.ts";

/** Native resources are closed on attention changes, session switches, and shutdown. */
export class SystemNotifications {
  private readonly active = new Map<string, Notification>();
  private session = "";

  constructor(private readonly open: (notice: SystemNotice) => void) {}

  show(notice: SystemNotice, shown: () => void): void {
    if (!Notification.isSupported() || this.active.has(notice.id)) return;
    let notification: Notification | undefined;
    try {
      notification = new Notification({
        title: notice.title,
        body: notice.body,
        silent: true,
      });
      if (this.active.size >= 5) {
        const [oldest, previous] = this.active.entries().next().value!;
        previous.removeAllListeners();
        previous.close();
        this.active.delete(oldest);
      }
      this.active.set(notice.id, notification);
      notification.once("show", shown);
      notification.once("click", () => this.open(notice));
      notification.once("close", () => this.active.delete(notice.id));
      notification.once("failed", () => this.active.delete(notice.id));
      notification.show();
    } catch {
      notification?.removeAllListeners();
      this.active.delete(notice.id);
    }
  }

  update(state: DesktopState): void {
    const session = `${state.settings.endpoint}:${state.sessionId}`;
    if (
      this.session !== session ||
      state.chatFocused ||
      !state.settings.notifications ||
      state.settings.quietUntil > Date.now()
    )
      this.close();
    this.session = session;
  }

  close(): void {
    for (const value of this.active.values()) {
      value.removeAllListeners();
      value.close();
    }
    this.active.clear();
  }
}
