import { Notification } from "electron";
import type { DesktopState } from "../shared/protocol.ts";
import type { SystemNotice } from "./notifications.ts";

/** Native resources are closed on attention changes, session switches, and shutdown. */
export class SystemNotifications {
  private readonly active = new Map<
    string,
    {
      notification: Notification;
      notice: SystemNotice;
      timer?: ReturnType<typeof setTimeout>;
    }
  >();
  private session = "";

  constructor(
    private readonly open: (notice: SystemNotice) => void,
    private readonly allowed: (notice: SystemNotice) => boolean,
  ) {}

  show(notice: SystemNotice, shown: () => void): void {
    if (
      !Notification.isSupported() ||
      this.active.has(notice.id) ||
      !this.allowed(notice)
    )
      return;
    if (notice.expiresAt !== undefined && Date.now() >= notice.expiresAt)
      return;
    let notification: Notification | undefined;
    try {
      notification = new Notification({
        title: notice.title,
        body: notice.body,
        silent: true,
      });
      if (this.active.size >= 5) {
        this.remove(this.active.keys().next().value!);
      }
      const timer =
        notice.expiresAt === undefined
          ? undefined
          : setTimeout(
              () => this.remove(notice.id),
              Math.max(0, notice.expiresAt - Date.now()),
            );
      this.active.set(notice.id, { notification, notice, timer });
      notification.once("show", () => {
        if (this.allowed(notice)) shown();
        else this.remove(notice.id);
      });
      notification.once("click", () => this.open(notice));
      notification.once("close", () => this.remove(notice.id));
      notification.once("failed", () => this.remove(notice.id));
      notification.show();
    } catch {
      notification?.removeAllListeners();
      this.remove(notice.id);
    }
  }

  update(state: DesktopState): void {
    const session = `${state.settings.endpoint}:${state.sessionId}:${state.settings.language}`;
    if (
      this.session !== session ||
      state.chatFocused ||
      !state.settings.notifications
    )
      this.close();
    this.session = session;
    for (const [id, entry] of this.active) {
      if (!this.allowed(entry.notice)) this.remove(id);
    }
  }

  close(): void {
    for (const id of this.active.keys()) this.remove(id);
  }

  private remove(id: string): void {
    const entry = this.active.get(id);
    if (!entry) return;
    clearTimeout(entry.timer);
    entry.notification.removeAllListeners();
    entry.notification.close();
    this.active.delete(id);
  }
}
