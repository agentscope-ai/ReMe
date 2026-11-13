import assert from "node:assert/strict";
import { register } from "node:module";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { DesktopController } from "../src/main/controller.ts";
import { Storage } from "../src/main/storage.ts";
import type { AnimationManifest } from "../src/shared/protocol.ts";
import { mockReMe } from "./e2e/mock-reme.mjs";

const electron = `data:text/javascript,${encodeURIComponent(`
  import { EventEmitter } from 'node:events';
  export class Notification extends EventEmitter {
    static instances = [];
    static isSupported() { return true; }
    closed = false;
    constructor() { super(); Notification.instances.push(this); }
    show() { queueMicrotask(() => { if (!this.closed) this.emit('show'); }); }
    close() { this.closed = true; }
  }
`)}`;
register(
  `data:text/javascript,${encodeURIComponent(`
  export async function resolve(specifier, context, next) {
    return specifier === 'electron'
      ? { url: ${JSON.stringify(electron)}, shortCircuit: true }
      : next(specifier, context);
  }
`)}`,
);
const { SystemNotifications } =
  await import("../src/main/system-notifications.ts");
const { Notification } = await import(electron);

async function until(check: () => boolean): Promise<void> {
  const deadline = Date.now() + 3000;
  while (!check()) {
    if (Date.now() > deadline)
      throw new Error("Notification did not reach expected state");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

test("immediate native notifications survive quiet mode and closing policy like immediate bubbles", async () => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "reme-immediate-notification-"),
  );
  const backend = await mockReMe();
  const storage = new Storage(directory);
  storage.settings.endpoint = backend.url;
  storage.settings.notifications = true;
  storage.settings.quietUntil = Date.now() + 3_600_000;
  const manifest = JSON.parse(
    await readFile(
      new URL(
        "../../assets/remy/anime/animations/v1/manifest.json",
        import.meta.url,
      ),
      "utf8",
    ),
  ) as AnimationManifest;
  let notifications: InstanceType<typeof SystemNotifications> | undefined;
  let shown = 0;
  const controller = new DesktopController(
    storage,
    (state) => notifications?.update(state),
    manifest,
    (notice, acknowledge) =>
      notifications!.show(notice, () => {
        shown++;
        acknowledge();
      }),
  );
  notifications = new SystemNotifications(
    () => {},
    (notice) => controller.canNotify(notice),
  );
  controller.setCompanionVisible(false);
  controller.start();
  try {
    await until(() => controller.state.connection === "connected");
    backend.notificationWindow({
      enabled: false,
      starts_at: new Date(Date.now() - 7200000).toISOString(),
      ends_at: new Date(Date.now() - 3600000).toISOString(),
      closing_message_id: "goodnight",
    });
    backend.upsert({
      id: "urgent",
      role: "assistant",
      source: "proactive",
      content: "约定的提醒",
      completed: true,
      created_at: new Date().toISOString(),
      delivery: "immediate",
      origin_session_id: "session-1",
      notification_eligible: true,
      chunks: [],
    });
    await until(() => shown === 1);
    assert.equal(Notification.instances.at(-1).closed, false);
    controller.setChatFocused(true);
    assert.equal(Notification.instances.at(-1).closed, true);
  } finally {
    notifications.close();
    await controller.close();
    await backend.close();
    await rm(directory, { recursive: true, force: true });
  }
});
