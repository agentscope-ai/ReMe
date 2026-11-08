import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { DesktopController } from "../src/main/controller.ts";
import { Storage } from "../src/main/storage.ts";
import type { AnimationManifest } from "../src/shared/protocol.ts";
import { mockReMe } from "./e2e/mock-reme.mjs";
import type { SystemNotice } from "../src/main/notifications.ts";
const manifest = JSON.parse(
  await readFile(
    new URL(
      "../../assets/remy/anime/animations/v1/manifest.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as AnimationManifest;

async function until(check: () => boolean, label: string): Promise<void> {
  const deadline = Date.now() + 8_000;
  while (!check()) {
    if (Date.now() > deadline) throw new Error(`Timed out: ${label}`);
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}

test("closing policy drops queued reminders and the deadline removes the active bubble", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-reminder-window-"));
  const backend = await mockReMe();
  const storage = new Storage(directory);
  storage.settings.endpoint = backend.url;
  const controller = new DesktopController(storage, () => {}, manifest);
  controller.setCompanionReady(true);
  controller.setChatVisible(true);
  controller.start();
  try {
    await until(
      () => controller.state.connection === "connected",
      "window snapshot",
    );
    backend.push("old-queued", "旧提醒");
    await until(() => controller.state.unread === 1, "queued old reminder");
    const starts_at = new Date(Date.now() - 3_600_000).toISOString();
    backend.notificationWindow({
      enabled: true,
      starts_at,
      ends_at: new Date(Date.now() + 3_600_000).toISOString(),
      closing_message_id: "closing",
    });
    backend.push("closing", "晚安");
    await until(
      () =>
        controller.state.messages.some((message) => message.id === "closing"),
      "closing message",
    );
    controller.setChatVisible(false);
    await until(
      () =>
        (controller.state.surface?.kind === "message"
          ? controller.state.surface.message.id
          : null) === "closing",
      "only closing may appear",
    );
    backend.push("too-late", "收尾后的提醒");
    await until(
      () =>
        controller.state.messages.some((message) => message.id === "too-late"),
      "late message retained in history",
    );
    assert.equal(
      controller.state.surface?.kind === "message"
        ? controller.state.surface.message.id
        : null,
      "closing",
    );
    backend.notificationWindow({
      enabled: true,
      starts_at,
      ends_at: new Date(Date.now() - 1).toISOString(),
      closing_message_id: "closing",
    });
    await until(
      () => controller.state.surface === null,
      "expired bubble removed",
    );
    backend.notificationWindow({
      enabled: true,
      starts_at,
      ends_at: new Date(Date.now() + 3_600_000).toISOString(),
    });
    await new Promise((resolve) => setTimeout(resolve, 200));
    assert.equal(controller.state.surface, null);
  } finally {
    await controller.close();
    await backend.close();
    await rm(directory, { recursive: true, force: true });
  }
});

test("failed settings writes preserve the active endpoint", async () => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "reme-desktop-settings-"),
  );
  const storage = new Storage(directory);
  const endpoint = storage.settings.endpoint;
  const flush = storage.flush;
  storage.flush = async () => {
    throw new Error("read-only settings");
  };
  const controller = new DesktopController(storage, () => {}, manifest);
  try {
    await assert.rejects(
      controller.settings({ endpoint: "http://127.0.0.1:9876" }),
      /read-only/,
    );
    assert.equal(storage.settings.endpoint, endpoint);
    assert.equal(controller.state.settings.endpoint, endpoint);
  } finally {
    storage.flush = flush;
    await controller.close();
    await rm(directory, { recursive: true, force: true });
  }
});

test("closing waits for settings persistence without starting a new connection", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-desktop-close-"));
  const storage = new Storage(directory);
  let release!: () => void;
  let writing = false;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  storage.flush = async () => {
    if (!writing) {
      writing = true;
      await gate;
    }
  };
  let publications = 0;
  const controller = new DesktopController(
    storage,
    () => {
      publications++;
    },
    manifest,
  );
  try {
    const changing = controller.settings({ endpoint: "http://127.0.0.1:9876" });
    await until(() => writing, "pending settings write");
    const closing = controller.close();
    release();
    await Promise.all([changing, closing]);
    assert.equal(publications, 0);
    await assert.rejects(controller.send("too late"), /不可用/);
  } finally {
    release();
    await controller.close();
    await rm(directory, { recursive: true, force: true });
  }
});

test(
  "controller applies subscription, local receipts, replay and endpoint isolation",
  { timeout: 20_000 },
  async () => {
    const directory = await mkdtemp(
      path.join(tmpdir(), "reme-desktop-controller-"),
    );
    const backend = await mockReMe();
    const second = await mockReMe();
    const storage = new Storage(directory);
    await storage.load();
    storage.settings.endpoint = backend.url;
    let publications = 0;
    const controller = new DesktopController(
      storage,
      () => {
        publications++;
      },
      manifest,
    );
    controller.setCompanionReady(true);
    controller.start();
    try {
      await until(
        () => controller.state.connection === "connected",
        "initial snapshot",
      );
      assert.equal(controller.state.unread, 0);
      assert.equal(controller.state.surface, null);
      backend.push("new", "还没完成", false);
      await until(
        () => controller.state.messages.some((message) => message.id === "new"),
        "partial upsert",
      );
      assert.equal(controller.state.unread, 0);
      controller.setCompanionVisible(false);
      backend.push("new", "最终提醒");
      backend.push("new", "最终提醒");
      await until(
        () => controller.state.unread === 1,
        "reminder while character is hidden",
      );
      assert.equal(controller.state.surface, null);
      assert.ok(
        Object.values(storage.presentations).some(
          (entry) => entry.receipts.new?.presented === false,
        ),
      );
      controller.setCompanionVisible(true);
      await until(
        () =>
          (controller.state.surface?.kind === "message"
            ? controller.state.surface.message.id
            : null) === "new",
        "completed reminder",
      );
      assert.equal(controller.state.unread, 1);
      const bubble = controller.bubbles.active!;
      const receipt = Object.values(storage.presentations).find(
        (entry) => entry.receipts.new,
      )?.receipts.new;
      assert.equal(
        receipt?.presented,
        false,
        "selection alone is not proof of presentation",
      );
      controller.bubbleShown("wrong-session", bubble.id, bubble.presentationId);
      assert.equal(receipt?.presented, false);
      controller.bubbleShown("session-1", bubble.id, bubble.presentationId);
      assert.equal(receipt?.presented, true);
      controller.bubbleAction("new", "dismiss");
      assert.equal(controller.state.unread, 1);
      assert.equal(controller.state.surface, null);
      controller.setChatVisible(true);
      controller.setChatFocused(true);
      controller.markRead("session-1", ["new"]);
      assert.equal(controller.state.unread, 0);
      const before = publications;
      controller.markRead("session-1", ["new"]);
      assert.equal(
        publications,
        before,
        "duplicate read acknowledgements must not create an IPC feedback loop",
      );
      controller.setChatVisible(false);
      backend.disconnect();
      backend.push("missed", "离线期间的新提醒");
      await until(
        () =>
          controller.state.messages.some((message) => message.id === "missed"),
        "cursor replay",
      );
      assert.equal(controller.state.unread, 1);
      assert.equal(
        controller.state.messages.filter((message) => message.id === "new")
          .length,
        1,
      );
      assert.ok(
        backend.cursors.some((cursor) => cursor?.startsWith("test-epoch:")),
      );
      await controller.settings({ endpoint: second.url });
      await until(
        () => controller.state.connection === "connected",
        "second endpoint",
      );
      assert.equal(
        controller.state.messages.some((message) => message.id === "missed"),
        false,
      );
      assert.equal(controller.state.unread, 0);
      assert.ok(
        Object.values(storage.presentations).some(
          (value) => value.receipts.new?.read,
        ),
      );
    } finally {
      await controller.close();
      await backend.close();
      await second.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);

test(
  "controller reconciles an accepted turn and explicitly aborts a pending generation",
  { timeout: 15_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-desktop-turn-"));
    const backend = await mockReMe();
    const storage = new Storage(directory);
    await storage.load();
    storage.settings.endpoint = backend.url;
    const controller = new DesktopController(storage, () => {}, manifest);
    controller.setCompanionReady(true);
    controller.start();
    try {
      await until(() => controller.state.connection === "connected", "connect");
      await controller.send("你好");
      assert.equal(Boolean(controller.state.pending), false);
      assert.equal(
        controller.state.messages.filter(
          (message) =>
            message.content === "我记住了。我们可以继续聊聊你的研究。",
        ).length,
        1,
      );
      await until(() => controller.state.status === "idle", "turn release");
      const pending = controller.send("请等我停止");
      await until(
        () => Boolean(controller.state.pending?.text),
        "partial reply",
      );
      controller.stopReply();
      await pending;
      assert.equal(controller.state.pending?.streaming, false);
      assert.equal(controller.state.pending?.error, "回复已停止");
    } finally {
      await controller.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);

test("immediate results become read only after their bubble is shown", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-immediate-read-"));
  const backend = await mockReMe();
  const storage = new Storage(directory);
  storage.settings.endpoint = backend.url;
  const controller = new DesktopController(storage, () => {}, manifest);
  controller.setCompanionReady(true);
  controller.setCompanionVisible(true);
  controller.start();
  try {
    await until(() => controller.state.connection === "connected", "connect");
    const message = {
      id: "immediate-result",
      role: "assistant" as const,
      source: "proactive" as const,
      content: "任务结果。",
      completed: true,
      created_at: new Date().toISOString(),
      delivery: "immediate" as const,
      origin_session_id: "source-session",
      notification_eligible: true,
      chunks: [],
    };
    backend.upsert(message);
    await until(
      () =>
        controller.state.surface?.kind === "message" &&
        controller.state.surface.message.id === message.id,
      "immediate bubble selected",
    );
    const surface = controller.state.surface;
    assert.ok(
      surface?.kind === "message" && surface.presentationId !== undefined,
    );
    assert.equal(controller.state.unread, 1);
    controller.bubbleShown("wrong-session", message.id, surface.presentationId);
    assert.equal(controller.state.unread, 1);
    controller.bubbleShown("session-1", message.id, surface.presentationId);
    assert.equal(controller.state.unread, 0);
    assert.ok(controller.state.readMessageIds.includes(message.id));
    assert.ok(
      controller.state.inboxMessages.some((item) => item.id === message.id),
    );
    const receipt = Object.values(storage.presentations).find(
      (entry) => entry.receipts[message.id],
    )!.receipts[message.id];
    assert.equal(receipt.presented, true);
    assert.equal(receipt.read, true);
    controller.bubbleAction(message.id, "dismiss");
    backend.upsert(message);
    await until(() => controller.state.surface === null, "dismissed bubble");
    assert.equal(controller.state.unread, 0);
  } finally {
    await controller.close();
    await backend.close();
    await rm(directory, { recursive: true, force: true });
  }
});

test("hidden reminders use native show acknowledgements and share the bubble cooldown", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-desktop-native-"));
  const backend = await mockReMe();
  const storage = new Storage(directory);
  await storage.load();
  storage.settings.endpoint = backend.url;
  storage.settings.notifications = true;
  const notices: { notice: SystemNotice; shown: () => void }[] = [];
  const controller = new DesktopController(
    storage,
    () => {},
    manifest,
    (notice, shown) => {
      notices.push({ notice, shown });
    },
  );
  controller.setCompanionVisible(false);
  controller.start();
  try {
    await until(() => controller.state.connection === "connected", "connect");
    backend.push("native", "native reminder");
    await until(() => notices.length === 1, "native dispatch");
    const receipt = Object.values(storage.presentations).find(
      (entry) => entry.receipts.native,
    )!.receipts.native;
    assert.equal(receipt.presented, false);
    assert.equal(controller.state.unread, 1);
    assert.equal(controller.canNotify(notices[0].notice), true);
    notices[0].shown();
    assert.equal(receipt.presented, true);
    assert.equal(receipt.read, false);
    backend.push("next", "another message");
    await until(() => controller.state.unread === 2, "next unread");
    assert.equal(
      notices.length,
      1,
      "native reminders must honor the same ten-minute cooldown",
    );
    backend.notificationWindow({
      enabled: true,
      starts_at: new Date(Date.now() - 3_600_000).toISOString(),
      ends_at: new Date(Date.now() + 3_600_000).toISOString(),
      closing_message_id: "closing",
    });
    await until(
      () => !controller.canNotify(notices[0].notice),
      "queued native reminder revoked at evening close",
    );
    await controller.newSession();
    await until(
      () => controller.state.sessionId === "session-2",
      "session switch",
    );
    assert.equal(controller.canOpenNotice(notices[0].notice), false);
    notices[0].shown();
    assert.equal(
      controller.state.unread,
      2,
      "switching sessions preserves unread messages from their source conversation",
    );
    assert.equal(
      receipt.read,
      false,
      "a late presentation acknowledgement does not mark the source message read",
    );
    assert.equal(
      controller.state.inboxMessages.find((message) => message.id === "native")
        ?.originSessionId,
      "session-1",
    );
    controller.setChatVisible(true);
    controller.setChatFocused(true);
    controller.markRead("session-1", ["native"]);
    assert.equal(receipt.read, true);
    assert.equal(
      controller.state.unread,
      1,
      "reading a source message updates only its own receipt",
    );
  } finally {
    await controller.close();
    await backend.close();
    await rm(directory, { recursive: true, force: true });
  }
});

test("language is published only after persistence succeeds; failure preserves the active preference", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-language-save-"));
  const storage = new Storage(directory);
  const flush = storage.flush;
  let release!: () => void;
  let writing = false;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  storage.flush = async () => {
    writing = true;
    await gate;
    throw new Error("read-only settings");
  };
  const controller = new DesktopController(storage, () => {}, manifest);
  try {
    const changing = controller.settings({ language: "en" });
    const rejected = assert.rejects(changing, /read-only/);
    await until(() => writing, "pending language write");
    controller.setChatVisible(true);
    assert.equal(
      controller.state.settings.language,
      "zh",
      "in-flight settings must not leak into IPC",
    );
    release();
    await rejected;
    assert.equal(storage.settings.language, "zh");
    assert.equal(controller.state.settings.language, "zh");
  } finally {
    release();
    storage.flush = flush;
    await controller.close();
    await rm(directory, { recursive: true, force: true });
  }
});

test("switching language preserves the PA subscription, animation instance, and original messages", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-language-live-"));
  const backend = await mockReMe();
  const storage = new Storage(directory);
  storage.settings.endpoint = backend.url;
  const controller = new DesktopController(storage, () => {}, manifest);
  controller.setCompanionReady(true);
  controller.start();
  try {
    await until(() => controller.state.connection === "connected", "connect");
    backend.activity("thinking");
    await until(
      () => controller.state.behavior.label === "正在认真想…",
      "thinking",
    );
    const cursor = controller.model.cursor;
    const animation = controller.state.behavior.animation.id;
    const history = controller.model.messages;
    const connections = backend.cursors.length;
    await controller.settings({ language: "en" });
    assert.equal(controller.state.behavior.label, "Thinking it through…");
    assert.equal(controller.state.behavior.animation.id, animation);
    assert.equal(controller.model.cursor, cursor);
    assert.equal(controller.model.messages, history);
    assert.equal(backend.cursors.length, connections);
    await assert.rejects(
      controller.settings({ language: "fr" as never }),
      /语言/,
    );
    assert.equal(controller.state.settings.language, "en");
  } finally {
    await controller.close();
    await backend.close();
    await rm(directory, { recursive: true, force: true });
  }
});
