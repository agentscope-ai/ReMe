import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { DesktopController } from "../src/main/controller.ts";
import { Storage } from "../src/main/storage.ts";
import type { AnimationManifest } from "../src/shared/protocol.ts";
import { mockReMe } from "./e2e/mock-reme.mjs";
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
    controller.start();
    try {
      await until(
        () => controller.state.connection === "connected",
        "initial snapshot",
      );
      assert.equal(controller.state.unread, 0);
      assert.equal(controller.state.bubble, null);
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
      assert.equal(controller.state.bubble, null);
      assert.ok(
        Object.values(storage.presentations).some(
          (entry) => entry.receipts.new?.presented === false,
        ),
      );
      controller.setCompanionVisible(true);
      await until(
        () => controller.state.bubble?.id === "new",
        "completed reminder",
      );
      assert.equal(controller.state.unread, 1);
      controller.bubbleAction("new", "dismiss");
      assert.equal(controller.state.unread, 1);
      assert.equal(controller.state.bubble, null);
      controller.setChatVisible(true);
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
