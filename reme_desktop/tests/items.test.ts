import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { DesktopController } from "../src/main/controller.ts";
import { Storage } from "../src/main/storage.ts";
import { UiError } from "../src/shared/i18n.ts";
import type {
  AnimationManifest,
  ProactiveInboxItem,
} from "../src/shared/protocol.ts";
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
const result = (index: number): ProactiveInboxItem & { content: string } => ({
  id: `task:results/${index}.md`,
  kind: "task_result",
  title: `Result ${index}`,
  preview: "Short preview",
  content: `Full result ${index}. [Source](https://example.com/${index})`,
  status: "received",
  created_at: new Date().toISOString(),
  task_id: "research",
  path: `results/${index}.md`,
  user_read_at: null,
  source_modified: false,
  source_present: true,
  receipt_status: null,
});
async function until(check: () => boolean) {
  const deadline = Date.now() + 8000;
  while (!check()) {
    assert.ok(Date.now() < deadline, "Timed out waiting for the subscription");
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}
async function setup(taskResults: ReturnType<typeof result>[] = []) {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-items-"));
  const backend = await mockReMe({ taskResults });
  const storage = new Storage(directory);
  storage.settings.endpoint = backend.url;
  const controller = new DesktopController(storage, () => {}, manifest);
  controller.setCompanionReady(true);
  controller.start();
  await until(() => controller.state.connection === "connected");
  return {
    controller,
    backend,
    storage,
    close: async () => {
      await controller.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    },
  };
}

for (const read of [false, true])
  test(`manual replay preserves ${read ? "read" : "unread"} state across sessions and automatic restrictions`, async () => {
    const h = await setup();
    const { controller, backend, storage } = h;
    try {
      backend.upsert({
        id: "old",
        role: "assistant",
        source: "proactive",
        content: "Stored reminder.",
        completed: true,
        delivery: "idle",
        notification_eligible: false,
        origin_session_id: "session-1",
        expires_at: new Date(Date.now() - 1000).toISOString(),
        chunks: [],
      });
      await until(() =>
        controller.state.inboxMessages.some((item) => item.id === "old"),
      );
      if (read) {
        controller.setChatVisible(true);
        controller.setChatFocused(true);
        controller.markRead("session-1", ["old"]);
      }
      await controller.newSession();
      await until(() => controller.state.sessionId === "session-2");
      await controller.settings({ quietUntil: Date.now() + 60_000 });
      backend.activity("thinking");
      await until(() => controller.state.status === "active");
      const target = { kind: "message", id: "old" } as const;
      await controller.remind(target);
      const first = controller.state.surface;
      assert.ok(first?.kind === "message");
      assert.deepEqual(first.target, target);
      assert.equal(first.canSnooze, false);
      controller.bubbleShown("session-2", "old", first.presentationId);
      assert.equal(controller.state.readMessageIds.includes("old"), read);
      const receipt = Object.values(storage.presentations).find(
        (p) => p.receipts.old,
      )!.receipts.old;
      assert.equal(receipt.read, read);
      assert.equal(receipt.presented, true);
      await controller.remind(target);
      const next = controller.state.surface;
      assert.ok(next?.kind === "message");
      assert.notEqual(next.presentationId, first.presentationId);
      controller.bubbleAction("old", "pause", true);
      assert.equal(
        controller.state.surface?.kind === "message" &&
          controller.state.surface.paused,
        true,
      );
      controller.bubbleAction("old", "dismiss");
      assert.notEqual(
        controller.state.surface?.kind === "message" &&
          controller.state.surface.message.id,
        "old",
      );
      assert.equal(controller.state.pending, null);
      assert.ok(!controller.state.messages.some((item) => item.id === "old"));
    } finally {
      await h.close();
    }
  });

test("task replay reads full content without changing its receipt or inserting a chat message", async () => {
  const task = result(1);
  const h = await setup([task]);
  try {
    const target = { kind: "task_result", id: task.id } as const;
    await h.controller.remind(target);
    const surface = h.controller.state.surface;
    assert.ok(surface?.kind === "message");
    assert.equal(surface.message.content, task.content);
    assert.deepEqual(surface.target, target);
    h.controller.bubbleShown("session-1", task.id, surface.presentationId);
    assert.equal(task.user_read_at, null);
    assert.equal(h.backend.taskReads.length, 0);
    h.controller.setChatVisible(true, target);
    assert.deepEqual(h.controller.state.focusTarget, target);
    assert.equal(h.controller.state.pending, null);
    assert.equal(
      h.controller.state.messages.some((item) => item.id === task.id),
      false,
    );
  } finally {
    await h.close();
  }
});

test("a delayed task replay cannot replace a more recent user selection", async () => {
  const task = result(2);
  const h = await setup([task]);
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  let waiting = false;
  try {
    h.backend.push("latest", "Latest selection");
    await until(() =>
      h.controller.state.inboxMessages.some((item) => item.id === "latest"),
    );
    const workspace = h.controller.workspace.bind(h.controller);
    h.controller.workspace = async (request) => {
      const response = await workspace(request);
      if (request.type === "proactiveInboxEntry") {
        waiting = true;
        await gate;
      }
      return response;
    };
    const delayed = h.controller.remind({ kind: "task_result", id: task.id });
    await until(() => waiting);
    await h.controller.remind({ kind: "message", id: "latest" });
    release();
    assert.equal(await delayed, false);
    assert.equal(
      h.controller.state.surface?.kind === "message" &&
        h.controller.state.surface.message.id,
      "latest",
    );
  } finally {
    release();
    await h.close();
  }
});

test("mark all read covers every task page and coalesces concurrent clicks while the workspace is hidden", async () => {
  const tasks = Array.from({ length: 106 }, (_, index) => result(index));
  tasks[0].user_read_at = new Date().toISOString();
  const h = await setup(tasks);
  try {
    h.backend.push("unread", "Unread reminder");
    await until(() => h.controller.state.unread === 1);
    h.controller.setChatVisible(false);
    await Promise.all([h.controller.markAllRead(), h.controller.markAllRead()]);
    assert.equal(h.controller.state.unread, 0);
    assert.ok(tasks.every((item) => item.user_read_at));
    assert.equal(h.backend.taskReads.length, 105);
    assert.ok(h.controller.state.inboxRevision > 0);
  } finally {
    await h.close();
  }
});

test("partial bulk-read failure is reported and the remaining items can be retried", async () => {
  const tasks = [result(1), result(2)];
  const h = await setup(tasks);
  try {
    h.backend.setTaskReadError(tasks[1].path!);
    await assert.rejects(
      h.controller.markAllRead(),
      /Mock task result read failed/,
    );
    assert.ok(tasks[0].user_read_at);
    assert.equal(tasks[1].user_read_at, null);
    h.backend.setTaskReadError("");
    await h.controller.markAllRead();
    assert.ok(tasks.every((item) => item.user_read_at));
  } finally {
    await h.close();
  }
});

test("a task response from a previous endpoint cannot produce a replay", async () => {
  const task = result(3);
  const h = await setup([task]);
  const next = await mockReMe();
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  let waiting = false;
  try {
    const workspace = h.controller.workspace.bind(h.controller);
    h.controller.workspace = async (request) => {
      const response = await workspace(request);
      waiting = true;
      await gate;
      return response;
    };
    const pending = h.controller.remind({ kind: "task_result", id: task.id });
    const rejected = assert.rejects(
      pending,
      (error) => error instanceof UiError && error.key === "itemUnavailable",
    );
    await until(() => waiting);
    await h.controller.settings({ endpoint: next.url });
    await until(() => h.controller.state.connection === "connected");
    release();
    await rejected;
    assert.equal(h.controller.state.surface, null);
  } finally {
    release();
    await h.close();
    await next.close();
  }
});

test("an explanation cannot cross endpoints even when both use the same session ID", async () => {
  const h = await setup();
  const next = await mockReMe();
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  let waiting = false;
  try {
    h.backend.push("old", "Private source from the original workspace");
    await until(() =>
      h.controller.state.inboxMessages.some((item) => item.id === "old"),
    );
    const item = h.controller.item.bind(h.controller);
    h.controller.item = async (target) => {
      const message = await item(target);
      waiting = true;
      await gate;
      return message;
    };
    const pending = h.controller.explain({ kind: "message", id: "old" });
    const rejected = assert.rejects(
      pending,
      (error) => error instanceof UiError && error.key === "itemUnavailable",
    );
    await until(() => waiting);
    await h.controller.settings({ endpoint: next.url });
    await until(() => h.controller.state.connection === "connected");
    assert.equal(h.controller.state.sessionId, "session-1");
    release();
    await rejected;
    assert.equal(h.controller.state.pending, null);
    assert.equal(
      h.controller.state.messages.some((message) => message.role === "user"),
      false,
    );
  } finally {
    release();
    await h.close();
    await next.close();
  }
});

test("bulk reading preserves an already-read reminder's explicit snooze", async () => {
  const h = await setup();
  try {
    h.backend.push("snoozed", "Remind me later");
    await until(() => h.controller.state.surface?.kind === "message");
    const surface = h.controller.state.surface;
    assert.ok(surface?.kind === "message");
    h.controller.bubbleShown("session-1", "snoozed", surface.presentationId);
    h.controller.setChatVisible(true);
    h.controller.setChatFocused(true);
    h.controller.markRead("session-1", ["snoozed"]);
    h.controller.bubbleAction("snoozed", "snooze");
    const receipt = Object.values(h.storage.presentations).find(
      (item) => item.receipts.snoozed,
    )!.receipts.snoozed;
    const deadline = receipt.snoozeUntil;
    assert.ok(deadline);
    await h.controller.markAllRead();
    assert.equal(receipt.snoozeUntil, deadline);
    assert.equal(receipt.read, true);
  } finally {
    await h.close();
  }
});

test("the app menu temporarily takes the surface without dismissing a replay", async () => {
  const task = result(7);
  const h = await setup([task]);
  try {
    await h.controller.remind({ kind: "task_result", id: task.id });
    const original = h.controller.state.surface;
    assert.ok(original?.kind === "message");
    h.controller.setMenuOpen(true);
    assert.equal(h.controller.state.surface?.kind, "menu");
    const firstMenu =
      h.controller.state.surface?.kind === "menu" &&
      h.controller.state.surface.id;
    h.controller.setMenuOpen(true);
    assert.notEqual(
      h.controller.state.surface?.kind === "menu" &&
        h.controller.state.surface.id,
      firstMenu,
    );
    h.controller.setMenuOpen(false);
    assert.deepEqual(h.controller.state.surface, original);
    assert.equal(task.user_read_at, null);
  } finally {
    await h.close();
  }
});
