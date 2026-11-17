import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { DesktopController } from "../src/main/controller.ts";
import { ReMeClient } from "../src/main/reme-client.ts";
import { Storage } from "../src/main/storage.ts";
import { UiError } from "../src/shared/i18n.ts";
import type {
  Presentation,
  SessionMessage,
  StreamChunk,
} from "../src/shared/protocol.ts";

const reminder: SessionMessage = {
  id: "reme-proactive-reminder-combined",
  role: "assistant",
  source: "proactive",
  content: "A relevant change",
  completed: true,
  delivery: "idle",
  notification_eligible: true,
  chunks: [],
};

test("explicit chat reading retries durable acknowledgements; historical suppression never sends them", async (t) => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "reme-read-receipt-test-"),
  );
  const manifest = JSON.parse(
    await readFile(
      new URL("../assets/remy/manifest.json", import.meta.url),
      "utf8",
    ),
  );
  const storage = new Storage(directory);
  let receive: ((chunk: StreamChunk) => void) | undefined;
  let offline = true;
  const calls: unknown[] = [];
  t.mock.method(
    ReMeClient.prototype,
    "call",
    async (job: string, body: unknown) => {
      if (job === "app_config")
        return {
          workspace_dir: "/receipt-test",
          jobs: Object.fromEntries(
            [
              "agent_events",
              "agent_chat",
              "agent_session",
              "agent_new_session",
              "agent_outbox",
              "agent_reminder",
              "agent_inbox",
            ].map((name) => [name, {}]),
          ),
        };
      assert.equal(job, "agent_reminder");
      calls.push(body);
      if (offline) throw new Error("temporary offline");
      return { id: reminder.id, items: [] };
    },
  );
  t.mock.method(
    ReMeClient.prototype,
    "stream",
    async (
      _job: string,
      _body: unknown,
      signal: AbortSignal,
      listener: (chunk: StreamChunk) => void,
    ) => {
      receive = listener;
      listener({
        chunk_type: "data",
        chunk: {
          type: "snapshot",
          agent_name: "Remy",
          session_id: "session",
          status: "idle",
          agents: [],
          activities: {},
          snapshot: {
            session_id: "session",
            state: { session_id: "session", status: "idle", active_flags: [] },
            messages: [reminder],
          },
        },
      });
      await new Promise<void>((resolve) => {
        if (signal.aborted) resolve();
        else signal.addEventListener("abort", () => resolve(), { once: true });
      });
    },
  );
  const controller = new DesktopController(storage, () => {}, manifest);
  t.after(async () => {
    await controller.close();
    await rm(directory, { recursive: true, force: true });
  });
  controller.start();
  await new Promise((resolve) => setImmediate(resolve));
  assert.ok(receive);
  assert.equal(controller.state.connection, "connected");
  assert.equal(controller.state.unread, 1);
  assert.deepEqual(controller.state.readMessageIds, []);
  // A historical snapshot is suppressed locally without claiming the user read it.
  assert.equal(calls.length, 0);
  await controller.markAllRead();
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0], {
    agent_name: "Remy",
    message_id: reminder.id,
    mark_read: true,
  });
  await storage.flush();
  const presentations = JSON.parse(
    await readFile(path.join(directory, "presentation.json"), "utf8"),
  ) as Record<string, Presentation>;
  const receipt = Object.values(presentations).flatMap((value) =>
    Object.values(value.receipts),
  )[0];
  assert.equal(receipt.userRead, true);
  assert.equal(receipt.readSynced, false);
  receive({
    chunk_type: "data",
    chunk: {
      type: "snapshot",
      agent_name: "Remy",
      session_id: "next-session",
      status: "idle",
      agents: [],
      activities: {},
      snapshot: {
        session_id: "next-session",
        state: { session_id: "next-session", status: "idle", active_flags: [] },
        messages: [],
      },
    },
  });
  offline = false;
  (controller as unknown as { readRetryAt: number }).readRetryAt = 0;
  await controller.markAllRead();
  assert.equal(calls.length, 2);
  await controller.markAllRead();
  assert.equal(calls.length, 2);
  await storage.flush();
  const final = JSON.parse(
    await readFile(path.join(directory, "presentation.json"), "utf8"),
  ) as Record<string, Presentation>;
  const acknowledged = Object.values(final).flatMap((value) =>
    Object.values(value.receipts),
  )[0];
  assert.equal(acknowledged.readSynced, true);
  const restored = new Storage(directory);
  await restored.load();
  assert.deepEqual(restored.presentations, storage.presentations);
});

test("storage rejects the obsolete presentation receipt format", async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-old-receipt-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  await writeFile(
    path.join(directory, "presentation.json"),
    JSON.stringify({
      scope: {
        initialized: true,
        receipts: { old: { read: true, presented: true } },
      },
    }),
  );
  await assert.rejects(
    new Storage(directory).load(),
    (error: unknown) =>
      error instanceof UiError && error.key === "invalidReceipts",
  );
});
