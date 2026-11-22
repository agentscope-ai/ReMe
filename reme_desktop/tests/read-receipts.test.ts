import { snapshot, clientInfo } from "./fixtures.ts";
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
  AgentMessage,
  StreamChunk,
} from "../src/shared/protocol.ts";

const reminder: AgentMessage = {
  id: "reme-proactive-reminder-combined",
  role: "assistant",
  source: "proactive",
  content: "A relevant change",
  completed: true,
  delivery: "idle",
  notification_eligible: true,
  origin_session_id: null,
  session_id: "session",
  items: [],
  turn_id: null,
  created_at: null,
  outcome: "completed",
  error: null,
};

test("explicit chat reading persists locally while historical presentation leaves messages unread", async (t) => {
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
  const calls: string[] = [];
  t.mock.method(ReMeClient.prototype, "call", async (job: string) => {
    calls.push(job);
    assert.equal(job, "get_client_info");
    return clientInfo;
  });
  t.mock.method(
    ReMeClient.prototype,
    "stream",
    async (
      _job: string,
      _body: unknown,
      signal: AbortSignal,
      listener: (chunk: StreamChunk) => void,
    ) => {
      listener({
        chunk_type: "data",
        chunk: snapshot([reminder]),
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
  assert.equal(controller.state.connection, "connected");
  assert.equal(controller.state.unread, 1);
  assert.deepEqual(controller.state.readMessageIds, []);
  // A historical snapshot is suppressed locally without claiming the user read it.
  controller.markAllRead();
  assert.equal(controller.state.unread, 0);
  assert.deepEqual(controller.state.readMessageIds, [reminder.id]);
  assert.deepEqual(calls, ["get_client_info"]);
  await storage.flush();
  const presentations = JSON.parse(
    await readFile(path.join(directory, "presentation.json"), "utf8"),
  ) as Record<string, Presentation>;
  const receipt = Object.values(presentations).flatMap((value) =>
    Object.values(value.receipts),
  )[0];
  assert.deepEqual(receipt, { presented: true, userRead: true });
  const restored = new Storage(directory);
  await restored.load();
  assert.deepEqual(restored.presentations, storage.presentations);
});

test("storage loads user reading from receipts with a legacy sync flag", async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-old-receipt-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  await writeFile(
    path.join(directory, "presentation.json"),
    JSON.stringify({
      scope: {
        initialized: true,
        receipts: {
          old: { presented: true, userRead: true, readSynced: false },
        },
      },
    }),
  );
  const storage = new Storage(directory);
  await storage.load();
  assert.equal(storage.presentation("scope").receipts.old.userRead, true);
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
