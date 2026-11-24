import { BehaviorController } from "../src/main/behavior.ts";
import { snapshot, clientInfo } from "../../client/tests/fixtures.ts";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { ClientController } from "../../client/src/controller.ts";
import { ReMeClient } from "../../client/src/reme-client.ts";
import { Storage } from "../src/main/storage.ts";
import { UiError } from "../../client/src/i18n.ts";
import type {
  Presentation,
  AgentMessage,
  StreamChunk,
} from "../../client/src/protocol.ts";

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
  const controller = new ClientController(
    storage,
    () => {},
    new BehaviorController(manifest),
  );
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

test("storage requires a user-read flag in presentation receipts", async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-invalid-receipt-"));
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
