import assert from "node:assert/strict";
import { readFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { DesktopController } from "../src/main/controller.ts";
import { ReMeClient } from "../src/main/reme-client.ts";
import { Storage } from "../src/main/storage.ts";
import { UiError } from "../src/shared/i18n.ts";
import type {
  AgentEvent,
  AgentView,
  SessionMessage,
  StreamChunk,
} from "../src/shared/protocol.ts";

const agents: AgentView[] = ["chat", "monitor"].map((agent_id) => ({
  agent_id,
  name: agent_id,
  kind: agent_id === "chat" ? "chat" : "event",
  enabled: true,
  running: false,
  status: "idle",
  session_id: agent_id + "-session",
  memory_path: `reme_agents/${agent_id}/MEMORY.md`,
  directory: `reme_agents/${agent_id}`,
  pending_events: 0,
  next_trigger_at: null,
  last_completed_at: null,
  error: null,
}));

const snapshot = (
  agentId: string,
  messages: SessionMessage[] = [],
): AgentEvent => ({
  type: "snapshot",
  agent_id: agentId,
  session_id: agentId + "-session",
  status: "idle",
  agents,
  activities: {},
  snapshot: {
    session_id: agentId + "-session",
    state: {
      session_id: agentId + "-session",
      status: "idle",
      active_flags: [],
    },
    messages,
  },
});

test("authoritative subscriptions reconcile replies and endpoint changes respect child turns", async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-controller-"));
  const manifest = JSON.parse(
    await readFile(
      new URL("../assets/remy/manifest.json", import.meta.url),
      "utf8",
    ),
  );
  const listeners = new Map<string, (chunk: StreamChunk) => void>();
  const calls: string[] = [];
  let holdChild = false;
  let waiting = false;
  t.mock.method(ReMeClient.prototype, "call", async (job: string) => {
    calls.push(job);
    assert.equal(job, "app_config");
    return {
      workspace_dir: "/test-workspace",
      jobs: Object.fromEntries(
        [
          "agent_events",
          "agent_chat",
          "agent_session",
          "agent_new_session",
          "agent_mailbox",
          "agent_event_list",
        ].map((name) => [name, { backend: "base" }]),
      ),
    };
  });
  const untilAbort = (signal: AbortSignal) =>
    new Promise<void>((resolve) => {
      if (signal.aborted) resolve();
      else signal.addEventListener("abort", () => resolve(), { once: true });
    });
  t.mock.method(
    ReMeClient.prototype,
    "stream",
    async (
      job: string,
      body: { agent_id?: string },
      signal: AbortSignal,
      receive: (chunk: StreamChunk) => void,
    ) => {
      const agentId = body.agent_id ?? "chat";
      if (job === "agent_events") {
        listeners.set(agentId, receive);
        receive({ chunk_type: "data", chunk: snapshot(agentId) });
        await untilAbort(signal);
      } else {
        receive({
          chunk_type: "reply_start",
          session_id: agentId + "-session",
          turn_id: "turn",
          chunk: "Answer",
        });
        if (agentId !== "chat" && holdChild) await untilAbort(signal);
        receive({
          chunk_type: "reply_end",
          status: waiting ? "waiting" : "completed",
          chunk: { result: "Answer" },
        });
      }
    },
  );
  const controller = new DesktopController(
    new Storage(directory),
    () => {},
    manifest,
  );
  t.after(async () => {
    await controller.close();
    await rm(directory, { recursive: true, force: true });
  });
  controller.start();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(controller.state.connection, "connected");
  await controller.send("Hello");
  // A command outcome never creates a second display transcript.
  assert.equal(controller.state.pending?.outcome, "completed");
  assert.equal("items" in controller.state.pending!, false);
  const message: SessionMessage = {
    id: "reply",
    role: "assistant",
    source: "chat",
    content: "Answer",
    turn_id: "turn",
    completed: true,
    delivery: "idle",
    notification_eligible: false,
    chunks: [],
  };
  const observed = t.mock.method(controller.speech, "observe");
  listeners.get("chat")!({
    chunk_type: "data",
    chunk: snapshot("chat", [message]),
  });
  assert.equal(controller.state.pending, null);
  assert.equal(controller.state.messages[0].id, "reply");
  assert.equal(observed.mock.calls.length, 1);
  assert.deepEqual(calls, ["app_config"]);

  waiting = true;
  await controller.send("Wait for confirmation");
  assert.equal(controller.state.pending?.outcome, "waiting");
  assert.equal(controller.state.pending?.error, undefined);
  waiting = false;

  controller.selectAgent("monitor");
  holdChild = true;
  const sending = controller.send("Check task");
  await assert.rejects(
    controller.settings({ endpoint: "http://127.0.0.1:2444" }),
    (error: unknown) =>
      error instanceof UiError && error.key === "stopBeforeEndpoint",
  );
  controller.stopReply();
  await sending;
});
