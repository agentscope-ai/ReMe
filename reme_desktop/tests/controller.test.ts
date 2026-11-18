import assert from "node:assert/strict";
import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
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
  SettingsUpdate,
} from "../src/shared/protocol.ts";

const agents: AgentView[] = ["Remy", "monitor"].map((agent_name) => ({
  agent_name,
  ...(agent_name === "Remy"
    ? { agent_type: "chat", memory_path: "reme_agents/Remy/MEMORY.md" }
    : {
        agent_type: "proactive",
        task_path: `reme_agents/${agent_name}/TASK.md`,
      }),
  enabled: true,
  running: false,
  status: "idle",
  session_id: agent_name + "-session",
  directory: `reme_agents/${agent_name}`,
  pending_events: 0,
  next_trigger_at: null,
  last_completed_at: null,
  error: null,
}));

const snapshot = (
  agentName: string,
  messages: SessionMessage[] = [],
): AgentEvent => ({
  type: "snapshot",
  agent_name: agentName,
  session_id: agentName + "-session",
  status: "idle",
  agents,
  activities: {},
  snapshot: {
    session_id: agentName + "-session",
    state: {
      session_id: agentName + "-session",
      status: "idle",
      active_flags: [],
    },
    messages,
  },
});

test("background transparency defaults off, persists both choices and rejects invalid updates", async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-appearance-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  await writeFile(
    path.join(directory, "settings.json"),
    JSON.stringify({ language: "en", theme: "dark" }),
  );
  const storage = new Storage(directory);
  assert.equal(storage.settings.transparentBackground, false);
  await storage.load();
  assert.equal(storage.settings.transparentBackground, false);
  const manifest = JSON.parse(
    await readFile(
      new URL("../assets/remy/manifest.json", import.meta.url),
      "utf8",
    ),
  );
  const controller = new DesktopController(storage, () => {}, manifest);
  t.after(() => controller.close());
  for (const transparentBackground of [true, false]) {
    await controller.settings({ transparentBackground });
    assert.equal(
      controller.state.settings.transparentBackground,
      transparentBackground,
    );
    const restored = new Storage(directory);
    await restored.load();
    assert.equal(
      restored.settings.transparentBackground,
      transparentBackground,
    );
    assert.equal(restored.settings.theme, "dark");
    assert.equal(restored.settings.language, "en");
  }
  await assert.rejects(
    controller.settings({
      transparentBackground: "true",
    } as unknown as SettingsUpdate),
    (error: unknown) =>
      error instanceof UiError && error.key === "invalidTransparency",
  );
  assert.equal(controller.state.settings.transparentBackground, false);
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
  let completeClear: (() => void) | undefined;
  let taskStream: ((chunk: StreamChunk) => void) | undefined;
  t.mock.method(ReMeClient.prototype, "call", async (job: string) => {
    calls.push(job);
    if (job === "create_chat_session")
      return new Promise<void>((resolve) => {
        completeClear = resolve;
      });
    assert.equal(job, "get_app_config");
    return {
      workspace_dir: "/test-workspace",
      jobs: Object.fromEntries(
        [
          "watch_agent_session",
          "ask_agent",
          "read_agent_session",
          "create_chat_session",
          "list_agent_outbox",
          "read_agent_reminder",
          "list_agent_inbox",
          "asr",
          "tts",
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
      body: { agent_name?: string },
      signal: AbortSignal,
      receive: (chunk: StreamChunk) => void,
    ) => {
      const agentName = body.agent_name ?? "Remy";
      if (job === "watch_agent_session") {
        listeners.set(agentName, receive);
        receive({ chunk_type: "data", chunk: snapshot(agentName) });
        await untilAbort(signal);
      } else {
        receive({
          chunk_type: "reply_start",
          session_id: agentName === "Remy" ? "Remy-session" : "query-fork",
          turn_id: "turn",
          chunk: "Answer",
        });
        if (agentName !== "Remy") {
          taskStream = receive;
          if (holdChild) await untilAbort(signal);
        }
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
  assert.equal(controller.state.speech.asr, true);
  assert.equal(controller.state.speech.tts, true);
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
  listeners.get("Remy")!({
    chunk_type: "data",
    chunk: snapshot("Remy", [message]),
  });
  assert.equal(controller.state.pending, null);
  assert.equal(controller.state.messages[0].id, "reply");
  assert.equal(observed.mock.calls.length, 1);
  assert.deepEqual(calls, ["get_app_config"]);

  waiting = true;
  await controller.send("Wait for confirmation");
  assert.equal(controller.state.pending?.outcome, "waiting");
  assert.equal(controller.state.pending?.error, undefined);
  waiting = false;

  controller.selectAgent("monitor");
  const active = snapshot("monitor");
  active.status = "active";
  listeners.get("monitor")!({ chunk_type: "data", chunk: active });
  await controller.send("Current progress?");
  assert.equal(controller.state.sessionId, "monitor-session");
  assert.equal(controller.state.pending?.sessionId, "query-fork");
  assert.deepEqual(
    controller.state.messages.map((message) => message.content),
    ["Current progress?", "Answer"],
  );
  assert.equal(controller.state.messages[1].completed, true);
  listeners.get("monitor")!({ chunk_type: "data", chunk: active });
  assert.equal(controller.state.messages[1].content, "Answer");
  holdChild = true;
  const sending = controller.send("Check task");
  taskStream!({
    chunk_type: "content",
    block_id: "answer",
    event: "delta",
    chunk: "Live progress",
  });
  assert.equal(controller.state.messages[1].content, "Live progress");
  await assert.rejects(
    controller.send("Another question"),
    (error: unknown) => error instanceof UiError && error.key === "busy",
  );
  await assert.rejects(
    controller.settings({ endpoint: "http://127.0.0.1:2444" }),
    (error: unknown) =>
      error instanceof UiError && error.key === "stopBeforeEndpoint",
  );
  controller.stopReply();
  await sending;

  controller.selectAgent("Remy");
  const clearing = controller.newSession();
  await new Promise((resolve) => setImmediate(resolve));
  assert.ok(completeClear);
  await controller.settings({ endpoint: "http://127.0.0.1:2444" });
  await new Promise((resolve) => setImmediate(resolve));
  await controller.send("Keep this new endpoint's reply");
  const pending = controller.state.pending;
  assert.ok(pending);
  completeClear();
  await assert.rejects(
    clearing,
    (error: unknown) =>
      error instanceof UiError && error.key === "sessionUnavailable",
  );
  assert.equal(controller.state.pending, pending);
});
