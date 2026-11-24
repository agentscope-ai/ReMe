import { BehaviorController } from "../src/main/behavior.ts";
import assert from "node:assert/strict";
import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { ClientController } from "../../client/src/controller.ts";
import { ReMeClient } from "../../client/src/reme-client.ts";
import { Storage } from "../src/main/storage.ts";
import { UiError } from "../../client/src/i18n.ts";
import type {
  AgentEvent,
  AgentView,
  AgentMessage,
  StreamChunk,
  SettingsUpdate,
} from "../../client/src/protocol.ts";

import {
  agent,
  message as makeMessage,
  snapshot as makeSnapshot,
  clientInfo,
} from "../../client/tests/fixtures.ts";
const agents: AgentView[] = ["Remy", "monitor"].map((name) =>
  agent(name, name + "-session"),
);
const snapshot = (
  agentName: string,
  messages: AgentMessage[] = [],
): AgentEvent => ({
  ...makeSnapshot(messages, agentName + "-session", agentName),
  agents,
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
  const controller = new ClientController(
    storage,
    () => {},
    new BehaviorController(manifest),
  );
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

test("animation playback waits for the companion renderer before starting its timeout", async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-animation-ready-"));
  const manifest = JSON.parse(
    await readFile(
      new URL("../assets/remy/manifest.json", import.meta.url),
      "utf8",
    ),
  );
  const controller = new ClientController(
    new Storage(directory),
    () => {},
    new BehaviorController(manifest),
  );
  t.after(async () => {
    await controller.close();
    await rm(directory, { recursive: true, force: true });
  });
  t.mock.timers.enable({ apis: ["Date"] });
  controller.state.connection = "connected";

  controller.preview("happy");
  t.mock.timers.tick(6000);
  controller.setCompanionReady(false);
  assert.equal(controller.state.behavior.animation.clip, "idle");

  controller.setCompanionReady(true);
  assert.equal(controller.state.behavior.animation.clip, "happy");
  const { id } = controller.state.behavior.animation;
  controller.reportPlayback(id, "started");
  assert.equal(controller.state.behavior.animation.id, id);

  controller.reportPlayback(id, "failed");
  assert.equal(controller.state.behavior.animation.clip, "idle");
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
  let backendInfo = { ...clientInfo, asr: true, tts: true };
  t.mock.method(ReMeClient.prototype, "call", async (job: string) => {
    calls.push(job);
    if (job === "create_chat_session")
      return new Promise<void>((resolve) => {
        completeClear = resolve;
      });
    assert.equal(job, "get_client_info");
    return backendInfo;
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
          chunk_type: "data",
          chunk: makeMessage({
            session_id: agentName === "Remy" ? "Remy-session" : "query-fork",
            turn_id: "turn",
            content: "",
            completed: false,
            outcome: null,
          }),
        });
        if (agentName !== "Remy") {
          taskStream = receive;
          if (holdChild) await untilAbort(signal);
        }
        receive({
          chunk_type: "data",
          chunk: makeMessage({
            session_id: agentName === "Remy" ? "Remy-session" : "query-fork",
            turn_id: "turn",
            content: "Answer",
            outcome: waiting ? "waiting" : "completed",
          }),
        });
      }
    },
  );
  const controller = new ClientController(
    new Storage(directory),
    () => {},
    new BehaviorController(manifest),
  );
  t.after(async () => {
    await controller.close();
    await rm(directory, { recursive: true, force: true });
  });
  controller.start();
  assert.equal(controller.state.clientInfo, null);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(controller.state.connection, "connected");
  assert.deepEqual(controller.state.clientInfo, backendInfo);
  assert.equal(controller.state.speech.asr, true);
  assert.equal(controller.state.speech.tts, true);
  await controller.send("Hello");
  // A command outcome never creates a second display transcript.
  assert.equal(controller.state.pending?.outcome, "completed");
  assert.equal("items" in controller.state.pending!, false);
  const message = makeMessage({
    id: "reply",
    session_id: "Remy-session",
    content: "Answer",
  });
  const observed = t.mock.method(controller.speech, "observe");
  listeners.get("Remy")!({
    chunk_type: "data",
    chunk: snapshot("Remy", [message]),
  });
  assert.equal(controller.state.pending, null);
  assert.equal(controller.state.messages[0].id, "reply");
  assert.equal(observed.mock.calls.length, 1);
  assert.deepEqual(calls, ["get_client_info"]);

  waiting = true;
  await controller.send("Wait for confirmation");
  assert.equal(controller.state.pending?.outcome, "waiting");
  assert.equal(controller.state.pending?.error, undefined);
  waiting = false;

  controller.selectAgent("monitor");
  const result = makeMessage({
    id: "native-result",
    session_id: "monitor-session",
    turn_id: "native-turn",
    content: "Task result",
  });
  for (const source of ["task", undefined] as const) {
    for (const phase of [
      "working",
      "failed",
      "interrupted",
      "completed",
    ] as const) {
      listeners.get("monitor")!({
        chunk_type: "data",
        chunk: {
          ...snapshot("monitor", [result]),
          activities: {
            monitor: {
              source,
              phase,
              turn_id: "batch",
              session_id: phase === "completed" ? "monitor-session" : "",
            },
          },
        },
      });
      assert.deepEqual(
        controller.state.messages.map(({ id, completed }) => [id, completed]),
        [["native-result", true]],
      );
      assert.equal(
        controller.state.taskRunStatus?.tone,
        phase === "completed"
          ? undefined
          : phase === "working"
            ? "working"
            : "error",
      );
    }
  }
  const active = snapshot("monitor");
  active.agent.status = "active";
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
    chunk_type: "data",
    chunk: makeMessage({
      session_id: "query-fork",
      content: "Live progress",
      completed: false,
      outcome: null,
    }),
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

  controller.selectPage("remy");
  assert.equal(controller.state.selectedAgentName, "Remy");
  assert.equal(controller.state.sessionId, "Remy-session");
  controller.selectPage("tasks");
  assert.equal(controller.state.selectedAgentName, "monitor");
  controller.openDocument("notes.md");
  assert.equal(controller.state.activePage, "knowledge");
  controller.setChatVisible(true, {
    kind: "mail",
    agentName: "monitor",
    id: "task-mail",
  });
  assert.equal(controller.state.activePage, "tasks");
  assert.equal(controller.state.focusTarget?.id, "task-mail");
  controller.setChatVisible(true, { kind: "message", id: "reminder" });
  assert.equal(controller.state.activePage, "remy");
  assert.equal(controller.state.selectedAgentName, "Remy");
  controller.setChatVisible(false);
  const chatSnapshot = makeSnapshot([], "Remy-session");
  listeners.get("Remy")!({
    chunk_type: "data",
    chunk: {
      ...chatSnapshot,
      agents,
      agent: { ...chatSnapshot.agent, status: "active" },
    },
  });
  assert.equal(controller.state.status, "active");
  assert.equal(Boolean(controller.state.pending?.streaming), false);
  const clearing = controller.newSession();
  await new Promise((resolve) => setImmediate(resolve));
  assert.ok(completeClear);
  backendInfo = {
    ...backendInfo,
    workspace_dir: "/new-workspace",
    timezone: "UTC",
  };
  await controller.settings({ endpoint: "http://127.0.0.1:2444" });
  assert.equal(controller.state.clientInfo, null);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(controller.state.clientInfo, backendInfo);
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
