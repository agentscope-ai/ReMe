import assert from "node:assert/strict";
import test from "node:test";
import { ReMeRuntime } from "./dist/runtime.js";

const CONFIG = {
  autoMemoryEnabled: true,
  autoMemoryInterval: 2,
  autoDreamEnabled: false,
  dreamIntervalMs: 0,
  dreamCron: "0 23 * * *",
  dreamHint: "",
};

test("submits completed turns to auto-memory in background batches", async () => {
  const calls = [];
  const client = {
    async autoMemory(messages, sessionId) {
      calls.push({ messages, sessionId });
      return { ok: true };
    },
  };
  const runtime = new ReMeRuntime(client, CONFIG, silentLogger());
  const session = { id: "session-one" };

  completeTurn(runtime, session, 1, 10);
  await runtime.stateFor(session).writes;
  assert.equal(calls.length, 0);

  completeTurn(runtime, session, 2, 20);
  await runtime.stateFor(session).writes;
  assert.equal(calls.length, 1);
  assert.equal(calls[0].messages.length, 4);
  assert.match(calls[0].sessionId, /^dsh-[a-f0-9]{24}$/);
});

test("requeues failed auto-memory batches and flushes them on disposal", async () => {
  let attempts = 0;
  const client = {
    async autoMemory() {
      attempts += 1;
      return { ok: attempts > 1, error: "offline" };
    },
  };
  const runtime = new ReMeRuntime(client, { ...CONFIG, autoMemoryInterval: 1 }, silentLogger());
  const session = { id: "retry-session" };
  completeTurn(runtime, session, 1, 10);
  await runtime.stateFor(session).writes;
  assert.equal(runtime.stateFor(session).pendingTurns.length, 1);

  await runtime.dispose(session);
  assert.equal(attempts, 2);
  assert.equal(runtime.states.has(session.id), false);
});

test("retries an in-flight failed batch before disposal completes", async () => {
  let attempts = 0;
  let markStarted;
  let releaseFirst;
  const started = new Promise(resolve => { markStarted = resolve; });
  const firstRequest = new Promise(resolve => { releaseFirst = resolve; });
  const client = {
    async autoMemory() {
      attempts += 1;
      if (attempts === 1) {
        markStarted();
        await firstRequest;
        return { ok: false, error: "offline" };
      }
      return { ok: true };
    },
  };
  const runtime = new ReMeRuntime(client, { ...CONFIG, autoMemoryInterval: 1 }, silentLogger());
  const session = { id: "in-flight-retry-session" };
  completeTurn(runtime, session, 1, 10);
  await started;

  const disposal = runtime.dispose(session);
  releaseFirst();
  await disposal;

  assert.equal(attempts, 2);
  assert.equal(runtime.states.has(session.id), false);
});

test("runs only one auto-dream task at a time", async () => {
  let calls = 0;
  let release;
  const client = {
    async autoDream() {
      calls += 1;
      await new Promise(resolve => { release = resolve; });
      return { ok: true };
    },
  };
  const runtime = new ReMeRuntime(client, CONFIG, silentLogger());
  const first = runtime.runDream();
  const second = runtime.runDream();
  assert.equal(calls, 1);
  release();
  await Promise.all([first, second]);
  assert.equal(calls, 1);
});

test("contains unexpected auto-dream client failures", async () => {
  const warnings = [];
  const runtime = new ReMeRuntime({
    async autoDream() { throw new Error("broken transport"); },
  }, CONFIG, {
    debug() {},
    warn(event, data) { warnings.push({ event, data }); },
    log() {},
  });
  await runtime.runDream();
  assert.equal(warnings.length, 1);
  assert.match(warnings[0].data.error, /broken transport/);
});

function completeTurn(runtime, session, turn, seq) {
  runtime.capture(session, { type: "turn/start", data: { turn } });
  runtime.capture(session, {
    type: "user/message",
    seq,
    data: {
      role: "user",
      content: [{ type: "text", text: `question ${turn}` }],
      source: { kind: "user" },
    },
  });
  runtime.capture(session, {
    type: "assistant/message",
    seq: seq + 1,
    data: {
      message: {
        role: "assistant",
        content: [{ type: "text", text: `answer ${turn}` }],
        source: { kind: "model" },
      },
    },
  });
  runtime.capture(session, {
    type: "turn/end",
    data: { turn, reason: { kind: "completed" } },
  });
}

function silentLogger() {
  return { debug() {}, warn() {}, log() {} };
}
