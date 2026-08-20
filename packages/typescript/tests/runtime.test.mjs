import assert from "node:assert/strict";
import test from "node:test";
import { ReMeRuntime } from "../dist/dsh/runtime.js";

const CONFIG = {
  autoMemoryEnabled: true,
  autoMemoryInterval: 2,
  autoDreamEnabled: false,
  shutdownTimeoutMs: 50,
  dreamIntervalMs: 0,
  dreamCron: "0 23 * * *",
  dreamHint: "",
  timezone: "Asia/Shanghai",
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
  const status = runtime.snapshot();
  assert.equal(status.autoMemory.queuedTurns, 0);
  assert.equal(status.autoMemory.recentTasks[0].phase, "completed");
  assert.equal(status.autoMemory.recentTasks[0].turns, 2);
  assert.equal(status.autoMemory.recentTasks[0].messages, 4);
});

test("requeues failed auto-memory batches and flushes them on disposal", async () => {
  let attempts = 0;
  const client = {
    async autoMemory() {
      attempts += 1;
      return { ok: attempts > 1, error: "offline" };
    },
  };
  const runtime = new ReMeRuntime(
    client,
    { ...CONFIG, autoMemoryInterval: 1 },
    silentLogger(),
  );
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
  const started = new Promise((resolve) => {
    markStarted = resolve;
  });
  const firstRequest = new Promise((resolve) => {
    releaseFirst = resolve;
  });
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
  const runtime = new ReMeRuntime(
    client,
    { ...CONFIG, autoMemoryInterval: 1 },
    silentLogger(),
  );
  const session = { id: "in-flight-retry-session" };
  completeTurn(runtime, session, 1, 10);
  await started;

  const disposal = runtime.dispose(session);
  releaseFirst();
  await disposal;

  assert.equal(attempts, 2);
  assert.equal(runtime.states.has(session.id), false);
});

test("splits auto-memory batches at workspace date boundaries", async () => {
  const calls = [];
  const client = {
    async autoMemory(messages, _sessionId, options) {
      calls.push({ messages, options });
      return { ok: true };
    },
  };
  const runtime = new ReMeRuntime(
    client,
    { ...CONFIG, autoMemoryInterval: 5 },
    silentLogger(),
  );
  const session = { id: "midnight-session" };

  completeTurn(runtime, session, 1, 10, Date.parse("2026-08-19T15:59:00Z"));
  completeTurn(runtime, session, 2, 20, Date.parse("2026-08-19T16:01:00Z"));
  await runtime.stateFor(session).writes;

  assert.equal(calls.length, 1);
  assert.equal(calls[0].options.date, "2026-08-19");
  await runtime.dispose(session);
  assert.equal(calls.length, 2);
  assert.equal(calls[1].options.date, "2026-08-20");
});

test("retries a failed partial prior-day batch after later activity", async () => {
  const calls = [];
  const client = {
    async autoMemory(messages, _sessionId, options) {
      calls.push({ messages, options });
      return { ok: calls.length > 1, error: "offline" };
    },
  };
  const runtime = new ReMeRuntime(
    client,
    { ...CONFIG, autoMemoryInterval: 5 },
    silentLogger(),
  );
  const session = { id: "midnight-retry-session" };

  completeTurn(runtime, session, 1, 10, Date.parse("2026-08-19T15:59:00Z"));
  completeTurn(runtime, session, 2, 20, Date.parse("2026-08-19T16:01:00Z"));
  await runtime.stateFor(session).writes;
  assert.equal(calls.length, 1);
  assert.equal(runtime.stateFor(session).pendingTurns.length, 2);

  for (let turn = 3; turn <= 7; turn += 1) {
    completeTurn(
      runtime,
      session,
      turn,
      turn * 10,
      Date.parse(`2026-08-20T00:0${turn}:00Z`),
    );
  }
  await runtime.stateFor(session).writes;

  assert.equal(calls.length, 3);
  assert.equal(calls[1].options.date, "2026-08-19");
  assert.equal(calls[1].messages.length, 2);
  assert.equal(calls[2].options.date, "2026-08-20");
  assert.equal(calls[2].messages.length, 10);
  assert.equal(runtime.stateFor(session).pendingTurns.length, 1);
});

test("bounds session disposal and aborts an unresponsive write", async () => {
  let observedSignal;
  const client = {
    async autoMemory(_messages, _sessionId, options) {
      observedSignal = options.signal;
      return new Promise(() => {});
    },
  };
  const runtime = new ReMeRuntime(
    client,
    {
      ...CONFIG,
      autoMemoryInterval: 1,
      shutdownTimeoutMs: 20,
    },
    silentLogger(),
  );
  const session = { id: "stuck-session" };
  completeTurn(runtime, session, 1, 10);

  const started = Date.now();
  await runtime.dispose(session);

  assert.equal(observedSignal.aborted, true);
  assert.ok(Date.now() - started < 500);
  assert.equal(runtime.states.has(session.id), true);
});

test("retains a failed final batch for a later plugin-shutdown retry", async () => {
  let attempts = 0;
  const client = {
    async autoMemory() {
      attempts += 1;
      return { ok: attempts > 2, error: "offline" };
    },
  };
  const runtime = new ReMeRuntime(
    client,
    { ...CONFIG, autoMemoryInterval: 1 },
    silentLogger(),
  );
  const session = { id: "retained-session" };
  completeTurn(runtime, session, 1, 10);
  await runtime.stateFor(session).writes;

  await runtime.dispose(session);
  assert.equal(attempts, 2);
  assert.equal(runtime.states.has(session.id), true);

  await runtime.disposeAll();
  assert.equal(attempts, 3);
  assert.equal(runtime.states.has(session.id), false);
});

test("runs only one auto-dream task at a time", async () => {
  let calls = 0;
  let release;
  const client = {
    async autoDream() {
      calls += 1;
      await new Promise((resolve) => {
        release = resolve;
      });
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
  const runtime = new ReMeRuntime(
    {
      async autoDream() {
        throw new Error("broken transport");
      },
    },
    CONFIG,
    {
      debug() {},
      warn(event, data) {
        warnings.push({ event, data });
      },
      log() {},
    },
  );
  await runtime.runDream();
  assert.equal(warnings.length, 1);
  assert.match(warnings[0].data.error, /broken transport/);
  assert.equal(runtime.snapshot().autoDream.lastResult, "failed");
  assert.match(runtime.snapshot().autoDream.lastError, /broken transport/);
});

test("applies changed batching and dream settings without replacing the runtime", async () => {
  const calls = [];
  let config = { ...CONFIG, autoMemoryInterval: 5, autoDreamEnabled: false };
  const runtime = new ReMeRuntime(
    {
      async autoMemory(messages) {
        calls.push(messages);
        return { ok: true };
      },
      async autoDream() {
        return { ok: true };
      },
    },
    () => config,
    silentLogger(),
  );
  const session = { id: "reconfigured-session" };
  runtime.start();
  completeTurn(runtime, session, 1, 10);
  assert.equal(calls.length, 0);

  config = {
    ...config,
    autoMemoryInterval: 1,
    autoDreamEnabled: true,
    dreamIntervalMs: 100000,
  };
  runtime.reconfigure();
  await runtime.stateFor(session).writes;
  assert.equal(calls.length, 1);
  assert.notEqual(runtime.dreamTimer, null);
  assert.equal(runtime.snapshot().autoDream.enabled, true);
  assert.ok(runtime.snapshot().autoDream.nextRunAt);
  await runtime.disposeAll();
});

function completeTurn(runtime, session, turn, seq, time) {
  runtime.capture(session, { type: "turn/start", data: { turn } });
  runtime.capture(session, {
    type: "user/message",
    seq,
    time,
    data: {
      role: "user",
      content: [{ type: "text", text: `question ${turn}` }],
      source: { kind: "user" },
    },
  });
  runtime.capture(session, {
    type: "assistant/message",
    seq: seq + 1,
    time: time === undefined ? undefined : time + 1000,
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
