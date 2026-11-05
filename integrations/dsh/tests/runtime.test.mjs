import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setImmediate as settle } from "node:timers/promises";
import { ReMeRuntime } from "../dist/runtime.js";

const CONFIG = {
  endpoint: "http://127.0.0.1:2333",
  autoMemoryEnabled: true,
  autoMemoryInterval: 2,
  autoMemoryRetryMs: 10,
  autoMemoryRetryMaxMs: 40,
  autoDreamEnabled: false,
  shutdownTimeoutMs: 50,
  dreamIntervalMs: 0,
  dreamCron: "0 23 * * *",
  dreamHint: "",
  timezone: "Asia/Shanghai",
};

function makeRuntime(
  t,
  client,
  config = CONFIG,
  logger = silentLogger(),
  directory,
) {
  const ownsDirectory = directory === undefined;
  directory ??= mkdtempSync(join(tmpdir(), "reme-dsh-runtime-"));
  const current = typeof config === "function" ? config : () => config;
  const runtime = new ReMeRuntime(
    client,
    () => ({ ...current(), autoMemoryQueueDir: directory }),
    logger,
  );
  t.after(async () => {
    await runtime.disposeAll();
    if (ownsDirectory) rmSync(directory, { recursive: true, force: true });
  });
  return runtime;
}

test("persists partial turns and submits one batch at the interval", async (t) => {
  const calls = [];
  const runtime = makeRuntime(t, {
    async autoMemory(messages, sessionId) {
      calls.push({ messages, sessionId });
      return { ok: true };
    },
  });
  const session = { id: "session-one" };
  completeTurn(runtime, session, 1, 10);
  await runtime.drain();
  assert.equal(calls.length, 0);
  assert.equal(runtime.snapshot().autoMemory.tasksRunning, 0);
  assert.equal(runtime.snapshot().autoMemory.queuedTurns, 1);
  completeTurn(runtime, session, 2, 20);
  await runtime.drain();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].messages.length, 4);
  assert.match(calls[0].sessionId, /^dsh-[a-f0-9]{24}$/);
  assert.equal(runtime.snapshot().autoMemory.queuedTurns, 0);
});

test("backs off across new turns and drains every batch after recovery", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let available = false;
  let attempts = 0;
  const runtime = makeRuntime(t, {
    async autoMemory() {
      attempts++;
      return { ok: available, error: "offline" };
    },
  });
  const session = { id: "outage" };
  for (let turn = 1; turn <= 200; turn++) {
    completeTurn(runtime, session, turn, turn * 10);
    await runtime.drain();
  }
  assert.equal(attempts, 1);
  assert.equal(runtime.snapshot().autoMemory.queuedTurns, 200);
  assert.equal(runtime.states.get(session.id).length, 0);
  t.mock.timers.tick(10);
  await runtime.drain();
  assert.equal(attempts, 2);
  available = true;
  t.mock.timers.tick(19);
  await settle();
  assert.equal(attempts, 2);
  t.mock.timers.tick(1);
  await runtime.drain();
  assert.equal(runtime.snapshot().autoMemory.queuedTurns, 0);
  assert.equal(attempts, 102);
});

for (const status of [413, 200]) {
  test(`isolates a rejected batch (HTTP ${status}) while preserving its session's order`, async (t) => {
    t.mock.timers.enable({ apis: ["setTimeout", "Date"] });
    let accepted = false;
    const calls = [];
    const runtime = makeRuntime(
      t,
      {
        async autoMemory(messages, _session, options) {
          const text = messages[0].content[0].text;
          calls.push({ text, requestId: options.requestId });
          return { ok: accepted || text !== "question 101", status };
        },
      },
      { ...CONFIG, autoMemoryInterval: 1 },
    );
    completeTurn(runtime, { id: "rejected" }, 101, 10);
    t.mock.timers.tick(1);
    completeTurn(runtime, { id: "rejected" }, 102, 20);
    t.mock.timers.tick(1);
    completeTurn(runtime, { id: "healthy" }, 201, 30);
    await runtime.drain();
    assert.deepEqual(
      calls.map((call) => call.text),
      ["question 101", "question 201"],
    );
    assert.equal(runtime.snapshot().autoMemory.queuedTurns, 2);

    t.mock.timers.tick(10);
    await runtime.drain();
    assert.deepEqual(calls[2], calls[0]);
    assert.equal(runtime.snapshot().autoMemory.queuedTurns, 2);
    accepted = true;
    t.mock.timers.tick(19);
    await runtime.drain();
    assert.equal(calls.length, 3);
    t.mock.timers.tick(1);
    await runtime.drain();
    assert.deepEqual(
      calls.slice(3).map((call) => call.text),
      ["question 101", "question 102"],
    );
    assert.equal(runtime.snapshot().autoMemory.queuedTurns, 0);
  });
}

for (const status of [429, 503]) {
  test(`backs off the whole endpoint for HTTP ${status}`, async (t) => {
    t.mock.timers.enable({ apis: ["setTimeout"] });
    let available = false;
    let calls = 0;
    const runtime = makeRuntime(
      t,
      {
        async autoMemory() {
          calls++;
          return { ok: available, status };
        },
      },
      { ...CONFIG, autoMemoryInterval: 1 },
    );
    completeTurn(runtime, { id: "first" }, 1, 10);
    await runtime.drain();
    completeTurn(runtime, { id: "second" }, 1, 10);
    await runtime.drain();
    assert.equal(calls, 1);
    assert.equal(runtime.snapshot().autoMemory.queuedTurns, 2);
    available = true;
    t.mock.timers.tick(10);
    await runtime.drain();
    assert.equal(calls, 3);
    assert.equal(runtime.snapshot().autoMemory.queuedTurns, 0);
  });
}

test("retries ended sessions without another turn and releases session state", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let available = false;
  const runtime = makeRuntime(t, {
    async autoMemory() {
      return { ok: available, error: "offline" };
    },
  });
  const session = { id: "ended" };
  completeTurn(runtime, session, 1, 10);
  await runtime.dispose(session);
  await runtime.drain();
  assert.equal(runtime.states.has(session.id), false);
  assert.equal(runtime.snapshot().autoMemory.queuedTurns, 1);
  available = true;
  t.mock.timers.tick(10);
  await runtime.drain();
  assert.equal(runtime.snapshot().autoMemory.queuedTurns, 0);
});

test("recovers queued and partial batches after restarting the Host", async (t) => {
  const directory = mkdtempSync(join(tmpdir(), "reme-dsh-restart-"));
  const original = makeRuntime(
    t,
    {
      async autoMemory() {
        return { ok: false, error: "offline" };
      },
    },
    CONFIG,
    silentLogger(),
    directory,
  );
  completeTurn(original, { id: "restart" }, 1, 10);
  await original.disposeAll();
  const calls = [];
  const recovered = makeRuntime(
    t,
    {
      async autoMemory(messages) {
        calls.push(messages);
        return { ok: true };
      },
    },
    CONFIG,
    silentLogger(),
    directory,
  );
  recovered.start();
  await recovered.drain();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].length, 2);
  assert.equal(recovered.snapshot().autoMemory.queuedTurns, 0);
  t.after(() => rmSync(directory, { recursive: true, force: true }));
});

test("keeps the same request ID and batch when retrying after a timeout", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const calls = [];
  const runtime = makeRuntime(t, {
    async autoMemory(messages, sessionId, options) {
      calls.push({ messages, sessionId, requestId: options.requestId });
      return { ok: calls.length > 1, error: "timeout" };
    },
  });
  const session = { id: "timeout" };
  completeTurn(runtime, session, 1, 10);
  completeTurn(runtime, session, 2, 20);
  await runtime.drain();
  await runtime.dispose(session);
  t.mock.timers.tick(10);
  await runtime.drain();
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[0], calls[1]);
  const tasks = runtime.snapshot().autoMemory.recentTasks;
  assert.equal(new Set(tasks.map((task) => task.id)).size, tasks.length);
});

test("keeps counts accurate beyond the twenty-entry diagnostic history", async (t) => {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const runtime = makeRuntime(
    t,
    {
      async autoMemory() {
        await gate;
        return { ok: true };
      },
    },
    { ...CONFIG, autoMemoryInterval: 1 },
  );
  const session = { id: "many-batches" };
  for (let n = 1; n <= 25; n++) completeTurn(runtime, session, n, n * 10);
  await settle();
  const status = runtime.snapshot().autoMemory;
  assert.equal(status.tasksRunning, 1);
  assert.equal(status.tasksQueued, 24);
  assert.equal(status.queuedTurns, 25);
  release();
  await runtime.drain();
  assert.equal(runtime.snapshot().autoMemory.recentTasks.length, 20);
});

test("does not put service reply text or file names into diagnostic snapshots", async (t) => {
  const runtime = makeRuntime(
    t,
    {
      async autoMemory() {
        return {
          ok: true,
          answer: "PRIVATE-ANSWER",
          metadata: { path: "PRIVATE-PATH" },
        };
      },
    },
    { ...CONFIG, autoMemoryInterval: 1 },
  );
  completeTurn(runtime, { id: "privacy" }, 1, 10);
  await runtime.drain();
  assert.doesNotMatch(JSON.stringify(runtime.snapshot()), /PRIVATE/);
});

test("keeps whitespace-bearing URL credentials out of snapshots and logs", async (t) => {
  const logs = [];
  const error = "refused http://alice:SECRET CANARY\n@localhost:2333";
  const runtime = makeRuntime(
    t,
    {
      async autoMemory() {
        throw new Error(error);
      },
      async autoDream() {
        return { ok: false, error };
      },
    },
    { ...CONFIG, autoMemoryInterval: 1 },
    {
      debug() {},
      warn(...args) {
        logs.push(args);
      },
    },
  );
  completeTurn(runtime, { id: "credentials" }, 1, 10);
  await runtime.drain();
  await runtime.runDream();
  const snapshot = runtime.snapshot();
  assert.equal(snapshot.autoMemory.recentTasks[0].phase, "failed");
  assert.equal(snapshot.autoDream.lastResult, "failed");
  assert.equal(logs.length, 2);
  assert.doesNotMatch(
    JSON.stringify({ snapshot, logs }),
    /alice|SECRET|CANARY/,
  );
});

test("splits dates and flushes a partial previous day without a new interval", async (t) => {
  const calls = [];
  const runtime = makeRuntime(
    t,
    {
      async autoMemory(messages, _id, options) {
        calls.push({ messages, date: options.date });
        return { ok: true };
      },
    },
    { ...CONFIG, autoMemoryInterval: 5 },
  );
  const session = { id: "midnight" };
  completeTurn(runtime, session, 1, 10, Date.parse("2026-08-19T15:59:00Z"));
  completeTurn(runtime, session, 2, 20, Date.parse("2026-08-19T16:01:00Z"));
  await runtime.drain();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].date, "2026-08-19");
  await runtime.dispose(session);
  await runtime.drain();
  assert.equal(calls[1].date, "2026-08-20");
});

test("aborts shutdown within its budget and leaves an unacknowledged batch on disk", async (t) => {
  let signal;
  const runtime = makeRuntime(
    t,
    {
      async autoMemory(_messages, _id, options) {
        signal = options.signal;
        return new Promise(() => {});
      },
    },
    { ...CONFIG, autoMemoryInterval: 1, shutdownTimeoutMs: 20 },
  );
  completeTurn(runtime, { id: "stuck" }, 1, 10);
  await settle();
  const started = Date.now();
  await runtime.disposeAll();
  assert.equal(signal.aborted, true);
  assert.equal(runtime.snapshot().phase, "stopped");
  assert.ok(Date.now() - started < 500);
  assert.equal(runtime.snapshot().autoMemory.queuedTurns, 1);
});

test("does not retry per ended session while its service is in backoff", async (t) => {
  let calls = 0;
  const runtime = makeRuntime(
    t,
    {
      async autoMemory() {
        calls++;
        return { ok: false };
      },
    },
    { ...CONFIG, autoMemoryInterval: 1 },
  );
  completeTurn(runtime, { id: "first" }, 1, 10);
  await runtime.drain();
  for (let n = 0; n < 5; n++) {
    const session = { id: "ended-" + n };
    completeTurn(runtime, session, 1, 10);
    await runtime.dispose(session);
  }
  assert.equal(calls, 1);
});

test("pauses further sends while disabled and resumes persisted work when enabled", async (t) => {
  let config = { ...CONFIG, autoMemoryInterval: 1 };
  let calls = 0;
  const runtime = makeRuntime(
    t,
    {
      async autoMemory() {
        calls++;
        return { ok: true };
      },
    },
    () => config,
  );
  runtime.capture({ id: "partial" }, { type: "turn/start" });
  runtime.capture(
    { id: "partial" },
    {
      type: "user/message",
      seq: 1,
      data: { content: "unfinished", source: { kind: "user" } },
    },
  );
  config = { ...config, autoMemoryEnabled: false };
  runtime.reconfigure();
  assert.equal(runtime.states.size, 0);
  runtime.outbox.append(
    config.endpoint,
    "session",
    "2026-10-21",
    [{ id: "m", name: "user", role: "user", content: [] }],
    1,
  );
  await runtime.drain();
  assert.equal(calls, 0);
  assert.equal(runtime.snapshot().autoMemory.queuedTurns, 1);
  config = { ...config, autoMemoryEnabled: true };
  runtime.reconfigure();
  await runtime.drain();
  assert.equal(calls, 1);
});

test("a failed old destination does not block a newly configured service", async (t) => {
  let config = { ...CONFIG, autoMemoryInterval: 1 };
  const calls = [];
  const runtime = makeRuntime(
    t,
    {
      async autoMemory(_messages, _session, options) {
        calls.push(options.endpoint);
        return { ok: options.endpoint === "http://new.test" };
      },
    },
    () => config,
  );
  completeTurn(runtime, { id: "old" }, 1, 10);
  await runtime.drain();
  config = { ...config, endpoint: "http://new.test" };
  completeTurn(runtime, { id: "new" }, 1, 10);
  await runtime.drain();
  assert.deepEqual(calls, [CONFIG.endpoint, "http://new.test"]);
  assert.equal(runtime.snapshot().autoMemory.queuedTurns, 1);
});

test("drains a new destination while the old request remains stalled", async (t) => {
  let config = { ...CONFIG, autoMemoryInterval: 1 };
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const calls = [];
  const active = new Map();
  const maximum = new Map();
  const runtime = makeRuntime(
    t,
    {
      async autoMemory(_messages, _session, options) {
        const endpoint = options.endpoint;
        calls.push(endpoint);
        active.set(endpoint, (active.get(endpoint) ?? 0) + 1);
        maximum.set(
          endpoint,
          Math.max(maximum.get(endpoint) ?? 0, active.get(endpoint)),
        );
        await (endpoint === CONFIG.endpoint ? gate : settle());
        active.set(endpoint, active.get(endpoint) - 1);
        return { ok: true };
      },
    },
    () => config,
  );
  try {
    completeTurn(runtime, { id: "old" }, 1, 10);
    await settle();
    config = { ...config, endpoint: "http://new.test" };
    runtime.reconfigure();
    completeTurn(runtime, { id: "new" }, 1, 10);
    completeTurn(runtime, { id: "new" }, 2, 20);
    await settle();
    assert.equal(runtime.snapshot().autoMemory.tasksRunning, 2);
    await settle();
    await settle();
    assert.deepEqual(calls, [
      CONFIG.endpoint,
      "http://new.test",
      "http://new.test",
    ]);
    assert.equal(runtime.snapshot().autoMemory.queuedTurns, 1);
    assert.equal(runtime.snapshot().autoMemory.tasksRunning, 1);
    assert.equal(runtime.snapshot().autoMemory.tasksQueued, 0);
    assert.deepEqual([...maximum.values()], [1, 1]);
  } finally {
    release();
    await runtime.drain();
  }
  assert.equal(runtime.snapshot().autoMemory.queuedTurns, 0);
});

test("aborts all destination workers at the shutdown deadline without discarding batches", async (t) => {
  let config = { ...CONFIG, autoMemoryInterval: 1, shutdownTimeoutMs: 20 };
  const signals = [];
  const runtime = makeRuntime(
    t,
    {
      async autoMemory(_messages, _session, options) {
        signals.push(options.signal);
        return new Promise(() => {});
      },
    },
    () => config,
  );
  completeTurn(runtime, { id: "old" }, 1, 10);
  await settle();
  config = { ...config, endpoint: "http://new.test" };
  runtime.reconfigure();
  completeTurn(runtime, { id: "new" }, 1, 10);
  await settle();
  assert.equal(signals.length, 2);
  await runtime.disposeAll();
  assert.ok(signals.every((signal) => signal.aborted));
  assert.equal(runtime.snapshot().phase, "stopped");
  assert.equal(runtime.snapshot().autoMemory.queuedTurns, 2);
});

test("runs only one auto-dream task at a time", async (t) => {
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
  const runtime = makeRuntime(t, client, CONFIG, silentLogger());
  const first = runtime.runDream();
  const second = runtime.runDream();
  assert.equal(calls, 1);
  release();
  await Promise.all([first, second]);
  assert.equal(calls, 1);
});

test("contains unexpected auto-dream client failures", async (t) => {
  const warnings = [];
  const runtime = makeRuntime(
    t,
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

test("applies changed batching and dream settings without replacing the runtime", async (t) => {
  const calls = [];
  let config = { ...CONFIG, autoMemoryInterval: 5, autoDreamEnabled: false };
  const runtime = makeRuntime(
    t,
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
  await runtime.drain();
  assert.equal(calls.length, 1);
  assert.notEqual(runtime.dreamTimer, null);
  assert.equal(runtime.snapshot().autoDream.enabled, true);
  assert.ok(runtime.snapshot().autoDream.nextRunAt);
  await runtime.disposeAll();
});

test("keeps one auto-dream schedule when reconfigured during a run", async (t) => {
  let calls = 0;
  let releaseFirst;
  const firstRequest = new Promise((resolve) => {
    releaseFirst = resolve;
  });
  let config = {
    ...CONFIG,
    autoDreamEnabled: true,
    dreamIntervalMs: 20,
    shutdownTimeoutMs: 100,
  };
  const runtime = makeRuntime(
    t,
    {
      async autoDream() {
        calls += 1;
        if (calls === 1) await firstRequest;
        return { ok: true };
      },
    },
    () => config,
    silentLogger(),
  );

  runtime.start();
  await delay(25);
  config = { ...config };
  runtime.reconfigure();
  releaseFirst();
  await delay(55);

  assert.ok(calls >= 2 && calls <= 3, `expected one schedule, got ${calls}`);
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

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
