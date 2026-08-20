import assert from "node:assert/strict";
import test from "node:test";
import { apply } from "../dist/dsh/index.js";

test("composes root-agent guidance and reme_search on supported DSH releases", async () => {
  const handlers = new Map();
  const tools = [];
  const cleanups = [];
  const ctx = {
    fiber: { state: 0 },
    logger: { debug() {}, warn() {}, log() {} },
    provide(name, value) {
      assert.equal(name, "remeMemory");
      assert.ok(value);
    },
    plugin() {
      return Promise.resolve();
    },
    inject() {},
    effect(execute) {
      const cleanup = execute();
      cleanups.push(cleanup);
      return cleanup;
    },
    tools: {
      register(tool) {
        tools.push(tool);
        return () => {};
      },
    },
    on(name, handler) {
      handlers.set(name, handler);
    },
  };
  apply(ctx, {
    autoMemoryEnabled: false,
    autoDreamEnabled: false,
    language: "zh",
  });
  assert.equal(tools.length, 1);
  assert.equal(tools[0].name, "reme_search");

  const injected = [];
  const agentCleanups = [];
  const agent = {
    status: "idle",
    session: { id: "root", header: {}, events: [] },
    inject(message) {
      injected.push(message);
    },
    ctx: {
      effect(execute) {
        const cleanup = execute();
        agentCleanups.push(cleanup);
        return cleanup;
      },
    },
  };
  handlers.get("agent/session-start")({ agent, source: "startup" });
  assert.equal(injected.length, 1);
  assert.equal(injected[0].source.kind, "plugin");
  assert.equal(injected[0].source.plugin, "reme-memory");
  assert.match(injected[0].content[0].text, /长期记忆/);

  await Promise.all(agentCleanups.map((cleanup) => cleanup()));
  await Promise.all(cleanups.map((cleanup) => cleanup()));
});

test("keeps prompt injection and capture out of subagents by default", async () => {
  const handlers = new Map();
  const ctx = {
    logger: { debug() {}, warn() {}, log() {} },
    provide() {},
    plugin() {
      return Promise.resolve();
    },
    inject() {},
    effect(execute) {
      return execute();
    },
    tools: {
      register() {
        return () => {};
      },
    },
    on(name, handler) {
      handlers.set(name, handler);
    },
  };
  apply(ctx, { autoDreamEnabled: false });
  let injected = false;
  handlers.get("agent/session-start")({
    agent: {
      status: "idle",
      session: { id: "child", header: { origin: "subagent" }, events: [] },
      inject() {
        injected = true;
      },
      ctx: {
        effect() {
          throw new Error("subagent must not install runtime state");
        },
      },
    },
    source: "startup",
  });
  assert.equal(injected, false);
});

test("registers a ReMe settings namespace and reads changed values for new sessions", () => {
  const handlers = new Map();
  let section;
  let notify;
  const ctx = {
    fiber: { state: 0 },
    logger: { debug() {}, warn() {}, log() {} },
    provide() {},
    plugin() {
      return Promise.resolve();
    },
    effect(execute) {
      return execute();
    },
    tools: {
      register() {
        return () => {};
      },
    },
    on(name, handler) {
      handlers.set(name, handler);
    },
    inject(names, callback) {
      if (!names.includes("settings")) return;
      callback({
        settings: {
          register(ns, _schema, options) {
            section = options.base;
            assert.equal(String(ns), "reme-memory");
            return {
              get: () => section,
              watch(listener) {
                notify = listener;
              },
            };
          },
        },
        effect(execute) {
          return execute();
        },
      });
    },
  };
  apply(ctx, {
    autoMemoryEnabled: false,
    autoDreamEnabled: false,
    language: "en",
  });
  section = { ...section, language: "zh" };
  notify();
  const injected = [];
  handlers.get("agent/session-start")({
    agent: {
      status: "idle",
      session: { id: "settings-session", header: {}, events: [] },
      inject(message) {
        injected.push(message);
      },
      ctx: {
        effect() {
          return () => {};
        },
      },
    },
  });
  assert.match(injected[0].content[0].text, /长期记忆/);
});
