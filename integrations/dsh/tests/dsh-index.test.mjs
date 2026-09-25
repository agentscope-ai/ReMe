import assert from "node:assert/strict";
import test from "node:test";
import { apply, Config } from "../dist/index.js";

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
  const nextStep = [];
  const agent = {
    status: "idle",
    session: { id: "root", header: {}, events: [] },
    inbox: { nextStep },
    inject(message) {
      injected.push(message);
      nextStep.push(message);
    },
    ctx: {
      effect(execute) {
        const cleanup = execute();
        agentCleanups.push(cleanup);
        return cleanup;
      },
    },
  };
  handlers.get("agent/created")({ agent, source: "startup" });
  assert.equal(injected.length, 1);
  assert.equal(injected[0].source.kind, "reme-memory");
  assert.match(injected[0].content[0].text, /长期记忆/);

  handlers.get("agent/created")({ agent, source: "resume" });
  assert.equal(injected.length, 1);

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
  handlers.get("agent/created")({
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

test("reads live DSH configuration for new sessions", async () => {
  const handlers = new Map();
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
  };
  const config = await Config["~standard"].validate({
    autoMemoryEnabled: false,
    autoDreamEnabled: false,
    language: "zh",
  });
  assert.equal(config.issues, undefined);
  apply(ctx, config.value);
  const injected = [];
  handlers.get("agent/created")({
    agent: {
      status: "idle",
      session: { id: "settings-session", header: {}, events: [] },
      inbox: { nextStep: [] },
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

  config.value.language[Symbol.for("cosmokit.volatile.write")]("en");
  handlers.get("loader/volatile-update")([["language"]]);
  handlers.get("agent/created")({
    agent: {
      status: "idle",
      session: { id: "updated-session", header: {}, events: [] },
      inbox: { nextStep: [] },
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
  assert.match(injected[1].content[0].text, /Long-term Memory/);
});
