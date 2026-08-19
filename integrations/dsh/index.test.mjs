import assert from "node:assert/strict";
import test from "node:test";
import { apply } from "./dist/index.js";

test("composes the root-agent guidance and reme_search tool on DSH rc.7", async () => {
  const handlers = new Map();
  const tools = [];
  const cleanups = [];
  const ctx = {
    logger: { debug() {}, warn() {}, log() {} },
    provide(name, value) {
      assert.equal(name, "remeMemory");
      assert.ok(value);
    },
    effect(execute) {
      const cleanup = execute();
      cleanups.push(cleanup);
      return cleanup;
    },
    tools: { register(tool) { tools.push(tool); } },
    on(name, handler) { handlers.set(name, handler); },
  };
  apply(ctx, { autoMemoryEnabled: false, autoDreamEnabled: false, language: "zh" });
  assert.equal(tools.length, 1);
  assert.equal(tools[0].name, "reme_search");

  const injected = [];
  const agentCleanups = [];
  const agent = {
    status: "idle",
    session: { id: "root", header: {}, events: [] },
    inject(message) { injected.push(message); },
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

  await Promise.all(agentCleanups.map(cleanup => cleanup()));
  await Promise.all(cleanups.map(cleanup => cleanup()));
});

test("keeps prompt injection and capture out of subagents by default", async () => {
  const handlers = new Map();
  const ctx = {
    logger: { debug() {}, warn() {}, log() {} },
    provide() {},
    effect(execute) { return execute(); },
    tools: { register() {} },
    on(name, handler) { handlers.set(name, handler); },
  };
  apply(ctx, { autoDreamEnabled: false });
  let injected = false;
  handlers.get("agent/session-start")({
    agent: {
      status: "idle",
      session: { id: "child", header: { origin: "subagent" }, events: [] },
      inject() { injected = true; },
      ctx: { effect() { throw new Error("subagent must not install runtime state"); } },
    },
    source: "startup",
  });
  assert.equal(injected, false);
});
