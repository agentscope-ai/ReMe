import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import plugin, {
  OPENCLAW_CONFIG_SCHEMA,
  OPENCLAW_CONFIG_UI_HINTS,
  OpenClawReMeRuntime,
  captureLastTurn,
  openClawSessionId,
  resolveOpenClawConfig,
} from "../dist/openclaw/index.js";

test("normalizes OpenClaw configuration and stable session ids", () => {
  const config = resolveOpenClawConfig(
    { endpoint: "http://localhost:2333///", searchLimit: 99 },
    {},
  );
  assert.equal(config.endpoint, "http://localhost:2333");
  assert.equal(config.searchLimit, 50);
  assert.equal(config.autoMemoryInterval, 5);
  assert.equal(config.autoDreamEnabled, true);
  assert.equal(config.autoRecall, true);
  assert.match(openClawSessionId("agent/session"), /^openclaw-[a-f0-9]{24}$/);
  assert.throws(
    () => resolveOpenClawConfig({ endpoint: "file:///tmp/reme" }, {}),
    /http\(s\)/,
  );
  assert.throws(
    () => resolveOpenClawConfig({ apiKey: "unsupported" }, {}),
    /Unknown ReMe config option/,
  );
  assert.throws(
    () => resolveOpenClawConfig({ autoCapture: true }, {}),
    /Unknown ReMe config option/,
  );
});

test("keeps the runtime schema aligned with the OpenClaw manifest", async () => {
  const manifest = JSON.parse(
    await readFile(new URL("../openclaw.plugin.json", import.meta.url), "utf8"),
  );
  assert.deepEqual(manifest.configSchema, OPENCLAW_CONFIG_SCHEMA);
  assert.deepEqual(manifest.uiHints, OPENCLAW_CONFIG_UI_HINTS);
  assert.deepEqual(manifest.contracts.tools, ["reme_search"]);
  assert.equal(manifest.activation.onStartup, true);
});

test("captures only the last OpenClaw user and assistant pair", () => {
  const messages = captureLastTurn(
    [
      { role: "user", content: "old question" },
      { role: "assistant", content: [{ type: "text", text: "old answer" }] },
      {
        id: "u2",
        role: "user",
        content: [{ type: "text", text: "new question" }],
      },
      { id: "a2", role: "assistant", content: "new answer" },
    ],
    "session",
  );
  assert.deepEqual(
    messages.map((message) => [message.role, message.content[0].text]),
    [
      ["user", "new question"],
      ["assistant", "new answer"],
    ],
  );
});

test("removes recalled context while preserving the current OpenClaw prompt", () => {
  const messages = captureLastTurn(
    [
      {
        role: "user",
        content:
          '<reme-context source="auto-recall">\nremembered deployment\n</reme-context>\n\nremember blue',
      },
      { role: "assistant", content: "noted" },
    ],
    "session",
  );
  assert.deepEqual(
    messages.map((message) => [message.role, message.content[0].text]),
    [
      ["user", "remember blue"],
      ["assistant", "noted"],
    ],
  );
});

test("uses the original prompt when other plugins prepend context around recall", () => {
  const messages = captureLastTurn(
    [
      {
        role: "user",
        content:
          "other plugin context\n\n" +
          '<reme-context source="auto-recall">\nremembered deployment\n</reme-context>\n\n' +
          "another plugin context\n\nremember blue",
      },
      { role: "assistant", content: "noted" },
    ],
    "session",
    "remember blue",
  );
  assert.deepEqual(
    messages.map((message) => [message.role, message.content[0].text]),
    [
      ["user", "remember blue"],
      ["assistant", "noted"],
    ],
  );
});

test("bounds prompts retained when an OpenClaw run ends before agent_end", () => {
  const runtime = new OpenClawReMeRuntime({}, resolveOpenClawConfig({}, {}), {
    warn() {},
  });
  for (let index = 0; index < 300; index += 1) {
    runtime.rememberPrompt(`prompt ${index}`, {
      agentId: "main",
      sessionId: `failed-session-${index}`,
      trigger: "user",
    });
  }

  assert.equal(
    runtime.takePrompt({ agentId: "main", sessionId: "failed-session-0" }),
    undefined,
  );
  assert.equal(
    runtime.takePrompt({ agentId: "main", sessionId: "failed-session-299" }),
    "prompt 299",
  );
});

test("batches OpenClaw memory per session and filters subagents", async () => {
  const calls = [];
  const client = {
    async autoMemory(messages, sessionId, options) {
      calls.push({ messages, sessionId, options });
      return { ok: true };
    },
  };
  const runtime = new OpenClawReMeRuntime(
    client,
    resolveOpenClawConfig(
      { autoMemoryInterval: 2, autoDreamEnabled: false },
      {},
    ),
    { warn() {} },
  );
  const context = {
    runId: "run-1",
    agentId: "main",
    sessionId: "session-1",
    sessionKey: "agent:main:session-1",
    trigger: "user",
  };
  runtime.capture(
    [
      { role: "user", content: "one" },
      { role: "assistant", content: "answer one" },
    ],
    context,
  );
  assert.equal(calls.length, 0);
  runtime.capture(
    [
      { role: "user", content: "two" },
      { role: "assistant", content: "answer two" },
    ],
    { ...context, runId: "run-2" },
  );
  await runtime.states.get("session-1").writes;
  assert.equal(calls.length, 1);
  assert.equal(calls[0].messages.length, 4);
  assert.equal(runtime.snapshot().autoMemory.queuedTurns, 0);
  assert.equal(
    runtime.accepts({
      agentId: "worker",
      sessionKey: "agent:main:subagent:worker",
      trigger: "user",
    }),
    false,
  );
});

test("runs one coalesced OpenClaw Auto Dream task", async () => {
  let resolveDream;
  let calls = 0;
  const runtime = new OpenClawReMeRuntime(
    {
      async autoDream() {
        calls += 1;
        return new Promise((resolve) => {
          resolveDream = resolve;
        });
      },
    },
    resolveOpenClawConfig({ autoMemoryEnabled: false }, {}),
    { warn() {}, debug() {} },
  );
  runtime.start();
  assert.equal(runtime.snapshot().phase, "running");
  assert.ok(runtime.snapshot().autoDream.nextRunAt);
  const first = runtime.runDream();
  const second = runtime.runDream();
  assert.equal(calls, 1);
  resolveDream({ ok: true });
  await Promise.all([first, second]);
  assert.equal(runtime.snapshot().autoDream.lastResult, "completed");
  await runtime.disposeAll();
});

test("registers OpenClaw recall, capture, tool, and shutdown lifecycle", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url, body: JSON.parse(init.body) });
    const answer = url.endsWith("/search") ? "remembered deployment" : "stored";
    return new Response(
      JSON.stringify({ success: true, answer, metadata: {} }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      },
    );
  };
  try {
    const hooks = new Map();
    const tools = [];
    let service;
    plugin.register({
      pluginConfig: {
        endpoint: "http://127.0.0.1:2333",
        autoMemoryInterval: 1,
        autoDreamEnabled: false,
      },
      logger: { info() {}, warn() {}, error() {} },
      registerTool(tool) {
        tools.push(tool);
      },
      on(name, handler) {
        hooks.set(name, handler);
      },
      registerService(value) {
        service = value;
      },
    });

    assert.equal(tools[0].name, "reme_search");
    await service.start();
    const recalled = await hooks.get("before_prompt_build")(
      { prompt: "deployment" },
      {
        runId: "run-1",
        trigger: "user",
        agentId: "main",
        sessionId: "session-1",
      },
    );
    assert.match(recalled.prependContext, /remembered deployment/);
    assert.match(recalled.prependSystemContext, /ReMe/);

    await hooks.get("agent_end")(
      {
        success: true,
        messages: [
          {
            role: "user",
            content:
              "other plugin context\n\n" +
              recalled.prependContext +
              "\n\ndeployment",
          },
          { role: "assistant", content: "noted" },
        ],
      },
      {
        runId: "run-1",
        trigger: "user",
        agentId: "main",
        sessionId: "session-1",
      },
    );
    await service.stop();

    assert.deepEqual(
      calls.map((call) => call.url),
      ["http://127.0.0.1:2333/search", "http://127.0.0.1:2333/auto_memory"],
    );
    assert.equal(calls[1].body.messages[0].content[0].text, "deployment");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
