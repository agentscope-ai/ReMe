import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import plugin, {
  OPENCLAW_CONFIG_SCHEMA,
  OpenClawReMeRuntime,
  captureLastTurn,
  openClawSessionId,
  resolveOpenClawConfig,
} from "../dist/openclaw/index.js";

test("normalizes OpenClaw configuration and stable session ids", () => {
  const config = resolveOpenClawConfig(
    { endpoint: "http://localhost:2333///", recallLimit: 99 },
    {},
  );
  assert.equal(config.endpoint, "http://localhost:2333");
  assert.equal(config.recallLimit, 50);
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
});

test("keeps the runtime schema aligned with the OpenClaw manifest", async () => {
  const manifest = JSON.parse(
    await readFile(new URL("../openclaw.plugin.json", import.meta.url), "utf8"),
  );
  assert.deepEqual(manifest.configSchema, OPENCLAW_CONFIG_SCHEMA);
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
      pluginConfig: { endpoint: "http://127.0.0.1:2333" },
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
    const recalled = await hooks.get("before_agent_start")(
      { prompt: "deployment" },
      { trigger: "user", agentId: "main", sessionId: "session-1" },
    );
    assert.match(recalled.prependContext, /remembered deployment/);

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
      { trigger: "user", agentId: "main", sessionId: "session-1" },
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
