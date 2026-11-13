import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
let server;
let api;
const originalFetch = globalThis.fetch;
before(async () => {
  server = await createServer({
    root: fileURLToPath(new URL("..", import.meta.url)),
    configFile: false,
    logLevel: "silent",
    server: { middlewareMode: true },
  });
  api = await server.ssrLoadModule("/app/api.ts");
});
after(async () => {
  globalThis.fetch = originalFetch;
  await server?.close();
});

test("malformed successful HTTP responses fail both ordinary and session reads", async () => {
  for (const payload of [null, [], {}, { success: "true" }, { success: 1 }]) {
    globalThis.fetch = async () => Response.json(payload);
    await assert.rejects(
      api.getAppConfig(),
      (error) => error instanceof Error && !(error instanceof TypeError),
    );
    await assert.rejects(
      api.readChatSession("session", new AbortController().signal),
      (error) => error instanceof Error && !(error instanceof TypeError),
    );
  }
  globalThis.fetch = async () => new Response("not json");
  await assert.rejects(api.getAppConfig());
  await assert.rejects(
    api.readChatSession("session", new AbortController().signal),
  );
});

test("response failures retain their service error and unsupported session capability stays explicit", async () => {
  globalThis.fetch = async () =>
    Response.json({ success: false, answer: "Read denied" });
  await assert.rejects(api.getAppConfig(), /Read denied/);
  globalThis.fetch = async () =>
    Response.json({
      success: false,
      metadata: { code: "session_not_supported" },
    });
  assert.equal(
    await api.readChatSession("session", new AbortController().signal),
    undefined,
  );
  globalThis.fetch = async () =>
    Response.json({ success: true, answer: { workspace_dir: "/workspace" } });
  assert.deepEqual(await api.getAppConfig(), { workspace_dir: "/workspace" });
});
