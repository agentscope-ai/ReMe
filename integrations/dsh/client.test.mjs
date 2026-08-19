import assert from "node:assert/strict";
import test from "node:test";
import { ReMeClient } from "./dist/client.js";

test("calls ReMe jobs with their native request and response envelopes", async () => {
  const calls = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    calls.push({ url, body: JSON.parse(init.body) });
    return new Response(JSON.stringify({ success: true, answer: "memory result", metadata: {} }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  try {
    const client = new ReMeClient({
      endpoint: "http://127.0.0.1:2333",
      requestTimeoutMs: 1000,
      backgroundTimeoutMs: 1000,
      apiKey: "",
    });
    const result = await client.search("deployment", { limit: 5, minScore: 0 });
    assert.equal(result.ok, true);
    assert.equal(result.answer, "memory result");
    assert.deepEqual(calls, [{
      url: "http://127.0.0.1:2333/search",
      body: { query: "deployment", limit: 5, min_score: 0 },
    }]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
