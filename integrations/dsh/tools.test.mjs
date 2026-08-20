import assert from "node:assert/strict";
import test from "node:test";
import { registerReMeTools } from "./dist/tools.js";

test("reme_search uses the ReMe search contract and renders model-facing text", async () => {
  const registered = [];
  const calls = [];
  registerReMeTools({
    tools: { register(tool) { registered.push(tool); } },
  }, {
    async search(query, options) {
      calls.push({ query, options });
      return { ok: true, answer: "daily/2026-08-19.md: remembered decision" };
    },
  }, { searchLimit: 5 });

  assert.equal(registered.length, 1);
  const tool = registered[0];
  assert.equal(tool.name, "reme_search");
  const controller = new AbortController();
  const result = await tool.execute(
    { query: " deployment decision ", limit: 100, min_score: -1 },
    { signal: controller.signal },
  );
  assert.equal(result, "daily/2026-08-19.md: remembered decision");
  assert.deepEqual(calls, [{
    query: "deployment decision",
    options: { limit: 50, minScore: 0, signal: controller.signal },
  }]);
  assert.deepEqual(tool.output.render({}, result), [{ type: "text", text: result }]);
});

test("reme_search fails closed on empty input and reports service errors", async () => {
  const registered = [];
  registerReMeTools({ tools: { register(tool) { registered.push(tool); } } }, {
    async search() { return { ok: false, error: "offline" }; },
  }, { searchLimit: 5 });
  const exec = { signal: new AbortController().signal };
  assert.match(await registered[0].execute({ query: "" }, exec), /cannot be empty/);
  assert.equal(await registered[0].execute({ query: "history" }, exec), "ReMe search failed: offline");
});

test("reme_search propagates caller cancellation", async () => {
  const registered = [];
  let observedSignal;
  registerReMeTools({ tools: { register(tool) { registered.push(tool); } } }, {
    async search(_query, options) {
      observedSignal = options.signal;
      return new Promise(resolve => {
        options.signal.addEventListener("abort", () => resolve({ ok: false, error: "cancelled" }), { once: true });
      });
    },
  }, { searchLimit: 5 });
  const controller = new AbortController();
  const request = registered[0].execute({ query: "history" }, { signal: controller.signal });
  controller.abort();
  await request;
  assert.equal(observedSignal, controller.signal);
});
