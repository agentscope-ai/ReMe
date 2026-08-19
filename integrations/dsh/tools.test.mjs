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
  const result = await tool.execute({ query: " deployment decision ", limit: 100, min_score: -1 }, {});
  assert.equal(result, "daily/2026-08-19.md: remembered decision");
  assert.deepEqual(calls, [{
    query: "deployment decision",
    options: { limit: 50, minScore: 0 },
  }]);
  assert.deepEqual(tool.output.render({}, result), [{ type: "text", text: result }]);
});

test("reme_search fails closed on empty input and reports service errors", async () => {
  const registered = [];
  registerReMeTools({ tools: { register(tool) { registered.push(tool); } } }, {
    async search() { return { ok: false, error: "offline" }; },
  }, { searchLimit: 5 });
  assert.match(await registered[0].execute({ query: "" }, {}), /cannot be empty/);
  assert.equal(await registered[0].execute({ query: "history" }, {}), "ReMe search failed: offline");
});
