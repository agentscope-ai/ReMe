import assert from "node:assert/strict";
import test from "node:test";
import { resolveConfig } from "./dist/config.js";

test("resolves the established ReMe host and port environment", () => {
  const config = resolveConfig({}, { REME_HOST: "memory.local", REME_PORT: "2444" });
  assert.equal(config.endpoint, "http://memory.local:2444");
  assert.equal(config.autoMemoryInterval, 5);
  assert.equal(config.dreamCron, "0 23 * * *");
});

test("normalizes bounded plugin configuration", () => {
  const config = resolveConfig({
    endpoint: "http://localhost:2333///",
    language: "zh",
    autoMemoryInterval: 0,
    searchLimit: 100,
    rootAgentsOnly: false,
  }, {});
  assert.equal(config.endpoint, "http://localhost:2333");
  assert.equal(config.language, "zh");
  assert.equal(config.autoMemoryInterval, 1);
  assert.equal(config.searchLimit, 50);
  assert.equal(config.rootAgentsOnly, false);
});
