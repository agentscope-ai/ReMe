import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("declares one installable DeepSeek Harness plugin", async () => {
  const manifest = JSON.parse(
    await readFile(new URL("../package.json", import.meta.url), "utf8"),
  );
  const patch = await readFile(
    new URL("../cordis.patch.yml", import.meta.url),
    "utf8",
  );
  assert.equal(manifest.name, "@agentscope-ai/reme-dsh-plugin");
  assert.equal(manifest.exports["."].import, "./dist/index.js");
  assert.equal(manifest.exports["./client"].default, "./dist/client.js");
  assert.equal(manifest.dsh.client.platform, "web");
  assert.equal(manifest.dsh.bundle.patch, "./cordis.patch.yml");
  assert.equal(manifest.dependencies, undefined);
  for (const dependency of [
    "@deepseek-ai/dsh-client-ui-primitives",
    "@deepseek-ai/dsh-llm",
    "@deepseek-ai/dsh-settings",
    "@deepseek-ai/dsh-tools",
    "@deepseek-ai/dsh-typert-protocol",
  ]) {
    assert.equal(manifest.peerDependencies[dependency], "^0.1.5-rc.2");
    assert.equal(manifest.peerDependenciesMeta[dependency]?.optional, true);
  }
  assert.equal(manifest.peerDependencies.openclaw, undefined);
  assert.match(patch, /remeMemory: true/);
  assert.doesNotMatch(patch, /@agentscope-ai\/reme\/dsh/);
  assert.equal(patch.match(/@agentscope-ai\/reme-dsh-plugin/g)?.length, 1);
  assert.doesNotMatch(patch, /reme-memory-client/);
});

test("builds a lazy DSH browser module for the ReMe settings card", async () => {
  const bundle = await readFile(
    new URL("../dist/client.js", import.meta.url),
    "utf8",
  );
  const statusPage = await readFile(
    new URL("../src/client/status-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(bundle, /window\.__ModuleLoader__\.load/);
  assert.match(bundle, /id: "@agentscope-ai\/reme-dsh-plugin"/);
  assert.match(bundle, /settings\.plugin\.item/);
  assert.match(bundle, /reme-status/);
  assert.match(bundle, /Personal Knowledge Base/);
  assert.match(statusPage, /个人知识库/);
  assert.match(bundle, /health_check/);
});
