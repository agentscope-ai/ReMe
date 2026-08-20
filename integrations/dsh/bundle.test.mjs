import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("declares an installable isolated DSH bundle compatible with rc.7 and later", async () => {
  const manifest = JSON.parse(await readFile(new URL("./package.json", import.meta.url), "utf8"));
  const patch = await readFile(new URL("./cordis.patch.yml", import.meta.url), "utf8");
  assert.equal(manifest.name, "@agentscope-ai/reme-dsh-memory");
  assert.equal(manifest.main, "./dist/index.js");
  assert.equal(manifest.types, "./dist/index.d.ts");
  assert.deepEqual(manifest.files, ["dist", "cordis.patch.yml", "README.md"]);
  assert.equal(manifest.dsh.bundle.patch, "./cordis.patch.yml");
  assert.equal(manifest.peerDependencies["@deepseek-ai/dsh-llm"], "^0.1.0-rc.7");
  assert.equal(manifest.peerDependencies["@deepseek-ai/dsh-tools"], "^0.1.0-rc.7");
  assert.match(patch, /remeMemory: true/);
  assert.match(patch, /@agentscope-ai\/reme-dsh-memory/);
});
