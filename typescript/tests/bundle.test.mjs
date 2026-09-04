import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("declares DSH and OpenClaw entries in one installable package", async () => {
  const manifest = JSON.parse(
    await readFile(new URL("../package.json", import.meta.url), "utf8"),
  );
  const patch = await readFile(
    new URL("../dsh/cordis.patch.yml", import.meta.url),
    "utf8",
  );
  const openClawManifest = JSON.parse(
    await readFile(new URL("../openclaw.plugin.json", import.meta.url), "utf8"),
  );
  assert.equal(manifest.name, "@agentscope-ai/reme");
  assert.equal(manifest.exports["./dsh"].import, "./dist/dsh/index.js");
  assert.equal(
    manifest.exports["./openclaw"].import,
    "./dist/openclaw/index.js",
  );
  assert.equal(manifest.exports["./client"].default, "./dist/dsh/client.js");
  assert.equal(manifest.exports["./package.json"], "./package.json");
  assert.equal(manifest.dsh.client.platform, "web");
  assert.ok(
    manifest.dsh.client.inject.includes(
      "@deepseek-ai/dsh-client-ui-settings-plugins",
    ),
  );
  assert.ok(
    manifest.dsh.client.inject.includes(
      "@deepseek-ai/dsh-client-ui-primitives",
    ),
  );
  assert.ok(
    manifest.dsh.client.inject.includes("@deepseek-ai/dsh-client-connection"),
  );
  assert.ok(
    !manifest.dsh.client.inject.includes("@deepseek-ai/dsh-client-runtime"),
  );
  assert.equal(manifest.dsh.bundle.patch, "./dsh/cordis.patch.yml");
  assert.deepEqual(manifest.openclaw.extensions, ["./dist/openclaw/index.js"]);
  assert.equal(
    manifest.peerDependencies["@deepseek-ai/dsh-llm"],
    "^0.1.2-rc.1",
  );
  assert.equal(
    manifest.peerDependencies["@deepseek-ai/dsh-tools"],
    "^0.1.2-rc.1",
  );
  assert.match(patch, /remeMemory: true/);
  assert.match(patch, /@agentscope-ai\/reme\/dsh/);
  assert.match(patch, /name: ["']@agentscope-ai\/reme["']$/m);
  assert.equal(openClawManifest.id, "reme");
  assert.equal(openClawManifest.kind, "memory");
});

test("builds a lazy DSH browser module for the ReMe settings card", async () => {
  const bundle = await readFile(
    new URL("../dist/dsh/client.js", import.meta.url),
    "utf8",
  );
  const statusPage = await readFile(
    new URL("../src/dsh/client/status-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(bundle, /window\.__ModuleLoader__\.load/);
  assert.match(bundle, /id: "@agentscope-ai\/reme"/);
  assert.match(bundle, /settings\.plugin\.item/);
  assert.match(bundle, /settings\.section/);
  assert.match(bundle, /reme-status/);
  assert.match(bundle, /Personal Knowledge Base/);
  assert.match(statusPage, /个人知识库/);
  assert.match(bundle, /health_check/);
  assert.match(bundle, /reme-settings-component-grid/);
});
