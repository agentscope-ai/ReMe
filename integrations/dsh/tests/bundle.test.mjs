import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import test from "node:test";
import vm from "node:vm";

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
    "@deepseek-ai/dsh-tools",
    "@deepseek-ai/dsh-typert-protocol",
  ]) {
    assert.equal(manifest.peerDependencies[dependency], "0.1.7-rc.2");
    assert.equal(manifest.peerDependenciesMeta[dependency]?.optional, true);
  }
  assert.equal(manifest.peerDependencies.openclaw, undefined);
  assert.match(patch, /remeMemory: true/);
  assert.match(patch, /id: reme-memory-scope[\s\S]*id: reme-memory\n/);
  assert.doesNotMatch(patch, /id: reme-memory-runtime/);
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
  assert.match(bundle, /plugins\.item/);
  assert.match(bundle, /reme-status/);
  assert.match(bundle, /Personal Knowledge Base/);
  assert.match(statusPage, /个人知识库/);
  assert.match(bundle, /health_check/);
});

test("registers a visible title for the Plugins page", async () => {
  const bundle = await readFile(
    new URL("../dist/client.js", import.meta.url),
    "utf8",
  );
  let module;
  const document = {
    head: { appendChild() {} },
    createElement() {
      return { dataset: {}, remove() {} };
    },
  };
  vm.runInNewContext(bundle, {
    document,
    window: {
      __ModuleLoader__: {
        load(value) {
          module = value;
        },
      },
    },
  });
  const nodeRequire = createRequire(import.meta.url);
  const client = module.factory((name) =>
    name === "@deepseek-ai/dsh-client-ui-primitives" ? {} : nodeRequire(name),
  );
  const registrations = [];
  const ctx = {
    locale: {
      bind() {
        return (key) => (key === "title" ? "ReMe Memory" : key);
      },
      register() {
        return () => {};
      },
    },
    configForms: {
      get() {
        return {};
      },
      whileServed(_namespaces, register) {
        return register();
      },
    },
    slots: {
      inject(_name, register) {
        return register();
      },
      register(options) {
        registrations.push(options);
        return () => {};
      },
    },
    get() {
      return { rpc: {} };
    },
    effect(execute) {
      return execute();
    },
  };
  client.apply(ctx);
  const item = registrations.find((options) => options.name === "plugins.item");
  assert.equal(item.id, "reme-memory");
  assert.equal(item.label(), "ReMe Memory");
});
