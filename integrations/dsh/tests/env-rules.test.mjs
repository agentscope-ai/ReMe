import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  canHostService,
  hasCoreFeatures,
} from "../dist/environment/dependency.js";
import {
  isLocalEndpoint,
  parseLocalEndpoint,
} from "../dist/environment/endpoint-rules.js";
import { ReMeEnvManager } from "../dist/environment/host/manager.js";

/** A manager whose process work is replaced by facts. */
function managerIn(t) {
  const directory = mkdtempSync(join(tmpdir(), "reme-rules-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const manager = new ReMeEnvManager(
    { DSH_HOME: directory },
    {
      fetchHealth: async () => ({ ok: false, version: "", error: "" }),
      portOpen: async () => false,
      spawnService: async () => ({ pid: 1 }),
      killTree() {},
      probePid() {},
      processImage: async () => "python.exe",
      processCommandLine: async () => "",
      exists: () => true,
      platform: "win32",
    },
  );
  return { manager, directory };
}

const INSTALLED = {
  installed: true,
  version: "0.4.2.1",
  versionOk: true,
  hasCore: true,
  cliPath: "",
  error: "",
};

test("the interpreter readiness has one definition", () => {
  // The Host refuses a start unless reme-ai is installed and new enough, so
  // the tab must offer exactly the same starts.
  assert.equal(canHostService(INSTALLED), true);
  assert.equal(canHostService(undefined), false);
  assert.equal(
    canHostService({ ...INSTALLED, installed: false, versionOk: false }),
    false,
  );
  assert.equal(canHostService({ ...INSTALLED, versionOk: false }), false);

  // The [core] extras are a stronger state: without them the service still
  // starts, but automatic memory and consolidation do not.
  assert.equal(hasCoreFeatures(INSTALLED), true);
  assert.equal(hasCoreFeatures({ ...INSTALLED, hasCore: false }), false);
  assert.equal(hasCoreFeatures({ ...INSTALLED, versionOk: false }), false);
});

test("only a loopback HTTP endpoint is a manageable local service", () => {
  assert.equal(isLocalEndpoint("http://127.0.0.1:2333"), true);
  assert.equal(isLocalEndpoint("http://localhost:2333"), true);
  assert.equal(isLocalEndpoint("https://127.0.0.1:2333"), false);
  assert.equal(isLocalEndpoint("http://10.0.0.5:2333"), false);
  assert.equal(isLocalEndpoint("http://[::1]:2333"), false);
  assert.equal(isLocalEndpoint("not-a-url"), false);

  // The Host and the tab read the same rule, so the port they report is the
  // same one.
  assert.deepEqual(parseLocalEndpoint("http://127.0.0.1:2334"), {
    host: "127.0.0.1",
    port: 2334,
  });
  assert.deepEqual(parseLocalEndpoint("http://localhost:2333"), {
    host: "127.0.0.1",
    port: 2333,
  });
});

test("an emptied secret never reaches the .env file", (t) => {
  const { manager, directory } = managerIn(t);
  const envPath = join(directory, ".env");
  writeFileSync(
    envPath,
    [
      "LLM_MODEL_NAME=qwen3.7-plus",
      "LLM_API_KEY=sk-keep",
      "EMBEDDING_API_KEY=EMPTY",
      "",
    ].join("\n"),
  );

  // "Leave blank to keep": the stored credential survives, so the other
  // changes are saved instead of being blocked by the empty value.
  const summary = manager.writeEnv(directory, {
    LLM_MODEL_NAME: "qwen3.8-plus",
    LLM_API_KEY: "",
  });
  const text = readFileSync(envPath, "utf8");
  assert.match(text, /^LLM_API_KEY=sk-keep$/m);
  assert.match(text, /^LLM_MODEL_NAME=qwen3.8-plus$/m);
  assert.equal(
    summary.entries.find((entry) => entry.key === "LLM_MODEL_NAME").value,
    "qwen3.8-plus",
  );
});

test("an emptied non-secret field is still an explicit clear", (t) => {
  const { manager, directory } = managerIn(t);
  const envPath = join(directory, ".env");
  writeFileSync(
    envPath,
    "LLM_API_KEY=sk-keep\nLLM_BASE_URL=https://example.com/v1\n",
  );

  manager.writeEnv(directory, { LLM_BASE_URL: "" });
  const text = readFileSync(envPath, "utf8");
  assert.match(text, /^LLM_BASE_URL=$/m);
  assert.match(text, /^LLM_API_KEY=sk-keep$/m);
});

test("the embedding key keeps the placeholder that lets ReMe start", (t) => {
  const { manager, directory } = managerIn(t);
  const envPath = join(directory, ".env");
  writeFileSync(envPath, "LLM_API_KEY=sk-keep\n");

  // ReMe 0.4.0.x needs a non-empty embedding key even without vector search,
  // so an unset one is written as the documented placeholder.
  manager.writeEnv(directory, {});
  assert.match(readFileSync(envPath, "utf8"), /^EMBEDDING_API_KEY=EMPTY$/m);

  // A stored key survives; the placeholder is not written over it.
  writeFileSync(envPath, "LLM_API_KEY=sk-keep\nEMBEDDING_API_KEY=real-key\n");
  manager.writeEnv(directory, {});
  assert.match(readFileSync(envPath, "utf8"), /^EMBEDDING_API_KEY=real-key$/m);
});

test("a required key that was never stored still fails the save", (t) => {
  const { manager, directory } = managerIn(t);
  const envPath = join(directory, ".env");
  writeFileSync(envPath, "EMBEDDING_API_KEY=EMPTY\n");

  // "Keep" can only keep what is stored; without a stored LLM key the file
  // cannot start ReMe, so the save is refused and the file is left alone.
  assert.throws(
    () => manager.writeEnv(directory, { LLM_API_KEY: "" }),
    /LLM_API_KEY is required/,
  );
  assert.match(readFileSync(envPath, "utf8"), /^EMBEDDING_API_KEY=EMPTY$/m);
});
