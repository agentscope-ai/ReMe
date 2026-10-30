import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  envSummary,
  LLM_PROTOCOLS,
  llmProbeSnippet,
  llmProtocolFor,
  maskSecret,
  mergeEnvFile,
  parseEnvFile,
  parseLlmTest,
  resolveWorkspaceDir,
  testLlm,
  validateEnvUpdate,
} from "../dist/environment/host/workspace.js";
import { MANAGED_KEYS } from "../dist/environment/env-fields.js";

test("resolves the workspace directory with ~/.reme defaults", () => {
  const home = process.platform === "win32" ? "C:\\Users\\u" : "/home/u";
  assert.equal(resolveWorkspaceDir("", home), join(home, ".reme"));
  assert.equal(resolveWorkspaceDir("~/.reme", home), join(home, ".reme"));
  assert.equal(resolveWorkspaceDir("~", home), home);
  assert.throws(() => resolveWorkspaceDir("relative/dir", home));
});

test("parses KEY=VALUE lines tolerating comments, blanks and quotes", () => {
  const parsed = parseEnvFile(
    [
      "# service credentials",
      "",
      "LLM_BACKEND=openai",
      'LLM_MODEL_NAME="qwen3.7-plus"',
      "LLM_API_KEY='sk-test'",
      "UNKNOWN_KEY=kept",
      "BROKEN LINE",
    ].join("\n"),
  );
  assert.deepEqual(parsed, {
    LLM_BACKEND: "openai",
    LLM_MODEL_NAME: "qwen3.7-plus",
    LLM_API_KEY: "sk-test",
    UNKNOWN_KEY: "kept",
  });
});

test("merges updates while preserving comments, order and unknown keys", () => {
  const existing = [
    "# managed by hand",
    "LLM_BACKEND=openai",
    "LLM_API_KEY=old-key",
    "",
    "EMBEDDING_API_KEY=EMPTY",
    "# keep me",
  ].join("\n");
  const merged = mergeEnvFile(existing, {
    LLM_API_KEY: "new-key",
    LLM_BASE_URL: "https://example.com/v1",
  });
  assert.match(merged, /^# managed by hand$/m);
  assert.match(merged, /^LLM_API_KEY=new-key$/m);
  assert.doesNotMatch(merged, /old-key/);
  assert.match(merged, /^EMBEDDING_API_KEY=EMPTY$/m);
  assert.match(merged, /^# keep me$/m);
  assert.match(merged, /^LLM_BASE_URL=https:\/\/example\.com\/v1$/m);
  assert.ok(merged.endsWith("\n"));
  // Idempotent updates do not duplicate keys.
  assert.equal(
    (
      mergeEnvFile(merged, { LLM_API_KEY: "new-key" }).match(
        /^LLM_API_KEY=/gm,
      ) ?? []
    ).length,
    1,
  );
});

test("a duplicated key is rewritten everywhere, so parsing agrees", () => {
  const existing = [
    "LLM_API_KEY=old-key",
    "LLM_MODEL_NAME=qwen3.7-plus",
    "LLM_API_KEY=older-key",
  ].join("\n");
  const merged = mergeEnvFile(existing, { LLM_API_KEY: "new-key" });
  assert.equal((merged.match(/^LLM_API_KEY=new-key$/gm) ?? []).length, 2);
  // Parsing keeps the last occurrence, and that occurrence is the new value.
  assert.equal(parseEnvFile(merged)["LLM_API_KEY"], "new-key");
  assert.equal((merged.match(/^LLM_API_KEY=/gm) ?? []).length, 2);
});

test("keeps the file's own line ending", () => {
  const merged = mergeEnvFile(
    "LLM_BACKEND=openai\r\nEMBEDDING_API_KEY=EMPTY\r\n",
    {
      LLM_BACKEND: "dashscope",
    },
  );
  assert.match(merged, /^LLM_BACKEND=dashscope\r$/m);
  assert.equal(merged.includes("\n\n"), false);
  assert.ok(merged.endsWith("\r\n"));
  assert.equal(mergeEnvFile("", { A: "1" }), "A=1\n");
});

test("masks secrets down to the last four characters", () => {
  assert.equal(maskSecret("sk-abcdefgh"), "••••efgh");
  assert.equal(maskSecret("abc"), "••••");
  assert.equal(maskSecret(""), "");
});

test("summaries expose masked secrets and plain non-secret values", (t) => {
  const directory = mkdtempSync(join(tmpdir(), "reme-env-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const path = join(directory, ".env");
  writeFileSync(
    path,
    "LLM_MODEL_NAME=qwen3.7-plus\nLLM_API_KEY=sk-super-secret\n",
  );
  const summary = envSummary(path, directory);
  assert.equal(summary.exists, true);
  assert.equal(summary.envPath, path);
  const model = summary.entries.find((entry) => entry.key === "LLM_MODEL_NAME");
  assert.equal(model.value, "qwen3.7-plus");
  assert.equal(model.secret, false);
  const key = summary.entries.find((entry) => entry.key === "LLM_API_KEY");
  assert.equal(key.set, true);
  assert.equal(key.secret, true);
  assert.equal(key.value, "");
  assert.equal(key.masked, "••••cret");
  assert.doesNotMatch(JSON.stringify(summary), /super-secret/);
});

test("the summary lists the managed keys in the shared order and masks the secrets", (t) => {
  // The Host projection and the browser form read the same key list, so the
  // summary has to name the same keys in the same order.
  const directory = mkdtempSync(join(tmpdir(), "reme-env-order-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const path = join(directory, ".env");
  writeFileSync(path, "EMBEDDING_API_KEY=EMPTY\nLLM_API_KEY=sk-super-secret\n");
  const summary = envSummary(path, directory);
  assert.deepEqual(
    summary.entries.map((entry) => entry.key),
    [...MANAGED_KEYS],
  );
  const key = summary.entries.find((entry) => entry.key === "LLM_API_KEY");
  assert.equal(key.secret, true);
  assert.equal(key.set, true);
  assert.equal(key.value, "");
  assert.equal(key.masked, "••••cret");
  assert.doesNotMatch(JSON.stringify(summary), /super-secret/);
  // A key the file does not hold is reported as unset, not as missing.
  const base = summary.entries.find((entry) => entry.key === "LLM_BASE_URL");
  assert.equal(base.set, false);
  assert.equal(base.value, "");
});

test("validates the embedding key that aborts ReMe 0.4.0.x startup", () => {
  assert.throws(() =>
    validateEnvUpdate({ LLM_API_KEY: "sk", EMBEDDING_API_KEY: "" }),
  );
  assert.throws(() =>
    validateEnvUpdate({ LLM_API_KEY: "", EMBEDDING_API_KEY: "EMPTY" }),
  );
  validateEnvUpdate({ LLM_API_KEY: "sk", EMBEDDING_API_KEY: "EMPTY" });
  validateEnvUpdate({ LLM_API_KEY: "sk", EMBEDDING_API_KEY: "real-key" });
});

test("parses LLM test results from noisy python output", () => {
  assert.deepEqual(
    parseLlmTest(
      'warn\n{"ok": true, "status": 200, "latencyMs": 321, "error": ""}\n',
    ),
    { ok: true, status: 200, latencyMs: 321, error: "" },
  );
  assert.deepEqual(parseLlmTest("no json"), {
    ok: false,
    status: 0,
    latencyMs: 0,
    error: "LLM test produced no result",
  });
});

test("runs the LLM probe inside the workspace and classifies failures", async (t) => {
  const directory = mkdtempSync(join(tmpdir(), "reme-llm-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const calls = [];
  const io = {
    run: async (exe, args, options) => {
      calls.push({ exe, args, options });
      const stdout =
        calls.length === 1
          ? '{"ok": true, "status": 200, "latencyMs": 120, "error": ""}\n'
          : '{"ok": false, "status": 401, "latencyMs": 40, "error": "HTTP 401"}\n';
      return { code: 0, stdout, stderr: "" };
    },
  };
  const ok = await testLlm(io, {
    workspaceDir: directory,
    pythonPath: "C:\\env\\python.exe",
  });
  assert.equal(ok.ok, true);
  assert.equal(ok.latencyMs, 120);
  assert.equal(calls[0].options.cwd, directory);
  assert.equal(calls[0].exe, "C:\\env\\python.exe");

  const denied = await testLlm(io, {
    workspaceDir: directory,
    pythonPath: "C:\\env\\python.exe",
  });
  assert.equal(denied.ok, false);
  assert.match(denied.error, /LLM_API_KEY/);

  const broken = await testLlm(
    {
      run: async () => ({
        code: null,
        stdout: "",
        stderr: "",
        error: "spawn ENOENT",
      }),
    },
    { workspaceDir: directory, pythonPath: "C:\\missing\\python.exe" },
  );
  assert.equal(broken.ok, false);
  assert.match(broken.error, /ENOENT/);
});

test("the LLM probe speaks the protocol of the selected backend", async (t) => {
  const directory = mkdtempSync(join(tmpdir(), "reme-llm-backend-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const calls = [];
  const io = {
    run: async (exe, args, options) => {
      calls.push({ exe, args, options });
      return {
        code: 0,
        stdout: '{"ok": true, "status": 200, "latencyMs": 12, "error": ""}\n',
        stderr: "",
      };
    },
  };
  const probe = async (backend) => {
    writeFileSync(
      join(directory, ".env"),
      [
        `LLM_BACKEND=${backend}`,
        "LLM_MODEL_NAME=test-model",
        "LLM_API_KEY=sk-test",
        "LLM_BASE_URL=https://provider.invalid",
      ].join("\n"),
    );
    calls.length = 0;
    await testLlm(io, {
      workspaceDir: directory,
      pythonPath: "C:\\env\\python.exe",
    });
    return String(calls[0]?.args[1] ?? "");
  };

  // The OpenAI-compatible providers keep the OpenAI-style request.
  const openai = await probe("openai");
  assert.match(openai, /\/chat\/completions/);
  assert.match(openai, /"Authorization"/);
  assert.match(openai, /Bearer /);

  // A native provider must not be probed with the OpenAI protocol: that would
  // report a working configuration as broken.
  const anthropic = await probe("anthropic");
  assert.match(anthropic, /\/v1\/messages/);
  assert.match(anthropic, /"x-api-key"/);
  assert.match(anthropic, /anthropic-version/);
  assert.doesNotMatch(anthropic, /\/chat\/completions/);

  const gemini = await probe("gemini");
  assert.match(gemini, /generateContent/);
  assert.match(gemini, /"x-goog-api-key"/);
  assert.doesNotMatch(gemini, /\/chat\/completions/);

  const ollama = await probe("ollama");
  assert.match(ollama, /\/api\/chat/);
  assert.doesNotMatch(ollama, /"Authorization"/);

  // An unknown backend falls back to ReMe's default rather than failing.
  const unknown = await probe("something-else");
  assert.match(unknown, /\/chat\/completions/);
});

test("the probe builds a valid request for every backend", async (t) => {
  const directory = mkdtempSync(join(tmpdir(), "reme-llm-request-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const model = "test-model";
  for (const backend of Object.keys(LLM_PROTOCOLS)) {
    const protocol = llmProtocolFor(backend);
    // The body is a JSON template: the placeholder is replaced with the
    // JSON-quoted model name, so the result must still be valid JSON.
    const body = protocol.body.replace("{model}", JSON.stringify(model));
    const parsed = JSON.parse(body);
    assert.equal(typeof parsed, "object", `${backend} body is JSON`);
    if (protocol.path.includes("{model}")) {
      // The model belongs in the path, URL-encoded.
      assert.ok(!protocol.body.includes("{model}"), `${backend} body`);
      const url = `https://provider.invalid${protocol.path.replace(
        "{model}",
        encodeURIComponent(model),
      )}`;
      assert.ok(url.includes(model), `${backend} url carries the model`);
    } else {
      assert.ok(
        protocol.body.includes("{model}"),
        `${backend} carries the model in the body`,
      );
    }
    // The URL is the base plus the provider's path, exactly as the provider's
    // own SDK builds it: nothing is trimmed away, so a base that already
    // carries the version suffix fails in the probe just like it fails in the
    // service.
    assert.ok(
      protocol.path.startsWith("/"),
      `${backend} path is appended to the base`,
    );
    const snippet = llmProbeSnippet(protocol);
    assert.doesNotMatch(
      snippet,
      /base\s*\[|base\.endswith\(|base\.removesuffix\(/,
      `${backend} does not rewrite the configured base`,
    );
  }
});

test("a backend that ignores LLM_BASE_URL is probed with its own endpoint", async (t) => {
  // ReMe's Gemini and Ollama credentials do not read LLM_BASE_URL, so the
  // probe must not read it either: a stale value left in the .env would send
  // the probe to a host the service never calls.
  const directory = mkdtempSync(join(tmpdir(), "reme-llm-base-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const calls = [];
  const io = {
    run: async (exe, args) => {
      calls.push({ exe, args });
      return {
        code: 0,
        stdout: '{"ok": true, "status": 200, "latencyMs": 12, "error": ""}\n',
        stderr: "",
      };
    },
  };
  const probe = async (backend, baseUrl) => {
    writeFileSync(
      join(directory, ".env"),
      [
        `LLM_BACKEND=${backend}`,
        "LLM_MODEL_NAME=test-model",
        "LLM_API_KEY=sk-test",
        ...(baseUrl === undefined ? [] : [`LLM_BASE_URL=${baseUrl}`]),
      ].join("\n"),
    );
    calls.length = 0;
    await testLlm(io, {
      workspaceDir: directory,
      pythonPath: "C:\\env\\python.exe",
    });
    return String(calls[0]?.args[1] ?? "");
  };

  // Without a base URL the provider's own endpoint is used.
  const gemini = await probe("gemini");
  assert.match(gemini, /generativelanguage\.googleapis\.com/);
  assert.doesNotMatch(gemini, /env\.get\("LLM_BASE_URL"/);

  // A stale base URL in the .env is ignored, exactly as ReMe ignores it.
  const stale = await probe("ollama", "https://stale.invalid");
  assert.match(stale, /localhost:11434/);
  assert.doesNotMatch(stale, /stale\.invalid/);

  // xAI is the same case: its credential carries an api host, not a base URL.
  const xai = await probe("xai", "https://stale.invalid");
  assert.match(xai, /api\.x\.ai/);
  assert.doesNotMatch(xai, /stale\.invalid/);
  assert.doesNotMatch(xai, /env\.get\("LLM_BASE_URL"/);

  // Ollama needs neither a base URL nor a key.
  const ollama = await probe("ollama");
  assert.match(ollama, /localhost:11434/);
  assert.match(ollama, /if False and not key/);

  // The OpenAI-compatible providers do read LLM_BASE_URL, so it stays required.
  const openai = await probe("openai");
  assert.match(openai, /missing\.append\("LLM_BASE_URL"\)/);
  assert.match(openai, /env\.get\("LLM_BASE_URL"/);
});
