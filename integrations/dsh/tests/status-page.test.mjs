import assert from "node:assert/strict";
import { before, test } from "node:test";
import { createRequire } from "node:module";
import { build } from "esbuild";
import React from "react";
import { act, create } from "react-test-renderer";
import { resolveConfig } from "../dist/config.js";

let Page;
let labels;
const require = createRequire(import.meta.url);
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
before(async () => {
  const result = await build({
    entryPoints: [
      new URL("../src/client/status-page.tsx", import.meta.url).pathname,
    ],
    bundle: true,
    write: false,
    platform: "node",
    format: "cjs",
    external: ["react", "@deepseek-ai/dsh-client-ui-primitives"],
  });
  const module = { exports: {} };
  const load = (name) =>
    name === "@deepseek-ai/dsh-client-ui-primitives"
      ? { MarkdownText: ({ text }) => React.createElement("pre", null, text) }
      : require(name);
  new Function("require", "module", "exports", result.outputFiles[0].text)(
    load,
    module,
    module.exports,
  );
  Page = module.exports.ReMeStatusPage;
  labels = module.exports.statusEn;
});
function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
function text(renderer) {
  return JSON.stringify(renderer.toJSON());
}
async function harness(t, fetcher) {
  const original = {
    fetch: globalThis.fetch,
    window: globalThis.window,
    document: globalThis.document,
  };
  const intervals = new Map();
  let interval = 0;
  globalThis.window = {
    setInterval: (callback) => {
      intervals.set(++interval, callback);
      return interval;
    },
    clearInterval: (id) => intervals.delete(id),
  };
  globalThis.document = { hidden: false };
  globalThis.fetch = fetcher;
  const listeners = new Set();
  let snapshot = {
    status: "ready",
    value: resolveConfig({ endpoint: "http://old.test:2333" }),
  };
  const scope = {
    getSnapshot: () => snapshot,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  const runtimeCalls = [];
  const rpc = {
    async call(_path, _method, _args, signal) {
      runtimeCalls.push(signal);
      return {
        ok: true,
        value: {
          phase: "running",
          autoMemory: {
            enabled: true,
            interval: 5,
            activeSessions: 0,
            queuedTurns: 0,
            tasksRunning: 0,
            tasksQueued: 0,
            recentTasks: [],
          },
          autoDream: {
            enabled: true,
            cron: "0 23 * * *",
            timezone: "UTC",
            running: false,
          },
        },
      };
    },
  };
  let renderer;
  await act(async () => {
    renderer = create(
      React.createElement(Page, { scope, rpc, t: (key) => labels[key] }),
    );
  });
  t.after(async () => {
    await act(async () => renderer.unmount());
    Object.assign(globalThis, original);
  });
  return {
    renderer,
    intervals,
    runtimeCalls,
    async settings(endpoint) {
      await act(async () => {
        snapshot = { status: "ready", value: resolveConfig({ endpoint }) };
        for (const listener of listeners) listener();
      });
    },
  };
}
function reply(input, init) {
  const url = new URL(input);
  const job = url.pathname;
  const body = JSON.parse(init.body);
  const host = url.hostname;
  if (job === "/health_check")
    return Response.json({
      success: true,
      answer: "healthy",
      metadata: {
        health: { version: `${host}-version`, healthy: true, components: {} },
      },
    });
  if (job === "/status")
    return Response.json({ success: true, answer: "status", metadata: {} });
  if (job === "/app_config")
    return Response.json({
      success: true,
      answer: { daily_dir: "daily", digest_dir: "digest" },
    });
  if (job === "/list")
    return Response.json({
      success: true,
      answer: "",
      metadata: { items: ["daily/fast.md", "daily/slow.md"] },
    });
  if (job === "/load")
    return Response.json({
      success: true,
      answer: `${host} ${body.path} body`,
    });
  if (job === "/auto_dream")
    return Response.json({ success: false, answer: "Dream refused" });
  throw new Error(`Unexpected test job ${job}`);
}
function button(renderer, predicate) {
  return renderer.root.findAllByType("button").find(predicate);
}
async function click(renderer, label) {
  await act(async () =>
    button(renderer, (entry) => entry.children.includes(label)).props.onClick(),
  );
}

test("a superseded endpoint cannot repaint status and runtime polling waits for full refresh", async (t) => {
  const gate = deferred();
  const signals = [];
  const h = await harness(t, async (input, init) => {
    if (new URL(input).hostname === "old.test") {
      signals.push(init.signal);
      await gate.promise;
    }
    return reply(input, init);
  });
  try {
    const calls = h.runtimeCalls.length;
    await act(async () => {
      for (const callback of h.intervals.values()) callback();
    });
    assert.equal(h.runtimeCalls.length, calls);
    await h.settings("http://new.test:2333");
    assert.ok(signals.every((signal) => signal.aborted));
    assert.ok(text(h.renderer).includes("new.test-version"));
    await act(async () => gate.resolve());
    assert.ok(!text(h.renderer).includes("old.test-version"));
    assert.ok(text(h.renderer).includes("new.test-version"));
  } finally {
    await act(async () => gate.resolve());
  }
});

test("slow file reads cannot replace a newer selected file or cross endpoint changes", async (t) => {
  const gate = deferred();
  const h = await harness(t, async (input, init) => {
    if (
      new URL(input).hostname === "old.test" &&
      JSON.parse(init.body).path === "daily/slow.md"
    )
      await gate.promise;
    return reply(input, init);
  });
  try {
    await click(h.renderer, labels.journal);
    await act(async () =>
      button(
        h.renderer,
        (entry) => entry.props.title === "daily/slow.md",
      ).props.onClick(),
    );
    await act(async () =>
      button(
        h.renderer,
        (entry) => entry.props.title === "daily/fast.md",
      ).props.onClick(),
    );
    await act(async () => gate.resolve());
    assert.ok(text(h.renderer).includes("old.test daily/fast.md body"));
    assert.ok(!text(h.renderer).includes("old.test daily/slow.md body"));
    await h.settings("http://new.test:2333");
    assert.ok(text(h.renderer).includes("new.test daily/fast.md body"));
    assert.ok(!text(h.renderer).includes("old.test daily/fast.md body"));
  } finally {
    await act(async () => gate.resolve());
  }
});

test("failed consolidation remains visible and repeated synchronous clicks start one request", async (t) => {
  let dreams = 0;
  const h = await harness(t, async (input, init) => {
    if (new URL(input).pathname === "/auto_dream") dreams++;
    return reply(input, init);
  });
  await click(h.renderer, labels.autoDream);
  const run = button(h.renderer, (entry) =>
    entry
      .findAllByType("strong")
      .some((child) => child.children.includes(labels.runDream)),
  );
  assert.ok(run);
  await act(async () => {
    run.props.onClick();
    run.props.onClick();
  });
  assert.equal(dreams, 1);
  assert.ok(text(h.renderer).includes("Dream refused"));
  assert.equal(
    button(h.renderer, (entry) =>
      entry
        .findAllByType("strong")
        .some((child) => child.children.includes(labels.runDream)),
    ).props.disabled,
    false,
  );
});
