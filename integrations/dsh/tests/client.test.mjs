import assert from "node:assert/strict";
import test from "node:test";
import { ReMeClient } from "../dist/reme/client.js";
import { safeError } from "../dist/endpoint.js";

test("redacts URL credentials containing spaces or stripped URL whitespace", async () => {
  const originalFetch = globalThis.fetch;
  const client = new ReMeClient({
    endpoint: "http://localhost:2333",
    requestTimeoutMs: 1000,
    backgroundTimeoutMs: 1000,
  });
  try {
    for (const password of [
      "SECRET CANARY",
      "SECRET\tCANARY",
      "SECRET\nCANARY",
    ]) {
      const url = `http://alice:${password}@localhost:2333/search`;
      assert.ok(new URL(url).password);
      const expected = "refused http://[redacted]@localhost:2333/search";
      globalThis.fetch = async () =>
        Response.json({ success: false, answer: `refused ${url}` });
      assert.equal((await client.search("fact")).error, expected);
      globalThis.fetch = async () => {
        throw new Error(`refused ${url}`);
      };
      assert.equal((await client.search("fact")).error, expected);
    }
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("rejects credentials before fetch and redacts URL credentials from service and transport errors", async () => {
  const originalFetch = globalThis.fetch;
  let calls = 0;
  const secretUrl = "http://alice:SECRET-CANARY@localhost:2333/search";
  globalThis.fetch = async () => {
    calls++;
    return Response.json({ success: false, answer: `refused ${secretUrl}` });
  };
  try {
    const client = new ReMeClient({
      endpoint: secretUrl,
      requestTimeoutMs: 1000,
      backgroundTimeoutMs: 1000,
    });
    assert.doesNotMatch(
      (await client.search("fact")).error,
      /alice|SECRET-CANARY/,
    );
    assert.equal(calls, 0);
    const valid = new ReMeClient({
      endpoint: "http://localhost:2333",
      requestTimeoutMs: 1000,
      backgroundTimeoutMs: 1000,
    });
    assert.doesNotMatch(
      (await valid.search("fact")).error,
      /alice|SECRET-CANARY/,
    );
    globalThis.fetch = async () => {
      throw new Error(`failed ${secretUrl}`);
    };
    assert.doesNotMatch(
      (await valid.search("fact")).error,
      /alice|SECRET-CANARY/,
    );
    assert.doesNotMatch(
      safeError(new Error(`failed ${secretUrl}`)),
      /alice|SECRET-CANARY/,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("uses the persisted destination and request ID when sending a recovered batch", async () => {
  const originalFetch = globalThis.fetch;
  let request;
  globalThis.fetch = async (url, init) => {
    request = { url, body: JSON.parse(init.body) };
    return Response.json({ success: true });
  };
  try {
    const client = new ReMeClient({
      endpoint: "http://new.test",
      requestTimeoutMs: 1000,
      backgroundTimeoutMs: 1000,
    });
    await client.autoMemory([], "session", {
      endpoint: "http://original.test",
      requestId: "persisted-batch",
    });
    assert.equal(request.url, "http://original.test/auto_memory");
    assert.equal(request.body.request_id, "persisted-batch");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("calls ReMe jobs with their native request and response envelopes", async () => {
  const calls = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    calls.push({ url, body: JSON.parse(init.body) });
    return new Response(
      JSON.stringify({ success: true, answer: "memory result", metadata: {} }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      },
    );
  };
  try {
    const client = new ReMeClient({
      endpoint: "http://127.0.0.1:2333",
      requestTimeoutMs: 1000,
      backgroundTimeoutMs: 1000,
    });
    const result = await client.search("deployment", { limit: 5, minScore: 0 });
    assert.equal(result.ok, true);
    assert.equal(result.answer, "memory result");
    assert.deepEqual(calls, [
      {
        url: "http://127.0.0.1:2333/search",
        body: { query: "deployment", limit: 5, min_score: 0 },
      },
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("combines caller cancellation with the request timeout", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (_url, init) =>
    new Promise((_resolve, reject) => {
      init.signal.addEventListener("abort", () => reject(init.signal.reason), {
        once: true,
      });
    });
  try {
    const client = new ReMeClient({
      endpoint: "http://127.0.0.1:2333",
      requestTimeoutMs: 1000,
      backgroundTimeoutMs: 1000,
    });
    const controller = new AbortController();
    const request = client.search("deployment", { signal: controller.signal });
    controller.abort(new Error("turn cancelled"));
    const result = await request;
    assert.equal(result.ok, false);
    assert.match(result.error, /turn cancelled/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("returns typed health, memory status, and redacted server configuration", async () => {
  const originalFetch = globalThis.fetch;
  const urls = [];
  globalThis.fetch = async (url) => {
    urls.push(url);
    if (url.endsWith("/health_check")) {
      return Response.json({
        success: true,
        metadata: {
          health: { version: "1.2.3", healthy: true, components: {} },
        },
      });
    }
    if (url.endsWith("/status")) {
      return Response.json({
        success: true,
        metadata: {
          status: {
            memory: {
              components: {},
              components_total_bytes: 10,
              components_total: "10 B",
              process_rss_bytes: 20,
              process_rss: "20 B",
            },
          },
        },
      });
    }
    return Response.json({
      success: true,
      answer: { workspace_dir: "/memory", token: "***" },
    });
  };
  try {
    let endpoint = "http://first.test";
    const client = new ReMeClient(() => ({
      endpoint,
      requestTimeoutMs: 1000,
      backgroundTimeoutMs: 1000,
    }));
    const health = await client.healthCheck();
    endpoint = "http://second.test";
    const status = await client.status();
    const appConfig = await client.appConfig();
    assert.equal(health.health.version, "1.2.3");
    assert.equal(status.memory.process_rss, "20 B");
    assert.deepEqual(appConfig.answer, {
      workspace_dir: "/memory",
      token: "***",
    });
    assert.deepEqual(urls, [
      "http://first.test/health_check",
      "http://second.test/status",
      "http://second.test/app_config",
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("lists and loads read-only ReMe workspace files", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url, body: JSON.parse(init.body) });
    if (url.endsWith("/list")) {
      return Response.json({
        success: true,
        metadata: { items: ["daily/2026-08-20/session.md", 1] },
      });
    }
    return Response.json({
      success: true,
      answer: "# Memory",
      metadata: {
        path: "daily/2026-08-20/session.md",
        mtime: "2026-08-20T12:00:00",
      },
    });
  };
  try {
    const client = new ReMeClient({
      endpoint: "http://127.0.0.1:2333",
      requestTimeoutMs: 1000,
      backgroundTimeoutMs: 1000,
    });
    const listing = await client.listFiles("daily", { limit: 1 });
    const file = await client.loadFile("daily/2026-08-20/session.md");
    assert.deepEqual(listing.files, ["daily/2026-08-20/session.md"]);
    assert.equal(listing.limited, true);
    assert.equal(file.content, "# Memory");
    assert.equal(file.path, "daily/2026-08-20/session.md");
    assert.deepEqual(calls[0], {
      url: "http://127.0.0.1:2333/list",
      body: {
        path: "daily",
        recursive: true,
        sort_by: "mtime",
        extensions: ["md", "markdown", "txt", "yaml", "yml"],
        limit: 1,
      },
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});
