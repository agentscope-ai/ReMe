import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

process.env.WRANGLER_LOG_PATH ??= ".wrangler/wrangler.log";
process.env.MINIFLARE_REGISTRY_PATH ??= ".wrangler/registry";
const { createTestHarness } = await import("wrangler");

const workerServer = createTestHarness({
  workers: [
    {
      configPath: new URL("../../dist/server/wrangler.json", import.meta.url),
    },
  ],
});

test.before(async () => {
  await workerServer.listen();
});

test.after(async () => {
  await workerServer.close();
});

async function render() {
  return workerServer.fetch("http://localhost/", {
    headers: { accept: "text/html" },
  });
}

test("server-renders the ReMe Studio shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  const packageJson = JSON.parse(
    await readFile(new URL("../../package.json", import.meta.url), "utf8"),
  );
  assert.match(html, /<title>ReMe Studio<\/title>/i);
  assert.match(
    html.replaceAll("<!-- -->", ""),
    new RegExp(`v${packageJson.version}`),
  );
  assert.match(html, /打开 对话/);
  assert.match(html, /打开 Proactive Chat/);
  assert.match(html, /文件保留在你的本地工作区/);
  assert.doesNotMatch(
    html,
    /codex-preview|Your site is taking shape|react-loading-skeleton/i,
  );
});
