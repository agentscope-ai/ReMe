import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  OpenAIDeepLinkHostStateSchema,
  OpenAIMessageOptionsSchema,
} from "@openai/mcp-extensions/app";

test("patched Zod validates MCP extension host payloads", () => {
  assert.equal(OpenAIDeepLinkHostStateSchema.safeParse({ url: "/reme/status" }).success, true);
  assert.equal(OpenAIDeepLinkHostStateSchema.safeParse({ url: 7 }).success, false);
  assert.equal(OpenAIMessageOptionsSchema.safeParse({ target: "active" }).success, true);
  assert.equal(OpenAIMessageOptionsSchema.safeParse({ target: "all" }).success, false);
});

test("self-contained Codex status app is built", async () => {
  const html = await readFile(new URL("../../plugins/reme/ui/status.html", import.meta.url), "utf8");
  assert.match(html, /<script\b/i);
  assert.doesNotMatch(html, /\/\* STATUS_APP \*\//);
});
