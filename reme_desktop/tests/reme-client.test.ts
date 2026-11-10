import assert from "node:assert/strict";
import test from "node:test";
import { ReMeClient } from "../src/main/reme-client.ts";
import { UiError } from "../src/shared/i18n.ts";

test("malformed JSON response shapes never acknowledge successful work", async (t) => {
  const client = new ReMeClient("http://localhost:2333");
  for (const payload of [null, [], {}, { success: "true" }]) {
    t.mock.method(globalThis, "fetch", async () => Response.json(payload));
    await assert.rejects(
      client.call("load", {}, new AbortController().signal),
      (error) => error instanceof UiError && error.key === "jobFailed",
    );
    t.mock.restoreAll();
  }
});

test("oversized document responses use a general response error instead of a speech error", async (t) => {
  t.mock.method(globalThis, "fetch", async () =>
    Response.json({ success: true, answer: "document" }),
  );
  const client = new ReMeClient("http://localhost:2333");
  await assert.rejects(
    client.request("load", {}, new AbortController().signal, { maxBytes: 5 }),
    (error) => error instanceof UiError && error.key === "responseTooLarge",
  );
});
