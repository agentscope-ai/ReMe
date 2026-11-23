import assert from "node:assert/strict";
import { test } from "node:test";
import { ReMeClient } from "../../client/src/reme-client.ts";
import { SessionModel } from "../../client/src/session-model.ts";
import { demoService } from "../demo/service.ts";

test("the playground serves the same snapshot and reply contract", async () => {
  const client = new ReMeClient("https://demo.example", demoService());
  const watch = new AbortController();
  const model = new SessionModel();
  const subscription = client.stream(
    "watch_agent_session",
    {},
    watch.signal,
    ({ chunk_type, chunk }) => {
      if (chunk_type === "data") model.apply(chunk);
    },
  );
  try {
    await client.stream(
      "ask_agent",
      { query: "Hello" },
      new AbortController().signal,
      () => {},
    );
    assert.equal(model.messages[0]?.content, "Hello");
    assert.equal(model.messages[1]?.outcome, "completed");
  } finally {
    watch.abort();
    await subscription.catch(() => {});
  }
});
