import assert from "node:assert/strict";
import test from "node:test";
import { UiError } from "../src/i18n.ts";
import { ReMeClient } from "../src/reme-client.ts";

test("administrator credentials stay scoped to the selected backend", async () => {
  const endpoint = "https://reme.example";
  const credentials = { endpoint, token: "test-admin" };
  const signal = new AbortController().signal;
  const requests: RequestInit[] = [];
  const fetchRequest: typeof fetch = async (_url, options) => {
    requests.push(options!);
    return Response.json({}, { status: requests.length === 1 ? 200 : 401 });
  };

  await new ReMeClient(endpoint, fetchRequest).manage(
    { type: "backend", action: "status", credentials },
    signal,
  );
  assert.equal(
    new Headers(requests[0].headers).get("authorization"),
    "Bearer test-admin",
  );
  assert.equal(requests[0].redirect, "error");
  await assert.rejects(
    new ReMeClient("https://different.example", fetchRequest).manage(
      { type: "backend", action: "status", credentials },
      signal,
    ),
    (error) => error instanceof UiError && error.key === "backendAuthRequired",
  );
  assert.equal(new Headers(requests[1].headers).get("authorization"), null);
});
