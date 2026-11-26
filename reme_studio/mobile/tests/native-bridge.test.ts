import assert from "node:assert/strict";
import test from "node:test";
import { ReMeClient } from "../../client/src/reme-client.ts";
import { UiError } from "../../client/src/i18n.ts";

test("iOS management preserves administrator tokens and their endpoint scope", async () => {
  const endpoint = "https://reme.example";
  const requests: Record<string, unknown>[] = [];
  const unread = new Set<number>();
  const previous = Object.getOwnPropertyDescriptor(globalThis, "window");
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      webkit: {
        messageHandlers: {
          remeStudio: {
            async postMessage(message: Record<string, unknown>) {
              const id = message.id as number;
              switch (message.action) {
                case "request.open":
                  requests.push(message);
                  unread.add(id);
                  return {
                    status: message.adminToken === "test-admin" ? 200 : 401,
                    contentType: "application/json",
                  };
                case "request.read":
                  return unread.delete(id) ? btoa('{"success":true}') : null;
                case "request.cancel":
                  unread.delete(id);
                  return true;
                default:
                  throw new Error(
                    `Unexpected native message: ${message.action}`,
                  );
              }
            },
          },
        },
      },
    },
  });
  try {
    const { iosFetch } = await import("../src/native-bridge.ts");
    assert.ok(iosFetch);
    const client = new ReMeClient(endpoint, iosFetch);
    const signal = new AbortController().signal;
    const credentials = { endpoint, token: "test-admin" };
    for (const action of ["config", "status", "restart"] as const) {
      assert.deepEqual(
        await client.manage({ type: "backend", action, credentials }, signal),
        { success: true },
      );
    }
    assert.equal(requests[0].url, `${endpoint}/admin/config`);
    assert.equal(requests[0].method, "GET");
    assert.equal(requests[2].method, "POST");
    for (const request of requests) {
      assert.equal(request.adminToken, credentials.token);
      assert.equal(Object.hasOwn(request, "authorization"), false);
    }
    await assert.rejects(
      new ReMeClient("https://different.example", iosFetch).manage(
        { type: "backend", action: "status", credentials },
        signal,
      ),
      (error) =>
        error instanceof UiError && error.key === "backendAuthRequired",
    );
    assert.equal(requests.at(-1)?.adminToken, "");

    const legacy = await iosFetch(`${endpoint}/admin/config`, {
      headers: { Authorization: "Bearer test-admin" },
    });
    assert.equal(legacy.status, 401);
    await legacy.body?.cancel();
    assert.equal(requests.at(-1)?.adminToken, "");
    assert.equal(Object.hasOwn(requests.at(-1)!, "authorization"), false);
  } finally {
    if (previous) Object.defineProperty(globalThis, "window", previous);
    else Reflect.deleteProperty(globalThis, "window");
  }
});
