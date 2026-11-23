import assert from "node:assert/strict";
import test from "node:test";
import { BrowserStorage } from "../src/browser-host.ts";

test("browser tabs save only their own settings and receipt changes", async (t) => {
  const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  const storage = new Map<string, string>();
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
    },
  });
  t.after(() => {
    if (original) Object.defineProperty(globalThis, "localStorage", original);
    else Reflect.deleteProperty(globalThis, "localStorage");
  });

  const endpoint = "http://127.0.0.1:2333";
  const scope = "test-session";
  const initial = new BrowserStorage(endpoint);
  initial.presentation(scope).receipts.first = {
    userRead: false,
    presented: false,
    snoozeUntil: 123,
  };
  initial.changed();
  await initial.flush();

  const reader = new BrowserStorage(endpoint);
  const other = new BrowserStorage(endpoint);
  reader.presentation(scope).receipts.first.userRead = true;
  reader.changed();
  await reader.flush();
  const readerSnapshot = storage.get("reme:presentations");
  assert.equal(storage.has("reme:settings"), false);

  other.settings.theme = "dark";
  other.changed();
  await other.flush();
  assert.equal(storage.get("reme:presentations"), readerSnapshot);

  const receipts = other.presentation(scope).receipts;
  receipts.first.presented = true;
  delete receipts.first.snoozeUntil;
  receipts.second = { userRead: true, presented: true };
  other.changed();
  await other.flush();

  reader.settings.voiceEnabled = false;
  reader.changed();
  await reader.flush();

  const reloaded = new BrowserStorage(endpoint);
  assert.equal(reloaded.settings.theme, "dark");
  assert.equal(reloaded.settings.voiceEnabled, false);
  assert.deepEqual(reloaded.presentation(scope).receipts, {
    first: { userRead: true, presented: true },
    second: { userRead: true, presented: true },
  });
});
