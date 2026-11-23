import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import { BrowserStorage } from "../src/browser-host.ts";

function useLocalStorage(t: TestContext): Map<string, string> {
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
  return storage;
}

test("browser tabs save only their own settings and receipt changes", async (t) => {
  const storage = useLocalStorage(t);
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

test("mobile saves a custom service address and web stays on its configured address", async (t) => {
  useLocalStorage(t);
  const initial = "http://127.0.0.1:2333";
  const remote = "http://192.0.2.10:2444";
  const mobile = new BrowserStorage();
  assert.equal(mobile.settings.endpoint, initial);
  mobile.settings.endpoint = remote;
  mobile.changed();
  await mobile.flush();

  const web = new BrowserStorage("https://studio.example.test");
  assert.equal(web.settings.endpoint, "https://studio.example.test");
  web.settings.theme = "dark";
  web.changed();
  await web.flush();

  const restarted = new BrowserStorage();
  assert.equal(restarted.settings.endpoint, remote);
  assert.equal(restarted.settings.theme, "dark");
});
