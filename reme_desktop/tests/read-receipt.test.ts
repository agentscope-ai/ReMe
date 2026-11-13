import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import { watchReadReceipt } from "../src/renderer/read-receipt.ts";

function environment(t: TestContext) {
  let focused = true;
  let visible = true;
  const window = new EventTarget();
  const document = new EventTarget();
  Object.defineProperties(document, {
    hasFocus: { value: () => focused },
    visibilityState: { get: () => (visible ? "visible" : "hidden") },
  });
  const original = ["window", "document"].map((key) =>
    Object.getOwnPropertyDescriptor(globalThis, key),
  );
  Object.defineProperties(globalThis, {
    window: { value: window, configurable: true },
    document: { value: document, configurable: true },
  });
  const restore = () => {
    ["window", "document"].forEach((key, index) => {
      if (original[index])
        Object.defineProperty(globalThis, key, original[index]!);
      else Reflect.deleteProperty(globalThis, key);
    });
  };
  t.mock.timers.enable({ apis: ["setTimeout"] });
  return {
    restore,
    async tick(ms: number) {
      t.mock.timers.tick(ms);
      await Promise.resolve();
      await Promise.resolve();
    },
    focus(value: boolean) {
      focused = value;
      window.dispatchEvent(new Event(value ? "focus" : "blur"));
    },
    visibility(value: boolean) {
      visible = value;
      document.dispatchEvent(new Event("visibilitychange"));
    },
  };
}

test("a receipt requires a continuous visible interval and resumes without a focus event", async (t) => {
  const env = environment(t);
  let reads = 0;
  const receipt = watchReadReceipt(async () => {
    reads++;
    return false;
  });
  t.after(() => {
    receipt.close();
    env.restore();
  });
  await env.tick(400);
  env.visibility(false);
  await env.tick(5000);
  assert.equal(reads, 0);
  env.visibility(true);
  await env.tick(599);
  assert.equal(reads, 0);
  await env.tick(1);
  assert.equal(reads, 1);
});

test("losing focus cancels the timer and returning focus starts a fresh interval", async (t) => {
  const env = environment(t);
  let reads = 0;
  const receipt = watchReadReceipt(async () => {
    reads++;
    return false;
  });
  t.after(() => {
    receipt.close();
    env.restore();
  });
  await env.tick(500);
  env.focus(false);
  await env.tick(2000);
  assert.equal(reads, 0);
  env.focus(true);
  await env.tick(599);
  assert.equal(reads, 0);
  await env.tick(1);
  assert.equal(reads, 1);
});

test("a failed receipt retries while visible without overlapping pending requests", async (t) => {
  const env = environment(t);
  let reads = 0;
  let reject!: (error: Error) => void;
  const failure = new Error("temporary receipt failure");
  const errors: unknown[] = [];
  const receipt = watchReadReceipt(
    async () => {
      reads++;
      if (reads === 1)
        await new Promise<void>((_, fail) => {
          reject = fail;
        });
      return false;
    },
    (error) => errors.push(error),
  );
  t.after(() => {
    receipt.close();
    env.restore();
  });
  await env.tick(600);
  env.focus(true);
  await env.tick(3000);
  assert.equal(reads, 1);
  reject(failure);
  await env.tick(0);
  assert.deepEqual(errors, [failure]);
  await env.tick(599);
  assert.equal(reads, 1);
  await env.tick(1);
  assert.equal(reads, 2);
});

test("closing a selected view aborts pending work and ignores its late failure", async (t) => {
  const env = environment(t);
  let reads = 0;
  let reject!: (error: Error) => void;
  let signal!: AbortSignal;
  const errors: unknown[] = [];
  const receipt = watchReadReceipt(
    async (current) => {
      signal = current;
      reads++;
      await new Promise<void>((_, fail) => {
        reject = fail;
      });
      return true;
    },
    (error) => errors.push(error),
  );
  t.after(() => {
    receipt.close();
    env.restore();
  });
  await env.tick(600);
  receipt.close();
  assert.equal(signal.aborted, true);
  reject(new Error("late failure"));
  await env.tick(0);
  env.focus(true);
  env.visibility(true);
  await env.tick(5000);
  assert.equal(reads, 1);
  assert.deepEqual(errors, []);
});

test("acknowledgement keeps retrying until the caller confirms there are no unread visible items", async (t) => {
  const env = environment(t);
  let reads = 0;
  const receipt = watchReadReceipt(async () => ++reads < 2);
  t.after(() => {
    receipt.close();
    env.restore();
  });
  await env.tick(600);
  assert.equal(reads, 1);
  await env.tick(600);
  assert.equal(reads, 2);
  await env.tick(5000);
  assert.equal(reads, 2);
  receipt.schedule();
  await env.tick(600);
  assert.equal(reads, 3);
});
