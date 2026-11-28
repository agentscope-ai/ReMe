import assert from "node:assert/strict";
import test from "node:test";
import { HoldToTalk } from "../src/mobile/hold-to-talk.ts";

function recording() {
  const calls: string[] = [];
  const gesture = new HoldToTalk({
    start: () => calls.push("start"),
    finish: () => calls.push("finish"),
    cancel: () => calls.push("cancel"),
    change: () => {},
  });
  return { gesture, calls };
}

test("a short tap edits text without opening a microphone", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { gesture, calls } = recording();
  gesture.press(40, 500);
  t.mock.timers.tick(200);
  assert.equal(gesture.release(), true);
  t.mock.timers.tick(1000);
  assert.deepEqual(calls, []);
});

test("holding starts once and releasing transcribes once, even after capture is lost", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { gesture, calls } = recording();
  gesture.press(40, 500);
  t.mock.timers.tick(360);
  assert.deepEqual(calls, ["start"]);
  assert.equal(gesture.release(), false);
  gesture.cancel();
  assert.deepEqual(calls, ["start", "finish"]);
});

test("sliding up cancels; sliding back restores release to transcribe", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { gesture, calls } = recording();
  gesture.press(40, 500);
  t.mock.timers.tick(360);
  gesture.move(40, 430);
  gesture.release();
  assert.deepEqual(calls, ["start", "cancel"]);
  gesture.press(40, 500);
  t.mock.timers.tick(360);
  gesture.move(40, 430);
  gesture.move(40, 470);
  gesture.release();
  assert.deepEqual(calls, ["start", "cancel", "start", "finish"]);
});

test("dragging before the hold threshold leaves the microphone closed", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { gesture, calls } = recording();
  gesture.press(40, 500);
  gesture.move(70, 500);
  t.mock.timers.tick(1000);
  assert.equal(gesture.release(), false);
  assert.deepEqual(calls, []);
});

test("changing pages, hiding the app or interrupting a touch cancels its recording", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const { gesture, calls } = recording();
  gesture.press(40, 500);
  gesture.cancel();
  t.mock.timers.tick(1000);
  assert.deepEqual(calls, []);
  gesture.press(40, 500);
  t.mock.timers.tick(360);
  gesture.cancel();
  gesture.release();
  assert.deepEqual(calls, ["start", "cancel"]);
});
