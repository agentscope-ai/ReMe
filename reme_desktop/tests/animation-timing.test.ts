import assert from "node:assert/strict";
import test from "node:test";
import {
  frameTimeline,
  timelineFrame,
} from "../src/renderer/animation/timing.ts";

test("frame timing catches up after delays, wraps loops and ends once-only playback on its final frame", () => {
  const timeline = frameTimeline([2, 3, 1], 10);
  assert.equal(timelineFrame(timeline, 0, "once").index, 0);
  assert.equal(timelineFrame(timeline, 200, "once").index, 1);
  assert.equal(timelineFrame(timeline, 550, "once").index, 2);
  assert.deepEqual(timelineFrame(timeline, 900, "once"), {
    index: 2,
    delay: Infinity,
    ended: true,
  });
  assert.equal(timelineFrame(timeline, 850, "loop").index, 1);
  assert.deepEqual(timelineFrame(timeline, 0, "hold"), {
    index: 2,
    delay: Infinity,
    ended: false,
  });
});

test("idle has varied blink rests and a stable timeline across window handoffs", () => {
  const ticks = [18, 12, 12, 2, 3, 2, 12, 11];
  const first = frameTimeline(ticks, 24, true, 1234);
  assert.deepEqual(first, frameTimeline(ticks, 24, true, 1234));
  assert.notDeepEqual(first, frameTimeline(ticks, 24, true, 5678));
  const pauses = first.frames
    .filter(({ index }) => index === 0)
    .map((frame, cycle) => {
      const before = cycle ? first.frames[cycle * ticks.length - 1].end : 0;
      return frame.end - before - (ticks[0] / 24) * 1000;
    });
  assert.equal(new Set(pauses).size, 16);
  assert.ok(pauses.every((pause) => pause >= 1800 && pause <= 6000));
  assert.equal(timelineFrame(first, first.duration, "loop").index, 0);
});
