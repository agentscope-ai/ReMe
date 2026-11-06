import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { BehaviorController } from "../src/main/behavior.ts";
import type {
  AgentActivity,
  AnimationManifest,
  DesktopState,
} from "../src/shared/protocol.ts";

const manifest = JSON.parse(
  await readFile(
    new URL(
      "../../assets/remy/anime/animations/v1/manifest.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as AnimationManifest;
const input = () =>
  ({
    connection: "connected",
    chatFocused: false,
    pending: null,
    bubble: null,
    settings: { endpoint: "http://localhost", quietUntil: 0, scale: 1 },
  }) as DesktopState;
const activity = (
  phase: AgentActivity["phase"],
  turn = "t",
  source: AgentActivity["source"] = "chat",
): AgentActivity => ({
  session_id: "s",
  turn_id: turn,
  source,
  phase,
  tool: null,
});

test("text followed by tools and reasoning follows the latest phase, with short-work smoothing", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  behavior.observe({ chat: activity("answering") }, true, 0);
  assert.equal(behavior.resolve(state, true, 0).animation.clip, "speaking");
  behavior.observe({ chat: activity("working") }, true, 100);
  assert.equal(behavior.resolve(state, true, 100).animation.clip, "thinking");
  behavior.observe({ chat: activity("remembering") }, true, 200);
  assert.equal(behavior.resolve(state, true, 200).animation.clip, "thinking");
  assert.equal(
    behavior.resolve(state, true, 500).animation.clip,
    "remembering",
  );
  behavior.observe({ chat: activity("thinking") }, true, 600);
  assert.equal(behavior.resolve(state, true, 900).animation.clip, "thinking");
});

test("dragging wins temporarily and stale ended callbacks cannot clear the current activity", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  behavior.observe({ chat: activity("thinking") }, true, 0);
  const thinking = behavior.resolve(state, true, 0).animation.id;
  behavior.dragging = true;
  assert.equal(behavior.resolve(state, true, 100).animation.clip, "dragged");
  behavior.observe({ chat: activity("answering") }, true, 200);
  behavior.dragging = false;
  behavior.ended(thinking, 300);
  assert.equal(behavior.resolve(state, true, 300).animation.clip, "speaking");
});

test("reminders enter, hold and exit with the bubble rather than disappearing after one clip", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  state.bubble = { id: "m", text: "reminder", paused: false };
  const enter = behavior.resolve(state, true, 0).animation;
  assert.equal(enter.segment, "enter");
  behavior.ended(enter.id, 1500);
  assert.equal(behavior.resolve(state, true, 1500).animation.segment, "hold");
  assert.equal(behavior.resolve(state, true, 7000).animation.mode, "hold");
  state.bubble = null;
  const exit = behavior.resolve(state, true, 7100).animation;
  assert.equal(exit.segment, "exit");
  behavior.ended(enter.id, 7150);
  assert.equal(behavior.resolve(state, true, 7150).animation.id, exit.id);
  behavior.ended(exit.id, 7500);
  assert.equal(behavior.resolve(state, true, 7500).animation.clip, "idle");
});

test("terminal feedback is live once per turn, never from historical snapshots", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  const completed = { chat: activity("completed") };
  behavior.observe(completed, false, 0);
  assert.equal(behavior.resolve(state, true, 0).animation.clip, "idle");
  behavior.observe({ chat: activity("completed", "new") }, true, 100);
  const done = behavior.resolve(state, true, 100).animation;
  assert.equal(done.clip, "happy");
  behavior.ended(done.id, 2000);
  behavior.observe({ chat: activity("completed", "new") }, true, 2100);
  assert.equal(behavior.resolve(state, true, 2100).animation.clip, "idle");
  behavior.observe({ chat: activity("failed", "failed") }, true, 2200);
  assert.equal(behavior.resolve(state, true, 2200).animation.clip, "confused");
});

test("ambient gestures are neutral and interruptible, and quiet mode is independent of sleep", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  const first = behavior.resolve(state, true, 15001).animation;
  assert.equal(first.mode, "once");
  assert.ok(manifest.ambient.includes(first.clip));
  behavior.ended(first.id, 18000);
  const next = behavior.resolve(state, true, 34000).animation;
  assert.notEqual(next.clip, first.clip);
  state.settings.quietUntil = 100000;
  assert.equal(behavior.resolve(state, true, 35000).animation.clip, "idle");
  state.chatFocused = true;
  behavior.touch(36000);
  assert.equal(
    behavior.resolve(state, true, 36000).animation.clip,
    "listening",
  );
  state.chatFocused = false;
  assert.equal(
    behavior.resolve(state, true, 337000).animation.clip,
    "sleeping",
  );
});

test("foreground work outranks background work and connection loss clears busy presentation", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  behavior.observe(
    { proactive: activity("remembering", "bg", "proactive") },
    true,
    0,
  );
  assert.equal(behavior.resolve(state, true, 0).animation.clip, "remembering");
  behavior.observe(
    {
      chat: activity("answering"),
      proactive: activity("remembering", "bg", "proactive"),
    },
    true,
    100,
  );
  assert.equal(behavior.resolve(state, true, 100).animation.clip, "speaking");
  state.connection = "reconnecting";
  assert.equal(behavior.resolve(state, true, 200).animation.clip, "idle");
  assert.equal(behavior.resolve(state, false, 300).animation.mode, "hold");
});
