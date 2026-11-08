import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { BehaviorController } from "../src/main/behavior.ts";
import { emptySpeech } from "../src/main/speech.ts";
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
    reminderId: null,
    settings: {
      language: "zh",
      endpoint: "http://localhost",
      quietUntil: 0,
      scale: 1,
    },
  }) as DesktopState & { reminderId: string | null };
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
  behavior.report(thinking, "ended", 300);
  assert.equal(behavior.resolve(state, true, 300).animation.clip, "speaking");
});

test("recording and actual audio playback temporarily own animation and return to current activity", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  state.speech = emptySpeech();
  behavior.observe({ chat: activity("remembering") }, true, 0);
  state.speech.capture = { id: "record", phase: "recording" };
  assert.equal(behavior.resolve(state, true, 0).animation.clip, "listening");
  state.speech.capture = { id: "record", phase: "transcribing" };
  assert.equal(behavior.resolve(state, true, 100).animation.clip, "thinking");
  state.speech.capture = null;
  state.speech.playback = {
    id: 1,
    messageId: "m",
    kind: "manual",
    phase: "playing",
    part: 1,
    total: 1,
  };
  assert.equal(behavior.resolve(state, true, 200).animation.clip, "speaking");
  state.speech.playback = null;
  assert.equal(
    behavior.resolve(state, true, 300).animation.clip,
    "remembering",
  );
});

test("reminders enter, hold and exit with the bubble rather than disappearing after one clip", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  state.reminderId = "m";
  const enter = behavior.resolve(state, true, 0).animation;
  assert.equal(enter.segment, "enter");
  behavior.report(enter.id, "ended", 1500);
  const hold = behavior.resolve(state, true, 1500).animation;
  assert.equal(hold.segment, "hold");
  behavior.report(hold.id, "started", 1500);
  assert.equal(behavior.resolve(state, true, 7000).animation.mode, "hold");
  state.reminderId = null;
  const exit = behavior.resolve(state, true, 7100).animation;
  assert.equal(exit.segment, "exit");
  behavior.report(enter.id, "ended", 7150);
  assert.equal(behavior.resolve(state, true, 7150).animation.id, exit.id);
  behavior.report(exit.id, "ended", 7500);
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
  behavior.report(done.id, "ended", 2000);
  behavior.observe({ chat: activity("completed", "new") }, true, 2100);
  assert.equal(behavior.resolve(state, true, 2100).animation.clip, "idle");
  behavior.observe({ chat: activity("failed", "failed") }, true, 2200);
  assert.equal(behavior.resolve(state, true, 2200).animation.clip, "confused");
});

test("ambient gestures are neutral and interruptible, and quiet mode is independent of sleep", () => {
  const behavior = new BehaviorController(
    { ...manifest, ambient: ["idle", "listening"] },
    0,
    () => 0,
  );
  const state = input();
  const first = behavior.resolve(state, true, 15001).animation;
  assert.equal(first.mode, "once");
  assert.ok(manifest.ambient.includes(first.clip));
  behavior.report(first.id, "ended", 18000);
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
    behavior.resolve(state, true, 937000).animation.clip,
    "sleeping",
  );
});

test("weighted fish gestures can repeat twice, yield to work, and never celebrate at random", () => {
  const fishManifest = {
    ...manifest,
    ambient: ["petting_fish", "listening", "idle"],
    ambient_weights: { petting_fish: 65, listening: 20, idle: 15 },
  };
  const behavior = new BehaviorController(fishManifest, 0, () => 0);
  const state = input();
  let now = 12001;
  for (const expected of ["petting_fish", "petting_fish", "listening"]) {
    const animation = behavior.resolve(state, true, now).animation;
    assert.equal(animation.clip, expected);
    assert.equal(animation.mode, "once");
    behavior.report(animation.id, "ended", now + 4000);
    behavior.resolve(state, true, now + 4000);
    now += 16001;
  }
  behavior.observe({ chat: activity("working") }, true, now);
  const working = behavior.resolve(state, true, now).animation;
  assert.equal(working.clip, "thinking");
  behavior.report(working.id, "started", now);
  assert.equal(
    behavior.resolve(state, true, now + 1_000_000).animation.clip,
    "thinking",
  );
  const weighted = new BehaviorController(fishManifest, 0, () => 0.5);
  assert.equal(
    weighted.resolve(state, true, 18501).animation.clip,
    "petting_fish",
  );
});

test("background tasks survive long inactivity, yield to chat, and return to remaining work", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  const first = {
    ...activity("answering", "task-turn", "task"),
    run_id: "one",
    task_id: "research",
  };
  const second = {
    ...first,
    run_id: "two",
    task_id: "watch",
    phase: "remembering" as const,
  };
  const tasks = { one: first, two: second };
  behavior.observe({}, true, 0, tasks);
  let resolved = behavior.resolve(state, true, 1_000_000);
  assert.equal(
    resolved.animation.clip,
    "remembering",
    "background text is not spoken to the user",
  );
  assert.match(resolved.label, /research/);
  behavior.report(resolved.animation.id, "started", 1_000_000);
  behavior.observe({ chat: activity("answering") }, true, 1_000_010, tasks);
  resolved = behavior.resolve(state, true, 1_000_010);
  assert.equal(resolved.animation.clip, "speaking");
  behavior.report(resolved.animation.id, "started", 1_000_010);
  behavior.observe({}, true, 1_000_020, { two: second });
  resolved = behavior.resolve(state, true, 1_000_020);
  assert.match(resolved.label, /watch/);
  assert.equal(resolved.animation.clip, "remembering");
  state.settings.quietUntil = 3_000_000;
  resolved = behavior.resolve(state, true, 2_000_000);
  assert.equal(
    resolved.animation.clip,
    "idle",
    "quiet mode hides gestures without pretending to sleep",
  );
  behavior.report(resolved.animation.id, "started", 2_000_000);
  behavior.observe({}, true, 2_000_001);
  assert.equal(behavior.resolve(state, true, 2_899_999).animation.clip, "idle");
  assert.equal(
    behavior.resolve(state, true, 2_900_000).animation.clip,
    "sleeping",
  );
});

test("active proactive work prevents sleep even when chat is focused or quiet", () => {
  for (const focused of [false, true]) {
    const behavior = new BehaviorController(manifest, 0, () => 0);
    const state = input();
    state.chatFocused = focused;
    state.settings.quietUntil = 3_000_000;
    behavior.observe(
      { proactive: activity("working", "background", "proactive") },
      true,
      0,
    );
    const resolved = behavior.resolve(state, true, 1_000_000);
    assert.equal(resolved.animation.clip, "idle");
    behavior.report(resolved.animation.id, "started", 1_000_000);
    state.settings.quietUntil = 0;
    state.chatFocused = false;
    assert.equal(
      behavior.resolve(state, true, 1_000_100).animation.clip,
      "thinking",
    );
  }
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

test("failed playback falls back without losing activity or letting stale failures disable another clip", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  behavior.observe({ chat: activity("thinking") }, true, 0);
  const thinking = behavior.resolve(state, true, 0).animation;
  behavior.report(thinking.id, "failed", 10);
  const fallback = behavior.resolve(state, true, 10);
  assert.equal(fallback.animation.clip, "idle");
  assert.ok(fallback.error);
  assert.equal(fallback.label, "正在认真想…");
  behavior.report(fallback.animation.id, "started", 10);
  behavior.observe({ chat: activity("answering") }, true, 100);
  const answering = behavior.resolve(state, true, 100).animation;
  behavior.report(thinking.id, "failed", 110);
  assert.equal(behavior.resolve(state, true, 110).animation.id, answering.id);
  assert.equal(
    behavior.resolve(state, true, 5101).animation.clip,
    "idle",
    "missing start acknowledgement must recover",
  );
});

test("a lost end acknowledgement retires a started gesture without disabling its artwork", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  behavior.play("happy", 0);
  const animation = behavior.resolve(state, true, 0).animation;
  behavior.report(animation.id, "started", 0);
  const recovered = behavior.resolve(state, true, 10_000);
  assert.equal(recovered.animation.clip, "idle");
  assert.equal(recovered.error, "");
  behavior.play("happy", 10_001);
  assert.equal(behavior.resolve(state, true, 10_001).animation.clip, "happy");
});

test("language changes update active and terminal captions without restarting playback", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  behavior.observe({ chat: activity("thinking") }, true, 0);
  const initial = behavior.resolve(state, true, 0);
  behavior.report(initial.animation.id, "started", 0);
  state.settings.language = "en";
  const english = behavior.resolve(state, true, 100);
  assert.equal(english.animation.id, initial.animation.id);
  assert.equal(english.label, "Thinking it through…");
  behavior.observe({ chat: activity("completed") }, true, 200);
  const completed = behavior.resolve(state, true, 200);
  assert.equal(completed.label, "All done");
  state.settings.language = "zh";
  const chinese = behavior.resolve(state, true, 300);
  assert.equal(chinese.animation.id, completed.animation.id);
  assert.equal(chinese.label, "完成啦");
});
