import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { BehaviorController } from "../src/main/behavior.ts";
import { emptySpeech } from "../src/main/speech.ts";
import { defaultSettings } from "../src/main/storage.ts";
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
      ...defaultSettings,
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
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  const first = behavior.resolve(state, true, 18001).animation;
  assert.equal(first.mode, "once");
  assert.ok(manifest.ambient.includes(first.clip));
  behavior.report(first.id, "ended", 21000);
  const next = behavior.resolve(state, true, 40000).animation;
  assert.notEqual(next.clip, first.clip);
  state.settings.quietUntil = 100000;
  assert.equal(behavior.resolve(state, true, 41000).animation.clip, "idle");
  state.chatFocused = true;
  behavior.touch(42000);
  assert.equal(
    behavior.resolve(state, true, 42000).animation.clip,
    "listening",
  );
  state.chatFocused = false;
  assert.equal(
    behavior.resolve(state, true, 943000).animation.clip,
    "sleeping",
  );
});

test("varied ambient gestures rest, avoid immediate repeats and respect individual cooldowns", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  let now = 18001;
  const seen = new Set<string>();
  const cooldowns = {
    petting_fish: 60_000,
    listening: 30_000,
    remembering: 75_000,
    thinking: 90_000,
  };
  const lastPlayed = new Map<string, number>();
  let previous = "";
  for (let count = 0; count < 20; count++) {
    const animation = behavior.resolve(state, true, now).animation;
    if (animation.mode !== "once") {
      now += 18_001;
      continue;
    }
    assert.ok(manifest.ambient.includes(animation.clip));
    assert.notEqual(animation.clip, previous);
    assert.ok(
      now - (lastPlayed.get(animation.clip) ?? -Infinity) >=
        cooldowns[animation.clip as keyof typeof cooldowns],
    );
    previous = animation.clip;
    lastPlayed.set(animation.clip, now);
    seen.add(animation.clip);
    assert.equal(animation.mode, "once");
    behavior.report(animation.id, "ended", now + 4000);
    assert.equal(
      behavior.resolve(state, true, now + 4000).animation.clip,
      "idle",
    );
    now += now < 120_000 ? 22_001 : 31_001;
  }
  assert.deepEqual([...seen].sort(), [...manifest.ambient].sort());
  behavior.observe({ chat: activity("working") }, true, now);
  const working = behavior.resolve(state, true, now).animation;
  assert.equal(working.clip, "thinking");
  behavior.report(working.id, "started", now);
  assert.equal(
    behavior.resolve(state, true, now + 1_000_000).animation.clip,
    "thinking",
  );
  const weighted = new BehaviorController(manifest, 0, () => 0.25);
  assert.equal(
    weighted.resolve(state, true, 23501).animation.clip,
    "petting_fish",
  );
});

test("thinking and tool work update captions without resetting the same animation", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  behavior.observe({ chat: activity("thinking") }, true, 0);
  const initial = behavior.resolve(state, true, 0).animation;
  behavior.report(initial.id, "started", 0);
  behavior.observe({ chat: activity("working") }, true, 1000);
  const working = behavior.resolve(state, true, 1000);
  assert.deepEqual(working.animation, initial);
  assert.equal(working.label, "正在查找和处理…");
  behavior.observe({ chat: activity("thinking") }, true, 2000);
  assert.deepEqual(behavior.resolve(state, true, 2000).animation, initial);
});

test("hidden playback never times out as broken artwork and resumes the latest activity", () => {
  for (const working of [false, true]) {
    const behavior = new BehaviorController(manifest, 0, () => 0);
    const state = input();
    if (working) behavior.observe({ chat: activity("thinking") }, true, 0);
    const before = behavior.resolve(state, true, 0).animation;
    const hidden = behavior.resolve(state, false, 6000);
    assert.equal(hidden.error, "");
    assert.equal(hidden.animation.mode, "hold");
    assert.equal(behavior.resolve(state, false, 60_000).error, "");
    behavior.report(before.id, "failed", 60_001);
    const shown = behavior.resolve(state, true, 60_002);
    assert.equal(shown.error, "");
    assert.equal(shown.animation.clip, working ? "thinking" : "idle");
    assert.equal(shown.animation.mode, "loop");
  }
});

test("activity levels control spacing and inactivity slows gestures without extending the sleep deadline", () => {
  for (const [level, delay] of [
    ["calm", 40_000],
    ["normal", 18_000],
    ["lively", 8_000],
  ] as const) {
    const behavior = new BehaviorController(manifest, 0, () => 0);
    const state = input();
    state.settings.activityLevel = level;
    behavior.resolve(state, true, 0);
    assert.equal(
      behavior.resolve(state, true, delay - 1).animation.clip,
      "idle",
    );
    const gesture = behavior.resolve(state, true, delay).animation;
    assert.equal(gesture.mode, "once");
    behavior.report(gesture.id, "ended", 120_000);
    behavior.resolve(state, true, 120_000);
    assert.equal(
      behavior.resolve(state, true, 120_000 + delay).animation.clip,
      "idle",
    );
    assert.equal(
      behavior.resolve(state, true, 120_000 + delay * 1.5).animation.mode,
      "once",
    );
    state.settings.sleepMinutes = 5;
    assert.equal(
      behavior.resolve(state, true, 300_000).animation.clip,
      "sleeping",
    );
    state.settings.sleepMinutes = 0;
    assert.notEqual(
      behavior.resolve(state, true, 301_000).animation.clip,
      "sleeping",
    );
  }
});

test("leaving focused chat or quiet mode gets a fresh rest before ambient playback", () => {
  const behavior = new BehaviorController(manifest, 0, () => 0);
  const state = input();
  state.chatFocused = true;
  behavior.resolve(state, true, 90_000);
  state.chatFocused = false;
  assert.equal(behavior.resolve(state, true, 90_001).animation.clip, "idle");
  assert.equal(behavior.resolve(state, true, 108_000).animation.mode, "once");
  state.settings.quietUntil = 200_000;
  behavior.resolve(state, true, 190_000);
  assert.equal(behavior.resolve(state, true, 200_001).animation.clip, "idle");
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
