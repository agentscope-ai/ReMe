import assert from "node:assert/strict";
import test from "node:test";
import { PresentationController } from "../src/main/presentation.ts";
import { displayMessages } from "../src/shared/messages.ts";
import { emptySpeech } from "../src/main/speech.ts";
import type {
  AgentActivity,
  DesktopState,
  SessionMessage,
  TaskActivities,
} from "../src/shared/protocol.ts";

const activity = (
  phase: AgentActivity["phase"],
  turn = "turn",
): AgentActivity => ({
  session_id: "session",
  turn_id: turn,
  source: "chat",
  phase,
});
const message = (content: string, completed = true): SessionMessage => ({
  id: "answer",
  role: "assistant",
  source: "chat",
  turn_id: "turn",
  content,
  completed,
  delivery: "idle",
  notification_eligible: true,
  chunks: [],
});
function setup() {
  const controller = new PresentationController();
  const state = {
    sessionId: "session",
    messages: [],
    pending: null,
    connection: "connected",
    chatVisible: true,
    chatFocused: false,
    settings: { language: "zh", quietUntil: 0 },
    speech: emptySpeech(),
  } as unknown as DesktopState;
  let current: AgentActivity | undefined;
  let history: SessionMessage[] = [];
  let tasks: TaskActivities = {};
  const observe = (
    phase: AgentActivity["phase"],
    now: number,
    live = true,
    turn = "turn",
  ) => {
    current = activity(phase, turn);
    controller.observe(current, live, now);
  };
  const resolve = (
    now: number,
    visible = true,
    dragging = false,
    reminder: import("../src/main/bubbles.ts").Reminder | null = null,
  ) => {
    state.messages = displayMessages(
      { ...state, messages: history },
      current,
      state.settings.language,
    );
    return controller.resolve(
      state,
      current,
      tasks,
      undefined,
      reminder,
      visible,
      dragging,
      now,
    );
  };
  return {
    controller,
    state,
    history: (messages: SessionMessage[]) => {
      history = messages;
    },
    observe,
    resolve,
    tasks: (value: TaskActivities) => {
      tasks = value;
    },
  };
}

test("both surfaces share exact phase text, unaffected by animation, language or drag transitions", () => {
  const h = setup();
  for (const [index, phase] of (
    ["thinking", "working", "remembering", "answering", "waiting"] as const
  ).entries()) {
    h.observe(phase, index * 100);
    const surface = h.resolve(index * 100);
    assert.equal(surface?.kind, "status");
    assert.equal(surface, h.state.messages.at(-1)?.status);
  }
  assert.equal(h.resolve(501, true, true), null);
  assert.equal(h.resolve(502)?.kind, "status");
  h.state.settings.language = "en";
  const english = h.resolve(503);
  assert.equal(english?.kind === "status" && english.text, "Waiting for you…");
  assert.equal(english, h.state.messages.at(-1)?.status);
  assert.equal(h.resolve(504, false), null);
});

test("local streams become native messages once, and final snapshots replace stale deltas on both sides", () => {
  const h = setup();
  h.state.pending = {
    id: "local",
    query: "hello",
    text: "",
    sessionId: "session",
    streaming: true,
  };
  assert.equal(h.resolve(0)?.kind, "status");
  h.observe("working", 10);
  h.state.pending.turnId = "turn";
  h.state.pending.text = "partial";
  const partial = h.resolve(20);
  assert.equal(partial?.kind, "message");
  if (partial?.kind !== "message") throw new Error("Missing reply");
  assert.equal(partial.message, h.state.messages.at(-1));
  assert.equal(partial.message.content, "partial");
  assert.equal(partial.message.status?.text, "正在查找和处理…");
  h.history([message("authoritative final")]);
  h.observe("completed", 30);
  const final = h.resolve(30);
  assert.equal(
    final?.kind === "message" && final.message.content,
    "authoritative final",
  );
  assert.equal(
    h.state.messages.filter((item) => item.role === "assistant").length,
    1,
  );
  h.state.pending = null;
  assert.equal(h.resolve(40)?.kind === "message", true);
});

test("outcomes expire after visible time, pause on hover and narration, and never replay from snapshots", () => {
  const h = setup();
  h.history([message("final")]);
  h.observe("completed", 0, false, "previous");
  assert.equal(h.resolve(0), null);
  h.observe("completed", 10);
  assert.equal(h.resolve(10)?.kind, "message");
  h.controller.action("answer", "pause", true, 1000);
  assert.equal(h.resolve(50_000)?.kind, "message");
  h.controller.action("answer", "pause", false, 50_000);
  h.state.speech.playback = {
    id: 1,
    messageId: "answer",
    kind: "reply",
    phase: "loading",
    part: 1,
    total: 1,
  };
  h.resolve(50_001);
  assert.equal(h.resolve(100_000)?.kind, "message");
  h.state.speech.playback = null;
  h.resolve(100_001);
  assert.equal(h.resolve(112_002), null);
  h.observe("completed", 112_003);
  assert.equal(
    h.resolve(112_003),
    null,
    "duplicate terminal events cannot restart the preview",
  );
  h.observe("completed", 112_004, false);
  assert.equal(h.resolve(112_004), null);
});

test("background status updates cannot replay a terminal reply restored from a snapshot", () => {
  const h = setup();
  h.history([message("historical reply")]);
  h.observe("completed", 0, false);
  assert.equal(h.resolve(0), null);
  h.tasks({
    run: {
      ...activity("working"),
      source: "task",
      task_id: "research",
      run_id: "run",
    },
  });
  h.observe("completed", 10);
  const surface = h.resolve(10);
  assert.equal(surface?.kind, "status");
  assert.equal(surface?.kind === "status" && surface.messageId, undefined);
  assert.match(surface?.kind === "status" ? surface.text : "", /research/);
});

test("dismissed replies stay dismissed across deltas, and a new turn has its own preview", () => {
  const h = setup();
  h.history([message("partial", false)]);
  h.observe("answering", 0);
  h.resolve(0);
  assert.equal(h.controller.action("stale", "dismiss", false, 1), false);
  assert.equal(h.controller.action("answer", "dismiss", false, 2), true);
  h.history([message("more partial", false)]);
  assert.equal(h.resolve(3), null);
  h.observe("completed", 4);
  assert.equal(h.resolve(4), null);
  h.observe("thinking", 5, true, "next");
  assert.equal(h.resolve(5)?.kind, "status");
  h.controller.reset();
  assert.equal(h.resolve(6), null);
});

test("failed and stopped replies retain partial content and the same outcome in both views", () => {
  for (const phase of ["failed", "interrupted"] as const) {
    const h = setup();
    h.history([message("partial")]);
    h.observe(phase, 0);
    const surface = h.resolve(0);
    assert.equal(surface?.kind, "message");
    if (surface?.kind !== "message") throw new Error("Missing outcome");
    assert.equal(surface.message, h.state.messages.at(-1));
    assert.equal(surface.message.status?.tone, "error");
    assert.equal(surface.message.streaming, false);
  }
  const h = setup();
  h.observe("failed", 0);
  assert.equal(
    h.resolve(0)?.kind,
    "status",
    "failures before any text still have feedback",
  );
});

test("open chat can mirror a reminder without changing its receipt or using reply cooldown", () => {
  const h = setup();
  h.history([{ ...message("a reminder"), source: "proactive" }]);
  const reminder = {
    id: "answer",
    text: "a reminder",
    presentationId: 1,
    paused: false,
  };
  const surface = h.resolve(0, true, false, reminder);
  assert.equal(surface?.kind, "message");
  if (surface?.kind !== "message") throw new Error("Missing reminder");
  assert.equal(surface.message, h.state.messages[0]);
  assert.equal(surface.message.source, "proactive");
  h.observe("thinking", 1);
  assert.equal(
    h.resolve(1, true, false, reminder)?.kind,
    "status",
    "foreground work has priority",
  );
});

test("background tasks and speech cannot overwrite the foreground conversation", () => {
  const h = setup();
  h.tasks({
    run: {
      ...activity("working"),
      source: "task",
      task_id: "research",
      run_id: "run",
    },
  });
  const background = h.resolve(0);
  assert.equal(background?.kind, "status");
  if (background?.kind !== "status")
    throw new Error("Missing background status");
  assert.match(background.text, /research/);
  h.state.speech.capture = { id: "record", phase: "recording" };
  h.observe("remembering", 1);
  const surface = h.resolve(1);
  assert.equal(surface, h.state.messages.at(-1)?.status);
  assert.equal(surface?.kind === "status" && surface.text, "正在回忆…");
});

test("a new local request cannot inherit the preceding turn's terminal phase", () => {
  const h = setup();
  h.observe("completed", 0, false, "previous");
  h.state.pending = {
    id: "local",
    query: "new question",
    text: "",
    sessionId: "session",
    streaming: true,
  };
  const surface = h.resolve(1);
  assert.equal(surface?.kind === "status" && surface.text, "正在认真想…");
  assert.equal(h.state.messages.at(-1)?.streaming, true);
  assert.equal(
    h.resolve(100_000)?.kind,
    "status",
    "long-running work cannot expire like a completed preview",
  );
});
