import assert from "node:assert/strict";
import test from "node:test";
import { ClientController } from "../src/controller.ts";
import { SessionModel } from "../src/session-model.ts";
import { displayMessages } from "../src/messages.ts";
import type { PendingTurn } from "../src/protocol.ts";
import { message, snapshot as makeSnapshot } from "./fixtures.ts";
const snapshot = (turn: string, sessionId = "session") =>
  makeSnapshot(
    [
      message({
        id: `${turn}:assistant`,
        turn_id: turn,
        content: turn,
        session_id: sessionId,
      }),
    ],
    sessionId,
    "monitor",
  );

test("automatic task inputs are distinguished from ordinary chat and task follow-ups", () => {
  const model = new SessionModel();
  model.apply(snapshot("turn"));
  model.messages.unshift({
    ...model.messages[0],
    id: "turn:user",
    role: "user",
    content: "The same text in every context",
    items: [],
  });
  const base = {
    sessionId: "session",
    messages: model.messages,
    pending: null,
  };
  const task = displayMessages({ ...base, task: true }, undefined, "zh");
  const chat = displayMessages(base, undefined, "zh");
  assert.equal(task[0].taskInput, true);
  assert.equal(task[1].taskInput, false);
  assert.equal(chat[0].taskInput, undefined);
  assert.equal(task[0].content, chat[0].content);
  const query = displayMessages(
    {
      ...base,
      task: true,
      messages: [],
      pending: {
        id: "follow-up",
        sessionId: "session",
        query: model.messages[0].content,
        streaming: true,
      },
    },
    undefined,
    "zh",
  );
  assert.equal(query[0].taskInput, undefined);
  assert.equal(query[0].content, task[0].content);
});

test("automatic stages preserve query errors until the next task session", async () => {
  const model = new SessionModel();
  model.apply(snapshot("old"));
  const pending: PendingTurn = {
    id: "local",
    sessionId: "session",
    turnId: "old",
    query: "old input",
    streaming: false,
    error: "Request failed",
    outcome: "failed",
  };
  const child = {
    model,
    pending: pending as PendingTurn | null,
    connected: false,
    error: null,
  };
  // Keep the real subscription supervisor and session projection; replace only its transport and publisher.
  const controller = Object.create(ClientController.prototype) as {
    children: Map<string, typeof child>;
    client: () => { stream: (...args: unknown[]) => Promise<void> };
    emit: () => void;
    watchChild: (id: string, signal: AbortSignal) => Promise<void>;
  };
  const abort = new AbortController();
  const observed: Array<{ pending: PendingTurn | null; messages: string[] }> =
    [];
  controller.children = new Map([["monitor", child]]);
  controller.emit = () =>
    observed.push({
      pending: child.pending,
      messages: child.model.messages.map((message) => message.id),
    });
  controller.client = () => ({
    stream: async (...args) => {
      const receive = args[3] as (chunk: unknown) => void;
      receive({ chunk_type: "data", chunk: snapshot("old") });
      receive({ chunk_type: "data", chunk: snapshot("new") });
      receive({ chunk_type: "data", chunk: snapshot("next", "next-session") });
      abort.abort();
    },
  });
  await controller.watchChild("monitor", abort.signal);
  assert.equal(observed[0].pending, pending);
  assert.equal(observed[1].pending, pending);
  assert.deepEqual(observed[1].messages, ["new:assistant"]);
  assert.equal(observed[2].pending, null);
  assert.deepEqual(observed[2].messages, ["next:assistant"]);
});
