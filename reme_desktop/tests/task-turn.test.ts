import assert from "node:assert/strict";
import test from "node:test";
import { DesktopController } from "../src/main/controller.ts";
import { SessionModel } from "../src/main/session-model.ts";
import type {
  AgentEvent,
  PendingTurn,
  SessionMessage,
} from "../src/shared/protocol.ts";

const reply = (turn: string): SessionMessage => ({
  id: `${turn}:assistant`,
  role: "assistant",
  turn_id: turn,
  source: "chat",
  content: turn,
  completed: true,
  delivery: "idle",
  notification_eligible: false,
  chunks: [],
});
const snapshot = (turn: string): AgentEvent => ({
  type: "snapshot",
  session_id: "session",
  status: "idle",
  activities: {},
  snapshot: {
    session_id: "session",
    state: { session_id: "session", status: "idle", active_flags: [] },
    messages: [reply(turn)],
  },
});

test("a later task round replaces an old failed draft without losing same-round errors", async () => {
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
  const controller = Object.create(DesktopController.prototype) as {
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
      abort.abort();
    },
  });
  await controller.watchChild("monitor", abort.signal);
  assert.equal(observed[0].pending, pending);
  assert.equal(observed[1].pending, null);
  assert.deepEqual(observed[1].messages, ["new:assistant"]);
});
