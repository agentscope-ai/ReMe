import assert from "node:assert/strict";
import test from "node:test";
import { applyStreamChunk } from "../app/chat-stream.ts";
import { mergeSessionMessages, mergeSessionTab } from "../app/session-sync.ts";

const entry = (id, role, content, extra = {}) => ({
  id,
  role,
  content,
  source: "chat",
  created_at: null,
  turn_id: "turn-1",
  completed: true,
  chunks: [],
  ...extra,
});
const snapshot = (messages) => ({
  session_id: "session-1",
  state: { session_id: "session-1", status: "idle", active_flags: [] },
  messages,
});
const turn = (id, text) => ({
  id,
  role: "assistant",
  content: "",
  turnId: "turn-1",
  blocks: [
    {
      id: "think:t1",
      type: "think",
      sourceType: "think",
      payloads: ["reason"],
      status: "done",
      expanded: false,
    },
    { id: "content:a1", type: "content", text },
  ],
});

test("session synchronization preserves streamed blocks and orders multiple proactive messages", () => {
  const streamed = turn("temporary-assistant", "B");
  const data = snapshot([
    entry("u1", "user", "A"),
    entry("a1", "assistant", "B"),
    entry("p1", "assistant", "C", { source: "proactive" }),
    entry("u2", "user", "D", { turn_id: "turn-2" }),
    entry("a2", "assistant", "E", { turn_id: "turn-2" }),
    entry("p2", "assistant", "F", { source: "proactive", turn_id: "turn-2" }),
  ]);
  const result = mergeSessionMessages([streamed], data);
  assert.deepEqual(
    result.map((message) => message.content),
    ["A", "B", "C", "D", "E", "F"],
  );
  assert.equal(result[1].blocks, streamed.blocks);
  assert.equal(result[2].source, "proactive");
  assert.equal(result[2].blocks[0].text, "C");
  assert.equal(result[5].source, "proactive");
  assert.equal(mergeSessionMessages(result, data), result);
});

test("native history replaces old browser-only history without duplicating answers", () => {
  const old = [
    { id: "old-user", role: "user", content: "A" },
    {
      id: "old-answer",
      role: "assistant",
      content: "",
      blocks: [{ id: "answer", type: "content", text: "B" }],
    },
  ];
  const result = mergeSessionMessages(
    old,
    snapshot([entry("u1", "user", "A"), entry("a1", "assistant", "B")]),
  );
  assert.deepEqual(
    result.map((message) => message.id),
    ["u1", "a1"],
  );
});

test("a request rejected before starting a native turn retains its visible error", () => {
  const rejected = {
    id: "rejected",
    role: "assistant",
    content: "",
    localOnly: true,
    blocks: [{ id: "error", type: "error", text: "Invalid credentials" }],
  };
  const result = mergeSessionMessages([rejected], snapshot([]));
  assert.deepEqual(result, [rejected]);
});

test("native acceptance before reply_start replaces the pending local turn", () => {
  const current = [
    { id: "old-user", role: "user", content: "A", turnId: "turn-1" },
    turn("old-answer", "B"),
    { id: "local-user", role: "user", content: "A", localOnly: true },
    {
      id: "local-answer",
      role: "assistant",
      content: "",
      localOnly: true,
      blocks: [{ id: "error", type: "error", text: "Connection interrupted" }],
    },
  ];
  const previous = [entry("u1", "user", "A"), entry("a1", "assistant", "B")];
  // An older identical prompt is insufficient evidence of this request's acceptance.
  assert.equal(mergeSessionMessages(current, snapshot(previous)).length, 4);
  const data = snapshot([
    ...previous,
    entry("p1", "assistant", "reminder", { source: "proactive" }),
    entry("u2", "user", "A", { turn_id: "turn-2" }),
    entry("a2", "assistant", "recovered", { turn_id: "turn-2" }),
  ]);
  const recovered = mergeSessionMessages(current, data);
  assert.deepEqual(
    recovered.map((message) => message.id),
    ["u1", "a1", "p1", "u2", "a2"],
  );
  assert.equal(recovered[4].blocks[0].text, "recovered");
  assert.equal(mergeSessionMessages(recovered, data), recovered);
});

test("unaccepted requests survive recovery of a different native turn", () => {
  const rejected = [
    { id: "local-user", role: "user", content: "rejected", localOnly: true },
    {
      id: "local-answer",
      role: "assistant",
      content: "",
      localOnly: true,
      blocks: [{ id: "error", type: "error", text: "Invalid credentials" }],
    },
  ];
  const data = snapshot([entry("u1", "user", "accepted")]);
  assert.deepEqual(mergeSessionMessages(rejected, data).slice(-2), rejected);
});

test("stale session snapshots cannot overwrite a new turn or a different session", () => {
  const tab = {
    id: "tab",
    type: "agent",
    sessionId: "session-1",
    title: "",
    messages: [turn("new-answer", "latest")],
  };
  const data = snapshot([entry("old-answer", "assistant", "old")]);
  assert.equal(mergeSessionTab(tab, data, "old-answer"), tab);
  assert.equal(
    mergeSessionTab({ ...tab, streaming: true }, data, "new-answer").messages,
    tab.messages,
  );
  assert.equal(
    mergeSessionTab({ ...tab, sessionId: "session-2" }, data, "new-answer")
      .messages,
    tab.messages,
  );
});

test("history polling preserves running tools until a native completion snapshot arrives", () => {
  const chunk = (status) => ({
    chunk_type: "tool_result",
    chunk: { id: "tool-1", type: "mcpToolCall", status },
    done: false,
    tool_call_id: "tool-1",
    tool_call_name: "search",
    event: "end",
    status: status === "inProgress" ? "running" : status,
  });
  const active = snapshot([
    entry("a1", "assistant", "", {
      completed: false,
      chunks: [chunk("inProgress")],
    }),
  ]);
  const running = mergeSessionMessages([], active);
  assert.equal(running[0].blocks[0].status, "running");
  const completed = snapshot([
    entry("a1", "assistant", "", { chunks: [chunk("completed")] }),
  ]);
  assert.equal(
    mergeSessionMessages(running, completed)[0].blocks[0].status,
    "done",
  );
});

test("native history repairs interrupted tools and restores missed failures", () => {
  const data = (status) =>
    snapshot([
      entry("a1", "assistant", "", {
        chunks: [
          {
            chunk_type: "tool_result",
            chunk: { id: "tool-1", type: "commandExecution", status },
            done: false,
            tool_call_id: "tool-1",
            event: "end",
            status,
          },
        ],
      }),
    ]);
  const interrupted = mergeSessionMessages([], data("failed"));
  interrupted[0].blocks.push({
    id: "error:connection",
    type: "error",
    text: "The connection ended before the reply completed.",
  });
  const recovered = mergeSessionMessages(interrupted, data("completed"));
  assert.equal(recovered[0].blocks[0].status, "done");
  assert.equal(recovered[0].blocks.length, 1);

  const failed = mergeSessionMessages(recovered, data("failed"));
  assert.equal(failed[0].blocks[0].status, "error");
});

test("history refresh preserves the exact live event trace and structured outcome", () => {
  const streamed = turn("temporary-assistant", "B");
  streamed.events = [
    { chunk_type: "data", chunk: { signature: "signed" }, done: false },
  ];
  streamed.reply = {
    session_id: "session-1",
    usage: null,
    status: "completed",
    error: null,
    result: "B",
    structured_output: false,
    raw: { signature: "signed" },
  };
  const result = mergeSessionMessages(
    [streamed],
    snapshot([entry("a1", "assistant", "B")]),
  );
  assert.deepEqual(result[0].events, streamed.events);
  assert.deepEqual(result[0].reply, streamed.reply);
});

test("native completion cannot overwrite a collected application-level terminal outcome", () => {
  for (const status of ["completed", "failed", "interrupted", "waiting"]) {
    const outcome = {
      session_id: "session-1",
      result: "invalid JSON",
      structured_output: false,
      usage: null,
      status,
      error: status === "failed" ? "Codex returned invalid JSON" : null,
      raw: { status: "completed", final_response: "invalid JSON" },
    };
    let live = { ...turn("temporary-assistant", "invalid JSON"), blocks: [] };
    const chunks = [
      {
        chunk_type: "content",
        event: "end",
        block_id: "answer",
        chunk: "invalid JSON",
        done: false,
      },
      {
        chunk_type: "reply_end",
        event: "end",
        status,
        chunk: outcome,
        done: false,
      },
    ];
    for (const chunk of chunks) live = applyStreamChunk(live, chunk);
    const data = snapshot([
      entry("a1", "assistant", "invalid JSON", {
        chunks: [
          chunks[0],
          {
            chunk_type: "reply_end",
            event: "end",
            status: "completed",
            chunk: null,
            done: false,
          },
        ],
      }),
    ]);
    const restored = mergeSessionMessages([live], data);
    assert.equal(restored[0].status, status);
    assert.equal(restored[0].reply.status, status);
    assert.deepEqual(restored[0].events, live.events);
    assert.deepEqual(
      restored[0].blocks.filter((block) => block.type === "data")[0].payloads,
      [false],
    );
    assert.deepEqual(
      restored[0].blocks
        .filter((block) => block.type === "error")
        .map((block) => block.text),
      status === "failed" ? [outcome.error] : [],
    );
    assert.equal(mergeSessionMessages(restored, data), restored);
  }
});
