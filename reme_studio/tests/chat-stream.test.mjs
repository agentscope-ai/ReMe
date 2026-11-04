import assert from "node:assert/strict";
import test from "node:test";
import {
  applyStreamChunk,
  chatStreamError,
  decodeSseEvent,
  formatStreamPayloads,
} from "../app/chat-stream.ts";

const initial = () => ({
  id: "assistant",
  role: "assistant",
  content: "",
  blocks: [],
});
const chunk = (chunk_type, value = "", extra = {}) => ({
  chunk_type,
  chunk: value,
  done: false,
  ...extra,
});

test("chat chunks retain arrival order and correlate tool results", () => {
  let message = initial();
  message = applyStreamChunk(
    message,
    chunk("think", "checking", { block_id: "thought-1" }),
  );
  message = applyStreamChunk(
    message,
    chunk("tool_call", '{"path":"daily"}', {
      tool_call_id: "tool-1",
      tool_call_name: "search",
    }),
  );
  message = applyStreamChunk(
    message,
    chunk("tool_result", ["daily/today.md"], { tool_call_id: "tool-1" }),
  );
  message = applyStreamChunk(
    message,
    chunk("content", "Found it.", { block_id: "answer-1" }),
  );
  message = applyStreamChunk(
    message,
    chunk(
      "data",
      { matches: 1 },
      { block_id: "data-1", media_type: "application/json" },
    ),
  );

  assert.deepEqual(
    message.blocks.map((block) => block.type),
    ["think", "tool", "content", "data"],
  );
  assert.equal(message.blocks[1].name, "search");
  assert.deepEqual(message.blocks[1].resultPayloads, [["daily/today.md"]]);
  assert.equal(message.blocks[2].text, "Found it.");
});

test("reply completion collapses every non-text block", () => {
  let message = initial();
  message = applyStreamChunk(
    message,
    chunk("think", "done", { block_id: "thought-1" }),
  );
  message = applyStreamChunk(
    message,
    chunk("usage", "", { input_tokens: 12, output_tokens: 4 }),
  );
  assert.ok(
    message.blocks.every(
      (block) =>
        block.type === "content" || block.type === "error" || block.expanded,
    ),
  );

  message = applyStreamChunk(message, chunk("reply_end"));
  assert.ok(
    message.blocks.every(
      (block) =>
        block.type === "content" || block.type === "error" || !block.expanded,
    ),
  );
  assert.ok(
    message.blocks.every(
      (block) =>
        block.type === "content" ||
        block.type === "error" ||
        block.status === "done",
    ),
  );
});

test("reply completion restores the final answer when content deltas were missed", () => {
  let message = initial();
  message = applyStreamChunk(
    message,
    chunk("think", "checking", { block_id: "thought-1" }),
  );
  message = applyStreamChunk(
    message,
    chunk("reply_end", "", { metadata: { answer: "Final answer." } }),
  );

  assert.equal(message.blocks.at(-1).type, "content");
  assert.equal(message.blocks.at(-1).text, "Final answer.");
  assert.equal(message.blocks[0].status, "done");
});

test("content deltas with the same block id are combined", () => {
  let message = initial();
  message = applyStreamChunk(
    message,
    chunk("content", "Hello", { block_id: "answer" }),
  );
  message = applyStreamChunk(
    message,
    chunk("content", " world", { block_id: "answer" }),
  );

  assert.equal(message.blocks.length, 1);
  assert.equal(message.blocks[0].text, "Hello world");
});

test("Codex hook notifications do not become chat data blocks", () => {
  let message = initial();
  for (const status of ["running", "completed"]) {
    const method = status === "running" ? "hook/started" : "hook/completed";
    message = applyStreamChunk(
      message,
      chunk(
        "data",
        {
          run: {
            id: "session-start:5:/etc/codex/hooks.json",
            eventName: "sessionStart",
            status,
          },
          threadId: "thread-1",
          turnId: "turn-1",
        },
        { metadata: { codex_method: method } },
      ),
    );
  }
  message = applyStreamChunk(message, chunk("content", "主人，你好。"));
  message = applyStreamChunk(message, chunk("reply_end"));

  assert.deepEqual(
    message.blocks.map((block) => block.type),
    ["content"],
  );
  assert.equal(message.blocks[0].text, "主人，你好。");
});

test("filtering hook diagnostics preserves structured output and errors", () => {
  let message = initial();
  message = applyStreamChunk(message, chunk("data", { matches: 1 }));
  message = applyStreamChunk(
    message,
    chunk(
      "data",
      { diff: "file changes" },
      {
        metadata: { codex_method: "turn/diff/updated" },
      },
    ),
  );
  message = applyStreamChunk(message, chunk("error", "Hook failed"));

  assert.deepEqual(
    message.blocks.map((block) => block.type),
    ["data", "error"],
  );
  assert.deepEqual(message.blocks[0].payloads, [
    { matches: 1 },
    { diff: "file changes" },
  ]);
  assert.equal(message.blocks[1].text, "Hook failed");
});

test("Codex tool snapshots finish before the reply ends", () => {
  let message = applyStreamChunk(
    initial(),
    chunk(
      "tool_call",
      { id: "tool-1", type: "mcpToolCall", status: "inProgress" },
      {
        tool_call_id: "tool-1",
        tool_call_name: "search",
      },
    ),
  );
  message = applyStreamChunk(
    message,
    chunk("tool_result", "Searching", { tool_call_id: "tool-1" }),
  );
  assert.equal(message.blocks[0].status, "running");
  message = applyStreamChunk(
    message,
    chunk(
      "tool_result",
      { id: "tool-1", type: "mcpToolCall", status: "inProgress" },
      {
        tool_call_id: "tool-1",
      },
    ),
  );
  assert.equal(message.blocks[0].status, "running");
  message = applyStreamChunk(
    message,
    chunk(
      "tool_result",
      { id: "tool-1", type: "mcpToolCall", status: "completed" },
      {
        tool_call_id: "tool-1",
      },
    ),
  );

  assert.equal(message.blocks[0].status, "done");
  assert.equal(message.blocks[0].expanded, false);
});

test("Codex failed and declined tools stay failed after reply completion", () => {
  for (const status of ["failed", "declined"]) {
    let message = applyStreamChunk(
      initial(),
      chunk(
        "tool_result",
        { id: "tool-1", type: "commandExecution", status },
        {
          tool_call_id: "tool-1",
          tool_call_name: "commandExecution",
        },
      ),
    );
    message = applyStreamChunk(message, chunk("reply_end"));

    assert.equal(message.blocks[0].status, "error");
  }
});

test("tool result data does not override the tool's execution state", () => {
  const message = applyStreamChunk(
    initial(),
    chunk(
      "tool_result",
      { id: "record-1", status: "failed" },
      {
        tool_call_id: "tool-1",
      },
    ),
  );
  assert.equal(message.blocks[0].status, "running");
});

test("payload formatting joins text deltas and separates structured snapshots", () => {
  assert.equal(formatStreamPayloads(["hel", "lo"]), "hello");
  assert.equal(
    formatStreamPayloads([{ status: "running" }, { status: "completed" }]),
    '{\n  "status": "running"\n}\n\n{\n  "status": "completed"\n}',
  );
  assert.equal(
    formatStreamPayloads(["progress", { result: 1 }]),
    'progress\n\n{\n  "result": 1\n}',
  );
});

test("SSE terminal marker becomes an explicit done chunk", () => {
  assert.deepEqual(decodeSseEvent("data:[DONE]"), {
    chunk_type: "done",
    chunk: "",
    done: true,
  });
  assert.deepEqual(
    decodeSseEvent('data:{"chunk_type":"content","chunk":"hi","done":false}'),
    {
      chunk_type: "content",
      chunk: "hi",
      done: false,
    },
  );
});

test("an aborted stream finishes without showing an error", async () => {
  const controller = new AbortController();
  controller.abort();

  const error = await chatStreamError(
    () => Promise.reject(new DOMException("Aborted", "AbortError")),
    controller.signal,
    "Chat failed",
  );

  assert.equal(error, undefined);
});

test("a failed stream returns its error message", async () => {
  const controller = new AbortController();

  const error = await chatStreamError(
    () => Promise.reject(new Error("Connection lost")),
    controller.signal,
    "Chat failed",
  );

  assert.equal(error, "Connection lost");
});
