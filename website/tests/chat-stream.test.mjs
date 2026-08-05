import assert from "node:assert/strict";
import test from "node:test";
import { applyStreamChunk, decodeSseEvent } from "../app/chat-stream.ts";

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
