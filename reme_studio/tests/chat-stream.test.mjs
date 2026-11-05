import assert from "node:assert/strict";
import test from "node:test";
import {
  applyStreamChunk,
  chatStreamError,
  decodeSseEvent,
  finishChatMessage,
  formatStreamPayloads,
  readChatStream,
  toggleChatBlock,
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

test("thinking finishes as each tool or answer phase starts", () => {
  let message = applyStreamChunk(
    initial(),
    chunk("think", "Checking the time", { block_id: "before-tool" }),
  );
  message = applyStreamChunk(
    message,
    chunk("tool_call", "date", { tool_call_id: "clock" }),
  );
  assert.equal(message.blocks[0].status, "done");
  assert.equal(message.blocks[0].expanded, false);
  assert.equal(message.blocks[1].status, "calling");

  message = applyStreamChunk(
    message,
    chunk("tool_result", "00:11", {
      tool_call_id: "clock",
      status: "completed",
    }),
  );
  message = applyStreamChunk(
    message,
    chunk("think", "Reporting the time", { block_id: "after-tool" }),
  );
  assert.equal(message.blocks[2].status, "streaming");
  message = applyStreamChunk(message, chunk("content", "现在是"));
  assert.deepEqual(
    message.blocks.slice(0, 3).map((block) => block.status),
    ["done", "done", "done"],
  );
  assert.equal(message.blocks[2].expanded, false);

  message = toggleChatBlock(message, message.blocks[0].id, true);
  message = applyStreamChunk(message, chunk("content", " 00:11。"));
  assert.equal(message.blocks[0].expanded, true);
});

test("thinking deltas and telemetry keep the current thought active", () => {
  for (const block_id of [undefined, "thought"]) {
    let message = applyStreamChunk(
      initial(),
      chunk("think", "First", { block_id }),
    );
    message = applyStreamChunk(
      message,
      chunk("think", " second", { block_id }),
    );
    message = applyStreamChunk(
      message,
      chunk("usage", "", { input_tokens: 4 }),
    );
    message = applyStreamChunk(message, chunk("data", { progress: 1 }));
    message = applyStreamChunk(message, chunk("content"));
    assert.deepEqual(message.blocks[0].payloads, ["First", " second"]);
    assert.equal(message.blocks[0].status, "streaming");
    assert.equal(message.blocks[0].expanded, true);
  }
});

test("a new thought settles the previous thought without finishing pending tools", () => {
  let message = applyStreamChunk(
    initial(),
    chunk("tool_call", "", { tool_call_id: "pending" }),
  );
  message = applyStreamChunk(
    message,
    chunk("think", "First", { block_id: "first" }),
  );
  message = applyStreamChunk(
    message,
    chunk("think", "Second", { block_id: "second" }),
  );
  assert.equal(message.blocks[0].status, "calling");
  assert.equal(message.blocks[1].status, "done");
  assert.equal(message.blocks[1].expanded, false);
  assert.equal(message.blocks[2].status, "streaming");
  assert.equal(message.blocks[2].expanded, true);
  message = applyStreamChunk(
    message,
    chunk("tool_result", "Background output", { tool_call_id: "pending" }),
  );
  assert.equal(message.blocks[2].status, "streaming");
  assert.equal(message.blocks[2].expanded, true);
  message = applyStreamChunk(message, chunk("content", "Partial answer"));
  assert.equal(message.blocks[0].status, "running");
  assert.equal(message.blocks[2].status, "done");
});

test("reply completion restores the final answer when content deltas were missed", () => {
  let message = initial();
  message = applyStreamChunk(
    message,
    chunk("think", "checking", { block_id: "thought-1" }),
  );
  message = applyStreamChunk(
    message,
    chunk(
      "reply_end",
      { result: "Final answer." },
      { event: "end", status: "completed" },
    ),
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
        { metadata: { codex_method: method, visibility: "diagnostic" } },
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
        status: "running",
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
        event: "end",
        status: "completed",
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
          event: "end",
          status: "failed",
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

test("Claude tools remain pending across message boundaries and report their actual result", () => {
  for (const is_error of [false, true]) {
    let message = applyStreamChunk(
      initial(),
      chunk("tool_call", "", { tool_call_id: "tool-1" }),
    );
    message = applyStreamChunk(message, chunk("reply_end"));
    assert.equal(message.blocks[0].status, "calling");

    message = applyStreamChunk(
      message,
      chunk(
        "tool_result",
        { tool_use_id: "tool-1", content: "Result", is_error },
        {
          tool_call_id: "tool-1",
          event: "end",
          status: is_error ? "failed" : "completed",
        },
      ),
    );
    assert.equal(message.blocks[0].status, is_error ? "error" : "done");
    message = applyStreamChunk(message, chunk("reply_end"));
    message = applyStreamChunk(message, chunk("done", "", { done: true }));
    assert.equal(message.blocks[0].status, is_error ? "error" : "done");
  }
});

test("Claude server tool results finish without an is_error field", () => {
  const message = applyStreamChunk(
    initial(),
    chunk(
      "tool_result",
      { tool_use_id: "server-1", content: { matches: 1 } },
      { tool_call_id: "server-1", event: "end", status: "completed" },
    ),
  );
  assert.equal(message.blocks[0].status, "done");
});

test("unrelated tool data cannot masquerade as a Claude error result", () => {
  const message = applyStreamChunk(
    initial(),
    chunk(
      "tool_result",
      { tool_use_id: "other-tool", content: "data", is_error: true },
      { tool_call_id: "tool-1" },
    ),
  );
  assert.equal(message.blocks[0].status, "running");
});

test("a transport failure marks pending blocks failed and preserves completed phases", () => {
  let message = initial();
  message = applyStreamChunk(
    message,
    chunk("think", "Checking", { block_id: "thought-1" }),
  );
  message = applyStreamChunk(
    message,
    chunk("tool_result", "", {
      tool_call_id: "completed",
      status: "completed",
    }),
  );
  message = applyStreamChunk(
    message,
    chunk("tool_call", "", { tool_call_id: "pending" }),
  );
  message = applyStreamChunk(
    message,
    chunk("think", "Still checking", { block_id: "pending-thought" }),
  );
  message = finishChatMessage(message, "Connection lost");

  assert.deepEqual(
    message.blocks.slice(0, 4).map((block) => block.status),
    ["done", "done", "error", "error"],
  );
  assert.equal(message.blocks.at(-1).text, "Connection lost");
  message = finishChatMessage(message);
  assert.equal(message.blocks[2].status, "error");
});

test("a server error followed by DONE does not mark a pending tool successful", () => {
  let message = applyStreamChunk(
    initial(),
    chunk("tool_call", "", { tool_call_id: "pending" }),
  );
  message = applyStreamChunk(message, chunk("error", "Agent failed"));
  message = applyStreamChunk(message, chunk("done", "", { done: true }));
  assert.equal(message.blocks[0].status, "error");
});

test("late tool output does not reopen a completed or failed execution", () => {
  for (const state of ["completed", "failed"]) {
    let message = applyStreamChunk(
      initial(),
      chunk("tool_result", "Result", {
        tool_call_id: "tool-1",
        status: state,
      }),
    );
    message = applyStreamChunk(
      message,
      chunk("tool_result", "Trailing output", { tool_call_id: "tool-1" }),
    );
    assert.equal(
      message.blocks[0].status,
      state === "failed" ? "error" : "done",
    );
    assert.equal(message.blocks[0].expanded, false);
  }
});

test("streaming preserves manual collapse and expansion of thinking and tool blocks", () => {
  for (const event of [
    chunk("think", "First", { block_id: "thought-1" }),
    chunk("tool_result", "First", { tool_call_id: "tool-1" }),
  ]) {
    let message = applyStreamChunk(initial(), event);
    const id = message.blocks[0].id;
    message = toggleChatBlock(message, id, false);
    message = applyStreamChunk(message, { ...event, chunk: "Next" });
    assert.equal(message.blocks[0].expanded, false);
    message = toggleChatBlock(message, id, true);
    message = applyStreamChunk(message, { ...event, chunk: "Last" });
    assert.equal(message.blocks[0].expanded, true);
  }
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

const sse = (event) => `data:${JSON.stringify(event)}\n\n`;
const byteStream = (text, fragment = false) => {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({
    start(controller) {
      if (fragment) {
        for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
      } else controller.enqueue(bytes);
      controller.close();
    },
  });
};

test("SSE reading preserves fragmented Chinese text and CRLF frames", async () => {
  const stream = byteStream(
    (sse(chunk("content", "你好")) + "data:[DONE]\n\n").replaceAll(
      "\n",
      "\r\n",
    ),
    true,
  );
  const chunks = [];
  await readChatStream(stream, (event) => chunks.push(event), "Interrupted");
  assert.equal(chunks[0].chunk, "你好");
  assert.equal(chunks[1].done, true);
  assert.equal(stream.locked, false);
});

test("SSE reading stops at DONE and cancels a connection that is still open", async () => {
  let cancelled = false;
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode("data:[DONE]\n\n"));
    },
    cancel() {
      cancelled = true;
    },
  });
  const chunks = [];
  await readChatStream(stream, (event) => chunks.push(event), "Interrupted");
  assert.deepEqual(
    chunks.map((event) => event.chunk_type),
    ["done"],
  );
  assert.equal(cancelled, true);
  assert.equal(stream.locked, false);
});

test("SSE EOF without DONE fails even after reply_end and retains partial output", async () => {
  for (const suffix of ["", sse(chunk("reply_end"))]) {
    let message = initial();
    const stream = byteStream(sse(chunk("content", "Partial reply")) + suffix);
    await assert.rejects(
      readChatStream(
        stream,
        (event) => {
          message = applyStreamChunk(message, event);
        },
        "Interrupted",
      ),
      /Interrupted/,
    );
    assert.equal(message.blocks[0].text, "Partial reply");
    assert.equal(stream.locked, false);
  }
});

test("SSE reading accepts a terminal event without a trailing blank line", async () => {
  for (const terminal of [
    "data:[DONE]",
    `data:${JSON.stringify(chunk("done", "", { done: true }))}`,
  ]) {
    const chunks = [];
    await readChatStream(
      byteStream(terminal),
      (event) => chunks.push(event),
      "Interrupted",
    );
    assert.equal(chunks[0].done, true);
  }
});

test("SSE reading preserves transport errors and releases the reader", async () => {
  const stream = new ReadableStream({
    start(controller) {
      controller.error(new Error("Connection lost"));
    },
  });
  await assert.rejects(
    readChatStream(stream, () => {}, "Interrupted"),
    /Connection lost/,
  );
  assert.equal(stream.locked, false);
});

test("malformed SSE cancels its connection and surfaces the parsing error", async () => {
  let cancelled = false;
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode("data:{broken}\n\n"));
    },
    cancel() {
      cancelled = true;
    },
  });
  await assert.rejects(
    readChatStream(stream, () => {}, "Interrupted"),
    SyntaxError,
  );
  assert.equal(cancelled, true);
  assert.equal(stream.locked, false);
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

test("authoritative text snapshots replace deltas, including an empty final text", () => {
  let message = initial();
  for (const event of [
    chunk("content", "draft", { block_id: "answer" }),
    chunk("content", "corrected", {
      block_id: "answer",
      event: "end",
      metadata: { phase: "final_answer" },
    }),
    chunk("content", "corrected", { block_id: "answer", event: "end" }),
    chunk("content", null, { block_id: "answer", event: "end" }),
  ])
    message = applyStreamChunk(message, event);
  assert.equal(message.blocks.length, 1);
  assert.equal(message.blocks[0].text, "corrected");
  assert.equal(message.blocks[0].metadata.phase, "final_answer");
  message = applyStreamChunk(
    message,
    chunk("content", "", { block_id: "answer", event: "end" }),
  );
  assert.equal(message.blocks[0].text, "");
});

test("tool snapshots replace streamed output and argument completion keeps the tool pending", () => {
  let message = applyStreamChunk(
    initial(),
    chunk("tool_call", '{"q":', { tool_call_id: "tool" }),
  );
  message = applyStreamChunk(
    message,
    chunk("tool_call", { q: "hello" }, { tool_call_id: "tool", event: "end" }),
  );
  assert.equal(message.blocks[0].status, "calling");
  message = applyStreamChunk(
    message,
    chunk("tool_result", "partial", {
      tool_call_id: "tool",
      status: "running",
    }),
  );
  message = applyStreamChunk(
    message,
    chunk(
      "tool_result",
      { output: "complete" },
      { tool_call_id: "tool", event: "end", status: "completed" },
    ),
  );
  assert.deepEqual(message.blocks[0].callPayloads, [{ q: "hello" }]);
  assert.deepEqual(message.blocks[0].resultPayloads, [{ output: "complete" }]);
  assert.equal(message.blocks[0].status, "done");
});

test("retryable errors do not turn a recovered reply into a failure", () => {
  let message = applyStreamChunk(
    initial(),
    chunk("error", "temporary", { metadata: { will_retry: true } }),
  );
  message = applyStreamChunk(message, chunk("content", "recovered"));
  message = applyStreamChunk(
    message,
    chunk("reply_end", null, { event: "end", status: "completed" }),
  );
  message = applyStreamChunk(message, chunk("done", "", { done: true }));
  assert.equal(message.status, "completed");
  assert.equal(message.blocks[0].metadata.will_retry, true);
});

test("reply interruption and input waits survive transport completion", () => {
  for (const status of ["interrupted", "waiting"]) {
    let message = applyStreamChunk(
      initial(),
      chunk("think", "working", { block_id: "thought" }),
    );
    message = applyStreamChunk(
      message,
      chunk("tool_call", null, { tool_call_id: "tool" }),
    );
    message = applyStreamChunk(
      message,
      chunk("reply_end", null, { event: "end", status }),
    );
    message = applyStreamChunk(message, chunk("done", "", { done: true }));
    assert.equal(message.status, status);
    assert.equal(message.blocks[0].status, "done");
    assert.equal(message.blocks[1].status, status);
  }
});

test("terminal outcomes retain raw data, scalar structured output and diagnostic events", () => {
  const outcome = {
    session_id: "session",
    result: "final answer",
    structured_output: false,
    usage: null,
    status: "waiting",
    error: null,
    raw: { cache: 7, signature: "signed", future: [false, null] },
  };
  const events = [
    chunk("reply_start", null, { event: "start", status: "running" }),
    chunk("data", null, {
      metadata: { visibility: "diagnostic", native: { future: [false, null] } },
    }),
    chunk("content", "checking", { block_id: "commentary" }),
    chunk("reply_end", outcome, { event: "end", status: "waiting" }),
    chunk("done", "", { done: true }),
  ];
  const message = events.reduce(applyStreamChunk, initial());
  assert.deepEqual(message.events, events);
  assert.deepEqual(message.reply, outcome);
  assert.equal(message.status, "waiting");
  assert.deepEqual(
    message.blocks.map((block) => block.type),
    ["content", "content", "data"],
  );
  assert.equal(message.blocks[1].text, "final answer");
  assert.deepEqual(message.blocks[2].payloads, [false]);
  assert.equal(formatStreamPayloads(message.blocks[2].payloads), "false");
  const replayed = JSON.parse(JSON.stringify(message.events)).reduce(
    applyStreamChunk,
    initial(),
  );
  assert.deepEqual(replayed, message);
});

test("null usage fields do not falsely settle an in-progress detail block", () => {
  const message = applyStreamChunk(
    initial(),
    chunk("usage", null, { input_tokens: null, output_tokens: null }),
  );
  assert.equal(message.blocks[0].status, "streaming");
  assert.equal(message.blocks[0].inputTokens, undefined);
});

test("stopping a running invocation preserves interruption rather than implying success", () => {
  let message = applyStreamChunk(
    initial(),
    chunk("reply_start", null, { status: "running" }),
  );
  message = applyStreamChunk(
    message,
    chunk("tool_call", "{}", { tool_call_id: "pending" }),
  );
  const stopped = finishChatMessage(message);
  assert.equal(stopped.status, "interrupted");
  assert.equal(stopped.blocks[0].status, "interrupted");
  assert.deepEqual(stopped.events, message.events);
  assert.equal(
    stopped.blocks.some((block) => block.type === "error"),
    false,
  );
});
