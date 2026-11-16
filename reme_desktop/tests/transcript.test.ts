import assert from "node:assert/strict";
import test from "node:test";
import {
  emptyReply,
  readReply,
  reduceReply,
  finishItems,
} from "../src/shared/transcript.ts";
import { UiError } from "../src/shared/i18n.ts";
import { readStream } from "../src/shared/stream.ts";
import type { StreamChunk } from "../src/shared/protocol.ts";

const text = (
  id: string,
  chunk: unknown,
  event: StreamChunk["event"] = "delta",
): StreamChunk => ({ chunk_type: "content", block_id: id, chunk, event });

test("parallel tool results update their original position without crossing text", () => {
  const reply = readReply([
    text("intro", "Checking", "start"),
    {
      chunk_type: "tool_call",
      tool_call_id: "a",
      tool_call_name: "read_file",
      chunk: { path: "a.md" },
      event: "start",
    },
    {
      chunk_type: "tool_call",
      tool_call_id: "b",
      tool_call_name: "exec_command",
      chunk: "pwd",
      event: "start",
    },
    text("progress", "Continuing", "start"),
    {
      chunk_type: "tool_result",
      tool_call_id: "b",
      chunk: "failed",
      status: "failed",
      event: "end",
    },
    {
      chunk_type: "tool_result",
      tool_call_id: "a",
      chunk: "found",
      status: "completed",
      event: "end",
    },
  ]);
  assert.deepEqual(
    reply.items.map((item) => [item.id, item.status]),
    [
      ["intro", "running"],
      ["a", "completed"],
      ["b", "failed"],
      ["progress", "running"],
    ],
  );
  const next = reduceReply(reply, {
    chunk_type: "tool_result",
    tool_call_id: "b",
    chunk: "late output",
    event: "delta",
    status: "running",
  });
  assert.equal(next.items[2].status, "failed");
  assert.equal(next.items[0], reply.items[0]);
});

test("null ends preserve text, empty snapshots clear it, and final corrections preserve commentary", () => {
  let reply = readReply([
    text("progress", "Checking"),
    text("progress", null, "end"),
    text("answer", "Draft"),
  ]);
  assert.equal(reply.text, "Checking\n\nDraft");
  reply = reduceReply(reply, {
    chunk_type: "reply_end",
    status: "completed",
    chunk: { result: "Final" },
  });
  assert.equal(reply.text, "Checking\n\nFinal");
  assert.equal(reply.items.at(-1)?.type, "text");
  assert.equal(reply.items.at(-1)?.status, "completed");
  reply = reduceReply(reply, text("answer", "", "end"));
  assert.equal(reply.text, "Checking\n\n");
});

test("snapshots are idempotent and preserve phases and only public summaries", () => {
  const chunks: StreamChunk[] = [
    {
      ...text("progress", "Checking", "end"),
      metadata: { phase: "commentary", native: { secret: "hidden" } },
    },
    {
      chunk_type: "think",
      block_id: "raw",
      chunk: "private chain",
      metadata: { kind: "reasoning_content" },
    },
    {
      chunk_type: "think",
      block_id: "summary",
      chunk: "Checking dependencies",
      metadata: { kind: "reasoning_summary" },
      event: "end",
    },
    { ...text("answer", "Done", "end"), metadata: { phase: "final_answer" } },
  ];
  const reply = readReply(chunks);
  const replay = chunks.reduce(reduceReply, reply);
  assert.deepEqual(replay, reply);
  assert.deepEqual(
    reply.items.map((item) => item.type),
    ["text", "summary", "text"],
  );
  assert.equal(JSON.stringify(reply).includes("hidden"), false);
  assert.equal(JSON.stringify(reply).includes("private chain"), false);
});

test("tool errors are distinct from the turn outcome and unresolved tools never become successful", () => {
  const reply = readReply([
    {
      chunk_type: "tool_call",
      tool_call_id: "a",
      tool_call_name: "bash",
      chunk: "exit 1",
      event: "start",
    },
    {
      chunk_type: "tool_result",
      tool_call_id: "a",
      chunk: { content: "error", is_error: true },
      event: "end",
    },
    {
      chunk_type: "tool_call",
      tool_call_id: "b",
      tool_call_name: "bash",
      chunk: "sleep 10",
      event: "start",
    },
    {
      chunk_type: "reply_end",
      status: "completed",
      chunk: { result: "Recovered" },
    },
  ]);
  assert.equal(reply.status, "completed");
  assert.equal(reply.error, undefined);
  assert.deepEqual(
    finishItems(reply.items, "completed").map((item) => item.status),
    ["failed", "unknown", "completed"],
  );
  assert.deepEqual(emptyReply().items, []);
});

test("SSE decodes fragmented UTF-8 and CRLF; EOF never implies successful completion", async () => {
  const bytes = new TextEncoder().encode(
    'data: {"chunk_type":"content","chunk":"你好"}\r\n\r\ndata: [DONE]\r\n\r\n',
  );
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
      controller.close();
    },
  });
  const chunks: StreamChunk[] = [];
  await readStream(stream, (chunk) => chunks.push(chunk));
  assert.deepEqual(
    chunks.map((chunk) => chunk.chunk_type),
    ["content", "done"],
  );
  assert.equal(chunks[0].chunk, "你好");
  const partial = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode(
          'data: {"chunk_type":"content","chunk":"partial"}\n\n',
        ),
      );
      controller.close();
    },
  });
  await assert.rejects(
    readStream(partial, () => {}),
    (error: unknown) =>
      error instanceof UiError && error.key === "streamInterrupted",
  );
});

test("image data becomes a bounded-label attachment and diagnostics stay hidden", () => {
  const reply = readReply([
    {
      chunk_type: "data",
      block_id: "image",
      media_type: "image/png",
      chunk: "aGVsbG8=",
      event: "end",
    },
    {
      chunk_type: "data",
      block_id: "diagnostic",
      chunk: { url: "https://example.com" },
      metadata: { visibility: "diagnostic" },
    },
  ]);
  assert.deepEqual(reply.items, [
    {
      id: "image",
      type: "artifact",
      url: "data:image/png;base64,aGVsbG8=",
      label: "Image",
      mediaType: "image/png",
      status: "completed",
    },
  ]);
});
