import assert from "node:assert/strict";
import test from "node:test";
import { readStream, reduceReply } from "../src/shared/stream.ts";
import type { StreamChunk } from "../src/shared/protocol.ts";

test("SSE survives split UTF-8 bytes, CRLF and transport DONE", async () => {
  const bytes = new TextEncoder().encode(
    ': heartbeat\r\n\r\ndata: {"chunk_type":"content","chunk":"你好"}\r\n\r\ndata: [DONE]\r\n\r\n',
  );
  const chunks: StreamChunk[] = [];
  await readStream(
    new ReadableStream({
      start(controller) {
        for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
        controller.close();
      },
    }),
    (chunk) => chunks.push(chunk),
  );
  assert.equal(chunks[0].chunk, "你好");
  assert.equal(chunks[1].chunk_type, "done");
});

test("an EOF without DONE is an interrupted transport", async () => {
  await assert.rejects(
    readStream(
      new ReadableStream({
        start(controller) {
          controller.close();
        },
      }),
      () => {},
    ),
    /结束标记/,
  );
});

test("oversized SSE frames are rejected before delivery, with or without a delimiter", async () => {
  const data =
    "data: " +
    JSON.stringify({
      chunk_type: "content",
      chunk: "x".repeat(8 * 1024 * 1024),
    });
  for (const ending of ["", "\n\n"]) {
    let cancelled = false;
    await assert.rejects(
      readStream(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode(data + ending));
          },
          cancel() {
            cancelled = true;
          },
        }),
        () => assert.fail("Oversized data must not reach the caller"),
      ),
      /流事件过大/,
    );
    assert.equal(cancelled, true);
  }
});

test("block end and reply_end snapshots override deltas; DONE alone is not success", () => {
  let progress = { text: "", blocks: new Map<string, string>() };
  progress = reduceReply(progress, {
    chunk_type: "content",
    block_id: "b",
    chunk: "旧增量",
  });
  progress = reduceReply(progress, {
    chunk_type: "content",
    block_id: "b",
    event: "end",
    chunk: "完整块",
  });
  assert.equal(progress.text, "完整块");
  progress = reduceReply(progress, {
    chunk_type: "done",
    chunk: "",
    done: true,
  });
  assert.equal("status" in progress, false);
  const final = reduceReply(progress, {
    chunk_type: "reply_end",
    status: "completed",
    chunk: { result: "最终答案", status: "completed" },
  });
  assert.equal(final.text, "最终答案");
  assert.equal(final.status, "completed");
});
