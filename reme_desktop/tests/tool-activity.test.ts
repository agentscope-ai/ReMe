import assert from "node:assert/strict";
import test from "node:test";
import { toolActions } from "../src/shared/tool-activity.ts";
import { readReply, reduceReply } from "../src/shared/transcript.ts";

const command = {
  type: "commandExecution",
  command: "cat src/chat.ts; ls src; rg stream src; npm test",
  commandActions: [
    { type: "read", name: "chat.ts", path: "src/chat.ts" },
    { type: "listFiles", path: "src" },
    { type: "search", path: "src", query: "stream" },
    { type: "unknown", command: "npm test" },
  ],
};

test("native command actions describe files, directories and searches independently of the tool name", () => {
  assert.deepEqual(toolActions("commandExecution", command), [
    { kind: "read", target: "src/chat.ts", query: undefined },
    { kind: "list", target: "src", query: undefined },
    { kind: "search", target: "src", query: "stream" },
    { kind: "command", target: "npm test", query: undefined },
  ]);
  assert.deepEqual(
    toolActions("fileChange", {
      type: "fileChange",
      changes: [{ path: "src/chat.ts" }],
    }),
    [{ kind: "edit", target: "src/chat.ts" }],
  );
});

test("output deltas retain action targets and completed snapshots replace their own details", () => {
  const start = readReply([
    {
      chunk_type: "tool_call",
      tool_call_id: "call",
      tool_call_name: "commandExecution",
      event: "start",
      chunk: command,
    },
  ]);
  const live = reduceReply(start, {
    chunk_type: "tool_result",
    tool_call_id: "call",
    event: "delta",
    chunk: "Output",
  });
  assert.equal(live.items[0].type, "tool");
  if (live.items[0].type !== "tool") return;
  assert.deepEqual(
    live.items[0].actions,
    toolActions("commandExecution", command),
  );
  const final = {
    chunk_type: "tool_result",
    tool_call_id: "call",
    tool_call_name: "commandExecution",
    event: "end",
    status: "completed",
    chunk: { ...command, aggregatedOutput: "Complete output" },
  } as const;
  const completed = reduceReply(live, final);
  assert.deepEqual(reduceReply(completed, final), completed);
  assert.equal(
    completed.items[0].type === "tool" && completed.items[0].output,
    "Complete output",
  );
});

test("named tools normalize image views and structured arguments without guessing shell semantics", () => {
  assert.deepEqual(
    toolActions("functions.view_image", { path: "resource/chart.png" }),
    [{ kind: "image", target: "resource/chart.png", query: undefined }],
  );
  assert.deepEqual(
    toolActions("read_file", { arguments: '{"path":"daily/report.md"}' }),
    [{ kind: "read", target: "daily/report.md", query: undefined }],
  );
  assert.deepEqual(toolActions("exec_command", { cmd: "cat README.md" }), [
    { kind: "command", target: "cat README.md", query: undefined },
  ]);
});

test("web actions retain search queries, opened pages and in-page searches", () => {
  assert.deepEqual(
    toolActions("webSearch", {
      type: "webSearch",
      action: { type: "search", query: "Android", queries: [] },
    }),
    [{ kind: "webSearch", query: "Android" }],
  );
  assert.deepEqual(
    toolActions("webSearch", {
      type: "webSearch",
      action: { type: "openPage", url: "https://example.com/source" },
    }),
    [{ kind: "browse", target: "https://example.com/source" }],
  );
  assert.deepEqual(
    toolActions("webSearch", {
      type: "webSearch",
      action: {
        type: "findInPage",
        url: "https://example.com/source",
        pattern: "ARM64",
      },
    }),
    [{ kind: "search", target: "https://example.com/source", query: "ARM64" }],
  );
});
