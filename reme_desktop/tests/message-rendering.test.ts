import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { SessionModel } from "../src/main/session-model.ts";
import { displayMessages } from "../src/shared/messages.ts";
import { reduceReply, type ReplyProgress } from "../src/shared/stream.ts";
import type {
  DisplayMessage,
  PendingTurn,
  SessionMessage,
  StreamChunk,
} from "../src/shared/protocol.ts";

const content = (id: string, text: string): StreamChunk => ({
  chunk_type: "content",
  event: "end",
  block_id: id,
  chunk: text,
});
const assistant: SessionMessage = {
  id: "turn:assistant",
  turn_id: "turn",
  role: "assistant",
  source: "chat",
  content: "正在检查。## 结果\n\n**已完成**。",
  chunks: [
    content("progress", "正在检查。"),
    { chunk_type: "tool_result", chunk: { secret: "provider details" } },
    content("answer", "## 结果\n\n**已完成**。"),
  ],
  completed: true,
  delivery: "idle",
  notification_eligible: false,
};
function modelFor(message: SessionMessage) {
  const model = new SessionModel();
  model.snapshot({
    session_id: "session",
    state: { session_id: "session", status: "idle", active_flags: [] },
    messages: [message],
  });
  return model;
}

test("history preserves every text message boundary and excludes provider traces", () => {
  const model = modelFor(assistant);
  const [message] = displayMessages(
    { ...model, pending: null },
    undefined,
    "zh",
  );
  assert.deepEqual(message.blocks, ["正在检查。", "## 结果\n\n**已完成**。"]);
  assert.equal(message.content, "正在检查。\n\n## 结果\n\n**已完成**。");
  assert.equal(model.messages[0].chunks.length, 2);
  assert.equal(
    JSON.stringify(model.messages).includes("provider details"),
    false,
  );
  model.apply({
    type: "message_upsert",
    session_id: "session",
    activities: {},
    message: {
      ...assistant,
      chunks: [...assistant.chunks, content("next", "下一条消息")],
    },
  });
  const [updated] = displayMessages(
    { ...model, pending: null },
    undefined,
    "zh",
  );
  assert.deepEqual(updated.blocks, [...message.blocks!, "下一条消息"]);
});

test("streaming blocks retain progress and results across the authoritative snapshot", () => {
  let progress: ReplyProgress = { text: "", blocks: new Map() };
  for (const chunk of [
    { ...content("progress", "正在"), event: "delta" as const },
    { ...content("progress", "检查。"), event: "delta" as const },
    content("progress", "正在检查。"),
    content("answer", "## 结果\n\n**已完成**。"),
  ])
    progress = reduceReply(progress, chunk);
  assert.equal(progress.text, "正在检查。\n\n## 结果\n\n**已完成**。");
  const pending: PendingTurn = {
    id: "local",
    turnId: "turn",
    sessionId: "session",
    query: "检查任务",
    text: progress.text,
    blocks: [...progress.blocks.values()],
    streaming: true,
  };
  const model = modelFor({ ...assistant, chunks: [], completed: false });
  const live = displayMessages({ ...model, pending }, undefined, "zh").find(
    (message) => message.role === "assistant",
  )!;
  assert.deepEqual(live.blocks, pending.blocks);
  assert.equal(live.streaming, true);
  const complete = displayMessages(
    { ...modelFor(assistant), pending },
    undefined,
    "zh",
  )[0];
  assert.deepEqual(complete.blocks, live.blocks);
  assert.equal(complete.streaming, false);
  const textOnly = displayMessages(
    {
      ...modelFor({ ...assistant, content: "纯文本消息", chunks: [] }),
      pending: null,
    },
    undefined,
    "zh",
  )[0];
  assert.equal(textOnly.content, "纯文本消息");
});

test("both roles render Markdown and each message has an independent Markdown document", async () => {
  const server = await createServer({
    configFile: false,
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  try {
    const { default: MessageBody } = await server.ssrLoadModule(
      "/src/renderer/message-body.tsx",
    );
    const markdown =
      "## 标题\n\n**重点**\n\n- 项目\n\n`code`\n\n| A | B |\n| - | - |\n| 1 | 2 |\n\n[来源](daily/result.md)\n\n[[daily/note|记录]]";
    for (const role of ["user", "assistant"] as const) {
      const message: DisplayMessage = {
        id: role,
        role,
        source: "chat",
        content: markdown,
        completed: true,
        streaming: false,
        status: null,
      };
      const render = (value: DisplayMessage) =>
        renderToStaticMarkup(
          createElement(MessageBody, {
            message: value,
            language: "zh",
            onLink: () => {},
          }),
        );
      const html = render(message);
      for (const tag of ["h2", "strong", "ul", "code", "table"])
        assert.match(html, new RegExp(`<${tag}[ >]`));
      assert.match(html, /href="daily\/result.md"/);
      assert.match(html, /#reme-file=daily%2Fnote/);
      const separate = render({
        ...message,
        blocks: ["```text\nunclosed code", "## 第二条消息"],
      });
      assert.match(separate, /<h2>第二条消息<\/h2>/);
      assert.equal(
        (separate.match(/class="message-content"/g) ?? []).length,
        2,
      );
    }
  } finally {
    await server.close();
  }
});

test("a task send immediately replaces the previous round while Remy keeps its history", () => {
  const model = modelFor(assistant);
  const pending: PendingTurn = {
    id: "next",
    sessionId: "session",
    query: "新一轮输入",
    text: "新一轮进度",
    blocks: ["新一轮进度"],
    streaming: true,
  };
  const task = () =>
    displayMessages(
      { ...model, pending, replaceOnSend: true },
      undefined,
      "zh",
    );
  assert.deepEqual(
    task().map((message) => message.content),
    ["新一轮输入", "新一轮进度"],
  );
  assert.equal(
    displayMessages({ ...model, pending }, undefined, "zh")[0].id,
    assistant.id,
  );
  pending.turnId = "next-turn";
  assert.deepEqual(
    task().map((message) => message.content),
    ["新一轮输入", "新一轮进度"],
  );
  pending.streaming = false;
  pending.error = "执行失败";
  const failed = task();
  assert.equal(failed.length, 2);
  assert.equal(failed[1].error, "执行失败");
  model.snapshot({
    session_id: "session",
    state: { session_id: "session", status: "idle", active_flags: [] },
    messages: [
      {
        ...assistant,
        id: "new-user",
        role: "user",
        turn_id: "next-turn",
        content: pending.query,
        chunks: [],
      },
      {
        ...assistant,
        id: "new-answer",
        turn_id: "next-turn",
        content: "新一轮结果",
        chunks: [content("answer", "新一轮结果")],
      },
    ],
  });
  assert.deepEqual(
    task().map((message) => message.id),
    ["new-user", "new-answer"],
  );
  assert.deepEqual(
    task().map((message) => message.content),
    ["新一轮输入", "新一轮结果"],
  );
});
