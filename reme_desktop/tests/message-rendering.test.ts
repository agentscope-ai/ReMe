import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { DesktopController } from "../src/main/controller.ts";
import { SessionModel } from "../src/main/session-model.ts";
import { displayMessages } from "../src/shared/messages.ts";
import {
  emptyReply,
  reduceReply,
  readReply,
} from "../src/shared/transcript.ts";
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
    {
      chunk_type: "tool_call",
      block_id: "tool",
      tool_call_name: "read_file",
      event: "start",
      chunk: { path: "note.md" },
      metadata: { native: { secret: "provider details" } },
    },
    {
      chunk_type: "tool_result",
      block_id: "tool",
      event: "end",
      chunk: "File content",
      status: "completed",
    },
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

test("history preserves typed timeline order and excludes native diagnostic metadata", () => {
  const model = modelFor(assistant);
  const [message] = displayMessages(
    { ...model, pending: null },
    undefined,
    "zh",
  );
  assert.deepEqual(
    message.items?.map((item) => item.type),
    ["text", "tool", "text"],
  );
  assert.equal(message.content, "正在检查。\n\n## 结果\n\n**已完成**。");
  assert.equal("chunks" in model.messages[0], false);
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
  assert.equal(updated.items?.at(-1)?.type, "text");
  assert.equal(updated.items?.length, 4);
});

test("streaming blocks retain progress and results across the authoritative snapshot", () => {
  let progress = emptyReply();
  const chunks: StreamChunk[] = [
    { ...content("progress", "正在"), event: "delta" as const },
    { ...content("progress", "检查。"), event: "delta" as const },
    content("progress", "正在检查。"),
    content("answer", "## 结果\n\n**已完成**。"),
  ];
  for (const chunk of chunks) progress = reduceReply(progress, chunk);
  assert.equal(progress.text, "正在检查。\n\n## 结果\n\n**已完成**。");
  const pending: PendingTurn = {
    id: "local",
    turnId: "turn",
    sessionId: "session",
    query: "检查任务",
    streaming: true,
  };
  const model = modelFor({ ...assistant, chunks, completed: false });
  const live = displayMessages({ ...model, pending }, undefined, "zh").find(
    (message) => message.role === "assistant",
  )!;
  assert.deepEqual(live.items, progress.items);
  assert.equal(live.streaming, true);
  const complete = displayMessages(
    { ...modelFor(assistant), pending },
    undefined,
    "zh",
  )[0];
  assert.deepEqual(
    complete.items?.filter((item) => item.type === "text"),
    live.items,
  );
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
        items: readReply([
          content("first", "```text\nunclosed code"),
          content("second", "## 第二条消息"),
        ]).items,
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

test("reasoning stays inside activity disclosure while prose, plans and completion remain visible in both languages", async () => {
  const server = await createServer({
    configFile: false,
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  try {
    const { default: MessageBody, timelineItems } = await server.ssrLoadModule(
      "/src/renderer/message-body.tsx",
    );
    const reply = readReply([
      content("intro", "我先核查来源。"),
      {
        chunk_type: "think",
        block_id: "r1",
        event: "end",
        chunk: "Compare the available sources",
        metadata: { kind: "reasoning_summary" },
      },
      {
        chunk_type: "tool_result",
        tool_call_id: "read",
        tool_call_name: "commandExecution",
        event: "end",
        status: "completed",
        chunk: {
          type: "commandExecution",
          commandActions: [
            { type: "read", path: "daily/report.md" },
            { type: "listFiles", path: "daily" },
            { type: "search", path: "daily", query: "小米" },
          ],
        },
      },
      {
        chunk_type: "think",
        block_id: "r2",
        event: "end",
        chunk: "Validate the result",
        metadata: { kind: "reasoning_summary" },
      },
      {
        chunk_type: "tool_result",
        tool_call_id: "cmd",
        tool_call_name: "commandExecution",
        event: "end",
        status: "completed",
        chunk: { type: "commandExecution", command: "date" },
      },
      content("progress", "来源已核实，正在整理结论。"),
      {
        chunk_type: "think",
        block_id: "plan",
        event: "end",
        chunk: "## 执行计划",
        metadata: { kind: "plan" },
      },
      content("final", "结果已经确认。"),
    ]);
    const entries = timelineItems(reply.items);
    assert.deepEqual(
      entries.map((entry: { type: string }) => entry.type),
      ["text", "activity", "text", "summary", "text"],
    );
    assert.equal(entries[1].items.length, 4);
    const message: DisplayMessage = {
      id: "reply",
      role: "assistant",
      source: "chat",
      content: reply.text,
      items: reply.items,
      completed: true,
      streaming: false,
      status: null,
    };
    const render = (
      value: DisplayMessage,
      language: "zh" | "en",
      compact = false,
    ) =>
      renderToStaticMarkup(
        createElement(MessageBody, {
          message: value,
          language,
          compact,
          onLink: () => {},
        }),
      );
    for (const language of ["zh", "en"] as const) {
      const html = render(message, language);
      assert.equal((html.match(/class="activity-group/g) ?? []).length, 1);
      assert.match(
        html,
        language === "zh"
          ? /已读取文件、运行了命令/
          : /Read files, Ran commands/,
      );
      assert.match(html, /<h2>执行计划<\/h2>/);
      assert.match(html, /来源已核实，正在整理结论/);
      assert.match(html, language === "zh" ? /已完成/ : /Completed/);
      assert.doesNotMatch(
        html,
        /Compare the available sources|Validate the result|reasoning-summary/,
      );
      assert.doesNotMatch(
        render(
          {
            ...message,
            status: {
              kind: "status",
              id: "wait",
              text: "Waiting",
              tone: "waiting",
            },
          },
          language,
        ),
        /turn-completion/,
      );
      assert.doesNotMatch(
        render({ ...message, error: "Connection lost" }, language),
        /turn-completion/,
      );
      assert.doesNotMatch(
        render(message, language, true),
        /turn-completion|activity-group/,
      );
    }
  } finally {
    await server.close();
  }
});

test("task fork replies display independently of automatic history", () => {
  const model = modelFor(assistant);
  const pending: PendingTurn = {
    id: "question",
    sessionId: "query-fork",
    turnId: "query-turn",
    query: "当前进展？",
    streaming: true,
  };
  let reply = reduceReply(emptyReply(), content("answer", "正在核实"));
  const task = () =>
    displayMessages(
      { sessionId: pending.sessionId, messages: [], pending, reply },
      undefined,
      "zh",
    );
  assert.deepEqual(
    task().map((message) => message.content),
    ["当前进展？", "正在核实"],
  );
  assert.equal(task()[1].streaming, true);
  assert.equal(
    displayMessages({ ...model, pending: null }, undefined, "zh")[0].id,
    assistant.id,
  );
  reply = reduceReply(reply, {
    chunk_type: "reply_end",
    status: "completed",
    chunk: { result: "已确认进展" },
  });
  pending.streaming = false;
  pending.outcome = "completed";
  assert.equal(task()[1].content, "已确认进展");
  assert.equal(task()[1].completed, true);
  assert.equal(task()[1].streaming, false);
  pending.outcome = "interrupted";
  pending.error = "查询已停止";
  assert.equal(task()[1].error, "查询已停止");
  assert.deepEqual(
    model.messages.map((message) => message.id),
    [assistant.id],
  );
});

test("separate provider messages in one turn form one ordered timeline", () => {
  const model = modelFor({
    ...assistant,
    id: "model-1",
    chunks: [content("intro", "检查中")],
    completed: true,
  });
  model.apply({
    type: "message_upsert",
    session_id: "session",
    activities: {},
    message: {
      ...assistant,
      id: "model-2",
      chunks: [content("answer", "完成")],
      completed: true,
    },
  });
  const messages = displayMessages(
    { ...model, pending: null },
    undefined,
    "zh",
  );
  assert.equal(messages.length, 1);
  assert.equal(messages[0].content, "检查中\n\n完成");
  assert.deepEqual(
    messages[0].items?.map((item) => item.id),
    ["intro", "answer"],
  );
});

test("Markdown stabilizes complete blocks and invalidates references when their definition arrives", async () => {
  const server = await createServer({
    configFile: false,
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  try {
    const { MarkdownBuffer } = await server.ssrLoadModule(
      "/src/renderer/markdown.tsx",
    );
    const buffer = new MarkdownBuffer();
    const first = buffer.update("# Title\n\nFirst paragraph\n\nLast", false);
    const next = buffer.update(
      "# Title\n\nFirst paragraph\n\nLast changed",
      false,
    );
    assert.equal(first[0], next[0]);
    assert.equal(first[1], next[1]);
    assert.match(next.at(-1).text, /Last changed/);
    const source =
      "# Title\n\nFirst paragraph\n\n[source][ref]\n\n[ref]: https://example.com";
    assert.deepEqual(buffer.update(source, true), [
      { offset: 0, text: source },
    ]);
    assert.deepEqual(buffer.update("Replaced", true), [
      { offset: 0, text: "Replaced" },
    ]);
    assert.deepEqual(buffer.update("Replaced and corrected", true), [
      { offset: 0, text: "Replaced and corrected" },
    ]);
  } finally {
    await server.close();
  }
});

test("a lost command connection does not settle a still-running authoritative turn", () => {
  const model = modelFor({
    ...assistant,
    chunks: [{ ...content("active", "Partial"), event: "delta" }],
    completed: false,
  });
  const pending: PendingTurn = {
    id: "request",
    turnId: "turn",
    sessionId: "session",
    query: "Check",
    streaming: false,
    error: "Connection lost; waiting for synchronization",
  };
  const reply = displayMessages({ ...model, pending }, undefined, "en").find(
    (message) => message.role === "assistant",
  )!;
  assert.equal(reply.streaming, true);
  assert.equal(reply.items?.[0].status, "running");
  assert.equal(reply.status?.tone, "waiting");
});

test("request bookkeeping never overwrites later model messages in an active turn", () => {
  const model = modelFor({
    ...assistant,
    id: "first",
    chunks: [content("intro", "Progress")],
    completed: true,
  });
  model.status = "active";
  model.apply({
    type: "message_upsert",
    session_id: "session",
    activities: {},
    message: {
      ...assistant,
      id: "second",
      chunks: [{ ...content("tail", "Live answer"), event: "delta" }],
      completed: false,
    },
  });
  const pending: PendingTurn = {
    id: "request",
    turnId: "turn",
    sessionId: "session",
    query: "Check",
    streaming: true,
  };
  const controller = Object.create(DesktopController.prototype) as {
    syncPending: (model: SessionModel, pending: PendingTurn) => void;
  };
  controller.syncPending(model, pending);
  const [message] = displayMessages({ ...model, pending }, undefined, "en");
  assert.equal(message.content, "Progress\n\nLive answer");
  assert.equal(message.streaming, true);
  assert.equal("items" in pending, false);
});
