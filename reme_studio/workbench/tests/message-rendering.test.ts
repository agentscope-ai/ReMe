import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { ClientController } from "../../client/src/controller.ts";
import { SessionModel } from "../../client/src/session-model.ts";
import { displayMessages } from "../../client/src/messages.ts";
import { emptyReply, reduceReply } from "../../client/src/transcript.ts";
import type {
  DisplayMessage,
  PendingTurn,
  AgentMessage,
  TextItem,
} from "../../client/src/protocol.ts";

import {
  message as makeMessage,
  snapshot,
} from "../../client/tests/fixtures.ts";
const content = (id: string, text: string): TextItem => ({
  id,
  type: "text",
  status: "completed",
  text,
  phase: null,
});
const assistant = makeMessage({
  id: "turn:assistant",
  content: "正在检查。\n\n## 结果\n\n**已完成**。",
  items: [
    content("progress", "正在检查。"),
    {
      id: "tool",
      type: "tool",
      name: "read_file",
      input: "note.md",
      output: "File content",
      status: "completed",
      actions: [{ kind: "read", target: "note.md", query: null }],
    },
    content("answer", "## 结果\n\n**已完成**。"),
  ],
});
function modelFor(message: AgentMessage) {
  const model = new SessionModel();
  model.apply(snapshot([message]));
  return model;
}

test("history preserves the backend's typed timeline order", () => {
  const model = modelFor(assistant);
  const [message] = displayMessages(
    { sessionId: model.sessionId, messages: model.messages, pending: null },
    undefined,
    "zh",
  );
  assert.deepEqual(
    message.items?.map((item) => item.type),
    ["text", "tool", "text"],
  );
  assert.equal(message.content, assistant.content);
  model.apply({
    ...snapshot([
      {
        ...assistant,
        items: [...assistant.items, content("next", "下一条消息")],
      },
    ]),
    type: "update",
  });
  assert.equal(model.messages[0].items.length, 4);
});

test("both roles render Markdown and each message has an independent Markdown document", async () => {
  const server = await createServer({
    configFile: false,
    root: fileURLToPath(new URL("../../", import.meta.url)),
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  try {
    const { default: MessageBody } = await server.ssrLoadModule(
      "/workbench/src/message-body.tsx",
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
        items: [
          content("first", "```text\nunclosed code"),
          content("second", "## 第二条消息"),
        ],
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

test("standalone reminder sources can move to the footer without rewriting the message", async () => {
  const server = await createServer({
    configFile: false,
    root: fileURLToPath(new URL("../../", import.meta.url)),
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  try {
    const { splitReminderSource } = await server.ssrLoadModule(
      "/workbench/src/message-body.tsx",
    );
    const path = "daily/2026-11-24/work_012500.md";
    const body = "This is the reminder.";
    const message: DisplayMessage = {
      id: "reminder",
      role: "assistant",
      source: "proactive",
      content: `${body}\n\n来源：work · ${path}`,
      completed: true,
      streaming: false,
      status: null,
    };
    const source = splitReminderSource(message);
    assert.equal(source.path, path);
    assert.equal(source.display.content, body);
    assert.ok(message.content.includes(path));

    const item = content("source", `**Source:** [Run record](${path})`);
    const structured = splitReminderSource({
      ...message,
      items: [content("body", body), item],
    });
    assert.equal(structured.path, path);
    assert.equal(structured.display.items[0].text, body);
    assert.equal(structured.display.items[1].text, "");
    assert.ok(item.text.includes(path));

    for (const text of [
      `${body}\nSource: ${path}`,
      `\`\`\`text\nSource: ${path}`,
      `Source: ${path} and summary.md`,
      "Source: ../outside.md",
      "Source: [External](https://example.com/doc.md)",
    ])
      assert.equal(
        splitReminderSource({ ...message, content: text }),
        undefined,
      );
  } finally {
    await server.close();
  }
});

test("reasoning stays inside activity disclosure while prose, plans and completion remain visible in both languages", async () => {
  const server = await createServer({
    configFile: false,
    root: fileURLToPath(new URL("../../", import.meta.url)),
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  try {
    const { default: MessageBody, timelineItems } = await server.ssrLoadModule(
      "/workbench/src/message-body.tsx",
    );
    const reply = {
      content: "我先核查来源。\n\n来源已核实，正在整理结论。\n\n结果已经确认。",
      items: [
        content("intro", "我先核查来源。"),
        {
          id: "r1",
          type: "summary",
          kind: "reasoning",
          text: "Compare the available sources",
          status: "completed",
        },
        {
          id: "read",
          type: "tool",
          name: "command",
          input: "",
          output: "",
          status: "completed",
          actions: [
            { kind: "read", target: "daily/report.md", query: null },
            { kind: "list", target: "daily", query: null },
            { kind: "search", target: "daily", query: "小米" },
          ],
        },
        {
          id: "r2",
          type: "summary",
          kind: "reasoning",
          text: "Validate the result",
          status: "completed",
        },
        {
          id: "cmd",
          type: "tool",
          name: "command",
          input: "date",
          output: "",
          status: "completed",
          actions: [{ kind: "command", target: "date", query: null }],
        },
        content("progress", "来源已核实，正在整理结论。"),
        {
          id: "plan",
          type: "summary",
          kind: "plan",
          text: "## 执行计划",
          status: "completed",
        },
        content("final", "结果已经确认。"),
      ],
    };
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
      content: reply.content,
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
        render(
          {
            ...message,
            items: [],
            content: " ",
            status: {
              kind: "status",
              id: "done",
              text: "Completed",
              tone: "success",
            },
          },
          language,
        ),
        /turn-completion|message-status/,
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
  let reply = reduceReply(emptyReply(), {
    chunk_type: "data",
    chunk: makeMessage({
      content: "正在核实",
      completed: false,
      outcome: null,
    }),
  });
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
    displayMessages(
      { ...model, sessionId: model.sessionId, pending: null },
      undefined,
      "zh",
    )[0].id,
    assistant.id,
  );
  reply = reduceReply(reply, {
    chunk_type: "data",
    chunk: makeMessage({ content: "已确认进展" }),
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

test("streaming Markdown reparses ambiguous tails and resolves definitions and corrected snapshots", async () => {
  const server = await createServer({
    configFile: false,
    root: fileURLToPath(new URL("../../", import.meta.url)),
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  try {
    const { default: MarkdownText } = await server.ssrLoadModule(
      "/workbench/src/markdown.tsx",
    );
    const render = (text: string, completed: boolean) =>
      renderToStaticMarkup(
        createElement(MarkdownText, { text, completed, onLink: () => {} }),
      );
    for (const source of [
      "hello\n#x",
      "#e\n5\n-p2=kj:c",
      "- one\n\n  continued\n- two",
      "| A | B |\n| - | - |\n| 1 | 2 |",
      "# Title\n\n[source][ref]\n\n[ref]: https://example.com",
      "```text\n# stays code\n```\n\n正文 [[daily/note|记录]]",
    ]) {
      const expected = render(source, true);
      for (const size of [1, 7]) {
        for (let end = size; end < source.length; end += size)
          render(source.slice(0, end), false);
        assert.equal(render(source, true), expected);
      }
    }
    const html = render("hello\n#x", true);
    assert.equal((html.match(/<p[ >]/g) ?? []).length, 1);
    assert.match(
      render("[source][ref]\n\n[ref]: https://example.com", true),
      /href="https:\/\/example.com"/,
    );
    assert.match(
      render("Replaced and corrected", true),
      /Replaced and corrected/,
    );
    assert.match(render("**重点", false), /<strong[^>]*>重点<\/strong>/);
  } finally {
    await server.close();
  }
});

test("a lost command connection does not settle a still-running authoritative turn", () => {
  const model = modelFor({
    ...assistant,
    content: "Partial",
    items: [{ ...content("active", "Partial"), status: "running" }],
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
  const reply = displayMessages(
    { ...model, sessionId: model.sessionId, pending },
    undefined,
    "en",
  ).find((message) => message.role === "assistant")!;
  assert.equal(reply.streaming, true);
  assert.equal(reply.items?.[0].status, "running");
  assert.equal(reply.status?.tone, "waiting");
});

test("request bookkeeping never overwrites later model messages in an active turn", () => {
  const model = modelFor({
    ...assistant,
    id: "first",
    content: "Progress",
    items: [content("intro", "Progress")],
    completed: true,
  });
  model.agent!.status = "active";
  model.apply({
    ...snapshot([
      {
        ...assistant,
        id: "first",
        content: "Progress\n\nLive answer",
        items: [
          content("intro", "Progress"),
          { ...content("tail", "Live answer"), status: "running" },
        ],
        completed: false,
      },
    ]),
    type: "update",
  });
  const pending: PendingTurn = {
    id: "request",
    turnId: "turn",
    sessionId: "session",
    query: "Check",
    streaming: true,
  };
  const controller = Object.create(ClientController.prototype) as {
    syncPending: (model: SessionModel, pending: PendingTurn) => void;
  };
  controller.syncPending(model, pending);
  const [message] = displayMessages(
    { ...model, sessionId: model.sessionId, pending },
    undefined,
    "en",
  );
  assert.equal(message.content, "Progress\n\nLive answer");
  assert.equal(message.streaming, true);
  assert.equal("items" in pending, false);
});
