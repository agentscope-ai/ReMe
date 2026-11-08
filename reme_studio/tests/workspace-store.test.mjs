import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { after, before, beforeEach, test } from "node:test";
import { createServer } from "vite";

const values = new Map();
let server;
let store;

before(async () => {
  globalThis.localStorage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
  server = await createServer({
    root: fileURLToPath(new URL("..", import.meta.url)),
    configFile: false,
    logLevel: "silent",
    server: { middlewareMode: true },
  });
  ({ useWorkspaceStore: store } = await server.ssrLoadModule("/app/store.ts"));
});

after(async () => {
  await server?.close();
  delete globalThis.localStorage;
});

beforeEach(() => {
  store.setState(store.getInitialState(), true);
  values.clear();
});

test("fresh Studio completes hydration without a workspace cache", async () => {
  await store.persist.rehydrate();

  assert.equal(store.persist.hasHydrated(), true);
  assert.deepEqual(store.getState().tabs, []);
  const id = store.getState().openAgent();
  assert.equal(store.getState().activeTabId, id);

  // Missing storage must not discard a tab opened before hydration finishes.
  values.clear();
  await store.persist.rehydrate();
  assert.equal(store.getState().activeTabId, id);
});

test("hydration keeps ordinary sessions and unsaved drafts while removing PA views", async () => {
  const draft = {
    id: "file:draft.md",
    type: "markdown",
    path: "draft.md",
    content: "unsaved draft",
    savedContent: "saved",
  };
  const chat = {
    id: "agent:chat",
    type: "agent",
    mode: "chat",
    title: "Existing chat",
    sessionId: "chat-session",
    messages: [{ id: "question", role: "user", content: "Keep this question" }],
  };
  const pa = { ...chat, id: "agent:pa", mode: "pa", sessionId: "pa-session" };
  values.set(
    "reme-workspace",
    JSON.stringify({
      version: 1,
      state: { tabs: [draft, chat, pa], activeTabId: pa.id },
    }),
  );

  await store.persist.rehydrate();

  assert.equal(store.persist.hasHydrated(), true);
  assert.deepEqual(
    store.getState().tabs.map((tab) => tab.id),
    [draft.id, chat.id],
  );
  assert.equal(store.getState().tabs[0].content, "unsaved draft");
  assert.equal(store.getState().tabs[1].sessionId, "chat-session");
  assert.deepEqual(store.getState().tabs[1].messages, chat.messages);
  assert.equal(store.getState().activeTabId, chat.id);
  assert.equal(typeof store.getState().openAgent, "function");
});

test("ordinary tabs bind independent sessions and keep their own messages", () => {
  const first = store.getState().openAgent();
  const second = store.getState().openAgent();
  assert.notEqual(first, second);

  for (const [id, session] of [
    [first, "first-session"],
    [second, "second-session"],
  ]) {
    store
      .getState()
      .addChatTurn(
        id,
        { id: `user:${id}`, role: "user", content: session },
        { id: `assistant:${id}`, role: "assistant", content: "", blocks: [] },
      );
    store.getState().applyChatChunk(id, `assistant:${id}`, {
      chunk_type: "reply_start",
      chunk: "",
      session_id: session,
      turn_id: `turn:${id}`,
      done: false,
    });
  }

  const [firstTab, secondTab] = store.getState().tabs;
  assert.equal(firstTab.sessionId, "first-session");
  assert.equal(secondTab.sessionId, "second-session");
  assert.equal(firstTab.messages[0].content, "first-session");
  assert.equal(secondTab.messages[0].content, "second-session");
  assert.equal(firstTab.messages[0].turnId, `turn:${first}`);
  assert.equal(secondTab.messages[0].turnId, `turn:${second}`);
});
