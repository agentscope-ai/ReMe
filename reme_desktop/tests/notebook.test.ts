import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { DesktopController } from "../src/main/controller.ts";
import { Storage } from "../src/main/storage.ts";
import { requestWorkspace } from "../src/main/workspace.ts";
import type { ReMeClient } from "../src/main/reme-client.ts";
import type { WorkspaceRequest } from "../src/shared/protocol.ts";
import {
  DEFAULT_PROJECT,
  projectLabel,
  projectValue,
} from "../src/shared/notebook.ts";

test("localized project names round trip through IPC with one stable default", async () => {
  const calls: [string, object][] = [];
  const client = {
    request: async (job: string, body: object) => {
      calls.push([job, body]);
      return { success: true, answer: {}, metadata: {} };
    },
  } as unknown as ReMeClient;
  const signal = new AbortController().signal;
  for (const language of ["zh", "en"] as const) {
    assert.equal(
      projectLabel(DEFAULT_PROJECT, language),
      language === "zh" ? "随笔" : "Notes",
    );
    for (const type of ["todo", "idea"] as const) {
      await requestWorkspace(
        client,
        {
          type,
          action: "create",
          text: "一句话",
          project: projectValue(
            projectLabel(DEFAULT_PROJECT, language),
            language,
          ),
        },
        signal,
      );
    }
    assert.equal(projectLabel("ReMe", language), "ReMe");
    assert.equal(projectValue("ReMe", language), "ReMe");
  }
  assert.equal(calls.length, 4);
  for (const [, body] of calls)
    assert.deepEqual(body, {
      action: "create",
      text: "一句话",
      project: "随笔",
    });
  await requestWorkspace(
    client,
    { type: "todo", action: "list", project: "  " },
    signal,
  );
  assert.deepEqual(calls.at(-1), ["todo", { action: "list", project: "随笔" }]);
});

test("notebook IPC preserves task patches, explicit clearing and source links", async () => {
  const calls: [string, object][] = [];
  const client = {
    request: async (job: string, body: object) => {
      calls.push([job, body]);
      return { success: true, answer: {}, metadata: {} };
    },
  } as unknown as ReMeClient;
  const signal = new AbortController().signal;
  await requestWorkspace(
    client,
    { type: "todo", action: "create", source_idea: "i1" },
    signal,
  );
  await requestWorkspace(
    client,
    {
      type: "todo",
      action: "update",
      id: "t1",
      status: "partial",
      progress_text: "接口完成，测试待补",
    },
    signal,
  );
  await requestWorkspace(
    client,
    {
      type: "todo",
      action: "update",
      id: "t1",
      due_date: null,
      source_idea: null,
    },
    signal,
  );
  await requestWorkspace(
    client,
    { type: "idea", action: "archive", id: "i1" },
    signal,
  );
  await requestWorkspace(
    client,
    {
      type: "todo",
      action: "list",
      source_idea: "i1",
      scope: "all",
      offset: 20,
      limit: 20,
    },
    signal,
  );
  assert.deepEqual(calls, [
    ["todo", { action: "create", source_idea: "i1" }],
    [
      "todo",
      {
        action: "update",
        id: "t1",
        status: "partial",
        progress_text: "接口完成，测试待补",
      },
    ],
    ["todo", { action: "update", id: "t1", due_date: null, source_idea: null }],
    ["idea", { action: "archive", id: "i1" }],
    [
      "todo",
      {
        action: "list",
        source_idea: "i1",
        scope: "all",
        offset: 20,
        limit: 20,
      },
    ],
  ]);
});

test("notebook IPC rejects untrusted fields and invalid actions before HTTP", async () => {
  let called = false;
  const client = {
    request: async () => {
      called = true;
    },
  } as unknown as ReMeClient;
  for (const request of [
    { type: "todo", action: "archive", id: "t1" },
    { type: "idea", action: "create", text: " " },
    { type: "todo", action: "create", text: "目标", id: "caller-id" },
    { type: "todo", action: "update", id: "t1" },
    { type: "todo", action: "update", id: "t1", status: "doing" },
    { type: "todo", action: "list", progress_text: "不能忽略" },
    { type: "todo", action: "list", limit: 101 },
    { type: "idea", action: "list", offset: -1 },
    { type: "idea", action: "list", status: "done" },
    { type: "todo", action: "create", text: "目标", _allowed_paths: ["/"] },
    { type: "idea", action: "get", id: "i1", path: "../outside.json" },
  ])
    await assert.rejects(
      requestWorkspace(
        client,
        request as WorkspaceRequest,
        new AbortController().signal,
      ),
    );
  assert.equal(called, false);
});

test("Tree Hollow shares Knowledge's default split and preserves Remy's narrower navigation", async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-notebook-view-"));
  const manifest = JSON.parse(
    await readFile(
      new URL("../assets/remy/manifest.json", import.meta.url),
      "utf8",
    ),
  );
  const controller = new DesktopController(
    new Storage(directory),
    () => {},
    manifest,
  );
  const server = await createServer({
    configFile: false,
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  t.after(async () => {
    await server.close();
    await controller.close();
    await rm(directory, { recursive: true, force: true });
  });
  const { default: Notebook } = await server.ssrLoadModule(
    "/src/renderer/notebook/index.tsx",
  );
  const { default: Knowledge } = await server.ssrLoadModule(
    "/src/renderer/knowledge/index.tsx",
  );
  const { default: Remy } = await server.ssrLoadModule(
    "/src/renderer/remy.tsx",
  );
  const state = {
    ...controller.state,
    activePage: "notebook",
    connection: "connected",
    workspace: "/workspace",
  };
  const notebook = renderToStaticMarkup(createElement(Notebook, { state }));
  const knowledge = renderToStaticMarkup(createElement(Knowledge, { state }));
  const remy = renderToStaticMarkup(
    createElement(Remy, { state, onSettings: () => {} }),
  );
  assert.match(notebook, /--sidebar-width:292px/);
  assert.match(knowledge, /--sidebar-width:292px/);
  assert.match(remy, /--sidebar-width:224px/);
  assert.match(notebook, /树洞/);
  assert.match(notebook, /aria-label="调整侧栏宽度"/);
  const collapsed = renderToStaticMarkup(
    createElement(Notebook, { state, collapsed: true }),
  );
  assert.match(collapsed, /id="sidebar-notebook"[^>]*hidden=""/);
});
