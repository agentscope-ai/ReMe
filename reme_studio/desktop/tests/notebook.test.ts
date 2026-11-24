import { fileURLToPath } from "node:url";
import { BehaviorController } from "../src/main/behavior.ts";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { ClientController } from "../../client/src/controller.ts";
import { Storage } from "../src/main/storage.ts";
import { requestWorkspace } from "../../client/src/workspace.ts";
import type { ReMeClient } from "../../client/src/reme-client.ts";
import type { WorkspaceRequest } from "../../client/src/protocol.ts";
import { clientInfo } from "../../client/tests/fixtures.ts";
import {
  DEFAULT_PROJECT,
  projectLabel,
  projectValue,
} from "../../client/src/notebook.ts";

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
  await requestWorkspace(
    client,
    { type: "todo", action: "list_projects", scope: "all" },
    signal,
  );
  assert.deepEqual(calls.at(-1), [
    "todo",
    { action: "list_projects", scope: "all" },
  ]);
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
      status: "in_progress",
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
    {
      type: "todo",
      action: "update",
      id: "t1",
      status: "closed",
      close_note: "需求撤销，取消执行",
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
        status: "in_progress",
        progress_text: "接口完成，测试待补",
      },
    ],
    ["todo", { action: "update", id: "t1", due_date: null, source_idea: null }],
    [
      "todo",
      {
        action: "update",
        id: "t1",
        status: "closed",
        close_note: "需求撤销，取消执行",
      },
    ],
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
    { type: "todo", action: "update", id: "t1", status: "done" },
    {
      type: "todo",
      action: "update",
      id: "t1",
      status: "pending",
      close_note: "已完成",
    },
    {
      type: "todo",
      action: "update",
      id: "t1",
      status: "closed",
      close_note: " ",
    },
    { type: "todo", action: "list", progress_text: "不能忽略" },
    { type: "todo", action: "list", limit: 101 },
    { type: "todo", action: "list_projects", limit: 20 },
    { type: "todo", action: "list_projects", scope: "invalid" },
    { type: "idea", action: "list_projects" },
    { type: "idea", action: "list", offset: -1 },
    { type: "idea", action: "list", status: "closed" },
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

test("Workspace modules share their default sidebar width and headings", async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-notebook-view-"));
  const manifest = JSON.parse(
    await readFile(
      new URL("../assets/remy/manifest.json", import.meta.url),
      "utf8",
    ),
  );
  const controller = new ClientController(
    new Storage(directory),
    () => {},
    new BehaviorController(manifest),
  );
  const server = await createServer({
    configFile: false,
    root: fileURLToPath(new URL("../../", import.meta.url)),
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  t.after(async () => {
    await server.close();
    await controller.close();
    await rm(directory, { recursive: true, force: true });
  });
  const { default: Notebook } = await server.ssrLoadModule(
    "/workbench/src/notebook/index.tsx",
  );
  const { default: Knowledge } = await server.ssrLoadModule(
    "/workbench/src/knowledge/index.tsx",
  );
  const { HostContext } = await server.ssrLoadModule("/workbench/src/host.ts");
  const render = (element: ReturnType<typeof createElement>) =>
    renderToStaticMarkup(
      createElement(HostContext.Provider, { value: {} }, element),
    );
  const { default: AgentWorkspace } = await server.ssrLoadModule(
    "/workbench/src/agent-workspace.tsx",
  );
  const { default: Settings } = await server.ssrLoadModule(
    "/workbench/src/settings.tsx",
  );
  const state = {
    ...controller.state,
    activePage: "notebook",
    connection: "connected",
    clientInfo: {
      ...clientInfo,
      jobs: [...clientInfo.jobs, "todo", "idea", "list_workspace", "load_file"],
    },
  };
  const notebook = render(createElement(Notebook, { state }));
  const knowledge = render(createElement(Knowledge, { state }));
  const remy = render(createElement(AgentWorkspace, { state, page: "remy" }));
  const tasks = render(createElement(AgentWorkspace, { state, page: "tasks" }));
  const settings = render(
    createElement(Settings, {
      state,
      version: "test",
      companionVisible: false,
      desktop: false,
      editableEndpoint: false,
    }),
  );
  for (const [html, eyebrow] of [
    [notebook, "NOTES &amp; TO-DOS"],
    [knowledge, "KNOWLEDGE &amp; HISTORY"],
    [remy, "AT YOUR SIDE"],
    [tasks, "IN THE BACKGROUND"],
    [settings, "YOUR SPACE"],
  ]) {
    assert.match(html, /--sidebar-width:256px/);
    assert.match(html, /class="module-sidebar-heading"/);
    assert.ok(
      html.includes(
        `<span class="eyebrow" aria-hidden="true">${eyebrow}</span>`,
      ),
    );
  }
  assert.doesNotMatch(knowledge, /YOUR SECOND MEMORY/);
  assert.match(notebook, /随记/);
  assert.match(notebook, /关闭备注（选填）/);
  assert.match(notebook, /aria-label="调整侧栏宽度"/);
  const collapsed = render(createElement(Notebook, { state, collapsed: true }));
  assert.match(collapsed, /id="sidebar-notebook"[^>]*hidden=""/);
});
