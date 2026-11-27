import { fileURLToPath } from "node:url";
import { BehaviorController } from "../src/main/behavior.ts";
import assert from "node:assert/strict";
import { readFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { ClientController } from "../../client/src/controller.ts";
import { Storage } from "../src/main/storage.ts";
import { agent as makeAgent, clientInfo } from "../../client/tests/fixtures.ts";

test("Remy keeps conversation navigation while proactive tasks use the full-width graph", async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-agent-workspace-"));
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
  const { HostContext } = await server.ssrLoadModule("/workbench/src/host.ts");
  const render = (element: ReturnType<typeof createElement>) =>
    renderToStaticMarkup(
      createElement(HostContext.Provider, { value: {} }, element),
    );
  const { default: Workbench } = await server.ssrLoadModule(
    "/workbench/src/workspace.tsx",
  );
  const { default: AgentWorkspace } = await server.ssrLoadModule(
    "/workbench/src/agent-workspace.tsx",
  );
  const agents = [
    makeAgent("Remy"),
    makeAgent("proactive_memory"),
    {
      ...makeAgent("proactive_news"),
      agent_type: "time",
      schedule: "0 8 * * *",
    },
  ];
  const state = {
    ...controller.state,
    activePage: "remy",
    connection: "connected",
    clientInfo,
    selectedAgentName: "Remy",
    agents,
    sessionId: "session",
  };
  const remy = render(
    createElement(AgentWorkspace, { state, onNavigate() {} }),
  );
  assert.match(remy, /<textarea\b/);
  assert.match(remy, /aria-current="page"[^>]*>.*对话<\/button>/);
  assert.doesNotMatch(remy, /conversation-reset|aria-label="清空上下文"/);
  assert.match(remy, /清空当前对话的上下文？/);
  const sidebarStart = remy.indexOf('<nav class="agent-sidebar"');
  const sidebar = remy.slice(
    sidebarStart,
    remy.indexOf("</nav>", sidebarStart),
  );
  assert.match(sidebar, /class="agent-sections"/);
  assert.match(sidebar, />收件箱<\/button>/);
  assert.match(sidebar, />提醒<\/button>/);
  assert.match(sidebar, />MEMORY\.md<\/button>/);
  assert.deepEqual(
    [
      ...sidebar.matchAll(
        /aria-label="([^"]+)操作" aria-haspopup="menu" aria-expanded="false"/g,
      ),
    ].map(([, name]) => name),
    ["对话", "提醒", "MEMORY.md", "收件箱"],
  );
  assert.doesNotMatch(sidebar, /管理任务|TASK\.md|发件箱|搜索任务/);

  const inbox = {
    waiting_count: 0,
    processing_count: 0,
    processed_count: 0,
    queues: {},
    items: [],
  };
  const taskState = {
    ...state,
    activePage: "tasks",
    proactive: {
      live: true,
      actions: [],
      sequence: 0,
      asOf: "2026-11-28T08:00:00Z",
      error: null,
      overview: {
        sources: [],
        agents: agents.slice(1).map((agent) => ({
          ...agent,
          active_run: null,
          last_run: null,
          deliveries: [],
          pending_delivery_count: 0,
          delivered_count: 0,
        })),
        event_inbox: inbox,
        remy_inbox: inbox,
        workspace: [],
        reminders: [],
        errors: {},
      },
    },
  };
  const html = render(
    createElement(Workbench, { version: "test", state: taskState }),
  );
  const start = html.indexOf('<div class="proactive-overview">');
  assert.ok(start > 0, "the task page mounts the workflow graph");
  const graph = html.slice(start, html.indexOf('data-state="inactive"', start));
  assert.match(graph, /data-has-sidebar="false"/);
  assert.match(graph, /aria-current="page"[^>]*>主动智能<\/span>/);
  for (const label of [
    "添加来源",
    "搜索任务",
    "新建任务",
    "管理 项目跟进",
    "管理 科技简报",
    "视图设置",
  ])
    assert.ok(graph.includes(`aria-label="${label}"`), label);
  for (const label of [
    "事件收件箱",
    "Proactive Runtime",
    "Remy 收件箱",
    "提醒",
    "工作区",
  ])
    assert.ok(graph.includes(label), label);
  assert.doesNotMatch(
    graph,
    /class="agent-(sidebar|list)"|flow-toolbar|flow-heading|高亮任务|aria-label="切换侧栏"/,
  );
  assert.doesNotMatch(graph, /<textarea\b|清空当前对话的上下文/);
});
