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

test("Remy keeps its navigation and proactive tasks retain a workflow overview alongside task pages", async (t) => {
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
  const { WorkspaceLayout } = await server.ssrLoadModule(
    "/workbench/src/workspace-titlebar.tsx",
  );
  const agents = [
    makeAgent("Remy"),
    makeAgent("proactive_memory"),
    {
      ...makeAgent("proactive_news"),
      agent_type: "time",
      schedule: "0 8 * * *",
    },
    makeAgent("我的自定义任务"),
    {
      ...makeAgent("我的定时任务"),
      agent_type: "time",
      enabled: false,
      schedule: "0 9 * * *",
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
    createElement(
      WorkspaceLayout.Provider,
      { value: { settings: state.settings, setSidebarCollapsed() {} } },
      createElement(AgentWorkspace, { state, onNavigate() {} }),
    ),
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
  assert.doesNotMatch(
    sidebar,
    /aria-haspopup="menu"|agent-row-menu|AT YOUR SIDE/,
  );
  const titlebar = remy.slice(0, sidebarStart);
  assert.match(titlebar, /class="remy-page-actions"/);
  assert.match(titlebar, /aria-label="对话操作"[^>]*aria-haspopup="menu"/);
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
  const start = html.indexOf('<div class="proactive-overview"');
  assert.ok(start > 0, "the task page mounts the workflow graph");
  const graph = html.slice(start, html.indexOf('data-state="inactive"', start));
  const taskPage = html.slice(
    html.lastIndexOf('class="split-pane ', start),
    start,
  );
  assert.match(taskPage, /id="sidebar-tasks"/);
  assert.match(taskPage, /data-has-sidebar="true"/);
  assert.match(taskPage, /aria-current="page"[^>]*>总览<\/span>/);
  assert.match(taskPage, /class="agent-list" aria-label="全部任务"/);
  assert.match(taskPage, /class="agent-row"[^>]*>.*总览<\/button>/);
  assert.doesNotMatch(
    taskPage,
    /sidebar-search|task-filter-tabs|aria-label="搜索任务"|agent-row-menu|aria-haspopup="menu"/,
  );
  for (const name of [
    "proactive_memory",
    "proactive_news",
    "我的自定义任务",
    "我的定时任务",
  ])
    assert.ok(
      taskPage.includes(`<span class="agent-name">${name}</span>`),
      name,
    );
  assert.match(graph, /data-embedded="true"/);
  assert.match(graph, /<h1>总览<\/h1>/);
  assert.doesNotMatch(graph, /class="workspace-titlebar"/);
  for (const label of [
    "添加来源",
    "新建任务",
    "管理 proactive_memory",
    "管理 proactive_news",
  ])
    assert.ok(graph.includes(`aria-label="${label}"`), label);
  for (const label of ["事件收件箱", "任务", "Remy 收件箱", "提醒"])
    assert.ok(graph.includes(label), label);
  assert.match(graph, /aria-label="数据流转流程"/);
  assert.match(graph, />01<.*>02<.*>03<.*>04<.*>05</);
  assert.match(graph, />全部任务<\/span>/);
  assert.doesNotMatch(graph, /flow-workspace|aria-label="视图设置"/);
  assert.doesNotMatch(
    graph,
    /class="agent-(sidebar|list)"|flow-toolbar|flow-heading|高亮任务|aria-label="切换侧栏"/,
  );
  assert.doesNotMatch(graph, /<textarea\b|清空当前对话的上下文/);

  const { default: TaskSidebar } = await server.ssrLoadModule(
    "/workbench/src/tasks/sidebar.tsx",
  );
  const tasks = render(
    createElement(TaskSidebar, {
      state: taskState,
      location: { agentName: "proactive_memory", section: "inbox" },
      onSelect() {},
      onToggle() {},
      onReveal() {},
      busy: false,
    }),
  );
  for (const label of ["进展", "TASK.md", "收件箱", "发件箱", "设置"])
    assert.ok(tasks.includes(`${label}</button>`), label);
  assert.match(tasks, /aria-current="page"[^>]*>.*收件箱<\/button>/);
  assert.match(tasks, /title="proactive_memory"[^>]*aria-expanded="true"/);
  assert.match(
    tasks,
    /class="task-pages-panel" data-expanded="false" aria-hidden="true" inert=""/,
  );
  assert.match(
    tasks,
    /role="switch" aria-checked="true"[^>]*aria-label="已开启"/,
  );
  assert.match(
    tasks,
    /role="switch" aria-checked="false"[^>]*aria-label="已暂停"/,
  );
  assert.ok(tasks.includes("打开任务目录</button>"));
  assert.doesNotMatch(tasks, /管理任务|Remy 的页面|MEMORY\.md|>提醒<\/button>/);

  const { default: TaskEditor } = await server.ssrLoadModule(
    "/workbench/src/tasks/editor.tsx",
  );
  const settings = render(
    createElement(TaskEditor, {
      state: taskState,
      agent: agents[1],
      onCreated() {},
    }),
  );
  assert.doesNotMatch(settings, /<textarea|保存说明/);
  assert.match(settings, /保存设置/);
  const create = render(
    createElement(TaskEditor, { state: taskState, onCreated() {} }),
  );
  assert.match(create, /<textarea/);
});
