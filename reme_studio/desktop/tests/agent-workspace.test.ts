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

test("main Chat and task workspaces expose their scoped pages and current conversation", async (t) => {
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
  for (const agentName of ["Remy", "finance"]) {
    const agent = makeAgent(agentName);
    const page = agentName === "Remy" ? "remy" : "tasks";
    const html = render(
      createElement(Workbench, {
        version: "test",
        state: {
          ...controller.state,
          activePage: page,
          connection: "connected",
          clientInfo,
          selectedAgentName: agentName,
          agents: [
            makeAgent("Remy"),
            { ...makeAgent("finance"), status: "error", error: "Agent error" },
            {
              ...makeAgent("tech_news"),
              agent_type: "time",
              schedule: "0 8 * * *",
            },
            makeAgent("work"),
          ],
          sessionId: "session",
        },
      }),
    );
    const titlebar = html.slice(
      html.indexOf("<header"),
      html.indexOf("</header>"),
    );
    assert.match(titlebar, new RegExp(`>${agent.agent_name}</button>`));
    assert.match(titlebar, /aria-current="page"[^>]*>(对话|进展)<\/span>/);
    assert.doesNotMatch(titlebar, /agent-status|清空上下文/);
    assert.match(html, /<textarea\b/);
    assert.doesNotMatch(html, /conversation-reset|aria-label="清空上下文"/);
    const sidebarStart = html.indexOf('<nav class="agent-sidebar"');
    const sidebar = html.slice(
      sidebarStart,
      html.indexOf("</nav>", sidebarStart),
    );
    if (agentName === "Remy") {
      assert.match(sidebar, /class="agent-sections"/);
      assert.match(sidebar, />收件箱<\/button>/);
      assert.match(sidebar, /aria-current="page"[^>]*>.*>对话<\/button>/);
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
      assert.doesNotMatch(sidebar, /finance|管理任务|TASK\.md|发件箱|搜索任务/);
      assert.match(html, /<dialog[^>]*>.*清空当前对话的上下文？.*<\/dialog>/);
      assert.match(
        sidebar,
        /class="agent-status"[^>]*data-badge="true"[^>]*>空闲<\/span>/,
      );
    } else {
      assert.match(sidebar, /role="searchbox" aria-label="搜索任务"/);
      assert.deepEqual(
        [
          ...sidebar.matchAll(
            /aria-pressed="(true|false)"[^>]*>(全部|事件触发|时间触发)</g,
          ),
        ].map(([, pressed, label]) => [label, pressed]),
        [
          ["全部", "true"],
          ["事件触发", "false"],
          ["时间触发", "false"],
        ],
      );
      assert.match(sidebar, /<ul class="agent-list">/);
      assert.deepEqual(
        [
          ...sidebar.matchAll(
            /class="agent-trigger"[^>]*>([^<]+)<\/span><span class="agent-name">([^<]+)</g,
          ),
        ].map(([, label, name]) => [label, name]),
        [
          ["Event", "finance"],
          ["Time", "tech_news"],
          ["Event", "work"],
        ],
      );
      assert.equal([...sidebar.matchAll(/aria-label="新建任务"/g)].length, 1);
      assert.match(sidebar, /aria-current="true" title="finance"/);
      assert.match(sidebar, /aria-label="管理 tech_news" aria-haspopup="menu"/);
      assert.doesNotMatch(
        sidebar,
        /agent-sections|>管理任务<\/button>|>TASK\.md<\/button>|>收件箱<\/button>|>发件箱<\/button>/,
      );
      assert.doesNotMatch(html, /清空当前对话的上下文/);
      const descriptionId = sidebar.match(
        /data-tone="error"[^>]*aria-label="异常"[^>]*aria-describedby="([^"]+)"/,
      )?.[1];
      assert.ok(descriptionId);
      assert.ok(
        sidebar.includes(
          `id="${descriptionId}" class="agent-status-detail" popover="hint" role="tooltip">Agent error</span>`,
        ),
      );
    }
  }
});
