import assert from "node:assert/strict";
import { readFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { DesktopController } from "../src/main/controller.ts";
import { Storage } from "../src/main/storage.ts";
import type { AgentView } from "../src/shared/protocol.ts";

test("main Chat and task workspaces expose their scoped pages and current conversation", async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-agent-workspace-"));
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
  const { default: Remy } = await server.ssrLoadModule(
    "/src/renderer/remy.tsx",
  );
  for (const agentName of ["Remy", "finance"]) {
    const agent: AgentView = {
      agent_name: agentName,
      ...(agentName === "Remy"
        ? { agent_type: "chat", memory_path: "reme_agents/Remy/MEMORY.md" }
        : {
            agent_type: "proactive",
            task_path: `reme_agents/${agentName}/TASK.md`,
          }),
      enabled: true,
      running: false,
      status: "idle",
      session_id: "session",
      directory: `reme_agents/${agentName}`,
      pending_events: 0,
      next_trigger_at: null,
      last_completed_at: null,
      error: null,
    };
    const html = renderToStaticMarkup(
      createElement(Remy, {
        state: {
          ...controller.state,
          connection: "connected",
          workspace: "/workspace",
          selectedAgentName: agentName,
          agents: [agent],
          sessionId: "session",
        },
        onSettings: () => {},
      }),
    );
    assert.match(
      html,
      new RegExp(`<h1[^>]*><button[^>]*>${agentName}</button></h1>`),
    );
    assert.match(html, /<textarea\b/);
    assert.match(html, /class="agent-tabs"/);
    assert.match(
      html,
      new RegExp(
        `aria-label="${agentName} · ${agentName === "Remy" ? "对话" : "进展"}" aria-current="page"`,
      ),
    );
    assert.doesNotMatch(html, /class="agent-sections"/);
  }
});
