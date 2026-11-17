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

test("main Chat and task workspaces open their conversation by default", async (t) => {
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
      agent_type: agentName === "Remy" ? "chat" : "proactive",
      enabled: true,
      running: false,
      status: "idle",
      session_id: "session",
      memory_path: `reme_agents/${agentName}/MEMORY.md`,
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
    assert.match(html, new RegExp(`<h1>${agentName}</h1>`));
    assert.match(html, /<textarea\b/);
  }
});
