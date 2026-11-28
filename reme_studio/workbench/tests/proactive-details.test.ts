import assert from "node:assert/strict";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { agent, clientInfo } from "../../client/tests/fixtures.ts";
import { defaultSettings } from "../../client/src/settings.ts";
import type { StudioState } from "../../client/src/protocol.ts";

test("graph details keep prose primary, provenance collapsed and mailbox headings singular", async (t) => {
  const server = await createServer({
    configFile: false,
    root: fileURLToPath(new URL("../../", import.meta.url)),
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  t.after(() => server.close());
  const { HostContext } = await server.ssrLoadModule("/workbench/src/host.ts");
  const { Facts, AgentDetail, EventDetail, ReminderDetail } =
    await server.ssrLoadModule("/workbench/src/proactive/details.tsx");
  const { default: ResourcePane } = await server.ssrLoadModule(
    "/workbench/src/resource-pane.tsx",
  );
  const state = {
    settings: { ...defaultSettings, language: "zh" },
    activePage: "tasks",
    connection: "connected",
    clientInfo,
  } as StudioState;
  const render = (component: unknown, props: Record<string, unknown>) =>
    renderToStaticMarkup(
      createElement(
        HostContext.Provider,
        { value: {} },
        createElement(component as never, props),
      ),
    );
  const linked = (paths: string[]) =>
    createElement(
      "div",
      {},
      [...new Set(paths)].map((path) =>
        createElement("a", { key: path }, path),
      ),
    );
  const source = {
    source_id: "screen",
    name: "屏幕截图",
    trigger_type: "event",
  };
  const sourceLabel = (value: typeof source) => value.name;
  const facts = render(Facts, {
    values: [
      ["empty", null],
      ["unset", ""],
      ["count", 0],
    ],
  });
  assert.doesNotMatch(facts, /empty|unset|—/);
  assert.match(facts, />0<\/dd>/);

  const runtime = {
    ...agent("proactive_memory"),
    active_run: null,
    last_run: {
      run_id: "internal-run",
      event_ids: ["internal-event"],
      source_refs: [source],
      phase: "completed",
      tool: null,
      started_at: null,
      completed_at: "2026-11-28T08:00:00Z",
      record_paths: ["daily/result.md"],
      notification_requested: true,
      error: null,
    },
    deliveries: [
      { notification: { record_path: "daily/result.md" }, status: "delivered" },
    ],
    pending_delivery_count: 0,
    delivered_count: 100,
  };
  const overview = render(AgentDetail, {
    agent: runtime,
    state,
    status: "等待新事件",
    linked,
    sourceLabel,
    onOutbox() {},
  });
  const primary = overview.split('<details class="flow-technical">')[0];
  assert.doesNotMatch(
    primary,
    /internal-run|internal-event|proactive_memory|TASK\.md|投递详情|打开任务|执行重试时间|等待至|当前步骤/,
  );
  assert.equal(
    overview.match(/daily\/result\.md/g)?.length,
    1,
    "an output is not repeated as a delivery row",
  );
  assert.match(
    overview,
    /<details class="flow-technical"><summary>原始信息<\/summary>/,
  );
  assert.match(overview, /internal-run/);

  const timestamp = "2026-11-28T08:00:00Z";
  const event = render(EventDetail, {
    state,
    linked,
    sourceLabel,
    onInbox() {},
    item: {
      id: "event-row",
      agent_name: "proactive_memory",
      state: "processed",
      received_at: timestamp,
      consume_ts: null,
      event: {
        description: "检查当前任务进度",
        event_ts: timestamp,
        paths: ["resource/input.md"],
      },
      notification: null,
      source_refs: [source, source],
      event_ids: ["internal-event"],
      run_ids: ["internal-run"],
      record_paths: [],
    },
  });
  assert.equal(
    event.match(/屏幕截图/g)?.length,
    1,
    "duplicate provenance references share one source badge",
  );
  assert.doesNotMatch(
    event.split('<details class="flow-technical">')[0],
    /事件发生|internal-event|internal-run/,
  );

  const reminder = render(ReminderDetail, {
    state,
    linked,
    sourceLabel,
    item: {
      id: "message",
      summary: "检查已完成。\n\n来源：daily/result.md",
      delivered_at: timestamp,
      source_refs: [],
      record_paths: ["daily/result.md"],
      event_ids: [],
      run_ids: [],
    },
  });
  assert.equal(
    reminder.match(/daily\/result\.md/g)?.length,
    1,
    "a standalone source footer becomes one file reference",
  );

  const mailbox = {
    state,
    view: "inbox",
    rows: [],
    selected: "",
    onSelect() {},
    controls: createElement("button", {}, "筛选"),
  };
  assert.match(render(ResourcePane, mailbox), /<h1>收件箱<\/h1>/);
  for (const view of ["inbox", "outbox", "reminders"])
    assert.ok(
      render(ResourcePane, { ...mailbox, view }).includes(
        `data-variant="${view}"`,
      ),
      `${view} uses its own palette in the shared timeline`,
    );
  const embedded = render(ResourcePane, { ...mailbox, embedded: true });
  assert.doesNotMatch(embedded, /<h1>/);
  assert.match(embedded, /筛选/);
});
