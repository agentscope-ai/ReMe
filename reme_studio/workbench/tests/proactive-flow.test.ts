import assert from "node:assert/strict";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Clock, Plug } from "lucide-react";
import { createServer } from "vite";
import { defaultSettings } from "../../client/src/settings.ts";
import type { StudioState } from "../../client/src/protocol.ts";
import {
  filterAgents,
  flowEdge,
  isTimeAction,
  matchesTransition,
  sourceIcon,
  transferOrigin,
} from "../src/proactive/flow.ts";
import type {
  OverviewTransition,
  OverviewInboxItem,
  OverviewReminder,
  ProactiveOverview,
} from "../../client/src/wire.ts";

const action = (
  type: OverviewTransition["type"],
  fields: Partial<OverviewTransition> = {},
): OverviewTransition => ({
  id: "action",
  type,
  occurred_at: "2026-11-28T08:00:00Z",
  source_id: null,
  agent_name: "notes",
  target_agent: null,
  event_ids: [],
  run_id: null,
  run_ids: [],
  delivery_id: null,
  message_id: null,
  paths: [],
  ...fields,
});

test("motion distinguishes receipt, confirmed delivery and final reminder creation", () => {
  assert.equal(
    flowEdge(
      action("event_admitted", { source_id: "preset:screen_capture_cron" }),
    ),
    "source:preset:screen_capture_cron",
  );
  assert.equal(
    flowEdge(action("event_admitted")),
    null,
    "unknown sources cannot produce fictional wires",
  );
  assert.equal(flowEdge(action("input_claimed")), "claim");
  assert.equal(flowEdge(action("input_claimed", { agent_name: "Remy" })), null);
  assert.equal(flowEdge(action("notification_delivered")), "deliver");
  assert.equal(flowEdge(action("reminder_sent")), "remind");
  for (const type of [
    "input_released",
    "input_processed",
    "notification_prepared",
    "run_completed",
    "run_failed",
    "workspace_read",
    "record_committed",
  ] as const)
    assert.equal(
      flowEdge(action(type)),
      null,
      `${type} does not move data between the five visible stages`,
    );
});

test("time remains identifiable on downstream actions without a source ID", () => {
  const overview = {
    sources: [{ source_id: "time:briefing", trigger_type: "time" }],
    agents: [
      { agent_name: "briefing", agent_type: "time" },
      { agent_name: "notes", agent_type: "event" },
    ],
  } as ProactiveOverview;
  assert.equal(
    isTimeAction(
      action("notification_delivered", { agent_name: "briefing" }),
      overview,
    ),
    true,
  );
  assert.equal(
    isTimeAction(
      action("event_admitted", { source_id: "time:briefing" }),
      overview,
    ),
    true,
  );
  assert.equal(isTimeAction(action("record_committed"), overview), false);
});

test("relay highlights follow event, delivery, run and message identities", () => {
  const item = {
    agent_name: "notes",
    event_ids: ["event"],
    run_ids: ["run"],
    chat_message_ids: ["message"],
    delivery: { id: "delivery" },
  } as OverviewInboxItem;
  assert.equal(
    matchesTransition(action("event_admitted", { event_ids: ["event"] }), item),
    true,
  );
  assert.equal(
    matchesTransition(
      action("input_claimed", { event_ids: ["event"], agent_name: "other" }),
      item,
    ),
    false,
  );
  assert.equal(
    matchesTransition(action("input_claimed", { event_ids: ["other"] }), item),
    false,
  );
  assert.equal(
    matchesTransition(
      action("notification_delivered", { delivery_id: "delivery" }),
      item,
    ),
    true,
  );
  assert.equal(
    matchesTransition(action("reminder_sent", { message_id: "message" }), item),
    true,
  );
  const remy = { ...item, agent_name: "Remy" };
  assert.equal(
    matchesTransition(
      action("notification_delivered", { run_id: "run" }),
      remy,
    ),
    true,
  );
  assert.equal(
    matchesTransition(
      action("reminder_sent", { run_ids: ["run"], agent_name: "Remy" }),
      remy,
    ),
    true,
  );
  assert.equal(
    matchesTransition(
      action("notification_delivered", {
        run_id: "other",
        event_ids: ["event"],
      }),
      remy,
    ),
    false,
  );
  const reminder = { id: "message", session_id: "session" } as OverviewReminder;
  assert.equal(
    matchesTransition(
      action("reminder_sent", { message_id: "message" }),
      reminder,
    ),
    true,
  );
  assert.equal(
    matchesTransition(
      action("reminder_sent", { message_id: "other" }),
      reminder,
    ),
    false,
  );
});

test("preset sources have distinct icons and custom sources have a local default", () => {
  const ids = [
    "screen_capture_cron",
    "codex_dialogue_cron",
    "claude_code_dialogue_cron",
    "finance_news_cron",
    "gold_quotes_cron",
    "oil_quotes_cron",
    "btc_quotes_cron",
    "nasdaq_quotes_cron",
  ];
  assert.equal(
    new Set(
      ids.map((id) =>
        sourceIcon({ source_id: `preset:${id}`, trigger_type: "event" }),
      ),
    ).size,
    ids.length,
  );
  assert.equal(
    sourceIcon({ source_id: "source_custom", trigger_type: "event" }),
    Plug,
  );
  assert.equal(
    sourceIcon({ source_id: "time:briefing", trigger_type: "time" }),
    Clock,
  );
});

test("task search matches original names within the selected trigger type", () => {
  const agents = [
    { agent_name: "proactive_memory", agent_type: "event" },
    { agent_name: "proactive_news", agent_type: "time" },
    { agent_name: "自定义任务_Custom", agent_type: "event" },
  ];
  assert.deepEqual(filterAgents(agents, " memory ", "all"), [agents[0]]);
  assert.deepEqual(filterAgents(agents, "PROACTIVE", "time"), [agents[1]]);
  assert.deepEqual(filterAgents(agents, "proactive_news", "event"), []);
  assert.deepEqual(filterAgents(agents, "custom", "all"), [agents[2]]);
  assert.deepEqual(filterAgents(agents, "自定义", "all"), [agents[2]]);
  assert.deepEqual(filterAgents(agents, "", "all"), agents);
});

test("stage headers reflect confirmed transitions across the five pipeline stages", async (t) => {
  const server = await createServer({
    configFile: false,
    root: fileURLToPath(new URL("../../", import.meta.url)),
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  t.after(() => server.close());
  const { default: FlowStageHeader } = await server.ssrLoadModule(
    "/workbench/src/proactive/stage-header.tsx",
  );
  const render = (transitions: OverviewTransition[], working = false) =>
    renderToStaticMarkup(
      createElement(
        "div",
        {},
        [0, 1, 2, 3, 4].map((step) =>
          createElement(FlowStageHeader, {
            key: step,
            step,
            count: 0,
            subtitle: "",
            language: "zh",
            working: step === 2 && working,
            flights: transitions.flatMap((action, index) => {
              const edge = flowEdge(action);
              return edge
                ? [{ action, edge, delay: index * 140, expires: 2000 }]
                : [];
            }),
          }),
        ),
      ),
    );
  const idle = render([]);
  assert.equal((idle.match(/class="flow-node-header"/g) ?? []).length, 5);
  assert.doesNotMatch(idle, /data-flowing|data-working/);
  assert.match(idle, />01<.*>02<.*>03<.*>04<.*>05</);
  const transitions = [
    action("event_admitted", { id: "admit", source_id: "time:briefing" }),
    action("input_claimed", { id: "claim" }),
    action("notification_delivered", { id: "deliver", agent_name: "briefing" }),
    action("reminder_sent", { id: "remind" }),
    action("record_committed", { id: "record" }),
  ];
  assert.deepEqual(
    transitions.flatMap((event) => {
      const edge = flowEdge(event);
      return edge ? [transferOrigin(edge)] : [];
    }),
    [0, 1, 2, 3],
  );
  const html = render(transitions, true);
  const stages = html.match(/<header .*?<\/header>/g)!;
  assert.equal(stages.length, 5);
  for (const stage of stages.slice(0, 4))
    assert.match(stage, /data-flowing="true"/);
  assert.doesNotMatch(stages[4], /data-flowing|data-working/);
  assert.match(stages[2], /data-working="true"/);
  for (const index of [0, 1, 3])
    assert.doesNotMatch(stages[index], /data-working/);

  const { default: ProactiveOverview } = await server.ssrLoadModule(
    "/workbench/src/proactive/index.tsx",
  );
  const { HostContext } = await server.ssrLoadModule("/workbench/src/host.ts");
  const inbox = {
    waiting_count: 0,
    processing_count: 0,
    processed_count: 0,
    queues: {},
    items: [],
  };
  const page = renderToStaticMarkup(
    createElement(
      HostContext.Provider,
      { value: {} },
      createElement(ProactiveOverview, {
        state: {
          settings: { ...defaultSettings, language: "zh" },
          activePage: "tasks",
          proactive: {
            live: true,
            actions: [],
            sequence: 1,
            asOf: null,
            error: null,
            overview: {
              sources: [],
              agents: [],
              event_inbox: inbox,
              remy_inbox: inbox,
              reminders: [],
              workspace: [],
              errors: { sources: "Collection unavailable" },
            },
          },
        } as unknown as StudioState,
        embedded: true,
      }),
    ),
  );
  assert.equal((page.match(/class="flow-node flow-/g) ?? []).length, 5);
  assert.equal(
    (page.match(/class="flow-panel-body" aria-hidden="false"/g) ?? []).length,
    5,
  );
  assert.match(page, /class="flow-fold-control" aria-expanded="true"/);
  assert.match(page, /收起内容/);
  assert.match(page, /data-collapsed="false" data-animate="true"/);
  assert.doesNotMatch(page, /flow-workspace|flow-links|视图设置|02 \/ INPUT/);
  assert.match(
    page,
    /role="alert"><strong>sources<\/strong> Collection unavailable/,
  );
});
