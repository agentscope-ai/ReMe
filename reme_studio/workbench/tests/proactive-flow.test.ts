import assert from "node:assert/strict";
import test from "node:test";
import {
  displayAgentName,
  filterAgents,
  flowEdge,
  isTimeAction,
  sourceIcon,
  sourceRoute,
} from "../src/proactive/flow.ts";
import { translator } from "../../client/src/i18n.ts";
import type {
  OverviewTransition,
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
  assert.equal(flowEdge(action("workspace_read")), "read");
  assert.equal(flowEdge(action("record_committed")), "write");
  assert.equal(flowEdge(action("notification_delivered")), "deliver");
  assert.equal(flowEdge(action("reminder_sent")), "remind");
  for (const type of [
    "input_released",
    "input_processed",
    "notification_prepared",
    "run_completed",
    "run_failed",
  ] as const)
    assert.equal(
      flowEdge(action(type)),
      null,
      `${type} is not a delivery or a file read`,
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
    "🔌",
  );
  assert.equal(
    sourceIcon({ source_id: "time:briefing", trigger_type: "time" }),
    "⏰",
  );
});

test("task search matches localized labels and raw names within the selected trigger type", () => {
  const agents = [
    { agent_name: "proactive_memory", agent_type: "event" },
    { agent_name: "proactive_news", agent_type: "time" },
    { agent_name: "Custom", agent_type: "event" },
  ];
  const label = (name: string) => displayAgentName(name, translator("zh"));
  assert.deepEqual(filterAgents(agents, " 项目 ", "all", label), [agents[0]]);
  assert.deepEqual(filterAgents(agents, "PROACTIVE", "time", label), [
    agents[1],
  ]);
  assert.deepEqual(filterAgents(agents, "科技", "event", label), []);
  assert.deepEqual(filterAgents(agents, "custom", "all", label), [agents[2]]);
  assert.deepEqual(filterAgents(agents, "", "all", label), agents);
});

test("source rails turn in either direction and remain valid for tight or aligned nodes", () => {
  for (const targetY of [20, 80]) {
    const route = sourceRoute({ x: 10, y: 50 }, { x: 90, y: targetY }, 60);
    assert.match(route, /^M10,50 H51 Q60,50 /);
    assert.ok(route.endsWith(`Q60,${targetY} 69,${targetY} H90`));
    assert.doesNotMatch(route, /NaN|Infinity/);
  }
  assert.equal(
    sourceRoute({ x: 10, y: 50 }, { x: 90, y: 50 }, 60),
    "M10,50 L90,50",
  );
  assert.equal(
    sourceRoute({ x: 90, y: 50 }, { x: 10, y: 20 }, 60),
    "M90,50 L10,20",
  );
  const tight = sourceRoute({ x: 10, y: 50 }, { x: 14, y: 51 }, 12);
  assert.match(tight, /H11\.5 Q12,50 12,50\.5 V50\.5/);
});
