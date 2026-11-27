import type {
  ConnectorView,
  OverviewTransition,
  ProactiveOverview,
} from "../../../client/src/wire.ts";
import type { translator } from "../../../client/src/i18n.ts";

export function displayAgentName(
  name: string,
  t: ReturnType<typeof translator>,
): string {
  const names = {
    proactive_memory: "flowProject",
    proactive_note: "flowNotes",
    proactive_fin: "flowMarkets",
    proactive_news: "flowBriefing",
  } as const;
  return name in names ? t(names[name as keyof typeof names]) : name;
}

export function filterAgents<
  T extends { agent_name: string; agent_type: string },
>(
  agents: T[],
  query: string,
  type: string,
  label: (name: string) => string,
): T[] {
  const needle = query.trim().toLocaleLowerCase();
  return agents.filter(
    (agent) =>
      (type === "all" || agent.agent_type === type) &&
      `${agent.agent_name} ${label(agent.agent_name)}`
        .toLocaleLowerCase()
        .includes(needle),
  );
}

/** Rounded source branches join separate event and time rails. */
export function sourceRoute(
  from: FlowPoint,
  to: FlowPoint,
  railX: number,
): string {
  const direction = Math.sign(to.y - from.y);
  const radius = Math.min(
    9,
    Math.abs(to.y - from.y) / 2,
    (railX - from.x) / 2,
    (to.x - railX) / 2,
  );
  if (!direction || radius <= 0) return `M${from.x},${from.y} L${to.x},${to.y}`;
  return `M${from.x},${from.y} H${railX - radius} Q${railX},${from.y} ${railX},${from.y + direction * radius} V${to.y - direction * radius} Q${railX},${to.y} ${railX + radius},${to.y} H${to.x}`;
}

export function sourceIcon(
  source: Pick<ConnectorView, "source_id" | "trigger_type">,
): string {
  if (source.trigger_type === "time") return "⏰";
  const id = source.source_id;
  if (id.includes("screen")) return "🖥️";
  if (id.includes("codex")) return "💬";
  if (id.includes("claude")) return "🧩";
  if (id.includes("news")) return "📰";
  if (id.includes("gold_quotes")) return "🥇";
  if (id.includes("oil_quotes")) return "🛢️";
  if (id.includes("btc_quotes")) return "🪙";
  if (id.includes("quotes")) return "📈";
  return "🔌";
}

/** Only confirmed backend actions produce motion; activity is never inferred as a file read. */
export function flowEdge(action: OverviewTransition): string | null {
  switch (action.type) {
    case "event_admitted":
      return action.source_id ? `source:${action.source_id}` : null;
    case "input_claimed":
      return !action.agent_name || action.agent_name === "Remy"
        ? null
        : "claim";
    case "notification_delivered":
      return "deliver";
    case "reminder_sent":
      return "remind";
    case "record_committed":
      return "write";
    case "workspace_read":
      return "read";
    default:
      return null;
  }
}

export function isTimeAction(
  action: OverviewTransition,
  overview: ProactiveOverview,
): boolean {
  return (
    overview.sources.some(
      (source) =>
        source.source_id === action.source_id && source.trigger_type === "time",
    ) ||
    overview.agents.some(
      (agent) =>
        agent.agent_name === action.agent_name && agent.agent_type === "time",
    )
  );
}

export type FlowPoint = { x: number; y: number };
export function curve(
  from: FlowPoint,
  to: FlowPoint,
  vertical = false,
): string {
  const bend = Math.max(
    24,
    Math.abs(vertical ? to.y - from.y : to.x - from.x) * 0.5,
  );
  return vertical
    ? `M${from.x},${from.y} C${from.x},${from.y + bend} ${to.x},${to.y - bend} ${to.x},${to.y}`
    : `M${from.x},${from.y} C${from.x + bend},${from.y} ${to.x - bend},${to.y} ${to.x},${to.y}`;
}
