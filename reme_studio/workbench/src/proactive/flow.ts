import {
  Bitcoin,
  Clock,
  Code,
  Droplet,
  Gem,
  MessageSquare,
  Monitor,
  Newspaper,
  Plug,
  TrendingUp,
  type LucideIcon,
} from "lucide-react";
import type {
  ConnectorView,
  OverviewInboxItem,
  OverviewReminder,
  OverviewTransition,
  ProactiveOverview,
} from "../../../client/src/wire.ts";

export type Flight = {
  action: OverviewTransition;
  edge: FlowEdge;
  expires: number;
  delay: number;
};

type FlowEdge = `source:${string}` | "claim" | "deliver" | "remind";

export const relayTiming = {
  depart: 320,
  arrive: 1280,
  receive: 600,
  duration: 2200,
} as const;

export function transferOrigin(edge: FlowEdge): 0 | 1 | 2 | 3 {
  if (edge.startsWith("source:")) return 0;
  return edge === "claim" ? 1 : edge === "deliver" ? 2 : 3;
}

/** Match receipts by their durable IDs, rather than highlighting unrelated rows. */
export function matchesTransition(
  action: OverviewTransition,
  item: OverviewInboxItem | OverviewReminder,
): boolean {
  if ("session_id" in item) return action.message_id === item.id;
  if (
    (action.delivery_id && action.delivery_id === item.delivery?.id) ||
    (action.message_id && item.chat_message_ids.includes(action.message_id))
  )
    return true;
  if (
    action.agent_name &&
    item.agent_name !== "Remy" &&
    action.agent_name !== item.agent_name
  )
    return false;
  if (action.type === "event_admitted" || action.type === "input_claimed")
    return action.event_ids.some((id) => item.event_ids.includes(id));
  return [action.run_id, ...action.run_ids].some(
    (id) => id != null && item.run_ids.includes(id),
  );
}

export function filterAgents<
  T extends { agent_name: string; agent_type: string },
>(agents: T[], query: string, type: string): T[] {
  const needle = query.trim().toLocaleLowerCase();
  return agents.filter(
    (agent) =>
      (type === "all" || agent.agent_type === type) &&
      agent.agent_name.toLocaleLowerCase().includes(needle),
  );
}

export function sourceIcon(
  source: Pick<ConnectorView, "source_id" | "trigger_type">,
): LucideIcon {
  if (source.trigger_type === "time") return Clock;
  const id = source.source_id;
  if (id.includes("screen")) return Monitor;
  if (id.includes("codex")) return MessageSquare;
  if (id.includes("claude")) return Code;
  if (id.includes("news")) return Newspaper;
  if (id.includes("gold_quotes")) return Gem;
  if (id.includes("oil_quotes")) return Droplet;
  if (id.includes("btc_quotes")) return Bitcoin;
  if (id.includes("quotes")) return TrendingUp;
  return Plug;
}

/** Only confirmed transfers animate the five-stage pipeline. */
export function flowEdge(action: OverviewTransition): FlowEdge | null {
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
