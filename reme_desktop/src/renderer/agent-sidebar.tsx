import { useState } from "react";
import {
  type LucideIcon,
  Bell,
  Bot,
  Clock3,
  FileText,
  Inbox,
  Radio,
  Send,
  Sparkles,
} from "lucide-react";
import type { AgentView, DesktopState } from "../shared/protocol.ts";
import { translator } from "../shared/i18n.ts";
import { Disclosure } from "./ui.tsx";

export type AgentSection =
  "chat" | "memory" | "reminders" | "mailbox" | "events";

export default function AgentSidebar({
  state,
  section,
  onSelect,
}: {
  state: DesktopState;
  section: AgentSection;
  onSelect: (agentId: string, section: AgentSection) => void;
}) {
  const t = translator(state.settings.language);
  const selected = state.selectedAgentId ?? "chat";
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const setOpen = (key: string, open: boolean) =>
    setExpanded((value) => ({ ...value, [key]: open }));
  const agents = state.agents ?? [];
  const chat = agents.find((agent) => agent.kind === "chat");
  const row = (
    agent: Pick<
      AgentView,
      "agent_id" | "name" | "kind" | "running" | "enabled" | "error"
    >,
  ) => {
    const isSelected = selected === agent.agent_id;
    const key = `agent:${agent.agent_id}`;
    const open = expanded[key] ?? false;
    const status = t(
      agent.error
        ? "agentError"
        : agent.running
          ? "agentRunning"
          : !agent.enabled
            ? "agentDisabled"
            : "agentIdle",
    );
    const Icon =
      agent.kind === "chat" ? Bot : agent.kind === "cron" ? Clock3 : Radio;
    const sections: Array<[AgentSection, LucideIcon, Parameters<typeof t>[0]]> =
      agent.kind === "chat"
        ? [
            ["memory", FileText, "memoryTab"],
            ["reminders", Bell, "reminders"],
            ["mailbox", Inbox, "inbox"],
          ]
        : [
            ["memory", FileText, "memoryTab"],
            ["events", Radio, "eventsTab"],
            ["mailbox", Send, "outbox"],
          ];
    return (
      <Disclosure
        key={agent.agent_id}
        label={agent.name}
        open={open}
        onOpenChange={(value) => setOpen(key, value)}
        className={`agent-node ${isSelected ? "active" : ""} ${agent.kind === "chat" ? "remy-node" : ""}`}
        triggerClassName="agent-toggle"
        action={
          <button
            className="agent-row"
            aria-pressed={isSelected}
            aria-current={isSelected && section === "chat" ? "page" : undefined}
            data-agent-id={agent.agent_id}
            onClick={() => onSelect(agent.agent_id, "chat")}
          >
            <span className="agent-avatar">
              <Icon size={17} />
            </span>
            <span className="agent-name">{agent.name}</span>
            <span
              className={`agent-status-dot ${agent.error ? "error" : agent.running ? "running" : !agent.enabled ? "disabled" : ""}`}
              title={status}
              aria-label={status}
            />
          </button>
        }
      >
        <nav className="agent-sections" aria-label={agent.name}>
          {sections.map(([id, SectionIcon, label]) => (
            <button
              key={id}
              aria-current={isSelected && section === id ? "page" : undefined}
              onClick={() => onSelect(agent.agent_id, id)}
            >
              <SectionIcon size={15} />
              <span>{t(label)}</span>
              {id === "reminders" && state.unread > 0 && (
                <span className="nav-count">{state.unread}</span>
              )}
            </button>
          ))}
        </nav>
      </Disclosure>
    );
  };
  return (
    <nav className="agent-sidebar" aria-label={t("agents")}>
      <div className="agent-sidebar-heading">
        <Sparkles size={14} />
        <span>{t("agents")}</span>
      </div>
      {row(
        chat ?? {
          agent_id: "chat",
          name: "Remy",
          kind: "chat",
          running: false,
          enabled: true,
          error: null,
        },
      )}
      {(["event", "cron"] as const).map((kind) => {
        const items = agents.filter((agent) => agent.kind === kind);
        const key = `group:${kind}`;
        return (
          <Disclosure
            key={kind}
            label={kind === "event" ? "Proactive" : "Cron"}
            open={expanded[key] ?? true}
            onOpenChange={(value) => setOpen(key, value)}
            className="agent-group"
            triggerClassName="agent-group-heading"
            summary={
              <>
                <span>{kind === "event" ? "Proactive" : "Cron"}</span>
                <small>{items.length}</small>
              </>
            }
          >
            <div className="agent-group-items">
              {items.length ? (
                items.map(row)
              ) : (
                <p className="agent-group-empty">{t("noAgents")}</p>
              )}
            </div>
          </Disclosure>
        );
      })}
      <div className="agent-sidebar-footer">
        <span
          className={`connection-dot ${state.connection === "connected" ? "online" : ""}`}
        />
        {t(state.connection === "connected" ? "connected" : "reconnecting")}
      </div>
    </nav>
  );
}
