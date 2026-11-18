import { useState } from "react";
import { Sparkles, Clock3, Radio } from "lucide-react";
import type { AgentView, DesktopState } from "../shared/protocol.ts";
import { translator } from "../shared/i18n.ts";
import { Disclosure } from "./ui.tsx";

export default function AgentSidebar({
  state,
  onSelect,
}: {
  state: DesktopState;
  onSelect: (agentName: string) => void;
}) {
  const t = translator(state.settings.language);
  const selected = state.selectedAgentName ?? "Remy";
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const setOpen = (key: string, open: boolean) =>
    setExpanded((value) => ({ ...value, [key]: open }));
  const agents = state.agents ?? [];
  const chat = agents.find((agent) => agent.agent_type === "chat");
  const row = (
    agent: Pick<
      AgentView,
      "agent_name" | "agent_type" | "running" | "enabled" | "error"
    >,
  ) => {
    const isSelected = selected === agent.agent_name;
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
      agent.agent_type === "chat"
        ? Sparkles
        : agent.agent_type === "cron"
          ? Clock3
          : Radio;
    return (
      <button
        key={agent.agent_name}
        className="agent-row"
        aria-current={isSelected ? "page" : undefined}
        data-agent-name={agent.agent_name}
        onClick={() => onSelect(agent.agent_name)}
      >
        <span className="agent-icon">
          <Icon size={18} />
        </span>
        <span className="agent-name">{agent.agent_name}</span>
        {(agent.running || agent.error || !agent.enabled) && (
          <span
            className={`agent-status-dot ${agent.error ? "error" : agent.running ? "running" : !agent.enabled ? "disabled" : ""}`}
            title={status}
            aria-label={status}
          />
        )}
      </button>
    );
  };
  return (
    <nav className="agent-sidebar" aria-label={t("agents")}>
      <div className="agent-navigation">
        {row(
          chat ?? {
            agent_name: "Remy",
            agent_type: "chat",
            running: false,
            enabled: true,
            error: null,
          },
        )}
        {(["proactive", "cron"] as const).map((agentType) => {
          const items = agents.filter(
            (agent) => agent.agent_type === agentType,
          );
          const key = `group:${agentType}`;
          return (
            <Disclosure
              key={agentType}
              label={t(
                agentType === "proactive" ? "proactiveTasks" : "scheduledTasks",
              )}
              open={expanded[key] ?? true}
              onOpenChange={(value) => setOpen(key, value)}
              className="agent-group"
              triggerClassName="agent-group-heading"
              summary={
                <>
                  <span>
                    {t(
                      agentType === "proactive"
                        ? "proactiveTasks"
                        : "scheduledTasks",
                    )}
                  </span>
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
      </div>
      <div className="agent-sidebar-footer">
        <span
          className={`connection-dot ${state.connection === "connected" ? "online" : ""}`}
        />
        {t(state.connection === "connected" ? "connected" : "reconnecting")}
      </div>
    </nav>
  );
}
