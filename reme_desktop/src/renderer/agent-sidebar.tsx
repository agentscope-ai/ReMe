import { useState } from "react";
import { Sparkles, Clock3, Radio, Plus, Ellipsis } from "lucide-react";
import type { AgentView, DesktopState } from "../shared/protocol.ts";
import { translator } from "../shared/i18n.ts";
import { Disclosure } from "./ui.tsx";
import ContextMenu from "./context-menu.tsx";
import type { TaskTarget } from "./tasks/dialog.tsx";

export default function AgentSidebar({
  state,
  onSelect,
  onManage,
  onToggle,
  onReveal,
  busy,
}: {
  state: DesktopState;
  onSelect: (agentName: string) => void;
  onManage: (target: TaskTarget) => void;
  onToggle: (agent: AgentView) => void;
  onReveal: (agentName: string) => void;
  busy: boolean;
}) {
  const t = translator(state.settings.language);
  const selected = state.selectedAgentName ?? "Remy";
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const setOpen = (key: string, open: boolean) =>
    setExpanded((value) => ({ ...value, [key]: open }));
  const agents = state.agents ?? [];
  const [menu, setMenu] = useState<{
    agentName: string;
    x: number;
    y: number;
  } | null>(null);
  const menuAgent = agents.find(
    (agent) => agent.agent_name === menu?.agentName,
  );
  const chat = agents.find((agent) => agent.agent_type === "chat");
  const row = (
    agent: Pick<
      AgentView,
      "agent_name" | "agent_type" | "status" | "enabled" | "error"
    >,
  ) => {
    const isChat = agent.agent_type === "chat";
    const isSelected = selected === agent.agent_name;
    const status = t(
      !agent.enabled
        ? "agentDisabled"
        : agent.error
          ? "agentError"
          : agent.status === "active"
            ? "agentRunning"
            : "agentIdle",
    );
    const Icon =
      agent.agent_type === "chat"
        ? Sparkles
        : agent.agent_type === "cron"
          ? Clock3
          : Radio;
    return (
      <div
        className="agent-row-wrapper"
        key={agent.agent_name}
        data-disabled={!agent.enabled || undefined}
        onContextMenu={
          agent.agent_type === "chat"
            ? undefined
            : (event) => {
                event.preventDefault();
                event.currentTarget.querySelector("button")?.focus();
                const bounds = event.currentTarget.getBoundingClientRect();
                setMenu({
                  agentName: agent.agent_name,
                  x: event.clientX || bounds.left,
                  y: event.clientY || bounds.bottom,
                });
              }
        }
      >
        <button
          key={agent.agent_name}
          className={`agent-row${isChat ? " agent-primary" : ""}`}
          aria-current={isSelected ? "page" : undefined}
          data-agent-name={agent.agent_name}
          onClick={() => onSelect(agent.agent_name)}
        >
          <span className="agent-icon">
            <Icon size={isChat ? 22 : 18} />
          </span>
          <span className="agent-label">
            <span className="agent-name">{agent.agent_name}</span>
            {isChat && (
              <span className="agent-caption">{t("personalAssistant")}</span>
            )}
          </span>
          {(agent.status === "active" || agent.error || !agent.enabled) && (
            <span
              className={`agent-status-dot ${!agent.enabled ? "disabled" : agent.error ? "error" : agent.status === "active" ? "running" : ""}`}
              title={
                !agent.enabled && agent.status === "active"
                  ? t("taskPauseHint")
                  : status
              }
              aria-label={status}
            />
          )}
        </button>
        {agent.agent_type !== "chat" && (
          <button
            className="agent-row-menu"
            aria-label={t("taskMenu", { name: agent.agent_name })}
            aria-haspopup="menu"
            aria-expanded={menu?.agentName === agent.agent_name}
            onClick={(event) => {
              const bounds = event.currentTarget.getBoundingClientRect();
              setMenu({
                agentName: agent.agent_name,
                x: bounds.left,
                y: bounds.bottom,
              });
            }}
          >
            <Ellipsis size={16} />
          </button>
        )}
      </div>
    );
  };
  return (
    <nav className="agent-sidebar" aria-label={t("agents")}>
      <div className="agent-navigation">
        {row(
          chat ?? {
            agent_name: "Remy",
            agent_type: "chat",
            status: "idle",
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
              actions={
                <button
                  className="agent-group-add"
                  aria-label={`${t("taskCreate")} · ${t(agentType === "proactive" ? "proactiveTasks" : "scheduledTasks")}`}
                  title={t("taskCreate")}
                  disabled={busy || state.connection !== "connected"}
                  onClick={() => onManage({ mode: "create", agentType })}
                >
                  <Plus size={15} />
                </button>
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
      {menu && menuAgent && menuAgent.agent_type !== "chat" && (
        <ContextMenu
          position={menu}
          onClose={() => setMenu(null)}
          items={[
            {
              id: "edit",
              label: t("taskManage"),
              disabled: busy,
              select: () =>
                onManage({ mode: "edit", agentName: menuAgent.agent_name }),
            },
            {
              id: "toggle",
              label: t(menuAgent.enabled ? "taskPause" : "taskEnable"),
              disabled: busy || state.connection !== "connected",
              select: () => onToggle(menuAgent),
            },
            {
              id: "directory",
              label: t("taskOpenDirectory"),
              select: () => onReveal(menuAgent.agent_name),
            },
            {
              id: "delete",
              label: t("taskDelete"),
              danger: true,
              disabled: busy,
              select: () =>
                onManage({ mode: "delete", agentName: menuAgent.agent_name }),
            },
          ]}
        />
      )}
      <div className="agent-sidebar-footer">
        <span
          className={`connection-dot ${state.connection === "connected" ? "online" : ""}`}
        />
        {t(state.connection === "connected" ? "connected" : "reconnecting")}
      </div>
    </nav>
  );
}
