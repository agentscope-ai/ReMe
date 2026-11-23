import { Fragment, useState, type ReactNode } from "react";
import { Clock3, Radio, Plus, Ellipsis } from "lucide-react";
import type { AgentView, StudioState } from "../../client/src/protocol.ts";
import { translator } from "../../client/src/i18n.ts";
import { Disclosure } from "./ui.tsx";
import ContextMenu from "./context-menu.tsx";
import type { TaskTarget } from "./tasks/dialog.tsx";

function AgentStatus({
  agent,
  t,
  compact = false,
}: {
  agent: AgentView;
  t: ReturnType<typeof translator>;
  compact?: boolean;
}) {
  const [tone, key] = !agent.enabled
    ? (["disabled", "agentDisabled"] as const)
    : agent.error || agent.status === "error"
      ? (["error", "agentError"] as const)
      : agent.status === "active"
        ? (["running", "agentRunning"] as const)
        : (["idle", "agentIdle"] as const);
  if (compact && tone === "idle") return null;
  const label = t(key);
  return (
    <span
      className="agent-status"
      data-tone={tone}
      role={compact ? "img" : undefined}
      aria-label={compact ? label : undefined}
      title={
        !agent.enabled && agent.status === "active"
          ? t("taskPauseHint")
          : agent.error || label
      }
    >
      <i aria-hidden />
      {!compact && label}
    </span>
  );
}

export default function AgentSidebar({
  state,
  page,
  sections,
  onSelect,
  onManage,
  onToggle,
  onReveal,
  busy,
}: {
  state: StudioState;
  page: "remy" | "tasks";
  sections: ReactNode;
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
  const remy = agents.find((agent) => agent.agent_name === "Remy");
  const [menu, setMenu] = useState<{
    agentName: string;
    x: number;
    y: number;
  } | null>(null);
  const menuAgent = agents.find(
    (agent) => agent.agent_name === menu?.agentName,
  );
  const row = (agent: AgentView) => {
    const isSelected = selected === agent.agent_name;
    const Icon = agent.agent_type === "cron" ? Clock3 : Radio;
    return (
      <Fragment key={agent.agent_name}>
        <div
          className="agent-row-wrapper"
          data-disabled={!agent.enabled || undefined}
          onContextMenu={(event) => {
            event.preventDefault();
            event.currentTarget.querySelector("button")?.focus();
            const bounds = event.currentTarget.getBoundingClientRect();
            setMenu({
              agentName: agent.agent_name,
              x: event.clientX || bounds.left,
              y: event.clientY || bounds.bottom,
            });
          }}
        >
          <button
            className="agent-row"
            data-selected={isSelected || undefined}
            aria-expanded={isSelected}
            title={t("progressTab")}
            onClick={() => onSelect(agent.agent_name)}
          >
            <span className="agent-icon">
              <Icon size={18} aria-hidden />
            </span>
            <span className="agent-name">{agent.agent_name}</span>
            <AgentStatus agent={agent} t={t} compact />
          </button>
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
        </div>
        {isSelected && sections}
      </Fragment>
    );
  };
  return (
    <nav
      className="agent-sidebar"
      aria-label={page === "remy" ? "Remy" : t("proactiveAi")}
    >
      <div className="agent-navigation">
        <div className="agent-sidebar-heading">
          <div className="agent-identity">
            <h2>{page === "remy" ? "Remy" : t("proactiveAi")}</h2>
            {page === "remy" && remy && <AgentStatus agent={remy} t={t} />}
          </div>
          {page === "remy" && <p>{t("personalAssistant")}</p>}
        </div>
        {page === "remy"
          ? sections
          : (["proactive", "cron"] as const).map((agentType) => {
              const items = agents.filter(
                (agent) => agent.agent_type === agentType,
              );
              const key = `group:${agentType}`;
              const label = t(
                agentType === "proactive" ? "eventOnlyTasks" : "scheduledTasks",
              );
              return (
                <Disclosure
                  key={agentType}
                  label={label}
                  open={expanded[key] ?? true}
                  onOpenChange={(value) => setOpen(key, value)}
                  className="agent-group"
                  triggerClassName="agent-group-heading"
                  summary={
                    <>
                      <span>{label}</span>
                      <span className="count-badge">{items.length}</span>
                    </>
                  }
                  actions={
                    <button
                      className="agent-group-add"
                      aria-label={`${t("taskCreate")} · ${label}`}
                      title={t("taskCreate")}
                      disabled={busy || state.connection !== "connected"}
                      onClick={() =>
                        onManage({
                          mode: "create",
                          withSchedule: agentType === "cron",
                        })
                      }
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
      {page === "tasks" && menu && menuAgent && (
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
    </nav>
  );
}
