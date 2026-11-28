import { useId, useState } from "react";
import {
  ChevronRight,
  FileText,
  FolderOpen,
  Inbox,
  MessageCircle,
  Pause,
  Play,
  Plus,
  Send,
  SlidersHorizontal,
  Workflow,
} from "lucide-react";
import type { AgentView, StudioState } from "../../../client/src/protocol.ts";
import { translator } from "../../../client/src/i18n.ts";
import { AgentStatus } from "../agent-sidebar.tsx";
import { SidebarHeading } from "../split-pane.tsx";
import { Switch } from "../ui.tsx";

export const taskSections = [
  ["chat", "progressTab", MessageCircle],
  ["document", "taskFileTab", FileText],
  ["inbox", "inbox", Inbox],
  ["outbox", "outbox", Send],
  ["manage", "settings", SlidersHorizontal],
] as const;
export type TaskSection = (typeof taskSections)[number][0];
export type TaskLocation =
  | { section: "overview" | "list" | "create" }
  | { section: TaskSection; agentName: string };

export default function TaskSidebar({
  state,
  location,
  onSelect,
  onToggle,
  onReveal,
  busy,
}: {
  state: StudioState;
  location: TaskLocation;
  onSelect: (location: TaskLocation) => void;
  onToggle: (agent: AgentView) => void;
  onReveal: (name: string) => void;
  busy: boolean;
}) {
  const t = translator(state.settings.language);
  const tasks =
    state.agents?.filter((agent) => agent.agent_type !== "chat") ?? [];
  const id = useId();
  const selectedName = "agentName" in location ? location.agentName : undefined;
  const [collapsedTask, setCollapsedTask] = useState<string | null>(null);
  const connected = state.connection === "connected";
  const toggle = (name: string) => {
    if (name === selectedName) {
      setCollapsedTask((current) => (current === name ? null : name));
    } else {
      setCollapsedTask(null);
      onSelect({ agentName: name, section: "chat" });
    }
  };
  return (
    <nav className="agent-sidebar task-sidebar" aria-label={t("proactiveAi")}>
      <SidebarHeading title={t("proactiveAi")}>
        <button
          type="button"
          className="icon-button"
          aria-label={t("taskCreate")}
          title={t("taskCreate")}
          disabled={busy || !connected}
          onClick={() => onSelect({ section: "create" })}
        >
          <Plus size={16} aria-hidden />
        </button>
      </SidebarHeading>
      <div className="task-overview-link">
        <button
          type="button"
          className="agent-row"
          aria-current={location.section === "overview" ? "page" : undefined}
          onClick={() => onSelect({ section: "overview" })}
        >
          <Workflow size={16} aria-hidden />
          {t("proactiveOverview")}
        </button>
      </div>
      <div className="agent-navigation">
        {tasks.length ? (
          <ul className="agent-list" aria-label={t("flowAllAgents")}>
            {tasks.map((agent) => {
              const name = agent.agent_name;
              const selected = selectedName === name;
              const expanded = selected && collapsedTask !== name;
              const pagesId = `${id}-${name}`;
              return (
                <li
                  key={name}
                  className="agent-row-wrapper"
                  data-disabled={!agent.enabled || undefined}
                  data-expanded={expanded}
                >
                  <button
                    type="button"
                    className="agent-row"
                    title={name}
                    aria-current={selected || undefined}
                    aria-expanded={expanded}
                    aria-controls={pagesId}
                    onClick={() => toggle(name)}
                    onContextMenu={(event) => {
                      event.preventDefault();
                      event.currentTarget.focus({ preventScroll: true });
                      toggle(name);
                    }}
                  >
                    <span
                      className="agent-trigger"
                      data-trigger={agent.agent_type}
                    >
                      {agent.agent_type === "time" ? "Time" : "Event"}
                    </span>
                    <span className="agent-name">{name}</span>
                    <AgentStatus agent={agent} t={t} compact />
                    <ChevronRight
                      className="task-expand-indicator"
                      size={14}
                      aria-hidden
                    />
                  </button>
                  <div
                    id={pagesId}
                    className="task-pages-panel"
                    data-expanded={expanded}
                    aria-hidden={!expanded}
                    inert={!expanded}
                  >
                    <div className="task-pages-content">
                      <ul
                        className="task-pages"
                        aria-label={t("agentPages", { name })}
                      >
                        {taskSections.map(([section, label, Icon]) => (
                          <li key={section}>
                            <button
                              type="button"
                              className="agent-row"
                              aria-current={
                                selected && location.section === section
                                  ? "page"
                                  : undefined
                              }
                              onClick={() =>
                                onSelect({ agentName: name, section })
                              }
                            >
                              <Icon size={14} aria-hidden />
                              {t(label)}
                            </button>
                          </li>
                        ))}
                        <li className="task-page-actions">
                          <label
                            className="task-enabled-row"
                            htmlFor={`${pagesId}-enabled`}
                          >
                            {agent.enabled ? (
                              <Play size={14} aria-hidden />
                            ) : (
                              <Pause size={14} aria-hidden />
                            )}
                            <span>
                              {t(
                                agent.enabled
                                  ? "taskEnabledStatus"
                                  : "taskPausedStatus",
                              )}
                            </span>
                            <Switch
                              id={`${pagesId}-enabled`}
                              checked={agent.enabled}
                              disabled={busy || !connected}
                              aria-label={t(
                                agent.enabled
                                  ? "taskEnabledStatus"
                                  : "taskPausedStatus",
                              )}
                              title={t(
                                agent.enabled ? "taskPause" : "taskEnable",
                              )}
                              onCheckedChange={() => onToggle(agent)}
                            />
                          </label>
                        </li>
                        <li>
                          <button
                            type="button"
                            className="agent-row"
                            onClick={() => onReveal(name)}
                          >
                            <FolderOpen size={14} aria-hidden />
                            {t("taskOpenDirectory")}
                          </button>
                        </li>
                      </ul>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="agent-list-empty" role="status">
            {t("noAgents")}
          </p>
        )}
      </div>
    </nav>
  );
}
