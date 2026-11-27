import { useEffect, useRef, useState, type ReactNode } from "react";
import { Search } from "lucide-react";
import type { AgentView, StudioState } from "../../../client/src/protocol.ts";
import { translator, errorText } from "../../../client/src/i18n.ts";
import { useHost } from "../host.ts";
import Chat from "../chat/index.tsx";
import AgentResources from "../agent-resources.tsx";
import TaskEditor from "../tasks/editor.tsx";
import { displayAgentName, filterAgents } from "./flow.ts";

export const taskViews = [
  ["overview", "flowTaskOverview"],
  ["chat", "progressTab"],
  ["inbox", "inbox"],
  ["outbox", "outbox"],
  ["manage", "settings"],
] as const;
export type TaskView = (typeof taskViews)[number][0];

export function TaskPanel({
  state,
  agent,
  view,
  onView,
  draft,
  onDraft,
  children,
}: {
  state: StudioState;
  agent: AgentView;
  view: TaskView;
  onView: (view: TaskView) => void;
  draft: string;
  onDraft: (draft: string) => void;
  children: ReactNode;
}) {
  const t = translator(state.settings.language);
  const { action } = useHost();
  const [error, setError] = useState<unknown>(null);
  const live = useRef(true);
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);
  useEffect(() => {
    if (view === "chat" && state.selectedAgentName !== agent.agent_name)
      void action({ type: "agent.select", agentName: agent.agent_name }).catch(
        (reason) => {
          if (live.current) setError(reason);
        },
      );
  }, [view, agent.agent_name, state.selectedAgentName]);
  return (
    <div className="flow-task-panel">
      <nav className="flow-task-tabs" aria-label={t("taskManage")}>
        {taskViews.map(([id, label]) => (
          <button
            key={id}
            type="button"
            aria-current={view === id ? "page" : undefined}
            onClick={() => onView(id)}
          >
            {t(label)}
          </button>
        ))}
      </nav>
      {error != null && (
        <p className="action-error" role="alert">
          {errorText(error, state.settings.language)}
        </p>
      )}
      <div className="flow-task-content" data-view={view}>
        {view === "overview" && children}
        {view === "manage" && (
          <TaskEditor
            key={agent.agent_name}
            state={state}
            agent={agent}
            onCreated={() => {}}
          />
        )}
        {(view === "inbox" || view === "outbox") && (
          <AgentResources
            key={`${agent.agent_name}:${view}`}
            state={state}
            agentName={agent.agent_name}
            agentLabel={(name) => displayAgentName(name, t)}
            active={state.activePage === "tasks"}
            embedded
            inbox={view === "inbox"}
          />
        )}
        {view === "chat" &&
          (state.selectedAgentName === agent.agent_name ? (
            <Chat
              state={state}
              active={state.activePage === "tasks"}
              draft={draft}
              onDraft={onDraft}
              canClearContext={false}
              clearDialogOpen={false}
              onClearDialogChange={() => {}}
            />
          ) : (
            <p className="document-loading">{t("flowConnecting")}</p>
          ))}
      </div>
    </div>
  );
}

export function TaskPicker({
  state,
  agents,
  onSelect,
}: {
  state: StudioState;
  agents: AgentView[];
  onSelect: (name: string) => void;
}) {
  const t = translator(state.settings.language);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const matches = filterAgents(agents, query, filter, (name) =>
    displayAgentName(name, t),
  );
  return (
    <div className="flow-task-picker">
      <label className="flow-picker-search">
        <Search size={16} />
        <input
          type="search"
          autoFocus
          aria-label={t("taskSearch")}
          placeholder={t("taskSearch")}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>
      <div className="source-tabs" role="group" aria-label={t("taskTriggers")}>
        {(
          [
            ["all", "resourceAll"],
            ["event", "taskEventTrigger"],
            ["time", "taskTimeTrigger"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            aria-pressed={filter === id}
            className={filter === id ? "active" : ""}
            onClick={() => setFilter(id)}
          >
            {t(label)}
          </button>
        ))}
      </div>
      {matches.map((agent) => (
        <button
          type="button"
          className="flow-picker-row"
          key={agent.agent_name}
          onClick={() => onSelect(agent.agent_name)}
        >
          <span>
            <strong>{displayAgentName(agent.agent_name, t)}</strong>
          </span>
          <span className="flow-type" data-time={agent.agent_type === "time"}>
            {t(
              agent.agent_type === "time"
                ? "taskTimeTrigger"
                : "taskEventTrigger",
            )}
          </span>
        </button>
      ))}
      {!matches.length && <p className="flow-hint">{t("taskNoMatches")}</p>}
    </div>
  );
}
