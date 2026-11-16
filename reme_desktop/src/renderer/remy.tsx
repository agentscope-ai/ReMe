import { useEffect, useRef, useState } from "react";
import type { DesktopState } from "../shared/protocol.ts";
import { errorText, translator } from "../shared/i18n.ts";
import { action } from "./desktop.ts";
import Chat from "./chat/index.tsx";
import SplitPane from "./split-pane.tsx";
import AgentSidebar, { type AgentSection } from "./agent-sidebar.tsx";
import AgentMemory from "./agent-memory.tsx";
import AgentResources from "./agent-resources.tsx";
import Reminders from "./reminders.tsx";

export default function Remy({
  state,
  onSettings,
  collapsed,
}: {
  state: DesktopState;
  onSettings: () => void;
  collapsed?: boolean;
}) {
  const t = translator(state.settings.language);
  const selected = state.selectedAgentId ?? "chat";
  const agent = state.agents?.find((item) => item.agent_id === selected);
  const [sections, setSections] = useState<Record<string, AgentSection>>({});
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [error, setError] = useState<unknown>(null);
  const scope = `${state.settings.endpoint}:${state.workspace}`;
  const section = sections[selected] ?? "chat";
  useEffect(() => {
    setSections({});
    setDrafts({});
    setError(null);
  }, [scope]);
  const focus = useRef(0);
  useEffect(() => {
    if (!state.focusRequest || focus.current === state.focusRequest) return;
    focus.current = state.focusRequest;
    setSections((value) => ({
      ...value,
      [selected]:
        state.focusTarget?.kind === "mail"
          ? selected === "chat"
            ? "inbox"
            : "outbox"
          : state.focusTarget?.kind === "message"
            ? "reminders"
            : "chat",
    }));
  }, [state.focusRequest, state.focusTarget]);
  const select = (agentId: string, value: AgentSection) => {
    setError(null);
    setSections((current) => ({ ...current, [agentId]: value }));
    if (agentId !== selected)
      void action({ type: "agent.select", agentId }).catch(setError);
  };
  const name = agent?.name ?? (selected === "chat" ? "Remy" : selected);
  const title =
    section === "chat"
      ? name
      : t(
          section === "memory"
            ? "memoryTab"
            : section === "reminders"
              ? "reminders"
              : section === "inbox"
                ? "inbox"
                : "outbox",
        );
  const status = t(
    agent?.error
      ? "agentError"
      : agent?.running
        ? "agentRunning"
        : agent?.enabled === false
          ? "agentDisabled"
          : "agentIdle",
  );
  return (
    <SplitPane
      name="remy"
      collapsed={collapsed}
      language={state.settings.language}
      defaultWidth={264}
      sidebar={
        <AgentSidebar state={state} section={section} onSelect={select} />
      }
    >
      <section className="agent-workspace">
        <header className="agent-toolbar">
          <div>
            {section !== "chat" && (
              <div className="agent-breadcrumb">{name}</div>
            )}
            <h1>{title}</h1>
          </div>
          <span
            className={`agent-status-pill ${agent?.running ? "running" : ""}`}
          >
            <i />
            {status}
          </span>
        </header>
        {error != null && (
          <p className="action-error" role="alert">
            {errorText(error, state.settings.language)}
          </p>
        )}
        <div className="agent-content" key={`${scope}:${selected}:${section}`}>
          {section === "chat" && (
            <Chat
              state={state}
              embedded
              active={state.activePage === "remy"}
              onSettings={onSettings}
              onExplain={(message) =>
                void action({
                  type: "item.action",
                  action: "explain",
                  target: { kind: "message", id: message.id },
                }).catch(setError)
              }
              draft={drafts[selected] ?? ""}
              onDraft={(value) =>
                setDrafts((current) => ({ ...current, [selected]: value }))
              }
            />
          )}
          {section === "memory" && agent && (
            <AgentMemory state={state} path={agent.memory_path} />
          )}
          {section === "reminders" && selected === "chat" && (
            <Reminders state={state} />
          )}
          {(section === "outbox" || section === "inbox") && (
            <AgentResources state={state} inbox={section === "inbox"} />
          )}
        </div>
      </section>
    </SplitPane>
  );
}
