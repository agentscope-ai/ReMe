import { useEffect, useRef, useState } from "react";
import type { DesktopState } from "../shared/protocol.ts";
import { errorText, translator } from "../shared/i18n.ts";
import { action } from "./desktop.ts";
import Chat from "./chat/index.tsx";
import SplitPane from "./split-pane.tsx";
import AgentSidebar from "./agent-sidebar.tsx";
import AgentDocument from "./agent-document.tsx";
import AgentResources from "./agent-resources.tsx";
import Reminders from "./reminders.tsx";
import type { TranslationKey } from "../shared/i18n.ts";

type AgentSection = "chat" | "document" | "reminders" | "outbox" | "inbox";

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
  const selected = state.selectedAgentName ?? "Remy";
  const agent = state.agents?.find((item) => item.agent_name === selected);
  const [sections, setSections] = useState<
    Partial<Record<string, AgentSection>>
  >({});
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
          ? selected === "Remy"
            ? "inbox"
            : "outbox"
          : state.focusTarget?.kind === "message"
            ? "reminders"
            : "chat",
    }));
  }, [state.focusRequest, state.focusTarget]);
  const select = (agentName: string, value: AgentSection) => {
    setError(null);
    setSections((current) => ({ ...current, [agentName]: value }));
    if (agentName !== selected)
      void action({ type: "agent.select", agentName }).catch(setError);
  };
  const name = agent?.agent_name ?? selected;
  const documentPath =
    agent?.agent_type === "chat" ? agent.memory_path : agent?.task_path;
  const isRemy = selected === "Remy";
  const tabs: Array<[AgentSection, TranslationKey]> = isRemy
    ? [
        ["reminders", "reminders"],
        ["document", "memoryTab"],
        ["inbox", "inbox"],
      ]
    : [
        ["document", "taskTab"],
        ["inbox", "inbox"],
        ["outbox", "outbox"],
      ];
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
      defaultWidth={224}
      minWidth={208}
      maxWidth={320}
      sidebar={
        <AgentSidebar
          state={state}
          onSelect={(agentName) => select(agentName, "chat")}
        />
      }
    >
      <section className="agent-workspace">
        <header className="agent-toolbar">
          <h1 title={name}>
            <button
              className="agent-home"
              aria-label={`${name} · ${t(isRemy ? "conversationTab" : "progressTab")}`}
              aria-current={section === "chat" ? "page" : undefined}
              onClick={() => select(selected, "chat")}
            >
              {name}
            </button>
          </h1>
          <div className="agent-tools">
            <nav className="agent-tabs" aria-label={t("agentPages", { name })}>
              {tabs.map(([id, label]) => (
                <button
                  key={id}
                  aria-current={section === id ? "page" : undefined}
                  onClick={() => select(selected, id)}
                >
                  {t(label)}
                  {id === "reminders" && state.unread > 0 && (
                    <span className="nav-count">{state.unread}</span>
                  )}
                </button>
              ))}
            </nav>
            <span
              className={`agent-status ${agent?.error ? "error" : agent?.running ? "running" : ""}`}
              title={agent?.error ?? undefined}
            >
              <i />
              {status}
            </span>
          </div>
        </header>
        {error != null && (
          <p className="action-error" role="alert">
            {errorText(error, state.settings.language)}
          </p>
        )}
        <div className="agent-content" key={`${scope}:${selected}`}>
          <div
            className="agent-chat"
            hidden={section !== "chat"}
            inert={section !== "chat"}
          >
            <Chat
              state={state}
              embedded
              active={state.activePage === "remy" && section === "chat"}
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
          </div>
          {section === "document" && documentPath && (
            <AgentDocument state={state} path={documentPath} />
          )}
          {section === "reminders" && selected === "Remy" && (
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
