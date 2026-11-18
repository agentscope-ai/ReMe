import { useEffect, useRef, useState } from "react";
import { Info } from "lucide-react";
import type { AgentView, DesktopState } from "../shared/protocol.ts";
import { errorText, translator } from "../shared/i18n.ts";
import { action } from "./desktop.ts";
import Chat from "./chat/index.tsx";
import SplitPane from "./split-pane.tsx";
import AgentSidebar from "./agent-sidebar.tsx";
import AgentDocument from "./agent-document.tsx";
import AgentResources from "./agent-resources.tsx";
import Reminders from "./reminders.tsx";
import type { TranslationKey } from "../shared/i18n.ts";
import TaskDialog, { type TaskTarget } from "./tasks/dialog.tsx";
import { requestResource } from "./resources.ts";

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
  const scopeRef = useRef(scope);
  scopeRef.current = scope;
  const [task, setTask] = useState<TaskTarget | null>(null);
  const [taskBusy, setTaskBusy] = useState(false);
  const [createdTask, setCreatedTask] = useState<string | null>(null);
  const [documentRevision, setDocumentRevision] = useState(0);
  const section = sections[selected] ?? "chat";
  useEffect(() => {
    setSections({});
    setDrafts({});
    setError(null);
    setTask(null);
    setTaskBusy(false);
    setCreatedTask(null);
  }, [scope]);
  useEffect(() => {
    if (
      createdTask &&
      state.agents?.some((item) => item.agent_name === createdTask)
    ) {
      setCreatedTask(null);
      void action({ type: "agent.select", agentName: createdTask }).catch(
        setError,
      );
    }
  }, [createdTask, state.agents]);
  const manage = async (operation: () => Promise<unknown>) => {
    if (taskBusy) return;
    setTaskBusy(true);
    setError(null);
    try {
      await operation();
    } catch (reason) {
      if (scopeRef.current === scope) setError(reason);
    } finally {
      if (scopeRef.current === scope) setTaskBusy(false);
    }
  };
  const toggleTask = (value: AgentView) =>
    void manage(() =>
      requestResource({
        type: "task",
        action: "update",
        agent_name: value.agent_name,
        settings: { enabled: !value.enabled },
      }),
    );
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
    agent?.enabled === false
      ? "agentDisabled"
      : agent?.error
        ? "agentError"
        : agent?.running
          ? "agentRunning"
          : "agentIdle",
  );
  return (
    <>
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
            onManage={setTask}
            onToggle={toggleTask}
            onReveal={(agentName) =>
              void manage(() => action({ type: "agent.reveal", agentName }))
            }
            busy={taskBusy}
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
              {!isRemy && (
                <button
                  className="agent-tool-button"
                  onClick={() => setTask({ mode: "edit", agentName: selected })}
                >
                  {t("taskManage")}
                </button>
              )}
              <nav
                className="agent-tabs"
                aria-label={t("agentPages", { name })}
              >
                {tabs.map(([id, label]) => (
                  <button
                    key={id}
                    className="agent-tool-button"
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
              {!isRemy && (
                <button
                  className="icon-button"
                  aria-label={t("taskDetails")}
                  onClick={() =>
                    setTask({ mode: "status", agentName: selected })
                  }
                >
                  <Info size={15} />
                </button>
              )}
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
              <AgentDocument
                state={state}
                path={documentPath}
                revision={documentRevision}
                onEdit={
                  isRemy
                    ? undefined
                    : () => setTask({ mode: "edit", agentName: selected })
                }
              />
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
      {task && (
        <TaskDialog
          key={`${scope}:${JSON.stringify(task)}`}
          target={task}
          state={state}
          onClose={() => setTask(null)}
          onCreated={(agentName) => {
            setCreatedTask(agentName);
            setTask(null);
          }}
          onSaved={() => setDocumentRevision((value) => value + 1)}
        />
      )}
    </>
  );
}
