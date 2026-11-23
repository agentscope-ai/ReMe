import { useEffect, useRef, useState } from "react";
import {
  Bell,
  BookOpenText,
  FileText,
  Inbox,
  MessageCircle,
  Send,
  Settings2,
} from "lucide-react";
import type { AgentView, StudioState } from "../../client/src/protocol.ts";
import { errorText, translator } from "../../client/src/i18n.ts";
import { useHost } from "./host.ts";
import Chat from "./chat/index.tsx";
import SplitPane from "./split-pane.tsx";
import AgentSidebar from "./agent-sidebar.tsx";
import AgentDocument from "./agent-document.tsx";
import AgentResources from "./agent-resources.tsx";
import Reminders from "./reminders.tsx";
import TaskDialog, { type TaskTarget } from "./tasks/dialog.tsx";
import { useResource } from "./resources.ts";

type AgentSection = "chat" | "document" | "reminders" | "outbox" | "inbox";
type AgentPage = "remy" | "tasks";

const remySections = [
  ["chat", "conversationTab", MessageCircle],
  ["reminders", "reminders", Bell],
  ["document", "memoryTab", BookOpenText],
  ["inbox", "inbox", Inbox],
] as const;

const taskSections = [
  ["document", "taskFileTab", FileText],
  ["inbox", "inbox", Inbox],
  ["outbox", "outbox", Send],
] as const;

export default function AgentWorkspace({
  state,
  page,
  collapsed,
  onNavigate,
}: {
  state: StudioState;
  page: AgentPage;
  collapsed?: boolean;
  onNavigate: () => void;
}) {
  const requestResource = useResource();
  const { action } = useHost();
  const t = translator(state.settings.language);
  const active = state.activePage === page;
  const isRemy = page === "remy";
  const selected = isRemy ? "Remy" : (state.selectedAgentName ?? "Remy");
  const agent = state.agents?.find((item) => item.agent_name === selected);
  const [sections, setSections] = useState<
    Partial<Record<string, AgentSection>>
  >({});
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [error, setError] = useState<unknown>(null);
  const live = useRef(false);
  const [task, setTask] = useState<TaskTarget | null>(null);
  const [taskBusy, setTaskBusy] = useState(false);
  const [createdTask, setCreatedTask] = useState<string | null>(null);
  const [documentRevision, setDocumentRevision] = useState(0);
  const hasTask = !isRemy && selected !== "Remy" && !!agent;
  const section = isRemy || hasTask ? (sections[selected] ?? "chat") : "chat";
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);
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
      if (live.current) setError(reason);
    } finally {
      if (live.current) setTaskBusy(false);
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
    if (!active || !state.focusRequest || focus.current === state.focusRequest)
      return;
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
  }, [active, selected, state.focusRequest, state.focusTarget]);
  const select = (agentName: string, value: AgentSection) => {
    setError(null);
    setSections((current) => ({ ...current, [agentName]: value }));
    if (agentName !== selected)
      void action({ type: "agent.select", agentName }).catch(setError);
  };
  const name = isRemy ? "Remy" : hasTask ? selected : t("proactiveAi");
  const documentPath = isRemy ? agent?.memory_path : agent?.task_path;
  const tabs = isRemy ? remySections : taskSections;
  const sectionLabel = tabs.find(([id]) => id === section)?.[1];
  const path = [
    ...(!isRemy && hasTask
      ? [{ label: t("proactiveAi"), revealSidebar: true }]
      : []),
    { label: name, onSelect: () => select(selected, "chat") },
    ...(isRemy || hasTask ? [{ label: t(sectionLabel ?? "progressTab") }] : []),
  ];
  const sectionNav = (
    <div
      className="agent-sections"
      aria-label={t("agentPages", { name })}
      role="group"
    >
      {!isRemy && (
        <button
          className="agent-row"
          onClick={() => setTask({ mode: "edit", agentName: selected })}
        >
          <Settings2 size={16} aria-hidden />
          {t("taskManage")}
        </button>
      )}
      {tabs.map(([id, label, Icon]) => (
        <button
          key={id}
          className="agent-row"
          aria-current={section === id ? "page" : undefined}
          onClick={() => {
            select(selected, id);
            onNavigate();
          }}
        >
          <Icon size={16} aria-hidden />
          {t(label)}
          {id === "reminders" && state.unread > 0 && (
            <span className="count-badge" data-tone="accent">
              {state.unread}
            </span>
          )}
        </button>
      ))}
    </div>
  );
  return (
    <>
      <SplitPane
        key={page}
        name={page}
        path={path}
        collapsed={collapsed}
        language={state.settings.language}
        defaultWidth={224}
        minWidth={208}
        maxWidth={320}
        sidebar={
          <AgentSidebar
            state={state}
            page={page}
            sections={sectionNav}
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
          {error != null && (
            <p className="action-error" role="alert">
              {errorText(error, state.settings.language)}
            </p>
          )}
          <div className="agent-content" key={selected}>
            {isRemy || hasTask ? (
              <>
                <div className="agent-chat" hidden={section !== "chat"}>
                  <Chat
                    state={state}
                    active={active && section === "chat"}
                    draft={drafts[selected] ?? ""}
                    onDraft={(value) =>
                      setDrafts((current) => ({
                        ...current,
                        [selected]: value,
                      }))
                    }
                  />
                </div>
                {section !== "chat" && (
                  <section
                    className="agent-resource-page"
                    aria-label={sectionLabel && t(sectionLabel)}
                  >
                    {section === "document" &&
                      (documentPath ? (
                        <AgentDocument
                          state={state}
                          path={documentPath}
                          revision={isRemy ? undefined : documentRevision}
                          onEdit={
                            isRemy
                              ? undefined
                              : () =>
                                  setTask({ mode: "edit", agentName: selected })
                          }
                        />
                      ) : (
                        <p className="document-loading">
                          {t("documentLoading")}
                        </p>
                      ))}
                    {section === "reminders" && isRemy && (
                      <Reminders state={state} />
                    )}
                    {(section === "outbox" || section === "inbox") && (
                      <AgentResources
                        key={section}
                        state={state}
                        inbox={section === "inbox"}
                      />
                    )}
                  </section>
                )}
              </>
            ) : (
              <div className="resource-empty">
                <p>{t("noAgents")}</p>
              </div>
            )}
          </div>
        </section>
      </SplitPane>
      {active && !isRemy && task && (
        <TaskDialog
          key={JSON.stringify(task)}
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
