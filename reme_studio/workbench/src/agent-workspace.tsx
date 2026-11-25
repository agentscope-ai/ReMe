import { useEffect, useRef, useState } from "react";
import { ArrowUpRight, CheckCheck, Eraser, RefreshCw } from "lucide-react";
import type { AgentView, StudioState } from "../../client/src/protocol.ts";
import { errorText, translator } from "../../client/src/i18n.ts";
import { useHost } from "./host.ts";
import Chat from "./chat/index.tsx";
import SplitPane from "./split-pane.tsx";
import type { MenuItem } from "./context-menu.tsx";
import AgentSidebar, {
  remySections,
  taskSections,
  type AgentSection,
  type RemySection,
} from "./agent-sidebar.tsx";
import AgentDocument from "./agent-document.tsx";
import AgentResources from "./agent-resources.tsx";
import Reminders from "./reminders.tsx";
import TaskEditor from "./tasks/editor.tsx";
import { useResource } from "./resources.ts";

type AgentPage = "remy" | "tasks";

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
    Partial<
      Record<AgentPage, { agentName: string | null; section: AgentSection }>
    >
  >({});
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [error, setError] = useState<unknown>(null);
  const live = useRef(false);
  const [taskBusy, setTaskBusy] = useState(false);
  const [createdTask, setCreatedTask] = useState<string | null>(null);
  const [inboxRevision, setInboxRevision] = useState(0);
  const [markingReminders, setMarkingReminders] = useState(false);
  const [clearDialogOpen, setClearDialogOpen] = useState(false);
  const hasTask = !isRemy && selected !== "Remy" && !!agent;
  const location = sections[page];
  const section =
    location?.section === "create" ||
    ((isRemy || hasTask) && location?.agentName === selected)
      ? location.section
      : "chat";
  useEffect(() => {
    if (!active || !isRemy || section !== "chat") setClearDialogOpen(false);
  }, [active, isRemy, section]);
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);
  useEffect(() => {
    if (
      active &&
      createdTask &&
      state.agents?.some((item) => item.agent_name === createdTask)
    ) {
      setCreatedTask(null);
      void action({ type: "agent.select", agentName: createdTask }).catch(
        setError,
      );
    }
  }, [active, createdTask, state.agents]);
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
      [page]: {
        agentName: selected,
        section:
          state.focusTarget?.kind === "mail"
            ? selected === "Remy"
              ? "inbox"
              : "outbox"
            : state.focusTarget?.kind === "message"
              ? "reminders"
              : "chat",
      },
    }));
  }, [active, page, selected, state.focusRequest, state.focusTarget]);
  const select = (agentName: string | null, value: AgentSection) => {
    setError(null);
    setCreatedTask(null);
    setSections((current) => ({
      ...current,
      [page]: { agentName, section: value },
    }));
    if (agentName && agentName !== selected)
      void action({ type: "agent.select", agentName }).catch(setError);
  };
  const name = isRemy ? "Remy" : hasTask ? selected : t("proactiveAi");
  const documentPath = isRemy ? agent?.memory_path : agent?.task_path;
  const sectionLabel =
    section === "create"
      ? "taskCreate"
      : (isRemy ? remySections : taskSections).find(
          ([id]) => id === section,
        )?.[1];
  const navigate = (id: AgentSection) => {
    select(selected, id);
    onNavigate();
  };
  const connected = state.connection === "connected";
  const canClearContext =
    connected && !!state.clientInfo && !state.pending?.streaming;
  const canMarkAllRead = connected && state.unread > 0 && !markingReminders;
  const markAllRead = () => {
    if (!canMarkAllRead) return;
    setError(null);
    setMarkingReminders(true);
    void action({ type: "items.read" })
      .catch(setError)
      .finally(() => setMarkingReminders(false));
  };
  const remyActions: Record<RemySection, MenuItem[]> = {
    chat: [
      {
        id: "clear-context",
        label: t("clearContext"),
        icon: Eraser,
        disabled: !canClearContext,
        select: () => {
          navigate("chat");
          setClearDialogOpen(true);
        },
      },
    ],
    reminders: [
      {
        id: "mark-all-read",
        label: t("markAllRead"),
        icon: CheckCheck,
        disabled: !canMarkAllRead,
        select: markAllRead,
      },
    ],
    document: [
      {
        id: "open-in-knowledge",
        label: t("openInKnowledge"),
        icon: ArrowUpRight,
        disabled: !connected || !documentPath,
        select: () => {
          if (documentPath)
            void action({ type: "document.open", path: documentPath }).catch(
              setError,
            );
        },
      },
    ],
    inbox: [
      {
        id: "refresh-inbox",
        label: t("refreshInbox"),
        icon: RefreshCw,
        disabled: !connected,
        select: () => {
          navigate("inbox");
          setInboxRevision((value) => value + 1);
        },
      },
    ],
  };
  const tasksRoot = { label: t("proactiveAi"), revealSidebar: true };
  const path =
    section === "create"
      ? [tasksRoot, { label: t("taskCreate") }]
      : [
          ...(!isRemy && hasTask ? [tasksRoot] : []),
          { label: name, onSelect: () => select(selected, "chat") },
          ...(isRemy || hasTask
            ? [{ label: t(sectionLabel ?? "progressTab") }]
            : []),
        ];
  return (
    <>
      <SplitPane
        key={page}
        name={page}
        path={path}
        collapsed={collapsed}
        language={state.settings.language}
        sidebar={
          <AgentSidebar
            state={state}
            page={page}
            section={section}
            remyActions={remyActions}
            onSelect={(agentName, section) => {
              select(agentName, section);
              onNavigate();
            }}
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
          <div className="agent-content">
            {(isRemy || hasTask) && (
              <div
                className="agent-chat"
                key={selected}
                hidden={section !== "chat"}
              >
                <Chat
                  state={state}
                  active={active && section === "chat"}
                  canClearContext={canClearContext}
                  clearDialogOpen={clearDialogOpen}
                  onClearDialogChange={setClearDialogOpen}
                  draft={drafts[selected] ?? ""}
                  onDraft={(value) =>
                    setDrafts((current) => ({
                      ...current,
                      [selected]: value,
                    }))
                  }
                />
              </div>
            )}
            {section !== "chat" ? (
              <section
                key={section === "create" ? section : `agent:${selected}`}
                className="agent-resource-page"
                aria-label={sectionLabel && t(sectionLabel)}
              >
                {(section === "create" || section === "manage") && (
                  <TaskEditor
                    agent={section === "manage" ? agent : undefined}
                    state={state}
                    onCreated={(agentName) => {
                      select(null, "chat");
                      setCreatedTask(agentName);
                    }}
                  />
                )}
                {section === "document" &&
                  (documentPath ? (
                    <AgentDocument
                      state={state}
                      path={documentPath}
                      onEdit={isRemy ? undefined : () => navigate("manage")}
                    />
                  ) : (
                    <p className="document-loading">{t("documentLoading")}</p>
                  ))}
                {section === "reminders" && isRemy && (
                  <Reminders
                    state={state}
                    canMarkAllRead={canMarkAllRead}
                    onMarkAllRead={markAllRead}
                  />
                )}
                {(section === "outbox" || section === "inbox") && (
                  <AgentResources
                    key={section}
                    state={state}
                    inbox={section === "inbox"}
                    revision={isRemy ? inboxRevision : undefined}
                  />
                )}
              </section>
            ) : (
              !isRemy &&
              !hasTask && (
                <div className="resource-empty">
                  <p>{t("noAgents")}</p>
                </div>
              )
            )}
          </div>
        </section>
      </SplitPane>
    </>
  );
}
