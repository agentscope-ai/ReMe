import { useContext, useEffect, useRef, useState } from "react";
import { ArrowUpRight, Plus, RefreshCw } from "lucide-react";
import type { AgentView, StudioState } from "../../../client/src/protocol.ts";
import { errorText, translator } from "../../../client/src/i18n.ts";
import AgentResources from "../agent-resources.tsx";
import Chat from "../chat/index.tsx";
import { useHost } from "../host.ts";
import { useNavigationBlocker, useNavigationState } from "../navigation.tsx";
import ProactiveOverview from "../proactive/index.tsx";
import { useResource } from "../resources.ts";
import SplitPane from "../split-pane.tsx";
import { WorkspaceLayout } from "../workspace-titlebar.tsx";
import MobileSections from "../mobile/sections.tsx";
import TaskEditor from "./editor.tsx";
import TaskDocument from "./document.tsx";
import TaskSidebar, { taskSections, type TaskLocation } from "./sidebar.tsx";
import "./workspace.css";

export default function Tasks({
  state,
  collapsed,
  onNavigate,
}: {
  state: StudioState;
  collapsed?: boolean;
  onNavigate: () => void;
}) {
  const { action } = useHost();
  const mobile = useContext(WorkspaceLayout)?.mobile;
  const request = useResource();
  const t = translator(state.settings.language);
  const active = state.activePage === "tasks";
  const [location, setLocation] = useNavigationState<TaskLocation>(
    "tasks",
    "location",
    { section: "overview" },
  );
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [revision, setRevision] = useState(0);
  const [editing, setEditing] = useState(false);
  const [navigationBlocked, setNavigationBlocked] = useState(false);
  useNavigationBlocker("tasks", active && editing);
  const name = "agentName" in location ? location.agentName : undefined;
  const agent = state.agents?.find((item) => item.agent_name === name);
  const { section } = location;
  const focus = useRef(0);
  const pendingFocus = Boolean(
    state.focusRequest &&
    state.focusRequest !== focus.current &&
    state.focusTarget,
  );
  useEffect(() => {
    if (
      !active ||
      pendingFocus ||
      editing ||
      !agent ||
      state.selectedAgentName === agent.agent_name
    )
      return;
    void action({ type: "agent.select", agentName: agent.agent_name }).catch(
      setError,
    );
  }, [
    active,
    pendingFocus,
    editing,
    agent?.agent_name,
    state.selectedAgentName,
  ]);
  useEffect(() => {
    if (
      !active ||
      !state.focusRequest ||
      focus.current === state.focusRequest ||
      !state.focusTarget
    )
      return;
    if (editing) {
      setNavigationBlocked(true);
      return;
    }
    focus.current = state.focusRequest;
    const agentName = state.selectedAgentName;
    if (!agentName || agentName === "Remy") return;
    setLocation({
      agentName,
      section: state.focusTarget.kind === "mail" ? "outbox" : "chat",
    });
  }, [
    active,
    editing,
    state.focusRequest,
    state.focusTarget,
    state.selectedAgentName,
  ]);
  const select = (value: TaskLocation) => {
    if (editing) {
      setNavigationBlocked(true);
      return;
    }
    setError(null);
    setLocation(value);
    onNavigate();
  };
  const manage = async (operation: () => Promise<unknown>) => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await operation();
    } catch (reason) {
      setError(reason);
    } finally {
      setBusy(false);
    }
  };
  const toggle = (value: AgentView) =>
    void manage(() =>
      request({
        type: "task",
        action: "update",
        agent_name: value.agent_name,
        settings: { enabled: !value.enabled },
      }),
    );
  const sectionLabel =
    section === "overview"
      ? "proactiveOverview"
      : section === "list"
        ? "flowAllAgents"
        : section === "create"
          ? "taskCreate"
          : taskSections.find(([id]) => id === section)![1];
  const path = [
    {
      label: t("proactiveAi"),
      revealSidebar: true,
      onSelect: () => select({ section: mobile ? "list" : "overview" }),
    },
    ...(name
      ? [
          {
            label: name,
            title: name,
            onSelect: () => select({ agentName: name, section: "chat" }),
          },
        ]
      : []),
    { label: t(sectionLabel) },
  ];
  const connected = state.connection === "connected";
  return (
    <SplitPane
      name="tasks"
      contentKey={JSON.stringify([location, editing])}
      language={state.settings.language}
      collapsed={collapsed}
      path={path}
      actions={
        <>
          {mobile && (
            <button
              type="button"
              className="titlebar-button"
              aria-label={t("taskCreate")}
              disabled={!connected || busy || editing}
              onClick={() => select({ section: "create" })}
            >
              <Plus size={22} aria-hidden />
            </button>
          )}
          {section === "document" && agent?.task_path && (
            <button
              type="button"
              className="titlebar-button"
              aria-label={t("openInKnowledge")}
              title={t("openInKnowledge")}
              disabled={!connected || editing}
              onClick={() =>
                void action({
                  type: "document.open",
                  path: agent.task_path!,
                }).catch(setError)
              }
            >
              <ArrowUpRight size={16} aria-hidden />
            </button>
          )}
          {(section === "inbox" || section === "outbox") && (
            <button
              type="button"
              className="titlebar-button"
              aria-label={t(
                section === "inbox" ? "refreshInbox" : "refreshOutbox",
              )}
              title={t(section === "inbox" ? "refreshInbox" : "refreshOutbox")}
              disabled={!connected}
              onClick={() => setRevision((value) => value + 1)}
            >
              <RefreshCw size={16} aria-hidden />
            </button>
          )}
        </>
      }
      sidebar={
        <TaskSidebar
          state={state}
          location={location}
          onSelect={select}
          onToggle={toggle}
          onReveal={(agentName) =>
            void manage(() => action({ type: "agent.reveal", agentName }))
          }
          busy={busy}
        />
      }
    >
      <section className="agent-workspace tasks-workspace">
        {section === "overview" || section === "list" ? (
          <MobileSections
            label={t("proactiveAi")}
            value={section}
            items={[
              { value: "overview", label: t("proactiveOverview") },
              { value: "list", label: t("flowAllAgents") },
            ]}
            onSelect={(section) => select({ section })}
          />
        ) : agent ? (
          <MobileSections
            label={t("agentPages", { name: agent.agent_name })}
            value={section}
            items={taskSections.map(([value, label]) => ({
              value,
              label: t(label),
            }))}
            disabled={editing}
            onSelect={(section) =>
              select({ agentName: agent.agent_name, section })
            }
          />
        ) : null}
        {error != null && (
          <p className="action-error" role="alert">
            {errorText(error, state.settings.language)}
          </p>
        )}
        {section === "overview" ? (
          <ProactiveOverview state={state} embedded />
        ) : section === "list" ? (
          <div className="mobile-task-list">
            <TaskSidebar
              state={state}
              location={location}
              onSelect={select}
              onToggle={toggle}
              onReveal={(agentName) =>
                void manage(() => action({ type: "agent.reveal", agentName }))
              }
              busy={busy}
            />
          </div>
        ) : section === "create" ? (
          <section className="agent-resource-page" aria-label={t("taskCreate")}>
            <TaskEditor
              state={state}
              onCreated={(agentName) => select({ agentName, section: "chat" })}
            />
          </section>
        ) : agent ? (
          <div className="agent-content" key={agent.agent_name}>
            {section === "chat" ? (
              state.selectedAgentName === agent.agent_name ? (
                <div className="agent-chat">
                  <Chat
                    state={state}
                    active={active}
                    draft={drafts[agent.agent_name] ?? ""}
                    onDraft={(value) =>
                      setDrafts((previous) => ({
                        ...previous,
                        [agent.agent_name]: value,
                      }))
                    }
                    canClearContext={false}
                    clearDialogOpen={false}
                    onClearDialogChange={() => {}}
                  />
                </div>
              ) : (
                <p className="document-loading">{t("flowConnecting")}</p>
              )
            ) : (
              <section
                className="agent-resource-page"
                aria-label={t(sectionLabel)}
                key={section}
              >
                {section === "document" &&
                  (agent.task_path ? (
                    <TaskDocument
                      state={state}
                      agent={agent}
                      editing={editing}
                      navigationBlocked={navigationBlocked}
                      onEditingChange={(value) => {
                        setEditing(value);
                        setNavigationBlocked(false);
                      }}
                    />
                  ) : (
                    <p className="document-loading">{t("documentLoading")}</p>
                  ))}
                {section === "manage" && (
                  <TaskEditor
                    state={state}
                    agent={agent}
                    onCreated={() => {}}
                  />
                )}
                {(section === "inbox" || section === "outbox") && (
                  <AgentResources
                    state={state}
                    agentName={agent.agent_name}
                    active={active}
                    inbox={section === "inbox"}
                    revision={revision}
                    showRefresh={false}
                  />
                )}
              </section>
            )}
          </div>
        ) : (
          <p className="document-loading">{t("noAgents")}</p>
        )}
      </section>
    </SplitPane>
  );
}
