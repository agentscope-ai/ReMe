import { useContext, useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  CheckCheck,
  Ellipsis,
  Eraser,
  Bell,
  Inbox,
  RefreshCw,
} from "lucide-react";
import type { StudioState } from "../../client/src/protocol.ts";
import { errorText, translator } from "../../client/src/i18n.ts";
import { useHost } from "./host.ts";
import Chat from "./chat/index.tsx";
import SplitPane from "./split-pane.tsx";
import ContextMenu, { type MenuPosition } from "./context-menu.tsx";
import AgentSidebar, {
  remySections,
  type RemySection,
} from "./agent-sidebar.tsx";
import AgentDocument from "./agent-document.tsx";
import AgentResources from "./agent-resources.tsx";
import Reminders from "./reminders.tsx";
import { useNavigationState } from "./navigation.tsx";
import { WorkspaceLayout } from "./workspace-titlebar.tsx";

/** Remy's conversation, document and mailbox navigation. */
export default function AgentWorkspace({
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
  const t = translator(state.settings.language);
  const active = state.activePage === "remy";
  const agent = state.agents?.find((item) => item.agent_name === "Remy");
  const [section, setSection] = useNavigationState<RemySection>(
    "remy",
    "section",
    "chat",
  );
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [inboxRevision, setInboxRevision] = useState(0);
  const [marking, setMarking] = useState(false);
  const [clearDialogOpen, setClearDialogOpen] = useState(false);
  const [menu, setMenu] = useState<MenuPosition | null>(null);
  const focus = useRef(0);
  useEffect(() => {
    if (!active || section !== "chat") {
      setClearDialogOpen(false);
      setMenu(null);
    }
  }, [active, section]);
  useEffect(() => {
    if (
      !active ||
      !state.focusRequest ||
      focus.current === state.focusRequest ||
      !state.focusTarget
    )
      return;
    focus.current = state.focusRequest;
    setSection(
      state.focusTarget.kind === "mail"
        ? "inbox"
        : state.focusTarget.kind === "message"
          ? "reminders"
          : "chat",
    );
  }, [active, state.focusRequest, state.focusTarget]);
  const connected = state.connection === "connected";
  const canClearContext =
    connected && !!state.clientInfo && !state.pending?.streaming;
  const canMarkAllRead = connected && state.unread > 0 && !marking;
  const markAllRead = () => {
    if (!canMarkAllRead) return;
    setMarking(true);
    setError(null);
    void action({ type: "items.read" })
      .catch(setError)
      .finally(() => setMarking(false));
  };
  const navigate = (value: RemySection) => {
    setError(null);
    setSection(value);
    onNavigate();
  };
  const label = remySections.find(([id]) => id === section)![1];
  return (
    <SplitPane
      name="remy"
      language={state.settings.language}
      collapsed={collapsed}
      path={[
        {
          label: "Remy",
          revealSidebar: true,
          onSelect: mobile ? () => navigate("chat") : undefined,
        },
        { label: t(label) },
      ]}
      actions={
        <div className="remy-page-actions">
          {mobile && section === "chat" && (
            <>
              <button
                type="button"
                className="titlebar-button"
                aria-label={t("inbox")}
                onClick={() => navigate("inbox")}
              >
                <Inbox size={21} aria-hidden />
              </button>
              <button
                type="button"
                className="titlebar-button"
                aria-label={t("reminders")}
                onClick={() => navigate("reminders")}
              >
                <Bell size={21} aria-hidden />
                {state.unread > 0 && (
                  <i className="mobile-unread" aria-hidden />
                )}
              </button>
            </>
          )}
          {section === "chat" && (
            <button
              type="button"
              className="titlebar-button"
              aria-label={t("sectionActions", { name: t("conversationTab") })}
              title={t("sectionActions", { name: t("conversationTab") })}
              aria-haspopup="menu"
              aria-expanded={Boolean(menu)}
              onClick={(event) => {
                event.currentTarget.focus({ preventScroll: true });
                const bounds = event.currentTarget.getBoundingClientRect();
                setMenu(
                  menu ? null : { x: bounds.right - 200, y: bounds.bottom + 6 },
                );
              }}
            >
              <Ellipsis size={17} aria-hidden />
            </button>
          )}
          {section === "reminders" && (
            <button
              type="button"
              className="titlebar-button"
              disabled={!canMarkAllRead}
              aria-label={t("markAllRead")}
              title={t("markAllRead")}
              onClick={markAllRead}
            >
              <CheckCheck size={16} aria-hidden />
              <span className="remy-action-label">{t("markAllRead")}</span>
            </button>
          )}
          {section === "document" && (
            <button
              type="button"
              className="titlebar-button"
              aria-label={t("openInKnowledge")}
              title={t("openInKnowledge")}
              disabled={!connected || !agent?.memory_path}
              onClick={() => {
                if (agent?.memory_path)
                  void action({
                    type: "document.open",
                    path: agent.memory_path,
                  }).catch(setError);
              }}
            >
              <ArrowUpRight size={17} aria-hidden />
            </button>
          )}
          {section === "inbox" && (
            <button
              type="button"
              className="titlebar-button"
              aria-label={t("refreshInbox")}
              title={t("refreshInbox")}
              disabled={!connected}
              onClick={() => setInboxRevision((value) => value + 1)}
            >
              <RefreshCw size={16} aria-hidden />
            </button>
          )}
        </div>
      }
      sidebar={
        <AgentSidebar state={state} section={section} onSelect={navigate} />
      }
    >
      <section className="agent-workspace">
        {error != null && (
          <p className="action-error" role="alert">
            {errorText(error, state.settings.language)}
          </p>
        )}
        <div className="agent-content">
          {section === "chat" && active && (
            <div className="agent-chat">
              <Chat
                state={state}
                active={active}
                canClearContext={canClearContext}
                clearDialogOpen={clearDialogOpen}
                onClearDialogChange={setClearDialogOpen}
                draft={draft}
                onDraft={setDraft}
              />
            </div>
          )}
          {section !== "chat" && (
            <section className="agent-resource-page" aria-label={t(label)}>
              {section === "document" &&
                (agent?.memory_path ? (
                  <AgentDocument
                    state={state}
                    agentName="Remy"
                    path={agent.memory_path}
                  />
                ) : (
                  <p className="document-loading">{t("documentLoading")}</p>
                ))}
              {section === "reminders" && <Reminders state={state} />}
              {section === "inbox" && (
                <AgentResources
                  state={state}
                  agentName="Remy"
                  inbox
                  revision={inboxRevision}
                  showRefresh={false}
                />
              )}
            </section>
          )}
        </div>
        {active && section === "chat" && menu && (
          <ContextMenu
            position={menu}
            size={{ width: 200, height: 380 }}
            onClose={() => setMenu(null)}
            items={[
              {
                id: "clear-context",
                label: t("clearContext"),
                icon: Eraser,
                disabled: !canClearContext,
                select: () => setClearDialogOpen(true),
              },
            ]}
          />
        )}
      </section>
    </SplitPane>
  );
}
