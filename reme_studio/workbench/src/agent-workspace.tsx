import { useEffect, useRef, useState } from "react";
import { ArrowUpRight, CheckCheck, Eraser, RefreshCw } from "lucide-react";
import type { StudioState } from "../../client/src/protocol.ts";
import { errorText, translator } from "../../client/src/i18n.ts";
import { useHost } from "./host.ts";
import Chat from "./chat/index.tsx";
import SplitPane from "./split-pane.tsx";
import type { MenuItem } from "./context-menu.tsx";
import AgentSidebar, {
  remySections,
  type RemySection,
} from "./agent-sidebar.tsx";
import AgentDocument from "./agent-document.tsx";
import AgentResources from "./agent-resources.tsx";
import Reminders from "./reminders.tsx";

/** Remy's persistent navigation; proactive tasks are owned by the full-width graph. */
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
  const t = translator(state.settings.language);
  const active = state.activePage === "remy";
  const agent = state.agents?.find((item) => item.agent_name === "Remy");
  const [section, setSection] = useState<RemySection>("chat");
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [inboxRevision, setInboxRevision] = useState(0);
  const [marking, setMarking] = useState(false);
  const [clearDialogOpen, setClearDialogOpen] = useState(false);
  const focus = useRef(0);
  useEffect(() => {
    if (!active || section !== "chat") setClearDialogOpen(false);
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
  const actions: Record<RemySection, MenuItem[]> = {
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
        disabled: !connected || !agent?.memory_path,
        select: () => {
          if (agent?.memory_path)
            void action({
              type: "document.open",
              path: agent.memory_path,
            }).catch(setError);
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
  const label = remySections.find(([id]) => id === section)![1];
  return (
    <SplitPane
      name="remy"
      language={state.settings.language}
      collapsed={collapsed}
      path={[{ label: "Remy", revealSidebar: true }, { label: t(label) }]}
      sidebar={
        <AgentSidebar
          state={state}
          section={section}
          actions={actions}
          onSelect={navigate}
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
              {section === "reminders" && (
                <Reminders
                  state={state}
                  canMarkAllRead={canMarkAllRead}
                  onMarkAllRead={markAllRead}
                />
              )}
              {section === "inbox" && (
                <AgentResources
                  state={state}
                  agentName="Remy"
                  inbox
                  revision={inboxRevision}
                />
              )}
            </section>
          )}
        </div>
      </section>
    </SplitPane>
  );
}
