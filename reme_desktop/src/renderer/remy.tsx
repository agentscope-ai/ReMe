import { useEffect, useRef, useState } from "react";
import { ArrowLeft, RefreshCw } from "lucide-react";
import type {
  DesktopState,
  DisplayMessage,
  ProactiveInboxDetail,
  ProactiveInboxItem,
  SessionSnapshot,
} from "../shared/protocol.ts";
import { errorText, translator } from "../shared/i18n.ts";
import { displayMessages } from "../shared/messages.ts";
import { action } from "./desktop.ts";
import { inboxDate, inboxRequest } from "./inbox.ts";
import Chat from "./chat/index.tsx";
import MessageBody from "./message-body.tsx";
import RemyInbox from "./remy-inbox.tsx";
import SplitPane from "./split-pane.tsx";

export default function Remy({
  state,
  onSettings,
  collapsed,
}: {
  state: DesktopState;
  onSettings: () => void;
  collapsed?: boolean;
}) {
  const language = state.settings.language;
  const t = translator(language);
  const [selected, setSelected] = useState("");
  const [detail, setDetail] = useState<ProactiveInboxDetail | null>(null);
  const [revision, setRevision] = useState(0);
  const [preview, setPreview] = useState<{
    sessionId: string;
    messages: DisplayMessage[];
  } | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const focusGeneration = useRef(0);
  const scope = `${state.settings.endpoint}:${state.workspace}`;

  useEffect(() => {
    focusGeneration.current++;
    setPreview(null);
    setDetail(null);
    setSelected("");
    setError(null);
    setLoading(false);
  }, [scope]);

  useEffect(() => {
    const id = state.focusMessageId;
    if (!id) return;
    setSelected("");
    setDetail(null);
    setError(null);
    const message = state.inboxMessages.find((item) => item.id === id);
    const origin = message?.originSessionId ?? state.sessionId;
    const generation = ++focusGeneration.current;
    if (
      !origin ||
      (origin === state.sessionId &&
        state.messages.some((item) => item.id === id))
    ) {
      setPreview(null);
      setLoading(false);
      return;
    }
    const open = async () => {
      setLoading(true);
      setPreview(null);
      try {
        let messages = state.messages;
        if (origin !== state.sessionId) {
          const snapshot = await inboxRequest<SessionSnapshot>({
            type: "conversation",
            sessionId: origin,
          });
          messages = displayMessages(
            { sessionId: origin, messages: snapshot.messages, pending: null },
            undefined,
            language,
          );
        }
        if (message && !messages.some((item) => item.id === id))
          messages = [...messages, message];
        if (generation === focusGeneration.current)
          setPreview({ sessionId: origin, messages });
      } catch (error) {
        if (generation === focusGeneration.current) setError(error);
      } finally {
        if (generation === focusGeneration.current) setLoading(false);
      }
    };
    void open();
    return () => {
      focusGeneration.current++;
    };
  }, [scope, state.focusMessageId, state.focusRequest]);

  useEffect(() => {
    if (
      !selected ||
      state.activePage !== "remy" ||
      state.connection !== "connected"
    )
      return;
    let live = true;
    let pending = false;
    const refresh = async () => {
      if (pending) return;
      pending = true;
      try {
        const value = await inboxRequest<ProactiveInboxDetail>({
          type: "proactiveInboxEntry",
          id: selected,
        });
        if (live) {
          setDetail(value);
          setError(null);
        }
      } catch (error) {
        if (live) setError(error);
      } finally {
        pending = false;
        if (live) setLoading(false);
      }
    };
    void refresh();
    const timer = setInterval(() => void refresh(), 5000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [scope, selected, revision, state.activePage, state.connection]);

  const result = detail?.item;
  useEffect(() => {
    if (
      !result?.path ||
      result.kind !== "task_result" ||
      result.id !== selected ||
      result.user_read_at ||
      state.activePage !== "remy" ||
      !state.chatVisible ||
      !state.chatFocused ||
      state.connection !== "connected"
    )
      return;
    let live = true;
    const path = result.path;
    const timer = setTimeout(() => {
      if (!document.hasFocus() || document.visibilityState !== "visible")
        return;
      void inboxRequest<{ user_read_at: string }>({
        type: "taskResult",
        path,
        markRead: true,
      })
        .then((value) => {
          if (!live) return;
          setDetail((current) =>
            current?.item.id === selected
              ? {
                  ...current,
                  item: { ...current.item, user_read_at: value.user_read_at },
                }
              : current,
          );
          setRevision((current) => current + 1);
        })
        .catch((error) => {
          if (live) setError(error);
        });
    }, 400);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [
    scope,
    result?.id,
    result?.path,
    result?.user_read_at,
    selected,
    revision,
    state.activePage,
    state.chatVisible,
    state.chatFocused,
    state.connection,
  ]);

  const openInput = (item: ProactiveInboxItem) => {
    focusGeneration.current++;
    setSelected(item.id);
    setPreview(null);
    setDetail(null);
    setLoading(true);
    setError(null);
    setRevision((current) => current + 1);
  };
  const returnToChat = () => {
    focusGeneration.current++;
    setPreview(null);
    setSelected("");
    setDetail(null);
    setLoading(false);
    setError(null);
  };
  const continueConversation = async () => {
    if (!preview) return;
    try {
      if (preview.sessionId !== state.sessionId)
        await action({ type: "session.select", sessionId: preview.sessionId });
      returnToChat();
    } catch (error) {
      setError(error);
    }
  };
  return (
    <SplitPane
      name="remy"
      collapsed={collapsed}
      language={language}
      sidebar={
        <RemyInbox
          key={scope}
          state={state}
          selected={selected}
          revision={revision}
          onOpen={openInput}
        />
      }
    >
      <div className="remy-detail">
        {(selected || preview) && (
          <div className="result-toolbar">
            <button className="subtle" onClick={returnToChat}>
              <ArrowLeft size={15} />
              {t("backToChat")}
            </button>
            {preview && (
              <button
                className="secondary"
                disabled={
                  Boolean(state.pending?.streaming) || state.status === "active"
                }
                onClick={() => void continueConversation()}
              >
                {t("continueConversation")}
              </button>
            )}
            {selected && (
              <button
                className="icon-button"
                aria-label={t("inboxRefresh")}
                title={t("inboxRefresh")}
                onClick={() => setRevision((current) => current + 1)}
              >
                <RefreshCw size={15} />
              </button>
            )}
          </div>
        )}
        {error != null && (
          <div className="connection-banner" role="alert">
            {errorText(error, language)}
          </div>
        )}
        {loading && (
          <div className="document-loading">
            {t(
              state.connection === "connected"
                ? "documentLoading"
                : "reconnecting",
            )}
          </div>
        )}
        {!loading && selected && (
          <div className="document-pane">
            {detail && (
              <article className="inbox-detail document-content">
                <div className="eyebrow">
                  {t("inbox")} / {t("taskResults")}
                </div>
                <h1 className="document-title">{detail.item.title}</h1>
                <div className="inbox-detail-meta">
                  <time>{inboxDate(detail.item.created_at, language)}</time>
                  <span>
                    {t(
                      detail.item.user_read_at
                        ? "inboxUserRead"
                        : "inboxUserUnread",
                    )}
                  </span>
                </div>
                {detail.warnings.length > 0 && (
                  <div className="inbox-warning" role="status">
                    {detail.warnings.map((warning, index) => (
                      <p key={index}>{warning}</p>
                    ))}
                  </div>
                )}
                <MessageBody
                  message={{
                    id: detail.item.id,
                    role: "assistant",
                    source: "proactive",
                    content: detail.content,
                    completed: true,
                    streaming: false,
                    status: null,
                  }}
                  language={language}
                  onLink={(url) => {
                    if (/^https?:\/\//i.test(url))
                      void action({ type: "link.open", url });
                  }}
                />
              </article>
            )}
          </div>
        )}
        {!loading && !selected && preview && (
          <Chat
            state={{
              ...state,
              sessionId: preview.sessionId,
              messages: preview.messages,
              pending: null,
              surface: null,
            }}
            readOnly
            active={state.activePage === "remy"}
            onSettings={onSettings}
          />
        )}
        <div
          className="remy-live-chat"
          hidden={loading || Boolean(selected || preview)}
          inert={loading || Boolean(selected || preview)}
        >
          <Chat
            state={state}
            active={
              state.activePage === "remy" && !loading && !selected && !preview
            }
            onSettings={onSettings}
          />
        </div>
      </div>
    </SplitPane>
  );
}
