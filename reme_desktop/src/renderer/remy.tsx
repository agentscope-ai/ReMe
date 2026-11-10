import { useEffect, useRef, useState } from "react";
import { ArrowLeft, ArrowUpRight } from "lucide-react";
import type {
  DesktopState,
  DisplayMessage,
  ProactiveInboxDetail,
  ProactiveInboxItem,
} from "../shared/protocol.ts";
import { errorText, translator } from "../shared/i18n.ts";
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
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const sending = useRef(false);
  const scope = `${state.settings.endpoint}:${state.workspace}`;
  const reminder = !selected
    ? state.inboxMessages.find((item) => item.id === state.focusMessageId)
    : undefined;
  const reminderRead = state.readMessageIds.includes(reminder?.id ?? "");
  const showingDetail = Boolean(selected || reminder);
  const busy = Boolean(state.pending?.streaming) || state.status === "active";
  const ready = state.connection === "connected" && state.status !== "error";

  useEffect(() => {
    setDetail(null);
    setSelected("");
    setError(null);
    setLoading(false);
  }, [scope]);

  useEffect(() => {
    if (!state.focusMessageId) return;
    setSelected("");
    setDetail(null);
    setError(null);
    setLoading(false);
  }, [scope, state.focusMessageId, state.focusRequest]);

  useEffect(() => {
    const sessionId = reminder?.originSessionId;
    if (
      !reminder?.completed ||
      !reminder.content ||
      !sessionId ||
      reminderRead ||
      state.activePage !== "remy" ||
      !state.chatVisible ||
      !state.chatFocused
    )
      return;
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    const acknowledge = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (!document.hasFocus() || document.visibilityState !== "visible")
          return;
        void action({
          type: "messages.read",
          sessionId,
          ids: [reminder.id],
        })
          .catch((error) => {
            if (live) setError(error);
          })
          .finally(() => {
            if (live) acknowledge();
          });
      }, 600);
    };
    acknowledge();
    window.addEventListener("focus", acknowledge);
    return () => {
      live = false;
      clearTimeout(timer);
      window.removeEventListener("focus", acknowledge);
    };
  }, [
    scope,
    reminder?.id,
    reminder?.originSessionId,
    reminder?.completed,
    reminder?.content,
    reminderRead,
    state.activePage,
    state.chatVisible,
    state.chatFocused,
  ]);

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
    setSelected(item.id);
    setDetail(null);
    setLoading(true);
    setError(null);
    setRevision((current) => current + 1);
  };
  const returnToChat = async () => {
    try {
      await action({ type: "chat.open" });
      setSelected("");
      setDetail(null);
      setLoading(false);
      setError(null);
    } catch (error) {
      setError(error);
    }
  };
  const explainReminder = async (message: DisplayMessage) => {
    if (sending.current || busy || !ready) return;
    sending.current = true;
    setError(null);
    try {
      await action({
        type: "chat.send",
        query: t("explainReminderQuery", { content: message.content }),
      });
    } catch (error) {
      setError(error);
    } finally {
      sending.current = false;
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
        {showingDetail && (
          <div className="result-toolbar">
            <button className="subtle" onClick={() => void returnToChat()}>
              <ArrowLeft size={15} />
              {t("backToChat")}
            </button>
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
          <div className="document-pane" key={selected}>
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
        {reminder && (
          <div
            className="document-pane"
            key={`${reminder.id}:${state.focusRequest}`}
          >
            <article
              className="inbox-detail document-content"
              data-message-id={reminder.id}
            >
              <div className="proactive-label">✦ {t("proactiveLabel")}</div>
              <div className="inbox-detail-meta">
                <time>{inboxDate(reminder.createdAt, language)}</time>
                <span>
                  {t(reminderRead ? "inboxUserRead" : "inboxUserUnread")}
                </span>
              </div>
              <MessageBody
                message={reminder}
                language={language}
                onLink={(url) =>
                  void action({ type: "link.open", url }).catch(setError)
                }
              />
              {reminder.completed && reminder.content && !reminder.error && (
                <div className="message-actions">
                  <button
                    type="button"
                    className="explain-reminder"
                    disabled={busy || !ready}
                    onClick={() => void explainReminder(reminder)}
                  >
                    {t("explainReminder")}
                    <ArrowUpRight size={13} aria-hidden />
                  </button>
                </div>
              )}
            </article>
          </div>
        )}
        <div
          className="remy-live-chat"
          hidden={loading || showingDetail}
          inert={loading || showingDetail}
        >
          <Chat
            state={state}
            active={state.activePage === "remy" && !loading && !showingDetail}
            onSettings={onSettings}
            onExplain={(message) => void explainReminder(message)}
          />
        </div>
      </div>
    </SplitPane>
  );
}
