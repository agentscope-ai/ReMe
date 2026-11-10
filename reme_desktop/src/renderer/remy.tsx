import { useEffect, useState } from "react";
import { ArrowLeft, ArrowUpRight } from "lucide-react";
import type {
  DesktopState,
  DisplayMessage,
  ProactiveInboxDetail,
  ItemTarget,
} from "../shared/protocol.ts";
import { errorText, translator } from "../shared/i18n.ts";
import { action } from "./desktop.ts";
import { inboxDate, inboxRequest } from "./inbox.ts";
import Chat from "./chat/index.tsx";
import MessageBody from "./message-body.tsx";
import RemyInbox from "./remy-inbox.tsx";
import SplitPane from "./split-pane.tsx";
import { watchReadReceipt } from "./read-receipt.ts";

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
  const selected =
    state.focusTarget?.kind === "task_result" ? state.focusTarget.id : "";
  const [detail, setDetail] = useState<ProactiveInboxDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const scope = `${state.settings.endpoint}:${state.workspace}`;
  const reminder =
    state.focusTarget?.kind === "message"
      ? state.inboxMessages.find((item) => item.id === state.focusTarget?.id)
      : undefined;
  const reminderRead = state.readMessageIds.includes(reminder?.id ?? "");
  const showingDetail = Boolean(selected || reminder);
  const busy = Boolean(state.pending?.streaming) || state.status === "active";
  const ready = state.connection === "connected" && state.status !== "error";

  useEffect(() => {
    setDetail(null);
    setError(null);
    setLoading(Boolean(selected));
  }, [scope, selected, state.focusTarget, state.focusRequest]);

  useEffect(() => {
    const sessionId = reminder?.originSessionId;
    if (
      !reminder?.completed ||
      !reminder.content.trim() ||
      !sessionId ||
      reminderRead ||
      state.activePage !== "remy" ||
      !state.chatVisible ||
      !state.chatFocused
    )
      return;
    const receipt = watchReadReceipt(async () => {
      await action({
        type: "messages.read",
        sessionId,
        ids: [reminder.id],
      });
      return true;
    }, setError);
    return receipt.close;
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
  }, [
    scope,
    selected,
    state.inboxRevision,
    state.focusRequest,
    state.activePage,
    state.connection,
  ]);

  const result = detail?.item;
  useEffect(() => {
    if (
      !result?.path ||
      !detail?.content.trim() ||
      result.kind !== "task_result" ||
      result.id !== selected ||
      result.user_read_at ||
      state.activePage !== "remy" ||
      !state.chatVisible ||
      !state.chatFocused ||
      state.connection !== "connected"
    )
      return;
    const path = result.path;
    const receipt = watchReadReceipt(async (signal) => {
      const value = await inboxRequest<{ user_read_at: string }>({
        type: "taskResult",
        path,
        markRead: true,
      });
      if (!signal.aborted) {
        setError(null);
        setDetail((current) =>
          current?.item.id === selected
            ? {
                ...current,
                item: { ...current.item, user_read_at: value.user_read_at },
              }
            : current,
        );
      }
      return !value.user_read_at;
    }, setError);
    return receipt.close;
  }, [
    scope,
    result?.id,
    result?.path,
    result?.user_read_at,
    detail?.content,
    selected,
    state.activePage,
    state.chatVisible,
    state.chatFocused,
    state.connection,
  ]);

  const explain = (target: ItemTarget) => {
    setError(null);
    void action({ type: "item.action", action: "explain", target }).catch(
      setError,
    );
  };
  const message: DisplayMessage | null =
    reminder ??
    (detail?.item.id === selected
      ? {
          id: detail.item.id,
          role: "assistant",
          source: "proactive",
          content: detail.content,
          createdAt: detail.item.created_at,
          completed: true,
          streaming: false,
          status: null,
        }
      : null);
  return (
    <SplitPane
      name="remy"
      collapsed={collapsed}
      language={language}
      sidebar={<RemyInbox key={scope} state={state} onError={setError} />}
    >
      <div className="remy-detail">
        {showingDetail && (
          <div className="result-toolbar">
            <button
              className="subtle"
              onClick={() => void action({ type: "chat.open" }).catch(setError)}
            >
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
        {!loading && message && (
          <div
            className="document-pane"
            key={`${message.id}:${state.focusRequest}`}
          >
            <article
              className="inbox-detail document-content"
              data-message-id={message.id}
            >
              {reminder ? (
                <div className="proactive-label">✦ {t("proactiveLabel")}</div>
              ) : (
                <>
                  <div className="eyebrow">
                    {t("inbox")} / {t("taskResults")}
                  </div>
                  <h1 className="document-title">{detail?.item.title}</h1>
                </>
              )}
              <div className="inbox-detail-meta">
                <time>{inboxDate(message.createdAt, language)}</time>
                <span>
                  {t(
                    (reminder ? reminderRead : detail?.item.user_read_at)
                      ? "inboxUserRead"
                      : "inboxUserUnread",
                  )}
                </span>
              </div>
              {!reminder && detail?.warnings.length ? (
                <div className="inbox-warning" role="status">
                  {detail.warnings.map((warning, index) => (
                    <p key={index}>{warning}</p>
                  ))}
                </div>
              ) : null}
              <MessageBody
                message={message}
                language={language}
                onLink={(url) =>
                  void action({ type: "link.open", url }).catch(setError)
                }
              />
              {message.completed && message.content && !message.error && (
                <div className="message-actions">
                  <button
                    type="button"
                    className="explain-reminder"
                    disabled={busy || !ready}
                    onClick={() =>
                      state.focusTarget && explain(state.focusTarget)
                    }
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
            onExplain={(message) =>
              explain({ kind: "message", id: message.id })
            }
          />
        </div>
      </div>
    </SplitPane>
  );
}
