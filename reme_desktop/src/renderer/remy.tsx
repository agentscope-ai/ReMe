import { useState } from "react";
import { Bell, ArrowUpRight, Sparkles } from "lucide-react";
import type { DesktopState } from "../shared/protocol.ts";
import { translator } from "../shared/i18n.ts";
import { action } from "./desktop.ts";
import Chat from "./chat/index.tsx";
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
  const t = translator(state.settings.language);
  const [filter, setFilter] = useState("all");
  const read = new Set(state.readMessageIds);
  const messages = state.messages
    .filter(
      (message) =>
        message.role === "assistant" &&
        message.source === "proactive" &&
        message.completed,
    )
    .filter((message) => filter === "all" || !read.has(message.id))
    .toReversed();
  const locale = state.settings.language === "zh" ? "zh-CN" : "en";
  let previousDay = "";
  return (
    <SplitPane
      name="remy"
      collapsed={collapsed}
      language={state.settings.language}
      sidebar={
        <>
          <div className="sidebar-heading">
            <span className="eyebrow">REMY / INBOX</span>
            <h1>
              {t("inbox")}
              <span className="count-pill">{state.unread}</span>
            </h1>
            <p>{t("inboxCaption")}</p>
          </div>
          <div className="inbox-filters">
            {["all", "unread"].map((value) => (
              <button
                key={value}
                className={filter === value ? "active" : ""}
                aria-pressed={filter === value}
                onClick={() => setFilter(value)}
              >
                {t(value === "all" ? "allUpdates" : "unreadUpdates")}
              </button>
            ))}
          </div>
          <div className="inbox-list">
            {messages.length === 0 && (
              <div className="sidebar-empty">
                <Bell size={26} strokeWidth={1.2} />
                <h3>{t(filter === "unread" ? "allCaughtUp" : "noUpdates")}</h3>
                <p>{t("noUpdatesHint")}</p>
              </div>
            )}
            {messages.map((message) => {
              const date = message.createdAt
                ? new Date(message.createdAt)
                : null;
              const valid = date && !Number.isNaN(date.getTime());
              const day = valid
                ? date.toDateString() === new Date().toDateString()
                  ? t("today")
                  : date.toLocaleDateString(locale, {
                      month: "short",
                      day: "numeric",
                    })
                : t("earlier");
              const group = day !== previousDay;
              previousDay = day;
              return (
                <div key={message.id}>
                  {group && <div className="list-section-label">{day}</div>}
                  <button
                    className={`inbox-card ${!read.has(message.id) ? "unread" : ""} ${state.focusMessageId === message.id ? "selected" : ""}`}
                    onClick={() =>
                      void action({ type: "chat.open", messageId: message.id })
                    }
                  >
                    <div className="inbox-card-top">
                      <Sparkles size={14} />
                      <span>Remy</span>
                      <time>
                        {valid
                          ? date.toLocaleTimeString(locale, {
                              hour: "2-digit",
                              minute: "2-digit",
                            })
                          : ""}
                      </time>
                      {!read.has(message.id) && <i className="unread-dot" />}
                    </div>
                    <p>{message.content}</p>
                    <span className="inbox-card-link">
                      {t("conversation")}
                      <ArrowUpRight size={12} />
                    </span>
                  </button>
                </div>
              );
            })}
          </div>
          <div className="sidebar-footer">
            <span
              className={`connection-dot ${state.connection === "connected" ? "online" : ""}`}
            />
            {t(state.connection === "connected" ? "connected" : "reconnecting")}
          </div>
        </>
      }
    >
      <Chat
        state={state}
        active={state.activePage === "remy"}
        onSettings={onSettings}
      />
    </SplitPane>
  );
}
