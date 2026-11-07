import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import { deserializeError, errorText, translator } from "../../shared/i18n.ts";
import remarkGfm from "remark-gfm";
import { action } from "../desktop.ts";
import { displayMessages } from "../../shared/messages.ts";
import type { DesktopState } from "../../shared/protocol.ts";
import { useSpeechPlayback } from "../speech/playback.ts";
import { useCapture } from "../speech/use-capture.ts";
import VoiceSettings from "../speech/settings.tsx";

export default function Chat({ state }: { state: DesktopState }) {
  const language = state.settings.language;
  const t = translator(language);
  const [query, setQuery] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [endpoint, setEndpoint] = useState(state.settings.endpoint);
  const [scale, setScale] = useState(state.settings.scale);
  const history = useRef<HTMLDivElement | null>(null);
  const input = useRef<HTMLTextAreaElement | null>(null);
  const queryRef = useRef(query);
  queryRef.current = query;
  useSpeechPlayback(state);
  const capture = useCapture(
    state,
    (text) => {
      const value = queryRef.current;
      const from = input.current?.selectionStart ?? value.length;
      const to = input.current?.selectionEnd ?? from;
      const composed = value.slice(0, from) + text + value.slice(to);
      queryRef.current = composed;
      setQuery(composed);
      if (state.settings.voiceAutoSend) void send(composed);
      requestAnimationFrame(() => {
        input.current?.focus();
        input.current?.setSelectionRange(
          from + text.length,
          from + text.length,
        );
      });
    },
    setError,
  );
  const voice = state.speech.playback;
  const displayedError =
    error ?? (state.speech.error ? deserializeError(state.speech.error) : null);
  const follow = useRef(true);
  const sending = useRef(false);
  const messages = displayMessages(state);
  const proactiveSignature = state.messages
    .filter((message) => message.source === "proactive")
    .map((message) => `${message.id}:${message.completed}`)
    .join("\n");
  const busy = Boolean(state.pending?.streaming) || state.status === "active";
  const ready = state.connection === "connected" && state.status !== "error";
  const run = async (value: Parameters<typeof action>[0]) => {
    setError(null);
    try {
      await action(value);
      return true;
    } catch (error) {
      setError(error);
      return false;
    }
  };

  useEffect(
    () => setEndpoint(state.settings.endpoint),
    [state.settings.endpoint],
  );
  useEffect(() => setScale(state.settings.scale), [state.settings.scale]);
  useEffect(
    () =>
      window.desktop.onCommand((command) => {
        if (command.type === "settings.open") setSettingsOpen(true);
      }),
    [],
  );
  useEffect(() => {
    if (!state.chatVisible || !state.focusMessageId) return;
    const element = [
      ...(history.current?.querySelectorAll<HTMLElement>("[data-message-id]") ??
        []),
    ].find((node) => node.dataset.messageId === state.focusMessageId);
    element?.scrollIntoView({ block: "center" });
    follow.current = false;
  }, [state.focusMessageId, state.chatVisible]);
  useEffect(() => {
    if (follow.current && history.current)
      history.current.scrollTop = history.current.scrollHeight;
  }, [state.messages, state.pending?.text, state.chatVisible]);
  useEffect(() => {
    if (state.chatVisible && !settingsOpen) input.current?.focus();
  }, [state.chatVisible, settingsOpen]);

  useEffect(() => {
    const root = history.current;
    if (!root || !state.sessionId || !state.chatVisible || !state.chatFocused)
      return;
    const visible = new Set<string>();
    const acknowledged = new Set<string>();
    let timer: ReturnType<typeof setTimeout>;
    const acknowledge = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (!document.hasFocus() || document.visibilityState !== "visible")
          return;
        const ids = [...visible].filter((id) => !acknowledged.has(id));
        if (ids.length) {
          ids.forEach((id) => acknowledged.add(id));
          void action({
            type: "messages.read",
            sessionId: state.sessionId!,
            ids,
          }).catch(() => ids.forEach((id) => acknowledged.delete(id)));
        }
      }, 600);
    };
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const id = (entry.target as HTMLElement).dataset.messageId!;
          if (entry.isIntersecting) visible.add(id);
          else visible.delete(id);
        }
        acknowledge();
      },
      { root, threshold: 0.3 },
    );
    root
      .querySelectorAll<HTMLElement>("[data-proactive=true] .proactive-label")
      .forEach((element) => observer.observe(element));
    window.addEventListener("focus", acknowledge);
    return () => {
      clearTimeout(timer);
      observer.disconnect();
      window.removeEventListener("focus", acknowledge);
    };
  }, [
    state.sessionId,
    proactiveSignature,
    state.chatVisible,
    state.chatFocused,
  ]);

  const send = async (text = queryRef.current) => {
    const value = text.trim();
    if (!value || busy || !ready || sending.current) return;
    sending.current = true;
    setQuery("");
    follow.current = true;
    if (!(await run({ type: "chat.send", query: value })))
      setQuery((current) => current || value);
    sending.current = false;
  };

  return (
    <main className="chat-shell">
      <header className="chat-titlebar">
        <div className="brand-mark" aria-hidden>
          R<span>·</span>
        </div>
        <div className="chat-heading">
          <h1>Remy</h1>
          <span>
            {state.connection !== "connected"
              ? t("waitingConnection")
              : state.behavior.label}
          </span>
        </div>
        <nav aria-label={t("chatActions")}>
          <button
            className="icon-button"
            aria-label={t("newConversation")}
            disabled={busy || !state.workspace}
            onClick={() => {
              follow.current = true;
              void run({ type: "session.new" });
            }}
          >
            ＋
          </button>
          <button
            className={`icon-button ${settingsOpen ? "selected" : ""}`}
            aria-label={t("settings")}
            aria-pressed={settingsOpen}
            onClick={() => setSettingsOpen(!settingsOpen)}
          >
            ⚙
          </button>
          <button
            className="icon-button"
            aria-label={t("closeChat")}
            onClick={() => void run({ type: "chat.close" })}
          >
            ×
          </button>
        </nav>
      </header>
      {settingsOpen && (
        <section className="settings" aria-label={t("desktopSettings")}>
          <label>
            {t("endpoint")}
            <div className="endpoint-row">
              <input
                value={endpoint}
                onChange={(event) => setEndpoint(event.target.value)}
                spellCheck={false}
                aria-label={t("endpoint")}
              />
              <button
                className="primary"
                disabled={Boolean(state.pending?.streaming)}
                onClick={() => void run({ type: "settings.update", endpoint })}
              >
                {t("connect")}
              </button>
            </div>
          </label>
          <label className="scale-row">
            {t("characterSize")} <span>{Math.round(scale * 100)}%</span>
            <input
              type="range"
              aria-label={t("characterSize")}
              min="0.7"
              max="1.5"
              step="0.05"
              value={scale}
              onChange={(event) => setScale(Number(event.target.value))}
              onPointerUp={() => void run({ type: "settings.update", scale })}
              onKeyUp={() => void run({ type: "settings.update", scale })}
            />
          </label>
          <div className="quiet-row">
            <span>{t("reminder")}</span>
            <button
              className="secondary"
              onClick={() =>
                void run({
                  type: "settings.update",
                  quietUntil:
                    state.settings.quietUntil > Date.now()
                      ? 0
                      : Date.now() + 30 * 60_000,
                })
              }
            >
              {state.settings.quietUntil > Date.now()
                ? t("resume")
                : t("quiet")}
            </button>
          </div>
          <label className="quiet-row">
            <span>
              {t("notifications")} <small>{t("notificationHint")}</small>
            </span>
            <input
              type="checkbox"
              aria-label={t("notifications")}
              checked={state.settings.notifications}
              onChange={(event) =>
                void run({
                  type: "settings.update",
                  notifications: event.target.checked,
                })
              }
            />
          </label>
          <VoiceSettings state={state} run={run} />
          {state.workspace && (
            <p className="workspace-path" title={state.workspace}>
              {t("workspace", { path: state.workspace })}
            </p>
          )}
          <p className="settings-note">{t("shortcutHint")}</p>
        </section>
      )}
      {state.connectionError && (
        <div className="connection-banner" role="status">
          <span>{state.connectionError}</span>
          <button onClick={() => setSettingsOpen(true)}>
            {t("connectionSettings")}
          </button>
        </div>
      )}
      {displayedError !== null && (
        <div className="action-error" role="alert">
          <span>{errorText(displayedError, language)}</span>
          <button
            aria-label={t("closeNotice")}
            onClick={() => {
              setError(null);
              void action({ type: "speech.clearError" });
            }}
          >
            ×
          </button>
        </div>
      )}
      <div
        ref={history}
        className="chat-history"
        aria-label={t("history")}
        onScroll={() => {
          if (history.current)
            follow.current =
              history.current.scrollHeight -
                history.current.scrollTop -
                history.current.clientHeight <
              48;
        }}
      >
        {messages.length === 0 && (
          <div className="empty-chat">
            <div className="empty-mark" aria-hidden>
              R<span>·</span>
            </div>
            <h2>{t("emptyTitle")}</h2>
            <p>{t("emptyDescription")}</p>
            {state.connection !== "connected" && (
              <p className="start-hint">
                {t("startService")}
                <code>reme start config=proactive</code>
              </p>
            )}
          </div>
        )}
        {messages.map((message) => (
          <article
            key={message.id}
            data-message-id={message.id}
            data-proactive={Boolean(message.proactive)}
            className={`message ${message.role}`}
          >
            {message.proactive && (
              <div className="proactive-label" data-message-id={message.id}>
                <span aria-hidden>✦</span> {t("proactiveLabel")}
              </div>
            )}
            <div className="message-body">
              {message.role === "user" ? (
                <p>{message.content}</p>
              ) : message.content ? (
                <Markdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    a: ({ href, children }) => (
                      <a
                        href={href}
                        onClick={(event) => {
                          event.preventDefault();
                          if (href) void run({ type: "link.open", url: href });
                        }}
                      >
                        {children}
                      </a>
                    ),
                    img: ({ alt }) => (
                      <span className="image-placeholder">
                        {alt || t("image")}
                      </span>
                    ),
                  }}
                >
                  {message.content}
                </Markdown>
              ) : message.streaming ? (
                <span className="thinking">
                  {t("thinkingPlaceholder")}
                  <span aria-hidden> · · ·</span>
                </span>
              ) : null}
            </div>
            {message.error && <p className="message-error">{message.error}</p>}
            {message.role === "assistant" &&
              !message.streaming &&
              !message.error &&
              state.messages.some(
                (entry) => entry.id === message.id && entry.completed,
              ) && (
                <button
                  type="button"
                  className="read-aloud"
                  disabled={!state.speech.tts || !ready}
                  aria-label={t(
                    voice?.messageId === message.id
                      ? "stopSpeech"
                      : "readAloud",
                  )}
                  title={t(
                    !state.speech.tts
                      ? "speechTtsUnavailable"
                      : voice?.messageId === message.id
                        ? "stopSpeech"
                        : "readAloud",
                  )}
                  aria-pressed={voice?.messageId === message.id}
                  onClick={() =>
                    void run(
                      voice?.messageId === message.id
                        ? { type: "speech.stop" }
                        : {
                            type: "speech.speak",
                            sessionId: state.sessionId!,
                            messageId: message.id,
                          },
                    )
                  }
                >
                  <span aria-hidden>
                    {voice?.messageId === message.id ? "■" : "♫"}
                  </span>
                  {t(
                    voice?.messageId === message.id
                      ? voice.phase === "loading"
                        ? "speechLoading"
                        : "stopSpeech"
                      : "readAloud",
                  )}
                </button>
              )}
          </article>
        ))}
      </div>
      <footer className="chat-composer">
        {(capture.phase || voice) && (
          <div className="voice-status" role="status">
            <span>
              {capture.phase
                ? t(
                    capture.phase === "starting"
                      ? "speechOpeningMic"
                      : capture.phase === "transcribing"
                        ? "speechTranscribing"
                        : "speechRecording",
                    { seconds: capture.seconds },
                  )
                : t(
                    voice!.phase === "loading"
                      ? "speechLoading"
                      : "speechPlaying",
                  )}
            </span>
            {capture.phase === "recording" && (
              <meter
                min={0}
                max={1}
                value={capture.level}
                aria-label={t("speechListening")}
              />
            )}
            {voice && voice.total > 1 && (
              <small>
                {t("speechProgress", { part: voice.part, total: voice.total })}
              </small>
            )}
            <button
              type="button"
              className="subtle"
              aria-label={t(capture.phase ? "cancelRecording" : "stopSpeech")}
              onClick={() =>
                capture.phase
                  ? capture.cancel()
                  : void run({ type: "speech.stop" })
              }
            >
              ×
            </button>
          </div>
        )}
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void send();
          }}
        >
          <textarea
            ref={input}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onPointerDown={() => void action({ type: "interaction" })}
            placeholder={busy ? t("draftWhileBusy") : t("messagePlaceholder")}
            aria-label={t("message")}
            rows={2}
            maxLength={100000}
            onKeyDown={(event) => {
              void action({ type: "interaction" });
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing &&
                event.nativeEvent.keyCode !== 229
              ) {
                event.preventDefault();
                void send();
              }
            }}
          />
          <button
            type="button"
            className={`mic-button ${capture.phase === "recording" ? "recording" : ""}`}
            aria-label={t(
              capture.phase === "recording" ? "finishRecording" : "holdToSpeak",
            )}
            aria-pressed={capture.phase === "recording"}
            title={t(
              state.speech.asr ? "voiceShortcut" : "speechAsrUnavailable",
            )}
            disabled={
              !ready || !state.speech.asr || capture.phase === "transcribing"
            }
            onPointerDown={(event) => {
              if (event.button !== 0) return;
              event.preventDefault();
              event.currentTarget.setPointerCapture(event.pointerId);
              capture.start();
            }}
            onPointerUp={() => capture.finish()}
            onPointerCancel={() => capture.cancel()}
            onLostPointerCapture={() => capture.finish()}
            onClick={(event) => {
              if (event.detail === 0) capture.toggle();
            }}
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              aria-hidden="true"
            >
              <rect x="9" y="2" width="6" height="12" rx="3" />
              <path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8" />
            </svg>
          </button>
          {state.pending?.streaming ? (
            <button
              type="button"
              className="stop-button"
              aria-label={t("stopReply")}
              onClick={() => void run({ type: "chat.stop" })}
            >
              ■
            </button>
          ) : (
            <button
              type="submit"
              className="send-button"
              aria-label={t("sendMessage")}
              disabled={busy || !ready || !query.trim()}
            >
              ↑
            </button>
          )}
        </form>
        <div className="composer-meta">
          <span
            className={`connection-dot ${state.connection === "connected" ? "online" : ""}`}
          />
          <span>
            {state.connection === "connected"
              ? t("memoryLocation")
              : t("waitingService")}
          </span>
          <span className="keyboard-hint">{t("newlineHint")}</span>
        </div>
      </footer>
    </main>
  );
}
