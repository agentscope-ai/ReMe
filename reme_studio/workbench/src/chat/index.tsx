import {
  useCallback,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  deserializeError,
  errorText,
  translator,
} from "../../../client/src/i18n.ts";
import { useHost } from "../host.ts";
import { watchReadReceipt } from "../read-receipt.ts";
import MessageBody, {
  formatTaskResult,
  isCompletedTurn,
  splitReminderSource,
} from "../message-body.tsx";
import type { StudioState } from "../../../client/src/protocol.ts";
import { useCapture } from "../speech/use-capture.ts";
import { messageKey } from "../../../client/src/message-key.ts";
import { useChatScroll } from "./use-chat-scroll.ts";
import {
  Bell,
  BookOpenText,
  CalendarClock,
  Check,
  ChevronRight,
  FileText,
  Headphones,
  List,
  LoaderCircle,
  Radar,
  RotateCcw,
  Sparkle,
  Square,
} from "lucide-react";
import { WorkspaceLayout } from "../workspace-titlebar.tsx";

const speechIcons = {
  idle: Headphones,
  loading: LoaderCircle,
  playing: Square,
  error: RotateCcw,
};

export default function Chat({
  state,
  active,
  draft,
  onDraft,
  canClearContext,
  clearDialogOpen,
  onClearDialogChange,
}: {
  draft: string;
  onDraft: (value: string) => void;
  state: StudioState;
  active: boolean;
  canClearContext: boolean;
  clearDialogOpen: boolean;
  onClearDialogChange: (open: boolean) => void;
}) {
  const { action, openRecorder } = useHost();
  const mobile = Boolean(useContext(WorkspaceLayout)?.openNavigation);
  const language = state.settings.language;
  const t = translator(language);
  const isRemy = (state.selectedAgentName ?? "Remy") === "Remy";
  const composerId = useId();
  const clearDialog = useRef<HTMLDialogElement>(null);
  const [clearing, setClearing] = useState(false);
  const [commandHidden, setCommandHidden] = useState(false);
  const [commandIndex, setCommandIndex] = useState(0);
  const query = draft;
  const setQuery = (value: string | ((current: string) => string)) => {
    const next = typeof value === "function" ? value(queryRef.current) : value;
    queryRef.current = next;
    setCommandHidden(false);
    setCommandIndex(0);
    onDraft(next);
  };
  const agent = state.agents?.find(
    (item) => item.agent_name === state.selectedAgentName,
  );
  const agentName = agent?.agent_name ?? "Remy";
  const automaticEvent = agent?.agent_type === "event" && !state.pending;
  const [error, setError] = useState<unknown>(null);
  const {
    history,
    content,
    following,
    jumpToLatest,
    stopFollowing,
    reveal,
    prepend,
  } = useChatScroll(state.sessionId, active);
  const [historyWindow, setHistoryWindow] = useState({
    sessionId: state.sessionId,
    start: Math.max(0, state.messages.length - 80),
  });
  const historyStart =
    historyWindow.sessionId === state.sessionId &&
    historyWindow.start < state.messages.length
      ? historyWindow.start
      : Math.max(0, state.messages.length - 80);
  const workspace = state.clientInfo?.workspace_dir;
  const historyMessages = useMemo(
    () =>
      state.messages.slice(historyStart).map((message) => ({
        message,
        source: splitReminderSource(message, workspace),
      })),
    [state.messages, historyStart, workspace],
  );
  const openLink = useCallback((url: string) => {
    void action({ type: "link.open", url }).catch(setError);
  }, []);
  const input = useRef<HTMLTextAreaElement | null>(null);
  const queryRef = useRef(query);
  queryRef.current = query;
  const capture = useCapture(
    { ...state, chatVisible: state.chatVisible && active },
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
  const focusedMessage = useRef("");
  const sending = useRef(false);
  const messages = state.messages;
  const focusedId =
    state.focusTarget?.kind === "message" ? state.focusTarget.id : null;
  const readableReminders = state.messages.filter(
    (message) =>
      message.source === "proactive" &&
      message.completed &&
      message.content.trim(),
  );
  const proactiveSignature = readableReminders
    .map((message) => message.id)
    .join("\n");
  const readIds = useRef(new Set<string>());
  readIds.current = new Set(state.readMessageIds);
  const streaming = Boolean(state.pending?.streaming);
  const busy = clearing || streaming || (isRemy && state.status === "active");
  const ready =
    state.connection === "connected" &&
    state.status !== "error" &&
    (isRemy || Boolean(state.sessionId));
  const clearDisabled = clearing || !canClearContext;
  const commands = isRemy
    ? [
        {
          name: "/clear",
          label: t("clearContext"),
          run: () => onClearDialogChange(true),
        },
      ]
    : [];
  const commandQuery = query.trim().toLowerCase();
  const matchingCommands =
    active && !commandHidden && commandQuery.startsWith("/")
      ? commands.filter(({ name }) => name.startsWith(commandQuery))
      : [];
  const selectedCommand = matchingCommands[commandIndex];
  const submitCommand =
    selectedCommand ?? commands.find(({ name }) => name === commandQuery);
  const commandId = (name: string) => `${composerId}-command-${name.slice(1)}`;
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

  useEffect(() => {
    if (!active || !state.chatVisible || !focusedId) return;
    const request = `${state.sessionId}:${state.focusRequest}:${focusedId}`;
    if (focusedMessage.current === request) return;
    const index = messages.findIndex((message) => message.id === focusedId);
    if (index >= 0 && index < historyStart) {
      stopFollowing();
      setHistoryWindow({
        sessionId: state.sessionId,
        start: Math.max(0, index - 10),
      });
      return;
    }
    const element = [
      ...(history.current?.querySelectorAll<HTMLElement>("[data-message-id]") ??
        []),
    ].find((node) => node.dataset.messageId === focusedId);
    if (element) {
      reveal(element);
      focusedMessage.current = request;
    }
  }, [
    focusedId,
    historyStart,
    state.focusRequest,
    state.chatVisible,
    state.messages,
    active,
    reveal,
    stopFollowing,
  ]);
  useLayoutEffect(() => {
    setHistoryWindow({
      sessionId: state.sessionId,
      start: Math.max(0, state.messages.length - 80),
    });
  }, [state.sessionId]);
  useLayoutEffect(() => {
    if (!state.pending?.streaming) return;
    jumpToLatest();
  }, [state.pending?.id, jumpToLatest]);
  useEffect(() => {
    if (following && state.messages.length - historyStart > 120)
      setHistoryWindow({
        sessionId: state.sessionId,
        start: state.messages.length - 80,
      });
  }, [state.messages.length, state.sessionId, historyStart, following]);
  const focusSession = isRemy ? state.sessionId : null;
  useEffect(() => {
    if (state.chatVisible && active && !clearing) input.current?.focus();
  }, [state.chatVisible, active, focusSession, clearing]);
  useEffect(() => {
    if (active && isRemy && clearDialogOpen) clearDialog.current?.showModal();
  }, [active, isRemy, clearDialogOpen]);

  useEffect(() => {
    const root = history.current;
    if (
      !active ||
      !root ||
      !state.sessionId ||
      !state.chatVisible ||
      !state.chatFocused
    )
      return;
    const readable = new Set(readableReminders.map((message) => message.id));
    const visible = new Set<string>();
    const receipt = watchReadReceipt(async () => {
      const ids = [...visible].filter((id) => !readIds.current.has(id));
      if (!ids.length) return false;
      await action({
        type: "messages.read",
        sessionId: state.sessionId!,
        ids,
      });
      return true;
    });
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const id = (entry.target as HTMLElement).dataset.readAnchor!;
          if (entry.isIntersecting) visible.add(id);
          else visible.delete(id);
        }
        receipt.schedule();
      },
      { root, threshold: 0.3 },
    );
    root
      .querySelectorAll<HTMLElement>("[data-read-anchor]")
      .forEach((element) => {
        if (readable.has(element.dataset.readAnchor!))
          observer.observe(element);
      });
    return () => {
      receipt.close();
      observer.disconnect();
    };
  }, [
    state.sessionId,
    proactiveSignature,
    state.chatVisible,
    state.chatFocused,
    active,
    historyStart,
  ]);

  const send = async (text = queryRef.current) => {
    const value = text.trim();
    if (!value || busy || !ready || sending.current) return;
    sending.current = true;
    setQuery("");
    jumpToLatest();
    if (!(await run({ type: "chat.send", query: value })))
      setQuery((current) => current || value);
    sending.current = false;
  };

  const chooseCommand = (command: (typeof commands)[number]) => {
    if (!active) return;
    setCommandHidden(true);
    command.run();
  };
  const submit = () => {
    if (submitCommand) chooseCommand(submitCommand);
    else void send();
  };
  const clearContext = async () => {
    if (!active || !isRemy || clearDisabled || sending.current) return;
    clearDialog.current?.close();
    const value = queryRef.current;
    sending.current = true;
    setClearing(true);
    setQuery("");
    if (!(await run({ type: "session.new" })) && !queryRef.current) {
      setQuery(value);
      setCommandHidden(true);
    }
    sending.current = false;
    setClearing(false);
  };

  const home = isRemy && messages.length === 0;
  const quickPrompts = home && (
    <div className="conversation-prompts">
      {(
        [
          ["promptScheduledQuery", "quickSchedule", CalendarClock],
          ["promptTemporaryQuery", "quickReminder", Bell],
          ["promptProactiveQuery", "quickWatch", Radar],
        ] as const
      ).map(([query, label, Icon]) => (
        <button
          key={query}
          type="button"
          title={t(query)}
          onClick={() => {
            setQuery(t(query));
            input.current?.focus();
          }}
        >
          <Icon size={15} strokeWidth={1.75} aria-hidden />
          <span>{t(label)}</span>
        </button>
      ))}
    </div>
  );
  return (
    <main className="conversation-pane" data-home={home}>
      {state.connectionError && (
        <div className="connection-banner" role="status">
          <span>{state.connectionError}</span>
          <button
            onClick={() => void run({ type: "page.select", page: "settings" })}
          >
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
      {home && (
        <div className="empty-chat">
          <h2>{t("welcomeTitle")}</h2>
          <p>{t("welcomeHint")}</p>
          {state.connection !== "connected" && (
            <p className="start-hint">
              {t("startService")}
              <code>reme start config=proactive</code>
            </p>
          )}
        </div>
      )}
      <div
        hidden={home}
        ref={history}
        className="chat-history"
        aria-label={t("history")}
      >
        <div ref={content} className="chat-content">
          {historyStart > 0 && (
            <button
              type="button"
              className="load-earlier"
              onClick={() => {
                prepend(() =>
                  setHistoryWindow({
                    sessionId: state.sessionId,
                    start: Math.max(0, historyStart - 80),
                  }),
                );
              }}
            >
              {t("loadEarlierMessages")}
            </button>
          )}
          {historyMessages.map(({ message, source }) => {
            const proactive = message.source === "proactive";
            const reminder = proactive && message.role === "assistant";
            const completed = isCompletedTurn(message);
            const result = automaticEvent
              ? formatTaskResult(message, language)
              : undefined;
            const canAct =
              message.role === "assistant" &&
              message.completed &&
              Boolean(message.content.trim()) &&
              !result &&
              !message.error &&
              message.status?.tone !== "error";
            const playback = voice?.messageId === message.id ? voice : null;
            const SpeechIcon = speechIcons[playback?.phase ?? "idle"];
            const speakLabel = playback
              ? playback.phase === "error"
                ? "speechRetry"
                : "stopSpeech"
              : "readAloud";
            const speechHint = t(
              state.speech.tts ? speakLabel : "speechTtsUnavailable",
            );
            const body = (
              <MessageBody
                message={source?.display ?? result ?? message}
                language={language}
                showCompletion={false}
                onLink={openLink}
              />
            );
            return (
              <article
                key={messageKey(message)}
                data-message-id={message.id}
                data-proactive={proactive}
                data-task-input={message.taskInput}
                className={`message ${message.role} ${focusedId === message.id ? "message-focused" : ""}`}
              >
                {proactive && (
                  <span
                    className="message-read-anchor"
                    data-read-anchor={message.id}
                    aria-hidden
                  />
                )}
                {proactive && message.role === "user" && (
                  <div className="proactive-label">
                    <Sparkle size={15} aria-hidden />
                    {t("automaticTurn")}
                  </div>
                )}
                {message.taskInput ? (
                  <details className="task-input">
                    <summary>
                      <List size={15} aria-hidden />
                      <span>{t("taskInput")}</span>
                      <ChevronRight
                        size={13}
                        className="activity-chevron"
                        aria-hidden
                      />
                    </summary>
                    {body}
                  </details>
                ) : (
                  body
                )}
                {(reminder || completed || canAct) && (
                  <footer className="message-footer">
                    {(reminder || completed) && (
                      <span
                        className="message-footer-status"
                        title={reminder ? t("proactiveLabel") : undefined}
                      >
                        {reminder ? (
                          <Sparkle size={15} aria-hidden />
                        ) : (
                          <Check size={15} aria-hidden />
                        )}
                        {t(reminder ? "proactiveStatus" : "completed")}
                      </span>
                    )}
                    {source && (
                      <a
                        className="message-source-link"
                        href={`#reme-file=${encodeURIComponent(source.path)}`}
                        aria-label={t("openReminderSource", {
                          path: source.path,
                        })}
                        title={t("openReminderSource", { path: source.path })}
                        onClick={(event) => {
                          event.preventDefault();
                          openLink(event.currentTarget.hash);
                        }}
                      >
                        <FileText size={14} aria-hidden />
                      </a>
                    )}
                    {canAct && (
                      <div className="message-actions">
                        <button
                          type="button"
                          disabled={
                            !state.speech.tts || !state.speech.ready || !ready
                          }
                          aria-label={t(speakLabel)}
                          data-tooltip={speechHint}
                          aria-pressed={Boolean(playback)}
                          onClick={() =>
                            void run(
                              playback
                                ? {
                                    type:
                                      playback.phase === "error"
                                        ? "speech.retry"
                                        : "speech.stop",
                                  }
                                : {
                                    type: "speech.speak",
                                    sessionId: state.sessionId!,
                                    messageId: message.id,
                                  },
                            )
                          }
                        >
                          <SpeechIcon
                            size={15}
                            aria-hidden
                            className={
                              playback?.phase === "loading"
                                ? "activity-spinner"
                                : undefined
                            }
                          />
                        </button>
                        {reminder && (
                          <button
                            type="button"
                            aria-label={t("explainReminder")}
                            data-tooltip={t("explainReminder")}
                            disabled={busy || !ready}
                            onClick={() =>
                              void run({
                                type: "item.action",
                                action: "explain",
                                target: { kind: "message", id: message.id },
                              })
                            }
                          >
                            <BookOpenText size={15} aria-hidden />
                          </button>
                        )}
                      </div>
                    )}
                  </footer>
                )}
              </article>
            );
          })}
          {state.taskRunStatus && (
            <div
              className={`task-run-status message-status ${state.taskRunStatus.tone}`}
              role="status"
            >
              {state.taskRunStatus.tone === "working" && (
                <LoaderCircle
                  size={15}
                  className="activity-spinner"
                  aria-hidden
                />
              )}
              {state.taskRunStatus.text}
            </div>
          )}
        </div>
        {!following && messages.length > 0 && (
          <button
            type="button"
            className="jump-to-latest"
            onClick={() => jumpToLatest(mobile ? "smooth" : "instant")}
          >
            {t("jumpToLatest")}
          </button>
        )}
      </div>
      {active && isRemy && (
        <dialog
          ref={clearDialog}
          className="workspace-dialog"
          aria-labelledby={`${composerId}-clear-title`}
          aria-describedby={`${composerId}-clear-description`}
          onClose={() => onClearDialogChange(false)}
        >
          <h2 id={`${composerId}-clear-title`}>{t("clearContextTitle")}</h2>
          <p id={`${composerId}-clear-description`}>
            {t("clearConversationHint")}
          </p>
          {clearDisabled && <p role="status">{t("sessionBusy")}</p>}
          <footer>
            <button
              type="button"
              className="secondary"
              onClick={() => clearDialog.current?.close()}
            >
              {t("cancel")}
            </button>
            <button
              type="button"
              className="primary"
              disabled={clearDisabled}
              onClick={() => void clearContext()}
            >
              {t("newConversation")}
            </button>
          </footer>
        </dialog>
      )}
      {quickPrompts}
      <footer
        className={`chat-composer${home && !mobile ? " welcome-composer" : ""}`}
      >
        {capture.notice && (
          <div className="voice-notice" role="status">
            {t("speechNoInput")}
          </div>
        )}
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
                      : voice!.phase === "error"
                        ? "speechPaused"
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
                {t(
                  voice.phase === "loading" && voice.preparing
                    ? "speechPreparingProgress"
                    : "speechProgress",
                  {
                    part:
                      voice.phase === "loading"
                        ? (voice.preparing ?? voice.part)
                        : voice.part,
                    total: voice.total,
                  },
                )}
              </small>
            )}
            {voice?.phase === "error" && (
              <button
                type="button"
                className="subtle"
                onClick={() => void run({ type: "speech.retry" })}
              >
                {t("speechRetry")}
              </button>
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
            submit();
          }}
        >
          {matchingCommands.length > 0 && (
            <div
              className="chat-command-menu"
              id={`${composerId}-commands`}
              role="listbox"
              aria-label={t("chatActions")}
            >
              {matchingCommands.map((command) => (
                <button
                  key={command.name}
                  type="button"
                  id={commandId(command.name)}
                  role="option"
                  aria-selected={command === selectedCommand}
                  tabIndex={-1}
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => chooseCommand(command)}
                >
                  <strong>{command.name}</strong>
                  <span>{command.label}</span>
                  {command === selectedCommand && <kbd aria-hidden>↵</kbd>}
                </button>
              ))}
            </div>
          )}
          <textarea
            ref={input}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onPointerDown={() => void action({ type: "interaction" })}
            placeholder={
              busy
                ? t("draftWhileBusy")
                : isRemy
                  ? t("agentMessagePlaceholder", { name: agentName })
                  : t("taskQueryPlaceholder")
            }
            aria-label={t("message")}
            aria-controls={selectedCommand && `${composerId}-commands`}
            aria-activedescendant={
              selectedCommand && commandId(selectedCommand.name)
            }
            aria-autocomplete={isRemy ? "list" : undefined}
            rows={2}
            maxLength={100000}
            onKeyDown={(event) => {
              void action({ type: "interaction" });
              if (
                event.nativeEvent.isComposing ||
                event.nativeEvent.keyCode === 229
              )
                return;
              if (selectedCommand && event.key === "Escape") {
                event.preventDefault();
                setCommandHidden(true);
                return;
              }
              if (selectedCommand && event.key === "Tab" && !event.shiftKey) {
                event.preventDefault();
                setQuery(selectedCommand.name);
                setCommandHidden(true);
                return;
              }
              if (
                selectedCommand &&
                (event.key === "ArrowDown" || event.key === "ArrowUp")
              ) {
                event.preventDefault();
                const count = matchingCommands.length;
                const direction = event.key === "ArrowDown" ? 1 : -1;
                setCommandIndex(
                  (current) => (current + direction + count) % count,
                );
                return;
              }
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
          />
          <button
            type="button"
            className={`mic-button ${capture.phase === "recording" ? "recording" : ""}`}
            aria-label={t(
              capture.phase === "recording"
                ? "finishRecording"
                : openRecorder
                  ? "startRecording"
                  : "holdToSpeak",
            )}
            aria-pressed={capture.phase === "recording"}
            title={t(
              !state.speech.asr
                ? "speechAsrUnavailable"
                : openRecorder
                  ? "startRecording"
                  : "voiceShortcut",
            )}
            disabled={
              !ready ||
              !state.speech.ready ||
              !state.speech.asr ||
              capture.phase === "transcribing"
            }
            onPointerDown={(event) => {
              if (openRecorder || event.button !== 0) return;
              event.preventDefault();
              event.currentTarget.setPointerCapture(event.pointerId);
              capture.start();
            }}
            onPointerUp={openRecorder ? undefined : capture.finish}
            onPointerCancel={openRecorder ? undefined : capture.cancel}
            onLostPointerCapture={openRecorder ? undefined : capture.finish}
            onClick={(event) => {
              if (openRecorder || event.detail === 0) capture.toggle();
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
              aria-label={submitCommand?.label ?? t("sendMessage")}
              disabled={!submitCommand && (busy || !ready || !query.trim())}
            >
              ↑
            </button>
          )}
        </form>
        <div className="composer-meta">
          <span className="composer-location">
            <span
              className={`connection-dot ${state.connection === "connected" ? "online" : ""}`}
              aria-hidden
            />
            {state.connection === "connected"
              ? t("memoryLocation")
              : t("waitingService")}
          </span>
          <span className="keyboard-hint">
            {state.speech.asr && (
              <ShortcutHint template={t("voiceShortcutHint")} keys="Ctrl + R" />
            )}
            <ShortcutHint template={t("newlineHint")} keys="Shift + Enter" />
          </span>
        </div>
      </footer>
    </main>
  );
}

function ShortcutHint({ template, keys }: { template: string; keys: string }) {
  const [before, after] = template.split("{shortcut}");
  return (
    <span>
      {before}
      <kbd>{keys}</kbd>
      {after}
    </span>
  );
}
