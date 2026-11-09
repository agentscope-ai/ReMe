import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Bootstrap } from "../../shared/protocol.ts";
import { Character, hitCharacter } from "../animation/player.tsx";
import { action } from "../desktop.ts";
import { translator } from "../../shared/i18n.ts";
import { ArrowUpRight, Sparkles, X, Volume2, Square } from "lucide-react";
import MessageBody from "../message-body.tsx";
import { surfaceKey } from "../../shared/surface.ts";

export default function Companion({ state, manifest, layout }: Bootstrap) {
  const active = layout.visible;
  const language = state.settings.language;
  const t = translator(language);
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const bubbleElement = useRef<HTMLElement | null>(null);
  const gesture = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const [error, setError] = useState<unknown>(null);
  const animation = state.behavior.animation;
  const dragging = animation.clip === "dragged";
  const surface = active ? state.surface : null;
  const bubble = surface?.kind === "message" ? surface : null;
  const message = bubble?.message;
  const reminder = message?.source === "proactive";
  const measurementKey = surfaceKey(surface);
  const run = (value: Parameters<typeof action>[0]) => {
    void action(value).catch((error: unknown) => setError(error));
  };

  const cancelDrag = () => {
    if (gesture.current) window.desktop.drag("end");
    gesture.current = null;
  };

  useLayoutEffect(() => {
    const element = bubbleElement.current;
    if (!element || !measurementKey) return;
    const measure = () => {
      const { width, height } = element.getBoundingClientRect();
      if (width && height)
        window.desktop.measureSurface({ key: measurementKey, width, height });
    };
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    measure();
    return () => observer.disconnect();
  }, [measurementKey]);

  useEffect(() => {
    if (!reminder || !bubble || !state.sessionId) return;
    let second = 0;
    const first = requestAnimationFrame(() => {
      second = requestAnimationFrame(() => {
        if (
          document.hidden ||
          !bubbleElement.current?.getBoundingClientRect().width
        )
          return;
        run({
          type: "bubble.shown",
          sessionId: state.sessionId!,
          id: message!.id,
          presentationId: bubble!.presentationId,
        });
      });
    });
    return () => {
      cancelAnimationFrame(first);
      cancelAnimationFrame(second);
    };
  }, [reminder, bubble?.presentationId, state.sessionId]);

  useEffect(() => {
    if (!active) return;
    let last = false;
    const move = (event: PointerEvent) => {
      const target = event.target as HTMLElement | null;
      const hit = Boolean(
        gesture.current ||
        target?.closest("[data-interactive]") ||
        hitCharacter(canvas.current, event.clientX, event.clientY),
      );
      if (hit !== last) {
        last = hit;
        window.desktop.pointer(hit);
      }
    };
    const leave = (event: PointerEvent) => {
      if (!event.relatedTarget && !gesture.current) {
        last = false;
        window.desktop.pointer(false);
      }
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerout", leave);
    return () => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerout", leave);
    };
  }, [active]);

  return (
    <main
      className="companion"
      aria-label={t("companion")}
      style={{
        display: active ? undefined : "none",
      }}
    >
      {bubble && message && (
        <section
          ref={bubbleElement}
          className="message-bubble"
          data-message-id={message.id}
          data-source={message.source}
          data-interactive
          style={{
            left: layout.bubble.left + layout.bubble.width / 2,
            top: layout.bubble.top,
            maxWidth: layout.bubble.maxWidth,
          }}
          aria-label={t(reminder ? "reminder" : "reply")}
          onPointerEnter={() =>
            run({ type: "bubble.pause", id: message.id, paused: true })
          }
          onPointerLeave={() =>
            run({ type: "bubble.pause", id: message.id, paused: false })
          }
        >
          <header>
            <span className="bubble-heading">
              <Sparkles size={13} aria-hidden />
              {t(reminder ? "reminderHeading" : "replyHeading")}
            </span>
            <button
              aria-label={t("dismissBubble")}
              className="icon-button"
              onClick={() => run({ type: "bubble.dismiss", id: message.id })}
            >
              <X size={13} />
            </button>
          </header>
          <MessageBody
            message={message}
            language={language}
            compact
            onLink={(url) => run({ type: "link.open", url })}
          />
          <footer>
            {state.speech.tts &&
              message.completed &&
              message.content &&
              !message.error &&
              message.status?.tone !== "error" && (
                <button
                  className="subtle"
                  disabled={!state.speech.ready}
                  aria-label={t(
                    state.speech.playback?.messageId === message.id
                      ? state.speech.playback.phase === "error"
                        ? "speechRetry"
                        : "stopSpeech"
                      : "readAloud",
                  )}
                  onClick={() =>
                    run(
                      state.speech.playback?.messageId === message.id
                        ? {
                            type:
                              state.speech.playback.phase === "error"
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
                  {state.speech.playback?.messageId === message.id &&
                  state.speech.playback.phase !== "error" ? (
                    <Square size={13} />
                  ) : (
                    <Volume2 size={14} />
                  )}
                </button>
              )}
            <button
              className="primary"
              onClick={() => run({ type: "chat.open", messageId: message.id })}
            >
              {t("expandReminder")} <ArrowUpRight size={13} aria-hidden />
            </button>
            {reminder && (
              <button
                className="subtle"
                onClick={() => run({ type: "bubble.snooze", id: message.id })}
              >
                {t("snoozeReminder")}
              </button>
            )}
          </footer>
        </section>
      )}
      {surface?.kind === "status" && (
        <button
          ref={bubbleElement as React.RefObject<HTMLButtonElement | null>}
          className={`status-notice ${surface.tone}`}
          data-interactive
          aria-label={t("statusOpen", { text: surface.text })}
          onClick={() =>
            run({ type: "chat.open", messageId: surface.messageId })
          }
          style={{
            left: layout.bubble.left + layout.bubble.width / 2,
            top: layout.bubble.top,
            maxWidth: layout.bubble.maxWidth,
          }}
        >
          <span className="status-light" aria-hidden />
          <span className="status-text" role="status" aria-live="polite">
            {surface.text}
          </span>
          <ArrowUpRight className="status-arrow" size={13} aria-hidden />
        </button>
      )}
      <button
        className={`character ${dragging ? "dragging" : ""} ${animation.clip === "happy" ? "reacting" : ""}`}
        aria-label={t("characterAction")}
        title={t("characterHint")}
        style={{
          left: layout.body.left,
          top: layout.body.top,
          width: layout.body.size,
          height: layout.body.size,
        }}
        onClick={(event) => {
          if (event.detail === 0) run({ type: "chat.open" });
        }}
        onContextMenu={(event) => {
          event.preventDefault();
          cancelDrag();
          if (hitCharacter(canvas.current, event.clientX, event.clientY))
            run({ type: "companion.menu" });
        }}
        onPointerDown={(event) => {
          if (
            event.button !== 0 ||
            !hitCharacter(canvas.current, event.clientX, event.clientY)
          )
            return;
          event.preventDefault();
          window.desktop.pointer(true);
          gesture.current = {
            x: event.screenX,
            y: event.screenY,
            moved: false,
          };
          event.currentTarget.setPointerCapture(event.pointerId);
          window.desktop.drag("start");
        }}
        onPointerMove={(event) => {
          const start = gesture.current;
          if (!start) return;
          if (Math.hypot(event.screenX - start.x, event.screenY - start.y) > 6)
            start.moved = true;
          if (start.moved) {
            window.desktop.drag("move");
          }
        }}
        onPointerUp={(event) => {
          if (event.button !== 0) return;
          const start = gesture.current;
          if (!start) return;
          window.desktop.drag("end");
          gesture.current = null;
          if (event.currentTarget.hasPointerCapture(event.pointerId))
            event.currentTarget.releasePointerCapture(event.pointerId);
          if (!start.moved) run({ type: "chat.open" });
        }}
        onPointerCancel={cancelDrag}
        onLostPointerCapture={cancelDrag}
      >
        <Character
          manifest={manifest}
          language={language}
          animation={animation}
          active={active}
          canvasRef={canvas}
          onPlayback={(id, status) =>
            run({ type: "playback.report", id, status })
          }
        />
      </button>
      <div
        className="companion-badges"
        data-interactive
        style={{
          left: layout.body.left + layout.body.size - 36,
          top: layout.body.top + 15,
        }}
      >
        {state.unread > 0 && (
          <button
            className="unread"
            aria-label={t("unreadBadge", { count: state.unread })}
            onClick={() => run({ type: "chat.open" })}
          >
            {state.unread > 9 ? "9+" : state.unread}
          </button>
        )}
        {state.connection !== "connected" && (
          <button
            className="offline-dot"
            aria-label={t("offlineAction")}
            onClick={() => run({ type: "chat.open" })}
          >
            !
          </button>
        )}
      </div>
      {(error !== null || state.behavior.error) && (
        <button
          className="companion-error"
          data-interactive
          onClick={() => {
            setError(null);
            run({ type: "chat.open" });
          }}
        >
          {state.behavior.error || t("openStatus")}
        </button>
      )}
    </main>
  );
}
