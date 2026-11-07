import { useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import type { Bootstrap } from "../../shared/protocol.ts";
import { Character, hitCharacter } from "../animation/player.tsx";
import { action } from "../desktop.ts";
import { translator } from "../../shared/i18n.ts";

export default function Companion({ state, manifest, layout }: Bootstrap) {
  const language = state.settings.language;
  const t = translator(language);
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const bubbleElement = useRef<HTMLElement | null>(null);
  const gesture = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const [error, setError] = useState<unknown>(null);
  const animation = state.behavior.animation;
  const dragging = animation.clip === "dragged";
  const surface = state.surface;
  const reminder = surface?.kind === "reminder" ? surface : null;
  const run = (value: Parameters<typeof action>[0]) => {
    void action(value).catch((error: unknown) => setError(error));
  };

  const openChat = () => run({ type: "chat.open" });
  const cancelDrag = () => {
    if (gesture.current) window.desktop.drag("end");
    gesture.current = null;
  };

  useEffect(() => {
    if (!reminder || !state.sessionId) return;
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
          id: reminder.id,
          presentationId: reminder.presentationId,
        });
      });
    });
    return () => {
      cancelAnimationFrame(first);
      cancelAnimationFrame(second);
    };
  }, [reminder?.presentationId, state.sessionId]);

  useEffect(() => {
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
  }, []);

  return (
    <main className="companion" aria-label={t("companion")}>
      {reminder && (
        <section
          ref={bubbleElement}
          className={`reminder ${layout.bubble.below ? "below" : ""}`}
          data-interactive
          style={
            {
              left: layout.bubble.left,
              top: layout.bubble.below
                ? layout.bubble.top
                : layout.body.top - 12,
              transform: layout.bubble.below ? undefined : "translateY(-100%)",
              width: layout.bubble.width,
              "--tail-left": `${layout.bubble.tail}px`,
            } as CSSProperties
          }
          aria-label={t("reminder")}
          onPointerEnter={() =>
            run({ type: "bubble.pause", id: reminder.id, paused: true })
          }
          onPointerLeave={() =>
            run({ type: "bubble.pause", id: reminder.id, paused: false })
          }
        >
          <header>
            <span className="eyebrow">{t("reminderHeading")}</span>
            <button
              aria-label={t("dismissReminder")}
              className="icon-button"
              onClick={() => run({ type: "bubble.dismiss", id: reminder.id })}
            >
              ×
            </button>
          </header>
          <p>{reminder.text.replace(/^#+\s*/gm, "").slice(0, 400)}</p>
          <footer>
            {state.speech.tts && (
              <button
                className="subtle"
                aria-label={t(
                  state.speech.playback?.messageId === reminder.id
                    ? "stopSpeech"
                    : "readAloud",
                )}
                onClick={() =>
                  run(
                    state.speech.playback?.messageId === reminder.id
                      ? { type: "speech.stop" }
                      : {
                          type: "speech.speak",
                          sessionId: state.sessionId!,
                          messageId: reminder.id,
                        },
                  )
                }
              >
                {state.speech.playback?.messageId === reminder.id ? "■" : "♫"}
              </button>
            )}
            <button
              className="primary"
              onClick={() => run({ type: "chat.open", messageId: reminder.id })}
            >
              {t("expandReminder")} <span aria-hidden>↗</span>
            </button>
            <button
              className="subtle"
              onClick={() => run({ type: "bubble.snooze", id: reminder.id })}
            >
              {t("snoozeReminder")}
            </button>
          </footer>
        </section>
      )}
      {surface?.kind === "status" && (
        <button
          className={`status-notice ${surface.tone} ${layout.bubble.below ? "below" : ""}`}
          data-interactive
          aria-label={t("statusOpen", { text: surface.text })}
          onClick={openChat}
          style={{
            left: layout.bubble.left,
            top: layout.bubble.below ? layout.bubble.top : layout.body.top - 12,
            width: layout.bubble.width,
            transform: layout.bubble.below ? undefined : "translateY(-100%)",
          }}
        >
          <span className="status-light" aria-hidden />
          <span role="status" aria-live="polite">
            {surface.text}
          </span>
          <span className="status-arrow" aria-hidden>
            ↗
          </span>
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
          if (event.detail === 0) openChat();
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
          if (!start.moved) openChat();
        }}
        onPointerCancel={cancelDrag}
        onLostPointerCapture={cancelDrag}
      >
        <Character
          manifest={manifest}
          language={language}
          animation={animation}
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
