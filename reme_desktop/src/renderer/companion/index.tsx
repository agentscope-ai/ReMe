import { useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import type { Bootstrap } from "../../shared/protocol.ts";
import { Character, hitCharacter } from "../animation/player.tsx";
import { action } from "../desktop.ts";

export default function Companion({ state, manifest, layout }: Bootstrap) {
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const gesture = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const [error, setError] = useState("");
  const animation = state.behavior.animation;
  const dragging = animation.clip === "dragged";
  const run = (value: Parameters<typeof action>[0]) => {
    void action(value).catch((error: unknown) => setError(String(error)));
  };

  const openChat = () => run({ type: "chat.open" });
  const cancelDrag = () => {
    if (gesture.current) window.desktop.drag("end");
    gesture.current = null;
  };

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
    <main className="companion" aria-label="Remy 桌面助手">
      {state.bubble && (
        <section
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
          aria-label="主动提醒"
          onPointerEnter={() =>
            run({ type: "bubble.pause", id: state.bubble!.id, paused: true })
          }
          onPointerLeave={() =>
            run({ type: "bubble.pause", id: state.bubble!.id, paused: false })
          }
        >
          <header>
            <span className="eyebrow">REMY 想到一件事</span>
            <button
              aria-label="收起提醒"
              className="icon-button"
              onClick={() =>
                run({ type: "bubble.dismiss", id: state.bubble!.id })
              }
            >
              ×
            </button>
          </header>
          <p>{state.bubble.text.replace(/^#+\s*/gm, "").slice(0, 400)}</p>
          <footer>
            <button
              className="primary"
              onClick={() =>
                run({ type: "chat.open", messageId: state.bubble!.id })
              }
            >
              展开聊聊 <span aria-hidden>↗</span>
            </button>
            <button
              className="subtle"
              onClick={() =>
                run({ type: "bubble.snooze", id: state.bubble!.id })
              }
            >
              10 分钟后
            </button>
          </footer>
        </section>
      )}
      <button
        className={`character ${dragging ? "dragging" : ""} ${animation.clip === "happy" ? "reacting" : ""}`}
        aria-label="和 Remy 聊聊，按住拖动可移动角色"
        title="左键开聊 · 按住拖动 · 右键菜单"
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
          animation={animation}
          canvasRef={canvas}
          onEnded={(id) => run({ type: "animation.ended", id })}
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
            aria-label={`${state.unread} 条未读提醒`}
            onClick={() => run({ type: "chat.open" })}
          >
            {state.unread > 9 ? "9+" : state.unread}
          </button>
        )}
        {state.connection !== "connected" && (
          <button
            className="offline-dot"
            aria-label="服务未连接，点击查看"
            onClick={() => run({ type: "chat.open" })}
          >
            !
          </button>
        )}
      </div>
      {error && (
        <button
          className="companion-error"
          data-interactive
          onClick={() => {
            setError("");
            run({ type: "chat.open" });
          }}
        >
          打开对话查看状态
        </button>
      )}
    </main>
  );
}
