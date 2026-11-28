import { useEffect, useRef, useState, type PointerEvent } from "react";

type Phase = "idle" | "pending" | "holding" | "canceling";
type Callbacks = {
  start: () => void;
  finish: () => void;
  cancel: () => void;
  change: (phase: Phase) => void;
};

/** A short tap edits text; only a stationary long press owns a recording. */
export class HoldToTalk {
  private phase: Phase = "idle";
  private timer?: ReturnType<typeof setTimeout>;
  private x = 0;
  private y = 0;

  constructor(private readonly callbacks: Callbacks) {}

  private change(phase: Phase) {
    if (this.phase === phase) return;
    this.phase = phase;
    this.callbacks.change(phase);
  }

  press(x: number, y: number) {
    this.cancel();
    this.x = x;
    this.y = y;
    this.change("pending");
    this.timer = setTimeout(() => {
      this.change("holding");
      this.callbacks.start();
    }, 360);
  }

  move(x: number, y: number) {
    if (this.phase === "pending") {
      if (Math.hypot(x - this.x, y - this.y) > 12) this.cancel();
    } else if (this.phase !== "idle") {
      this.change(this.y - y >= 64 ? "canceling" : "holding");
    }
  }

  release() {
    const phase = this.phase;
    clearTimeout(this.timer);
    this.change("idle");
    if (phase === "holding") this.callbacks.finish();
    else if (phase === "canceling") this.callbacks.cancel();
    return phase === "pending";
  }

  cancel() {
    const recording = this.phase === "holding" || this.phase === "canceling";
    clearTimeout(this.timer);
    this.change("idle");
    if (recording) this.callbacks.cancel();
  }
}

export function useHoldToTalk({
  enabled,
  scope,
  start,
  finish,
  cancel,
}: {
  enabled: boolean;
  scope: string;
  start: () => void;
  finish: () => void;
  cancel: () => void;
}) {
  const latest = useRef({ enabled, start, finish, cancel });
  latest.current = { enabled, start, finish, cancel };
  const [phase, setPhase] = useState<Phase>("idle");
  const [gesture] = useState(
    () =>
      new HoldToTalk({
        start: () => latest.current.start(),
        finish: () => latest.current.finish(),
        cancel: () => latest.current.cancel(),
        change: setPhase,
      }),
  );
  useEffect(() => () => gesture.cancel(), [gesture, scope]);
  useEffect(() => {
    if (!enabled && phase === "pending") gesture.cancel();
  }, [enabled, phase, gesture]);
  useEffect(() => {
    const hide = () => {
      if (document.hidden) gesture.cancel();
    };
    document.addEventListener("visibilitychange", hide);
    return () => document.removeEventListener("visibilitychange", hide);
  }, [gesture]);
  return {
    held: phase === "holding" || phase === "canceling",
    canceling: phase === "canceling",
    handlers: {
      onPointerDown(event: PointerEvent<HTMLTextAreaElement>) {
        if (!latest.current.enabled || !event.isPrimary || event.button !== 0)
          return;
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        gesture.press(event.clientX, event.clientY);
      },
      onPointerMove(event: PointerEvent<HTMLTextAreaElement>) {
        if (event.currentTarget.hasPointerCapture(event.pointerId))
          gesture.move(event.clientX, event.clientY);
      },
      onPointerUp(event: PointerEvent<HTMLTextAreaElement>) {
        if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
        const tap = gesture.release();
        event.currentTarget.releasePointerCapture(event.pointerId);
        if (tap) event.currentTarget.focus();
      },
      onPointerCancel: () => gesture.cancel(),
      onLostPointerCapture: () => gesture.cancel(),
      onContextMenu(event: React.MouseEvent<HTMLTextAreaElement>) {
        if (latest.current.enabled || phase !== "idle") event.preventDefault();
      },
    },
  };
}
