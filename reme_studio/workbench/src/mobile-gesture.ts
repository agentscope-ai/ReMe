import { useEffect, type RefObject } from "react";

export function useMobileDrawerGesture(
  root: RefObject<HTMLDivElement | null>,
  drawer: RefObject<HTMLElement | null>,
  open: boolean,
  setOpen: (open: boolean) => void,
) {
  useEffect(() => {
    const element = root.current;
    const panel = drawer.current;
    if (!element || !panel) return;

    let drag: {
      id: number;
      x: number;
      y: number;
      width: number;
      startedAt: number;
      tracking: boolean;
    } | null = null;
    const reset = () => {
      drag = null;
      delete element.dataset.dragging;
      element.style.removeProperty("--mobile-drag-offset");
      element.style.removeProperty("--mobile-drag-progress");
    };
    const start = (event: TouchEvent) => {
      reset();
      if (
        event.touches.length !== 1 ||
        !(event.target instanceof Element) ||
        event.target.closest(
          "dialog, input, textarea, select, [contenteditable], .conversation-prompts",
        )
      )
        return;
      const touch = event.touches[0];
      const style = getComputedStyle(element);
      const edge = Math.max(
        Number.parseFloat(style.getPropertyValue("--mobile-back-gesture")) || 0,
        Number.parseFloat(style.getPropertyValue("--mobile-safe-left")) || 0,
      );
      if (
        !open &&
        touch.clientX > edge + 30 &&
        !event.target.closest(".mobile-titlebar")
      )
        return;
      drag = {
        id: touch.identifier,
        x: touch.clientX,
        y: touch.clientY,
        width: panel.offsetWidth,
        startedAt: event.timeStamp,
        tracking: false,
      };
    };
    const move = (event: TouchEvent) => {
      if (!drag) return;
      const touch = event.touches[0];
      if (event.touches.length !== 1 || touch.identifier !== drag.id) {
        reset();
        return;
      }
      const dx = touch.clientX - drag.x;
      const dy = touch.clientY - drag.y;
      if (!drag.tracking) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) < 10) return;
        if (
          !event.cancelable ||
          Math.abs(dx) < Math.abs(dy) * 1.35 ||
          (open ? dx >= 0 : dx <= 0)
        ) {
          reset();
          return;
        }
        drag.tracking = true;
        element.dataset.dragging = "";
      }
      event.preventDefault();
      const offset = Math.min(
        drag.width,
        Math.max(0, (open ? drag.width : 0) + dx),
      );
      element.style.setProperty("--mobile-drag-offset", `${offset}px`);
      element.style.setProperty(
        "--mobile-drag-progress",
        String(offset / drag.width),
      );
    };
    const end = (event: TouchEvent) => {
      if (drag?.tracking) {
        const touch = Array.from(event.changedTouches).find(
          ({ identifier }) => identifier === drag?.id,
        );
        if (touch) {
          const distance = (touch.clientX - drag.x) * (open ? -1 : 1);
          const elapsed = Math.max(1, event.timeStamp - drag.startedAt);
          if (
            distance > drag.width * 0.3 ||
            (distance > 24 && distance / elapsed > 0.5)
          )
            setOpen(!open);
        }
      }
      reset();
    };

    element.addEventListener("touchstart", start, { passive: true });
    element.addEventListener("touchmove", move, { passive: false });
    element.addEventListener("touchend", end);
    element.addEventListener("touchcancel", reset);
    return () => {
      element.removeEventListener("touchstart", start);
      element.removeEventListener("touchmove", move);
      element.removeEventListener("touchend", end);
      element.removeEventListener("touchcancel", reset);
      reset();
    };
  }, [root, drawer, open, setOpen]);
}
