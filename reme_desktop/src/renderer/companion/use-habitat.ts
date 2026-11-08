import { useLayoutEffect } from "react";
import type { DesktopState } from "../../shared/protocol.ts";

/** The content rail and composer define free space, independently of message text. */
export function useHabitat(state: DesktopState): void {
  useLayoutEffect(() => {
    const motion = matchMedia("(prefers-reduced-motion: reduce)");
    const pane = document.querySelector(".conversation-pane");
    const history = document.querySelector(".chat-history");
    const form = document.querySelector(".chat-composer form");
    let previous = "";
    let live = true;
    const measure = () => {
      if (!live) return;
      let slot = null;
      if (
        state.activePage === "remy" &&
        state.chatVisible &&
        pane &&
        history &&
        form
      ) {
        const bounds = pane.getBoundingClientRect();
        const messages = history.getBoundingClientRect();
        const composer = form.getBoundingClientRect();
        const edge = Math.max(
          messages.right - parseFloat(getComputedStyle(history).paddingRight),
          composer.right,
        );
        const x = Math.ceil(edge + 24),
          y = Math.ceil(messages.top + 12);
        const width = Math.floor(bounds.right - 16 - x);
        const height = Math.floor(bounds.bottom - 24 - y);
        if (width > 0 && height > 0) slot = { x, y, width, height };
      }
      const value = { slot, reducedMotion: motion.matches };
      const encoded = JSON.stringify(value);
      if (encoded !== previous) {
        previous = encoded;
        window.desktop.measureHabitat(value);
      }
    };
    const observer = new ResizeObserver(measure);
    for (const element of [pane, history, form])
      if (element) observer.observe(element);
    window.addEventListener("resize", measure);
    motion.addEventListener("change", measure);
    measure();
    return () => {
      live = false;
      observer.disconnect();
      window.removeEventListener("resize", measure);
      motion.removeEventListener("change", measure);
    };
  }, [state.activePage, state.chatVisible]);
}
