import { useCallback, useEffect, useLayoutEffect, useRef } from "react";
import { useStickToBottom } from "use-stick-to-bottom";

function hasSelection(root: HTMLElement | null): boolean {
  const selection = window.getSelection();
  return Boolean(
    root &&
    selection &&
    !selection.isCollapsed &&
    (root.contains(selection.anchorNode) || root.contains(selection.focusNode)),
  );
}

/** One scroll owner; history paging and message links are explicit reader commands. */
export function useChatScroll(sessionId: string | null, active: boolean) {
  const visible = useRef(active);
  visible.current = active;
  const {
    scrollRef: history,
    contentRef: content,
    isAtBottom: following,
    scrollToBottom,
    stopScroll: stopFollowing,
  } = useStickToBottom({
    initial: "instant",
    resize: "instant",
    targetScrollTop: (target, { scrollElement }) =>
      !visible.current || hasSelection(scrollElement)
        ? scrollElement.scrollTop
        : target,
  });
  const anchor = useRef<{ id: string; top: number } | null>(null);
  const context = useRef<{ sessionId: string | null; active: boolean } | null>(
    null,
  );
  const resume = useRef(true);
  const reading = useRef({ top: 0, following: true });

  useEffect(() => {
    const root = history.current;
    if (!root) return;
    const remember = () => {
      if (visible.current && root.clientHeight > 0)
        reading.current = {
          top: root.scrollTop,
          following:
            root.scrollHeight - root.scrollTop - root.clientHeight < 48,
        };
    };
    root.addEventListener("scroll", remember, { passive: true });
    return () => root.removeEventListener("scroll", remember);
  }, [history]);

  const jumpToLatest = useCallback(() => {
    if (visible.current && !hasSelection(history.current))
      void scrollToBottom("instant");
  }, [history, scrollToBottom]);

  useLayoutEffect(() => {
    const previous = context.current;
    const changed = !previous || previous.sessionId !== sessionId;
    context.current = { sessionId, active };
    if (changed) {
      anchor.current = null;
      reading.current = { top: 0, following: true };
    }
    if (!active) {
      if (previous?.active || changed)
        resume.current = changed || reading.current.following;
      stopFollowing();
    } else if (changed || (!previous?.active && resume.current)) jumpToLatest();
    else if (!previous?.active && history.current) {
      stopFollowing();
      history.current.scrollTop = reading.current.top;
    }
  }, [sessionId, active, jumpToLatest, stopFollowing, history]);

  useEffect(() => {
    const onSelection = () => {
      if (visible.current && hasSelection(history.current)) stopFollowing();
    };
    document.addEventListener("selectionchange", onSelection);
    return () => document.removeEventListener("selectionchange", onSelection);
  }, [history, stopFollowing]);

  const reveal = useCallback(
    (element: HTMLElement) => {
      stopFollowing();
      const root = history.current;
      if (root)
        root.scrollTop +=
          element.getBoundingClientRect().top -
          root.getBoundingClientRect().top;
    },
    [history, stopFollowing],
  );

  const prepend = useCallback(
    (load: () => void) => {
      stopFollowing();
      const root = history.current;
      const top = root?.getBoundingClientRect().top ?? 0;
      const first = [
        ...(root?.querySelectorAll<HTMLElement>("article[data-message-id]") ??
          []),
      ].find((element) => element.getBoundingClientRect().bottom > top);
      anchor.current = first
        ? {
            id: first.dataset.messageId!,
            top: first.getBoundingClientRect().top,
          }
        : null;
      load();
    },
    [history, stopFollowing],
  );

  useLayoutEffect(() => {
    const root = history.current;
    const pending = anchor.current;
    if (!root || !pending) return;
    anchor.current = null;
    const element = [
      ...root.querySelectorAll<HTMLElement>("article[data-message-id]"),
    ].find((element) => element.dataset.messageId === pending.id);
    if (element)
      root.scrollTop += element.getBoundingClientRect().top - pending.top;
  });

  return {
    history,
    content,
    following,
    jumpToLatest,
    stopFollowing,
    reveal,
    prepend,
  };
}
