import { useLayoutEffect, useRef, type RefObject } from "react";

/** Animate a changed destination without remounting its editor, requests or scroll state. */
export function useMobileContentMotion(
  target: RefObject<HTMLElement | null>,
  destination: string,
  enabled: boolean,
) {
  const previous = useRef(destination);
  useLayoutEffect(() => {
    if (previous.current === destination) return;
    previous.current = destination;
    const element = target.current;
    if (
      !enabled ||
      !element?.getClientRects().length ||
      matchMedia("(prefers-reduced-motion: reduce)").matches
    )
      return;
    const animation = element.animate(
      [
        { opacity: 0, translate: "0 6px" },
        { opacity: 1, translate: "0 0" },
      ],
      { duration: 200, easing: "cubic-bezier(.22,.75,.22,1)" },
    );
    return () => animation.cancel();
  }, [destination, enabled, target]);
}
