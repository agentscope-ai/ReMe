import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Check, ChevronLeft, ChevronRight } from "lucide-react";
import "./context-menu.css";

export interface MenuItem {
  id: string;
  label: string;
  disabled?: boolean;
  checked?: boolean;
  children?: MenuItem[];
  select?: () => void;
}

export default function ContextMenu({
  items,
  position,
  size,
  onClose,
}: {
  items: MenuItem[];
  position: { x: number; y: number };
  size?: { width: number; height: number };
  onClose: () => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef(document.activeElement as HTMLElement | null);
  const close = useRef(onClose);
  close.current = onClose;
  const [path, setPath] = useState<string[]>([]);
  const [viewport, setViewport] = useState(0);
  let page = items;
  let title = "";
  for (const id of path) {
    const parent = page.find((item) => item.id === id);
    if (!parent?.children) break;
    title = parent.label;
    page = parent.children;
  }
  useLayoutEffect(() => {
    const element = root.current;
    if (!element) return;
    const bounds = element.getBoundingClientRect();
    element.style.left = `${Math.max(8, Math.min(position.x, innerWidth - bounds.width - 8))}px`;
    element.style.top = `${Math.max(8, Math.min(position.y, innerHeight - bounds.height - 8))}px`;
    element
      .querySelector<HTMLButtonElement>("button:not(:disabled)")
      ?.focus({ preventScroll: true });
  }, [position.x, position.y, size?.width, size?.height, path, viewport]);
  useEffect(() => {
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) close.current();
    };
    const dismiss = () => close.current();
    const resize = () => setViewport((value) => value + 1);
    document.addEventListener("pointerdown", outside);
    window.addEventListener("blur", dismiss);
    window.addEventListener("resize", resize);
    return () => {
      document.removeEventListener("pointerdown", outside);
      window.removeEventListener("blur", dismiss);
      window.removeEventListener("resize", resize);
      if (
        document.hasFocus() &&
        trigger.current?.isConnected &&
        (document.activeElement === document.body ||
          root.current?.contains(document.activeElement))
      )
        trigger.current.focus({ preventScroll: true });
    };
  }, []);
  return createPortal(
    <div
      ref={root}
      className="context-menu"
      style={size ? { width: size.width, maxHeight: size.height } : undefined}
      role="menu"
      data-interactive
      onContextMenu={(event) => event.preventDefault()}
      onKeyDown={(event) => {
        if (event.key === "Escape" || event.key === "ArrowLeft") {
          event.preventDefault();
          if (path.length) setPath(path.slice(0, -1));
          else onClose();
        } else if (event.key === "ArrowRight") {
          const button = document.activeElement;
          if (
            button instanceof HTMLButtonElement &&
            button.getAttribute("aria-haspopup") === "menu"
          ) {
            event.preventDefault();
            button.click();
          }
        } else if (
          ["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)
        ) {
          event.preventDefault();
          const buttons = [
            ...root.current!.querySelectorAll<HTMLButtonElement>(
              "button:not(:disabled)",
            ),
          ];
          const current = buttons.indexOf(
            document.activeElement as HTMLButtonElement,
          );
          const index =
            event.key === "Home"
              ? 0
              : event.key === "End"
                ? buttons.length - 1
                : (current +
                    (event.key === "ArrowDown" ? 1 : -1) +
                    buttons.length) %
                  buttons.length;
          buttons[index]?.focus();
          buttons[index]?.scrollIntoView({ block: "nearest" });
        } else if (event.key === "Tab") onClose();
      }}
    >
      {path.length > 0 && (
        <button
          className="context-menu-back"
          role="menuitem"
          onClick={() => setPath(path.slice(0, -1))}
        >
          <ChevronLeft size={14} />
          <span>{title}</span>
        </button>
      )}
      {page.map((item) => (
        <button
          key={item.id}
          role={item.checked === undefined ? "menuitem" : "menuitemcheckbox"}
          data-menu-id={item.id}
          aria-checked={item.checked}
          aria-haspopup={item.children ? "menu" : undefined}
          disabled={item.disabled}
          onClick={() => {
            if (item.children) setPath([...path, item.id]);
            else {
              onClose();
              item.select?.();
            }
          }}
        >
          <span className="context-menu-check">
            {item.checked && <Check size={13} />}
          </span>
          <span>{item.label}</span>
          {item.children && <ChevronRight size={13} />}
        </button>
      ))}
    </div>,
    document.body,
  );
}
