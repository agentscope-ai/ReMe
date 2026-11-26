import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  type LucideIcon,
} from "lucide-react";
import "./context-menu.css";

export interface MenuItem {
  id: string;
  label: string;
  status?: string;
  icon?: LucideIcon;
  disabled?: boolean;
  checked?: boolean;
  children?: MenuItem[];
  select?: () => void;
}

export type MenuPosition = { x: number; y: number };

export default function ContextMenu({
  items,
  position,
  open = true,
  size,
  onClose,
}: {
  items: MenuItem[];
  position?: MenuPosition;
  open?: boolean;
  size?: { width: number; height: number };
  onClose: () => void;
}) {
  const inline = position === undefined;
  const x = position?.x ?? 0;
  const y = position?.y ?? 0;
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLElement | null>(null);
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
    if (open) trigger.current = document.activeElement as HTMLElement | null;
  }, [open]);
  useLayoutEffect(() => {
    const element = root.current;
    if (!element || inline) return;
    const view = window.visualViewport;
    const inset = 8;
    const minX = (view?.offsetLeft ?? 0) + inset;
    const minY = (view?.offsetTop ?? 0) + inset;
    const width = Math.max(0, (view?.width ?? innerWidth) - 2 * inset);
    const height = Math.max(0, (view?.height ?? innerHeight) - 2 * inset);
    element.style.maxWidth = `${width}px`;
    element.style.maxHeight = `${Math.min(size?.height ?? 380, height)}px`;
    const bounds = element.getBoundingClientRect();
    element.style.left = `${Math.max(minX, Math.min(x, minX + width - bounds.width))}px`;
    element.style.top = `${Math.max(minY, Math.min(y, minY + height - bounds.height))}px`;
  }, [inline, x, y, size?.width, size?.height, path, viewport]);
  useLayoutEffect(() => {
    const element = root.current;
    if (!element || !open) return;
    const first = element.querySelector<HTMLButtonElement>(
      "button:not(:disabled)",
    );
    (first ?? element).focus({ preventScroll: !inline });
  }, [inline, open, x, y, path]);
  useEffect(() => {
    if (!open) return;
    const source = trigger.current;
    const outside = (event: Event) => {
      const target = event.target as Node;
      if (
        !root.current?.contains(target) &&
        !(inline && source?.contains(target))
      )
        close.current();
    };
    const dismiss = () => close.current();
    const resize = () => setViewport((value) => value + 1);
    const view = window.visualViewport;
    // Dismiss inline menus at click time so rows stay in place while targeting them.
    const outsideEvent = inline ? "click" : "pointerdown";
    document.addEventListener(outsideEvent, outside, inline);
    if (inline) document.addEventListener("contextmenu", outside, true);
    if (!inline) {
      window.addEventListener("blur", dismiss);
      window.addEventListener("resize", resize);
      view?.addEventListener("resize", resize);
      view?.addEventListener("scroll", resize);
    }
    return () => {
      document.removeEventListener(outsideEvent, outside, inline);
      if (inline) document.removeEventListener("contextmenu", outside, true);
      if (!inline) {
        window.removeEventListener("blur", dismiss);
        window.removeEventListener("resize", resize);
        view?.removeEventListener("resize", resize);
        view?.removeEventListener("scroll", resize);
      }
      if (
        document.hasFocus() &&
        source?.isConnected &&
        (document.activeElement === document.body ||
          root.current?.contains(document.activeElement))
      )
        source.focus({ preventScroll: true });
    };
  }, [inline, open]);
  const menu = (
    <div
      ref={root}
      className="context-menu"
      data-inline={inline || undefined}
      style={size ? { width: size.width } : undefined}
      role="menu"
      tabIndex={-1}
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
          <span className="context-menu-label">{title}</span>
        </button>
      )}
      {page.map(({ icon: Icon, ...item }) => (
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
              if (!inline) onClose();
              item.select?.();
            }
          }}
        >
          {(!inline || Icon || item.checked !== undefined) && (
            <span className="context-menu-icon" aria-hidden>
              {item.checked ? (
                <Check size={13} />
              ) : (
                Icon && <Icon size={14} strokeWidth={1.75} />
              )}
            </span>
          )}
          <span className="context-menu-label">{item.label}</span>
          {item.status && (
            <span className="context-menu-status">{item.status}</span>
          )}
          {item.children && <ChevronRight size={13} />}
        </button>
      ))}
    </div>
  );
  return inline ? (
    <div
      className="context-menu-disclosure"
      data-open={open || undefined}
      inert={!open}
    >
      {menu}
    </div>
  ) : (
    createPortal(menu, document.body)
  );
}
