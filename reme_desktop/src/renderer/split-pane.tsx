import { useEffect, useRef, useState, type ReactNode } from "react";
import { translator, type Language } from "../shared/i18n.ts";

export default function SplitPane({
  name,
  sidebar,
  children,
  language,
  collapsed = false,
  defaultWidth = 292,
  minWidth = 240,
  maxWidth = 400,
  minContentWidth = 280,
  className = "",
  resizeLabel,
}: {
  name: string;
  sidebar: ReactNode;
  children: ReactNode;
  language: Language;
  collapsed?: boolean;
  defaultWidth?: number;
  minWidth?: number;
  maxWidth?: number;
  minContentWidth?: number;
  className?: string;
  resizeLabel?: string;
}) {
  const root = useRef<HTMLDivElement>(null);
  const [available, setAvailable] = useState<number>();
  const [preferredWidth, setPreferredWidth] = useState(() => {
    try {
      const saved = Number(localStorage.getItem(`split:${name}`));
      return Math.max(
        minWidth,
        Math.min(
          maxWidth,
          Number.isFinite(saved) && saved > 0 ? saved : defaultWidth,
        ),
      );
    } catch {
      return defaultWidth;
    }
  });
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) =>
      setAvailable(entry.contentRect.width),
    );
    observer.observe(root.current!);
    return () => observer.disconnect();
  }, []);
  const maximum = Math.max(
    minWidth,
    Math.min(maxWidth, (available || Infinity) - minContentWidth),
  );
  const width = Math.min(preferredWidth, maximum);
  const resize = (value: number) => {
    const next = Math.max(minWidth, Math.min(maximum, value));
    setPreferredWidth(next);
    try {
      localStorage.setItem(`split:${name}`, String(next));
    } catch {
      /* In-memory resizing still works. */
    }
  };
  return (
    <div
      ref={root}
      className={`split-pane ${className}`}
      style={{ "--sidebar-width": `${width}px` } as React.CSSProperties}
    >
      <aside
        id={`sidebar-${name}`}
        className="workspace-sidebar"
        hidden={collapsed}
        inert={collapsed}
      >
        {sidebar}
      </aside>
      <div
        className="split-handle"
        hidden={collapsed}
        role="separator"
        aria-orientation="vertical"
        aria-label={resizeLabel ?? translator(language)("resizeSidebar")}
        aria-valuemin={minWidth}
        aria-valuemax={maximum}
        aria-valuenow={width}
        tabIndex={0}
        onKeyDown={(event) => {
          if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
            event.preventDefault();
            resize(
              event.key === "Home"
                ? minWidth
                : event.key === "End"
                  ? maximum
                  : width + (event.key === "ArrowRight" ? 16 : -16),
            );
          }
        }}
        onPointerDown={(event) => {
          if (event.button === 0) {
            event.preventDefault();
            event.currentTarget.focus();
            event.currentTarget.setPointerCapture(event.pointerId);
          }
        }}
        onPointerMove={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId))
            resize(event.clientX - root.current!.getBoundingClientRect().left);
        }}
        onPointerUp={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            resize(event.clientX - root.current!.getBoundingClientRect().left);
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
        }}
      />
      <div className="workspace-main">{children}</div>
    </div>
  );
}
