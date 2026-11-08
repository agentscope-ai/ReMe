import { useState, type ReactNode } from "react";
import { translator, type Language } from "../shared/i18n.ts";

export default function SplitPane({
  name,
  sidebar,
  children,
  language,
  collapsed = false,
  defaultWidth = 292,
}: {
  name: string;
  sidebar: ReactNode;
  children: ReactNode;
  language: Language;
  collapsed?: boolean;
  defaultWidth?: number;
}) {
  const [width, setWidth] = useState(() => {
    try {
      return Math.max(
        240,
        Math.min(
          400,
          Number(localStorage.getItem(`split:${name}`)) || defaultWidth,
        ),
      );
    } catch {
      return defaultWidth;
    }
  });
  const resize = (value: number) => {
    const next = Math.max(240, Math.min(400, value));
    setWidth(next);
    try {
      localStorage.setItem(`split:${name}`, String(next));
    } catch {
      /* In-memory resizing still works. */
    }
  };
  return (
    <div
      className="split-pane"
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
        aria-label={translator(language)("resizeSidebar")}
        aria-valuemin={240}
        aria-valuemax={400}
        aria-valuenow={width}
        tabIndex={0}
        onKeyDown={(event) => {
          if (["ArrowLeft", "ArrowRight"].includes(event.key)) {
            event.preventDefault();
            resize(width + (event.key === "ArrowRight" ? 16 : -16));
          }
        }}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId))
            resize(event.clientX);
        }}
        onPointerUp={(event) =>
          event.currentTarget.releasePointerCapture(event.pointerId)
        }
      />
      <div className="workspace-main">{children}</div>
    </div>
  );
}
