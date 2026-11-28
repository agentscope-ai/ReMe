import { useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { X } from "lucide-react";
import { translator, type Language } from "../../client/src/i18n.ts";
import { useMobileContentMotion } from "./mobile/motion.ts";
import WorkspaceTitlebar, {
  WorkspaceLayout,
  type Breadcrumb,
} from "./workspace-titlebar.tsx";

export function SidebarHeading({
  eyebrow,
  title,
  children,
}: {
  eyebrow?: string;
  title: string;
  children?: ReactNode;
}) {
  return (
    <header className="module-sidebar-heading">
      {eyebrow && (
        <span className="eyebrow" aria-hidden>
          {eyebrow}
        </span>
      )}
      <div className="module-sidebar-title">
        <h2>{title}</h2>
        {children}
      </div>
    </header>
  );
}

export default function SplitPane({
  name,
  sidebar,
  children,
  language,
  collapsed = false,
  defaultWidth = 256,
  defaultRatio,
  minWidth = 224,
  maxWidth = 400,
  minContentWidth = 280,
  className = "",
  resizeLabel,
  path,
  actions,
  mobileView = "content",
  mobileNavigation,
  contentKey,
}: {
  name: string;
  sidebar: ReactNode;
  children: ReactNode;
  language: Language;
  collapsed?: boolean;
  defaultWidth?: number;
  defaultRatio?: number;
  minWidth?: number;
  maxWidth?: number;
  minContentWidth?: number;
  className?: string;
  resizeLabel?: string;
  path?: readonly Breadcrumb[];
  actions?: ReactNode;
  mobileView?: "navigation" | "content";
  mobileNavigation?: ReactNode;
  contentKey?: string;
}) {
  const layout = useContext(WorkspaceLayout);
  const hasTitlebar = Boolean(layout && path);
  const mobile = Boolean(hasTitlebar && layout?.mobile);
  const inlineNavigation = mobile && mobileView === "navigation";
  const menu = useRef<HTMLDialogElement>(null);
  const sidebarId = `sidebar-${name}`;
  const root = useRef<HTMLDivElement>(null);
  const main = useRef<HTMLDivElement>(null);
  const sidebarRoot = useRef<HTMLElement>(null);
  useMobileContentMotion(
    inlineNavigation ? sidebarRoot : main,
    contentKey ?? JSON.stringify(path?.map((item) => item.label) ?? []),
    mobile,
  );
  const [available, setAvailable] = useState<number>();
  const storageKey =
    defaultRatio === undefined ? `split:${name}` : `split:${name}:ratio`;
  const [resizing, setResizing] = useState(false);
  const [preferred, setPreferred] = useState(() => {
    try {
      const saved = Number(localStorage.getItem(storageKey));
      return Number.isFinite(saved) &&
        saved > 0 &&
        (defaultRatio === undefined || saved < 1)
        ? saved
        : undefined;
    } catch {
      return undefined;
    }
  });
  useEffect(() => {
    if (mobile) return;
    const observer = new ResizeObserver(([entry]) =>
      setAvailable(entry.contentRect.width),
    );
    observer.observe(root.current!);
    return () => observer.disconnect();
  }, [mobile]);
  useEffect(() => {
    const dialog = menu.current;
    if (!dialog) return;
    if (!collapsed && !dialog.open) dialog.showModal();
    else if (collapsed && dialog.open) dialog.close();
  }, [collapsed, mobile, inlineNavigation]);
  const availableWidth =
    available || (defaultRatio ? defaultWidth / defaultRatio : Infinity);
  const maximum = Math.max(
    minWidth,
    Math.min(maxWidth, availableWidth - minContentWidth),
  );
  const clamp = (value: number) => Math.max(minWidth, Math.min(maximum, value));
  const width = clamp(
    defaultRatio
      ? availableWidth * (preferred ?? defaultRatio)
      : (preferred ?? defaultWidth),
  );
  const resize = (value: number) => {
    const next = clamp(value);
    const preference = defaultRatio ? next / availableWidth : next;
    setPreferred(preference);
    try {
      localStorage.setItem(storageKey, String(preference));
    } catch {
      /* In-memory resizing still works. */
    }
  };
  const panel = (
    <aside
      ref={sidebarRoot}
      id={sidebarId}
      className="workspace-sidebar"
      hidden={!mobile && collapsed}
      inert={!mobile && collapsed}
      tabIndex={hasTitlebar ? -1 : undefined}
    >
      {sidebar}
    </aside>
  );
  return (
    <div
      ref={root}
      data-resizing={resizing || undefined}
      data-mobile-view={mobile ? mobileView : undefined}
      className={`split-pane ${hasTitlebar ? "workspace-split" : ""} ${className}`}
      style={{ "--sidebar-width": `${width}px` } as React.CSSProperties}
    >
      {layout && path && (
        <WorkspaceTitlebar
          {...layout}
          path={path}
          sidebar={inlineNavigation || sidebar == null ? undefined : sidebarId}
          collapsed={collapsed}
        >
          {actions}
        </WorkspaceTitlebar>
      )}
      {mobile && mobileNavigation}
      {mobile && !inlineNavigation ? (
        <dialog
          ref={menu}
          className="mobile-page-menu"
          aria-label={path?.[0]?.label}
          onClose={() => layout?.setSidebarCollapsed(true)}
          onClick={(event) => {
            if (event.target === event.currentTarget) menu.current?.close();
          }}
        >
          <div className="mobile-page-menu-content">
            <header className="mobile-page-menu-heading">
              <h2>{path?.[0]?.label}</h2>
              <button
                type="button"
                className="icon-button"
                aria-label={translator(language)("hideSidebar")}
                onClick={() => menu.current?.close()}
              >
                <X size={20} aria-hidden />
              </button>
            </header>
            {panel}
          </div>
        </dialog>
      ) : (
        panel
      )}
      <div
        className="split-handle"
        hidden={collapsed || mobile}
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
            setResizing(true);
          }
        }}
        onLostPointerCapture={() => setResizing(false)}
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
      <div
        className="workspace-main"
        ref={main}
        hidden={inlineNavigation}
        inert={inlineNavigation}
      >
        {children}
      </div>
    </div>
  );
}
