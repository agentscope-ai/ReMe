import { createContext, useState, useSyncExternalStore } from "react";
import { Maximize2, Menu, Minus, Moon, PanelLeft, Sun, X } from "lucide-react";
import { errorText, translator } from "../../client/src/i18n.ts";
import type { StudioAction, Settings } from "../../client/src/protocol.ts";
import { useHost } from "./host.ts";

export type Breadcrumb = {
  label: string;
  title?: string;
  onSelect?: () => void;
  revealSidebar?: boolean;
};

export const WorkspaceLayout = createContext<{
  settings: Pick<Settings, "language" | "theme">;
  platform?: string;
  setSidebarCollapsed: (collapsed: boolean) => void;
  mobileSidebar?: HTMLDivElement | null;
  openNavigation?: () => void;
  navigationOpen?: boolean;
} | null>(null);

export const githubUrl = "https://github.com/agentscope-ai/ReMe";

const darkPreference = "(prefers-color-scheme: dark)";

function subscribeToColorScheme(onChange: () => void) {
  const preference = window.matchMedia(darkPreference);
  preference.addEventListener("change", onChange);
  return () => preference.removeEventListener("change", onChange);
}

export default function WorkspaceTitlebar({
  path,
  sidebar,
  collapsed,
  settings,
  platform,
  setSidebarCollapsed,
  openNavigation,
  navigationOpen,
}: NonNullable<React.ContextType<typeof WorkspaceLayout>> & {
  path: readonly Breadcrumb[];
  sidebar: string;
  collapsed: boolean;
}) {
  const { action, desktop } = useHost();
  const { language, theme } = settings;
  const t = translator(language);
  const isChinese = language === "zh";
  const systemDark = useSyncExternalStore(
    subscribeToColorScheme,
    () => window.matchMedia(darkPreference).matches,
    () => false,
  );
  const dark = theme === "dark" || (theme === "system" && systemDark);
  const ThemeIcon = dark ? Moon : Sun;
  const [error, setError] = useState<unknown>(null);
  const [saving, setSaving] = useState(false);
  const run = async (value: StudioAction | (() => Promise<void>)) => {
    setError(null);
    const updating =
      typeof value !== "function" && value.type === "settings.update";
    if (updating) setSaving(true);
    try {
      await (typeof value === "function" ? value() : action(value));
    } catch (reason) {
      setError(reason);
    } finally {
      if (updating) setSaving(false);
    }
  };
  const toggleLabel = t(collapsed ? "showSidebar" : "hideSidebar");
  const shortcut = platform === "darwin" ? "⇧⌘S" : "Ctrl+Shift+S";
  const languageLabel = t("switchLanguage", {
    language: t(isChinese ? "english" : "chinese"),
  });
  if (openNavigation) {
    const current = path.at(-1)?.label;
    return (
      <header className="workspace-titlebar mobile-titlebar">
        <button
          type="button"
          aria-label={t("navigation")}
          aria-controls="mobile-navigation"
          aria-expanded={navigationOpen}
          onClick={openNavigation}
        >
          <Menu size={22} aria-hidden />
        </button>
        {current !== t("conversationTab") && <span>{current}</span>}
      </header>
    );
  }
  return (
    <header
      className="workspace-titlebar"
      data-sidebar-collapsed={collapsed}
      onDoubleClick={(event) => {
        if (desktop && !(event.target as HTMLElement).closest("button"))
          void run(desktop.toggleMaximize);
      }}
    >
      <div className="workspace-titlebar-side">
        <button
          type="button"
          className="titlebar-button"
          aria-label={toggleLabel}
          aria-expanded={!collapsed}
          aria-controls={sidebar}
          title={platform ? `${toggleLabel} (${shortcut})` : toggleLabel}
          aria-keyshortcuts={
            platform
              ? platform === "darwin"
                ? "Meta+Shift+S"
                : "Control+Shift+S"
              : undefined
          }
          onClick={() => setSidebarCollapsed(!collapsed)}
        >
          <PanelLeft size={17} aria-hidden />
        </button>
        <span className="workspace-brand" hidden={collapsed}>
          ReMe Studio
        </span>
      </div>
      <nav className="workspace-breadcrumb" aria-label={t("workspacePath")}>
        <ol>
          {path.map(({ label, title, onSelect, revealSidebar }, index) => {
            const current = index === path.length - 1;
            return (
              <li key={index}>
                {index > 0 && (
                  <span className="path-divider" aria-hidden>
                    /
                  </span>
                )}
                {(onSelect || revealSidebar) && !current ? (
                  <button
                    type="button"
                    title={title ?? label}
                    onClick={() => {
                      onSelect?.();
                      if (revealSidebar) {
                        setSidebarCollapsed(false);
                        requestAnimationFrame(() =>
                          document.getElementById(sidebar)?.focus(),
                        );
                      }
                    }}
                  >
                    {label}
                  </button>
                ) : (
                  <span
                    aria-current={current ? "page" : undefined}
                    title={title ?? label}
                  >
                    {label}
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      </nav>
      <div className="workspace-titlebar-actions">
        <button
          type="button"
          className="titlebar-button"
          aria-label={languageLabel}
          title={languageLabel}
          disabled={saving}
          onClick={() =>
            void run({
              type: "settings.update",
              language: isChinese ? "en" : "zh",
            })
          }
        >
          <span className="titlebar-language-mark" aria-hidden>
            {isChinese ? "中" : "A"}
          </span>
        </button>
        <button
          type="button"
          role="switch"
          className="titlebar-button"
          aria-checked={dark}
          aria-label={t("darkTheme")}
          title={t(dark ? "lightTheme" : "darkTheme")}
          disabled={saving}
          onClick={() =>
            void run({
              type: "settings.update",
              theme: dark ? "light" : "dark",
            })
          }
        >
          <ThemeIcon size={16} strokeWidth={1.75} aria-hidden />
        </button>
        <button
          type="button"
          className="titlebar-button"
          aria-label={t("openGithub")}
          title={t("openGithub")}
          onClick={() => void run({ type: "link.open", url: githubUrl })}
        >
          <span className="github-mark" aria-hidden />
        </button>
        {desktop && platform && platform !== "darwin" && (
          <div className="window-actions">
            <button
              type="button"
              className="icon-button"
              aria-label={t("minimizeWindow")}
              onClick={() => void run(desktop.minimize)}
            >
              <Minus size={15} aria-hidden />
            </button>
            <button
              type="button"
              className="icon-button"
              aria-label={t("maximizeWindow")}
              onClick={() => void run(desktop.toggleMaximize)}
            >
              <Maximize2 size={14} aria-hidden />
            </button>
            <button
              type="button"
              className="icon-button"
              aria-label={t("closeChat")}
              onClick={() => void run(desktop.close)}
            >
              <X size={16} aria-hidden />
            </button>
          </div>
        )}
      </div>
      {error !== null && (
        <div className="titlebar-error action-error" role="alert">
          {errorText(error, language)}
        </div>
      )}
    </header>
  );
}
