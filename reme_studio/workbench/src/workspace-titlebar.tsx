import {
  createContext,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import {
  ArrowLeft,
  ArrowRight,
  ChevronRight,
  Maximize2,
  Minus,
  Moon,
  PanelLeft,
  Sun,
  X,
} from "lucide-react";
import { errorText, translator } from "../../client/src/i18n.ts";
import type { StudioAction, Settings } from "../../client/src/protocol.ts";
import { useHost } from "./host.ts";
import { useNavigation } from "./navigation.tsx";
import MobileTitlebar from "./mobile/titlebar.tsx";

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
  mobile?: boolean;
} | null>(null);

export const githubUrl = "https://github.com/agentscope-ai/ReMe";

const darkPreference = "(prefers-color-scheme: dark)";

function subscribeToColorScheme(onChange: () => void) {
  const preference = window.matchMedia(darkPreference);
  preference.addEventListener("change", onChange);
  return () => preference.removeEventListener("change", onChange);
}

export type TitlebarProps = NonNullable<
  React.ContextType<typeof WorkspaceLayout>
> & {
  path: readonly Breadcrumb[];
  sidebar?: string;
  collapsed?: boolean;
  children?: ReactNode;
  status?: ReactNode;
};

export default function WorkspaceTitlebar(props: TitlebarProps) {
  return props.mobile ? (
    <MobileTitlebar {...props} />
  ) : (
    <DesktopTitlebar {...props} />
  );
}

function DesktopTitlebar({
  path,
  sidebar,
  collapsed = true,
  settings,
  platform,
  setSidebarCollapsed,
  children,
  status,
}: TitlebarProps) {
  const { action, desktop } = useHost();
  const navigation = useNavigation();
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
  const issue = error ?? navigation?.error;
  const historyControls = (
    <div
      className="workspace-history"
      role="group"
      aria-label={t("navigationHistory")}
    >
      <button
        type="button"
        className="titlebar-button"
        aria-label={t("goBack")}
        title={t("goBack")}
        disabled={!navigation?.canBack}
        onClick={navigation?.back}
      >
        <ArrowLeft size={17} aria-hidden />
      </button>
      <button
        type="button"
        className="titlebar-button"
        aria-label={t("goForward")}
        title={t("goForward")}
        disabled={!navigation?.canForward}
        onClick={navigation?.forward}
      >
        <ArrowRight size={17} aria-hidden />
      </button>
    </div>
  );
  return (
    <header
      className="workspace-titlebar"
      data-has-sidebar={Boolean(sidebar)}
      data-sidebar-collapsed={collapsed}
      onDoubleClick={(event) => {
        if (desktop && !(event.target as HTMLElement).closest("button"))
          void run(desktop.toggleMaximize);
      }}
    >
      <div className="workspace-titlebar-side">
        <span className="workspace-brand">ReMe Studio</span>
      </div>
      <nav className="workspace-breadcrumb" aria-label={t("workspacePath")}>
        <ol>
          <li className="workspace-root">
            <span>{t("workspaceTab")}</span>
          </li>
          {path.map(({ label, title, onSelect, revealSidebar }, index) => {
            const current = index === path.length - 1;
            return (
              <li key={index}>
                <ChevronRight
                  className="path-divider"
                  size={15}
                  strokeWidth={1.5}
                  aria-hidden
                />
                {(onSelect || revealSidebar) && !current ? (
                  <button
                    type="button"
                    title={title ?? label}
                    onClick={() => {
                      onSelect?.();
                      if (revealSidebar && sidebar) {
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
        {status}
      </nav>
      <div className="workspace-titlebar-actions">
        {historyControls}
        {sidebar && (
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
        )}
        {children}
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
      {issue != null && (
        <div className="titlebar-error action-error" role="alert">
          {errorText(issue, language)}
        </div>
      )}
    </header>
  );
}
