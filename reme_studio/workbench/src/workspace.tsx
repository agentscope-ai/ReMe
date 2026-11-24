import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import * as Tabs from "@radix-ui/react-tabs";
import {
  ArrowLeft,
  BookOpen,
  ChevronRight,
  MessageCircle,
  Sparkles,
  Leaf,
  Settings as SettingsIcon,
} from "lucide-react";
import { translator } from "../../client/src/i18n.ts";
import type { StudioState, StudioPage } from "../../client/src/protocol.ts";
import { WORKBENCH_PAGES } from "../../client/src/protocol.ts";
import { useHost } from "./host.ts";
import { useSpeechPlayback } from "./speech/playback.ts";
import { useMobileDrawerGesture } from "./mobile-gesture.ts";
import AgentWorkspace from "./agent-workspace.tsx";
import Settings from "./settings.tsx";
import Knowledge from "./knowledge/index.tsx";
import Notebook from "./notebook/index.tsx";
import { WorkspaceLayout } from "./workspace-titlebar.tsx";

export default function Workbench({
  state,
  version,
  platform,
  companionVisible = false,
  railAccessory,
  mobile = false,
}: {
  state: StudioState;
  version: string;
  platform?: string;
  companionVisible?: boolean;
  railAccessory?: ReactNode;
  mobile?: boolean;
}) {
  const { action, onCommand } = useHost();
  const desktop = platform !== undefined;
  const t = translator(state.settings.language);
  // File-backed pages own their drafts and requests for exactly one workspace.
  const scope = JSON.stringify([
    state.settings.endpoint,
    state.clientInfo?.workspace_dir,
  ]);
  const [collapsed, setCollapsed] = useState<
    Partial<Record<StudioPage, boolean>>
  >(() => {
    try {
      return Object.fromEntries(
        WORKBENCH_PAGES.flatMap((page) => {
          const saved = localStorage.getItem(`sidebar:${page}:collapsed`);
          return saved === null ? [] : [[page, saved === "true"]];
        }),
      );
    } catch {
      return {};
    }
  });
  const [compact, setCompact] = useState(false);
  const [mobileMenu, setMobileMenu] = useState<"primary" | "secondary" | null>(
    null,
  );
  const [mobileSidebar, setMobileSidebar] = useState<HTMLDivElement | null>(
    null,
  );
  const mobileRoot = useRef<HTMLDivElement>(null);
  const mobileDrawer = useRef<HTMLElement>(null);
  const mobileMenuOpen = mobileMenu !== null;
  const setMobileMenuOpen = useCallback((open: boolean) => {
    setMobileMenu(open ? "primary" : null);
  }, []);
  useMobileDrawerGesture(
    mobileRoot,
    mobileDrawer,
    mobileMenuOpen,
    setMobileMenuOpen,
  );
  useEffect(() => {
    const viewport = window.matchMedia("(max-width: 640px)");
    const update = () => setCompact(viewport.matches);
    update();
    viewport.addEventListener("change", update);
    return () => viewport.removeEventListener("change", update);
  }, []);
  const sidebarCollapsed = (page: StudioPage) =>
    mobile ? state.activePage !== page : (collapsed[page] ?? compact);
  const setSidebarCollapsed = useCallback(
    (value: boolean, page = state.activePage) => {
      if (mobile) {
        setMobileMenu(value ? null : "secondary");
        return;
      }
      setCollapsed((current) => ({ ...current, [page]: value }));
      try {
        localStorage.setItem(`sidebar:${page}:collapsed`, String(value));
      } catch {
        /* In-memory state still works. */
      }
    },
    [mobile, state.activePage],
  );
  const activeSidebarCollapsed = sidebarCollapsed(state.activePage);
  const closeCompactSidebar = () => {
    if (mobile || (compact && !activeSidebarCollapsed))
      setSidebarCollapsed(true);
  };
  useEffect(() => {
    if (mobileMenu === null) return;
    const back = (event: Event) => {
      event.preventDefault();
      setMobileMenu(mobileMenu === "secondary" ? "primary" : null);
    };
    window.addEventListener("reme-mobile-back", back);
    const escape = (event: KeyboardEvent) => {
      if (
        event.key === "Escape" &&
        !event.defaultPrevented &&
        !document.querySelector("dialog[open]")
      )
        back(event);
    };
    window.addEventListener("keydown", escape);
    return () => {
      window.removeEventListener("reme-mobile-back", back);
      window.removeEventListener("keydown", escape);
    };
  }, [mobileMenu]);
  const previousMenu = useRef(mobileMenu);
  useEffect(() => {
    const root = mobileRoot.current;
    if (root && previousMenu.current !== mobileMenu) {
      const selector =
        mobileMenu === "secondary"
          ? ".mobile-menu-back"
          : mobileMenu === "primary"
            ? '.mobile-modules [aria-current="page"]'
            : ".mobile-titlebar button";
      for (const element of root.querySelectorAll<HTMLElement>(selector)) {
        if (!element.getClientRects().length) continue;
        element.focus({ preventScroll: true });
        break;
      }
    }
    previousMenu.current = mobileMenu;
  }, [mobileMenu]);
  useEffect(() => {
    if (!platform) return;
    const toggle = (event: KeyboardEvent) => {
      if (
        event.defaultPrevented ||
        event.isComposing ||
        event.repeat ||
        event.altKey ||
        !event.shiftKey ||
        event.code !== "KeyS"
      )
        return;
      if (
        platform === "darwin"
          ? !event.metaKey || event.ctrlKey
          : !event.ctrlKey || event.metaKey
      )
        return;
      event.preventDefault();
      setSidebarCollapsed(!activeSidebarCollapsed);
    };
    window.addEventListener("keydown", toggle);
    return () => window.removeEventListener("keydown", toggle);
  }, [platform, activeSidebarCollapsed, setSidebarCollapsed]);
  useEffect(() => {
    if (state.documentTarget && !compact)
      setCollapsed((value) => ({ ...value, knowledge: false }));
  }, [state.documentTarget, compact]);
  const lastPage = useRef<Exclude<StudioPage, "settings">>("remy");
  const settingsOpen = state.activePage === "settings";
  const agentPage =
    state.activePage === "tasks" ||
    (state.activePage !== "remy" &&
      state.selectedAgentName &&
      state.selectedAgentName !== "Remy")
      ? "tasks"
      : "remy";
  useEffect(() => {
    if (state.activePage !== "settings") lastPage.current = state.activePage;
  }, [state.activePage]);
  useSpeechPlayback(state);
  const select = (page: StudioPage) =>
    void action({ type: "page.select", page });
  useEffect(
    () =>
      onCommand((command) => {
        if (command.type === "settings.open") select("settings");
      }),
    [],
  );
  const content = (
    <Tabs.Root
      inert={mobileMenu !== null}
      orientation="vertical"
      value={state.activePage}
      onValueChange={(value) => select(value as StudioPage)}
      className="workspace-shell"
      data-transparent-background={state.settings.transparentBackground}
    >
      {!mobile && (
        <div className="workspace-rail">
          <Tabs.List className="workspace-tabs" aria-label={t("navigation")}>
            <Tabs.Trigger
              className="rail-button remy-mark"
              value="remy"
              aria-label="Remy"
            >
              <span className="rail-label" aria-hidden>
                Remy
              </span>
              {state.unread > 0 && <i className="tab-dot" aria-hidden />}
            </Tabs.Trigger>
            <Tabs.Trigger
              className="rail-button"
              value="tasks"
              aria-label={t("proactiveAi")}
            >
              <Sparkles size={18} aria-hidden />
              <span className="rail-label" aria-hidden>
                {t("proactiveAi")}
              </span>
            </Tabs.Trigger>
            <Tabs.Trigger
              className="rail-button"
              value="notebook"
              aria-label={t("notebook")}
            >
              <Leaf size={18} aria-hidden />
              <span className="rail-label" aria-hidden>
                {t("notebook")}
              </span>
            </Tabs.Trigger>
            <Tabs.Trigger
              className="rail-button"
              value="knowledge"
              aria-label={t("knowledge")}
            >
              <BookOpen size={18} aria-hidden />
              <span className="rail-label" aria-hidden>
                {t("knowledge")}
              </span>
            </Tabs.Trigger>
          </Tabs.List>
          <div className="rail-actions">
            {railAccessory}
            <button
              type="button"
              className="rail-button"
              aria-label={t("settings")}
              aria-pressed={settingsOpen}
              aria-controls="desktop-settings"
              onClick={() =>
                select(settingsOpen ? lastPage.current : "settings")
              }
            >
              <SettingsIcon size={18} aria-hidden />
              <span className="rail-label" aria-hidden>
                {t("settings")}
              </span>
            </button>
          </div>
        </div>
      )}
      <div
        className="workspace-content"
        hidden={settingsOpen}
        inert={settingsOpen}
      >
        <Tabs.Content value={agentPage} forceMount className="workspace-page">
          <AgentWorkspace
            key={scope}
            state={state}
            page={agentPage}
            collapsed={sidebarCollapsed(agentPage)}
            onNavigate={closeCompactSidebar}
          />
        </Tabs.Content>
        <Tabs.Content value="notebook" forceMount className="workspace-page">
          <Notebook
            key={scope}
            state={state}
            collapsed={sidebarCollapsed("notebook")}
            onNavigate={closeCompactSidebar}
          />
        </Tabs.Content>
        <Tabs.Content value="knowledge" forceMount className="workspace-page">
          <Knowledge
            key={scope}
            state={state}
            collapsed={sidebarCollapsed("knowledge")}
            onNavigate={closeCompactSidebar}
          />
        </Tabs.Content>
      </div>
      <div
        id="desktop-settings"
        className="workspace-page settings-page"
        hidden={!settingsOpen}
        inert={!settingsOpen}
      >
        <Settings
          state={state}
          version={version}
          companionVisible={companionVisible}
          desktop={desktop}
          editableEndpoint={desktop || mobile}
          collapsed={sidebarCollapsed("settings")}
          onNavigate={closeCompactSidebar}
        />
      </div>
    </Tabs.Root>
  );
  const modules = [
    ["remy", "Remy", MessageCircle],
    ["tasks", t("proactiveAi"), Sparkles],
    ["notebook", t("notebook"), Leaf],
    ["knowledge", t("knowledge"), BookOpen],
    ["settings", t("settings"), SettingsIcon],
  ] as const;
  return (
    <WorkspaceLayout.Provider
      value={{
        settings: state.settings,
        platform,
        setSidebarCollapsed,
        mobileSidebar,
        openNavigation: mobile ? () => setMobileMenu("primary") : undefined,
        navigationOpen: mobileMenuOpen,
      }}
    >
      {mobile ? (
        <div
          ref={mobileRoot}
          className="mobile-workbench"
          data-menu={mobileMenu ?? undefined}
        >
          <aside
            ref={mobileDrawer}
            id="mobile-navigation"
            className="mobile-drawer"
            inert={!mobileMenuOpen}
            aria-label={t("navigation")}
          >
            <div
              className="mobile-menu-panel mobile-menu-primary"
              inert={mobileMenu !== "primary"}
            >
              <div className="mobile-brand">
                <span aria-hidden>R</span>
                <strong>ReMe</strong>
              </div>
              <nav className="mobile-modules" aria-label={t("navigation")}>
                {modules.map(([page, label, Icon]) => (
                  <button
                    key={page}
                    type="button"
                    data-module={page}
                    aria-current={
                      state.activePage === page ? "page" : undefined
                    }
                    onClick={() => {
                      select(page);
                      setMobileMenu("secondary");
                    }}
                  >
                    <span className="mobile-module-icon">
                      <Icon size={19} aria-hidden />
                    </span>
                    <span>{label}</span>
                    <span className="mobile-module-status" aria-hidden>
                      {page === "remy" && state.unread > 0 && (
                        <i className="mobile-unread" />
                      )}
                      <ChevronRight size={15} aria-hidden />
                    </span>
                  </button>
                ))}
              </nav>
            </div>
            <div
              className="mobile-menu-panel mobile-menu-secondary"
              inert={mobileMenu !== "secondary"}
            >
              <button
                className="mobile-menu-back"
                type="button"
                onClick={() => setMobileMenu("primary")}
              >
                <ArrowLeft size={18} aria-hidden />
                {t("workspaceTab")}
              </button>
              <div ref={setMobileSidebar} className="mobile-sidebar-host" />
            </div>
          </aside>
          {content}
          <button
            className="mobile-drawer-dismiss"
            type="button"
            inert={!mobileMenuOpen}
            aria-label={t("hideSidebar")}
            onClick={() => setMobileMenu(null)}
          />
        </div>
      ) : (
        content
      )}
    </WorkspaceLayout.Provider>
  );
}
