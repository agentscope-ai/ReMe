import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import * as Tabs from "@radix-ui/react-tabs";
import {
  BookOpen,
  Sparkles,
  Leaf,
  Settings as SettingsIcon,
} from "lucide-react";
import { translator } from "../../client/src/i18n.ts";
import type { StudioState, StudioPage } from "../../client/src/protocol.ts";
import { WORKBENCH_PAGES } from "../../client/src/protocol.ts";
import { useHost } from "./host.ts";
import { useSpeechPlayback } from "./speech/playback.ts";
import { MobileBackNavigation, MobileTabBar } from "./mobile/navigation.tsx";
import AgentWorkspace from "./agent-workspace.tsx";
import Tasks from "./tasks/index.tsx";
import Settings from "./settings.tsx";
import BackendSetup from "./backend-setup.tsx";
import { BackendAccess } from "./backend-access.tsx";
import type { NavigationGuard } from "../../client/src/backend.ts";
import Knowledge from "./knowledge/index.tsx";
import Notebook from "./notebook/index.tsx";
import { WorkspaceLayout } from "./workspace-titlebar.tsx";
import { NavigationProvider } from "./navigation.tsx";

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
  const desktop = !mobile && platform !== undefined;
  const t = translator(state.settings.language);
  const [setupBlocking, setSetupBlocking] = useState(true);
  const [backendAccess, setBackendAccess] = useState<{
    endpoint: string;
    token: string;
  }>();
  // File-backed pages own their drafts and requests for exactly one workspace.
  const scope = JSON.stringify([
    state.settings.endpoint,
    state.clientInfo?.workspace_dir,
  ]);
  const [collapsed, setCollapsed] = useState<
    Partial<Record<StudioPage, boolean>>
  >(() => {
    if (mobile) return {};
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
  useEffect(() => {
    if (mobile) return;
    const viewport = window.matchMedia("(max-width: 640px)");
    const update = () => setCompact(viewport.matches);
    update();
    viewport.addEventListener("change", update);
    return () => viewport.removeEventListener("change", update);
  }, [mobile]);
  const sidebarCollapsed = (page: StudioPage) =>
    mobile && state.activePage !== page
      ? true
      : (collapsed[page] ?? (mobile || compact));
  const setSidebarCollapsed = useCallback(
    (value: boolean, page = state.activePage) => {
      setCollapsed((current) => ({ ...current, [page]: value }));
      if (mobile) return;
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
  }, [platform, activeSidebarCollapsed, setSidebarCollapsed, state.activePage]);
  useEffect(() => {
    if (!mobile && state.documentTarget && !compact)
      setCollapsed((value) => ({ ...value, knowledge: false }));
  }, [state.documentTarget, compact, mobile]);
  const lastPage = useRef<Exclude<StudioPage, "settings">>("remy");
  const settingsOpen = state.activePage === "settings";
  useEffect(() => {
    if (state.activePage !== "settings") lastPage.current = state.activePage;
  }, [state.activePage]);
  useSpeechPlayback(state);
  const guard = useRef<NavigationGuard | null>(null);
  const registerGuard = useCallback((value: NavigationGuard | null) => {
    guard.current = value;
  }, []);
  const navigate = useCallback((next: () => void) => {
    if (guard.current) guard.current(next);
    else next();
  }, []);
  const select = (page: StudioPage) => {
    if (page !== state.activePage)
      navigate(() => void action({ type: "page.select", page }));
  };
  useEffect(
    () =>
      onCommand((command) => {
        if (command.type === "settings.open") select("settings");
      }),
    [],
  );
  const content = (
    <Tabs.Root
      inert={setupBlocking}
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
              aria-label={t("memory")}
            >
              <BookOpen size={18} aria-hidden />
              <span className="rail-label" aria-hidden>
                {t("memory")}
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
        <Tabs.Content value="remy" forceMount className="workspace-page">
          <AgentWorkspace
            key={scope}
            state={state}
            collapsed={sidebarCollapsed("remy")}
            onNavigate={closeCompactSidebar}
          />
        </Tabs.Content>
        <Tabs.Content value="tasks" forceMount className="workspace-page">
          <Tasks
            key={scope}
            state={state}
            collapsed={sidebarCollapsed("tasks")}
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
          windowTransparency={platform === "darwin"}
          open={settingsOpen}
          editableEndpoint
          navigate={navigate}
          registerGuard={registerGuard}
          collapsed={sidebarCollapsed("settings")}
          onNavigate={closeCompactSidebar}
        />
      </div>
      {mobile && <MobileTabBar state={state} onSelect={select} />}
    </Tabs.Root>
  );
  const layout = (
    <WorkspaceLayout.Provider
      value={{
        settings: state.settings,
        platform,
        setSidebarCollapsed,
        mobile,
      }}
    >
      <BackendSetup
        key={state.settings.endpoint}
        state={state}
        editableEndpoint
        onBlockingChange={setSetupBlocking}
      />
      {mobile ? <div className="mobile-workbench">{content}</div> : content}
    </WorkspaceLayout.Provider>
  );
  const endpoint = state.settings.endpoint;
  return (
    <BackendAccess.Provider
      value={{
        endpoint,
        token: backendAccess?.endpoint === endpoint ? backendAccess.token : "",
        setToken: (token) => setBackendAccess({ endpoint, token }),
      }}
    >
      <NavigationProvider
        key={scope}
        page={state.activePage}
        onSelect={(page) => action({ type: "page.select", page })}
        navigate={navigate}
      >
        {layout}
        {mobile && <MobileBackNavigation />}
      </NavigationProvider>
    </BackendAccess.Provider>
  );
}
