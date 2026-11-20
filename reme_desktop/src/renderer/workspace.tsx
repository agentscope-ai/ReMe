import { useEffect, useLayoutEffect, useRef, useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import {
  BookOpen,
  Leaf,
  Settings as SettingsIcon,
  Sparkles,
  Minus,
  Maximize2,
  X,
  PanelLeft,
} from "lucide-react";
import { translator } from "../shared/i18n.ts";
import type { Bootstrap, DesktopPage } from "../shared/protocol.ts";
import { action } from "./desktop.ts";
import { useSpeechPlayback } from "./speech/playback.ts";
import Remy from "./remy.tsx";
import Settings from "./settings.tsx";
import Knowledge from "./knowledge/index.tsx";
import Notebook from "./notebook/index.tsx";

export default function Workspace(bootstrap: Bootstrap) {
  const { state, platform, version } = bootstrap;
  const t = translator(state.settings.language);
  // File-backed pages own their drafts and requests for exactly one workspace.
  const scope = JSON.stringify([
    state.settings.endpoint,
    state.clientInfo?.workspace_dir,
  ]);
  const [collapsed, setCollapsed] = useState<
    Partial<Record<DesktopPage, boolean>>
  >(() => {
    try {
      return Object.fromEntries(
        (["remy", "notebook", "knowledge", "settings"] as const).map((page) => [
          page,
          localStorage.getItem(`sidebar:${page}:collapsed`) === "true",
        ]),
      );
    } catch {
      return {};
    }
  });
  const toggleSidebar = () => {
    const page = state.activePage;
    const value = !collapsed[page];
    setCollapsed({ ...collapsed, [page]: value });
    try {
      localStorage.setItem(`sidebar:${page}:collapsed`, String(value));
    } catch {
      /* In-memory state still works. */
    }
  };
  useEffect(() => {
    if (state.documentTarget)
      setCollapsed((value) => ({ ...value, knowledge: false }));
  }, [state.documentTarget]);
  const lastPage = useRef<Exclude<DesktopPage, "settings">>("remy");
  const settingsOpen = state.activePage === "settings";
  useEffect(() => {
    if (state.activePage !== "settings") lastPage.current = state.activePage;
  }, [state.activePage]);
  const tabsRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const tabs = tabsRef.current!;
    const updateIndicator = () => {
      const active = tabs.querySelector<HTMLButtonElement>(
        '[role="tab"][data-state="active"]',
      );
      tabs.dataset.active = String(!!active);
      if (active) {
        tabs.style.setProperty("--tab-left", `${active.offsetLeft}px`);
        tabs.style.setProperty("--tab-width", `${active.offsetWidth}px`);
      }
    };
    updateIndicator();
    const observer = new ResizeObserver(updateIndicator);
    observer.observe(tabs);
    tabs
      .querySelectorAll('[role="tab"]')
      .forEach((tab) => observer.observe(tab));
    return () => observer.disconnect();
  }, [state.activePage]);
  useSpeechPlayback(state);
  const select = (page: DesktopPage) =>
    void action({ type: "page.select", page });
  useEffect(
    () =>
      window.desktop.onCommand((command) => {
        if (command.type === "settings.open") select("settings");
      }),
    [],
  );
  return (
    <Tabs.Root
      value={state.activePage}
      onValueChange={(value) => select(value as DesktopPage)}
      className="workspace-shell"
      data-transparent-background={state.settings.transparentBackground}
    >
      <header
        className="workspace-titlebar"
        onDoubleClick={(event) => {
          if (!(event.target as HTMLElement).closest("button"))
            void action({ type: "window.maximize" });
        }}
      >
        <div className="workspace-brand">
          <button
            className="icon-button sidebar-toggle"
            aria-label={t(
              collapsed[state.activePage] ? "showSidebar" : "hideSidebar",
            )}
            title={t(
              collapsed[state.activePage] ? "showSidebar" : "hideSidebar",
            )}
            aria-expanded={!collapsed[state.activePage]}
            aria-controls={`sidebar-${state.activePage}`}
            onClick={toggleSidebar}
          >
            <PanelLeft size={18} />
          </button>
          <span className="brand-symbol" aria-hidden>
            <Sparkles size={17} />
          </span>
          <span>ReMe Desktop</span>
        </div>
        <Tabs.List
          ref={tabsRef}
          className="workspace-tabs"
          aria-label={t("navigation")}
        >
          <Tabs.Trigger value="remy">
            <Sparkles size={15} />
            Remy{state.unread > 0 && <i className="tab-dot" />}
          </Tabs.Trigger>
          <Tabs.Trigger value="notebook">
            <Leaf size={15} />
            {t("notebook")}
          </Tabs.Trigger>
          <Tabs.Trigger value="knowledge">
            <BookOpen size={15} />
            {t("knowledge")}
          </Tabs.Trigger>
        </Tabs.List>
        <div className="window-actions">
          <button
            className="icon-button settings-button"
            aria-label={t("settings")}
            title={t("settings")}
            aria-pressed={settingsOpen}
            aria-controls="desktop-settings"
            onClick={() => select(settingsOpen ? lastPage.current : "settings")}
          >
            <SettingsIcon size={18} />
          </button>
          {platform !== "darwin" && (
            <>
              <button
                className="icon-button"
                aria-label={t("minimizeWindow")}
                onClick={() => void action({ type: "window.minimize" })}
              >
                <Minus size={15} />
              </button>
              <button
                className="icon-button"
                aria-label={t("maximizeWindow")}
                onClick={() => void action({ type: "window.maximize" })}
              >
                <Maximize2 size={14} />
              </button>
              <button
                className="icon-button"
                aria-label={t("closeChat")}
                onClick={() => void action({ type: "chat.close" })}
              >
                <X size={16} />
              </button>
            </>
          )}
        </div>
      </header>
      <div
        className="workspace-content"
        hidden={settingsOpen}
        inert={settingsOpen}
      >
        <Tabs.Content value="remy" forceMount className="workspace-page">
          <Remy key={scope} state={state} collapsed={collapsed.remy} />
        </Tabs.Content>
        <Tabs.Content value="notebook" forceMount className="workspace-page">
          <Notebook key={scope} state={state} collapsed={collapsed.notebook} />
        </Tabs.Content>
        <Tabs.Content value="knowledge" forceMount className="workspace-page">
          <Knowledge
            key={scope}
            state={state}
            collapsed={collapsed.knowledge}
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
          companionVisible={bootstrap.layout.visible}
          collapsed={collapsed.settings}
        />
      </div>
    </Tabs.Root>
  );
}
