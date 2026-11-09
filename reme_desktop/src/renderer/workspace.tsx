import { useEffect, useRef, useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import {
  BookOpen,
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

export default function Workspace(bootstrap: Bootstrap) {
  const { state, platform, version } = bootstrap;
  const t = translator(state.settings.language);
  const [collapsed, setCollapsed] = useState<
    Partial<Record<DesktopPage, boolean>>
  >(() => {
    try {
      return Object.fromEntries(
        (["remy", "knowledge", "settings"] as const).map((page) => [
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
  const lastPage = useRef<"remy" | "knowledge">("remy");
  const settingsOpen = state.activePage === "settings";
  useEffect(() => {
    if (state.activePage !== "settings") lastPage.current = state.activePage;
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
          <span className="brand-symbol">
            <Sparkles size={17} />
          </span>
          <span>
            Remy<span className="brand-subtitle">ReMe</span>
          </span>
        </div>
        <Tabs.List className="workspace-tabs" aria-label={t("navigation")}>
          <Tabs.Trigger value="remy">
            <Sparkles size={15} />
            Remy{state.unread > 0 && <i className="tab-dot" />}
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
          <Remy
            state={state}
            collapsed={collapsed.remy}
            onSettings={() => select("settings")}
          />
        </Tabs.Content>
        <Tabs.Content value="knowledge" forceMount className="workspace-page">
          <Knowledge state={state} collapsed={collapsed.knowledge} />
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
          collapsed={collapsed.settings}
        />
      </div>
    </Tabs.Root>
  );
}
