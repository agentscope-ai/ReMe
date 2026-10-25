"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { IS_DEMO } from "./studio-mode";
import { loadWorkspaceFile, openWorkspaceFile } from "./open-file";
import { SparkMenuExpandLine, SparkMenuFoldLine } from "@agentscope-ai/icons";
import { BookOpenText, Settings, Sparkles } from "lucide-react";
import FilesNavigator from "./files-workspace/FilesNavigator";
import MemoryGraphView from "./files-workspace/MemoryGraphView";
import { clampNavigatorWidth } from "./files-workspace/panel-resize";
import { useI18n, useLanguageStore } from "./i18n";
import { useWorkspaceStore } from "./store";
import { useThemeStore } from "./theme";
import type { WorkspaceTab } from "./types";
import ThemeMenu from "./workspace-theme";
import Tabs from "./workspace-tabs";
import Chat from "./workspace-chat";
import packageJson from "../package.json";

const SettingsCenter = IS_DEMO
  ? null
  : dynamic(() => import("./settings-center"), { ssr: false });
const DemoBar = IS_DEMO
  ? dynamic(() => import("../demo/DemoBar"), { ssr: false })
  : null;

const TabbedEditor = dynamic(() => import("./files-workspace/TabbedEditor"), {
  ssr: false,
});

function GitHubIcon({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M12 .7a11.5 11.5 0 0 0-3.64 22.4c.58.1.79-.25.79-.56v-2.23c-3.23.7-3.91-1.37-3.91-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.16.08 1.78 1.2 1.78 1.2 1.04 1.77 2.72 1.26 3.38.96.1-.75.4-1.26.74-1.55-2.58-.29-5.29-1.29-5.29-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.47.11-3.05 0 0 .97-.31 3.16 1.18A11 11 0 0 1 12 6.11c.98 0 1.96.13 2.88.39 2.2-1.49 3.16-1.18 3.16-1.18.63 1.58.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.4-2.72 5.38-5.3 5.67.42.36.79 1.07.79 2.16v3.25c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z" />
    </svg>
  );
}

function Workspace() {
  const { language, setLanguage, t } = useI18n();
  const hydrateLanguage = useLanguageStore((state) => state.hydrate);
  const hydrateTheme = useThemeStore((state) => state.hydrate);
  const [navOpen, setNavOpen] = useState(true);
  const [navigatorWidth, setNavigatorWidth] = useState(260);
  const [resizingNavigator, setResizingNavigator] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { tabs, activeTabId, openAgent } = useWorkspaceStore();
  const active = tabs.find((tab) => tab.id === activeTabId);
  useEffect(() => {
    let mounted = true;
    void Promise.resolve(useWorkspaceStore.persist.rehydrate()).then(
      async () => {
        if (IS_DEMO && !useWorkspaceStore.getState().tabs.length && mounted) {
          await openWorkspaceFile("START_HERE.md");
        }
        const restoredFiles = useWorkspaceStore
          .getState()
          .tabs.filter(
            (tab): tab is Extract<WorkspaceTab, { type: "markdown" }> =>
              tab.type === "markdown" && Boolean(tab.loading),
          );
        await Promise.all(
          restoredFiles.map((tab) =>
            loadWorkspaceFile(tab.id, tab.path, () => mounted),
          ),
        );
      },
    );
    return () => {
      mounted = false;
    };
  }, []);
  useEffect(() => {
    hydrateLanguage();
  }, [hydrateLanguage]);
  useEffect(() => {
    hydrateTheme();
  }, [hydrateTheme]);
  useEffect(() => {
    const saved = Number(localStorage.getItem("reme-navigator-width"));
    if (!Number.isFinite(saved) || saved <= 0) return;
    const frame = requestAnimationFrame(() =>
      setNavigatorWidth(clampNavigatorWidth(saved, window.innerWidth)),
    );
    return () => cancelAnimationFrame(frame);
  }, []);

  const resizeNavigator = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const navigator = event.currentTarget.parentElement;
    const surface = navigator?.parentElement;
    if (!navigator || !surface) return;
    setResizingNavigator(true);
    const startX = event.clientX;
    const initialWidth = navigator.getBoundingClientRect().width;
    const containerWidth = surface.getBoundingClientRect().width;
    const move = (nextEvent: PointerEvent) => {
      setNavigatorWidth(
        clampNavigatorWidth(
          initialWidth + nextEvent.clientX - startX,
          containerWidth,
        ),
      );
    };
    const stop = (nextEvent: PointerEvent) => {
      const finalWidth = clampNavigatorWidth(
        initialWidth + nextEvent.clientX - startX,
        containerWidth,
      );
      setNavigatorWidth(finalWidth);
      localStorage.setItem("reme-navigator-width", String(finalWidth));
      setResizingNavigator(false);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
  };

  const resizeNavigatorWithKeyboard = (
    event: React.KeyboardEvent<HTMLDivElement>,
  ) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const containerWidth =
      event.currentTarget.parentElement?.parentElement?.getBoundingClientRect()
        .width ?? window.innerWidth;
    setNavigatorWidth((current) => {
      const next = clampNavigatorWidth(
        current + (event.key === "ArrowRight" ? 24 : -24),
        containerWidth,
      );
      localStorage.setItem("reme-navigator-width", String(next));
      return next;
    });
  };

  return (
    <main
      className={`shell ${navOpen ? "" : "nav-closed"} ${
        resizingNavigator ? "navigator-is-resizing" : ""
      }`}
      style={
        { "--navigator-width": `${navigatorWidth}px` } as React.CSSProperties
      }
    >
      <header className="topbar">
        <button
          className={`menu ${navOpen ? "is-open" : ""}`}
          onClick={() => setNavOpen(!navOpen)}
          aria-label={t("toggleNavigator")}
          aria-expanded={navOpen}
          aria-controls="workspace-navigator"
        >
          <span className="menu-icons" aria-hidden="true">
            <SparkMenuFoldLine size={20} className="menu-fold-icon" />
            <SparkMenuExpandLine size={20} className="menu-expand-icon" />
          </span>
        </button>
        <strong>
          <span className="brand-icon" aria-hidden="true" />
          <span className="brand-title">ReMe Studio</span>
          <span className="app-version-divider" aria-hidden="true" />
          <span className="app-version">v{packageJson.version}</span>
        </strong>
        <span>
          {active?.type === "markdown"
            ? active.path
            : active?.type === "graph"
            ? `${active.root} · ${t("memoryGraph")}`
            : active?.title || t("workspace")}
        </span>
        <div className="topbar-actions">
          <nav className="resource-links" aria-label={t("officialWebsite")}>
            <a
              href="https://reme.agentscope.io"
              target="_blank"
              rel="noreferrer"
            >
              <BookOpenText size={17} />
              <span>{t("officialWebsite")}</span>
            </a>
            <i aria-hidden="true" />
            <a
              href="https://github.com/modelscope/ReMe"
              target="_blank"
              rel="noreferrer"
            >
              <GitHubIcon />
              <span>{t("github")}</span>
            </a>
          </nav>
          <div className="topbar-divider" aria-hidden="true" />
          <div
            className="language-switch"
            aria-label={t("switchLanguage")}
            role="group"
          >
            <button
              className={language === "zh" ? "active" : ""}
              onClick={() => setLanguage("zh")}
            >
              中文
            </button>
            <button
              className={language === "en" ? "active" : ""}
              onClick={() => setLanguage("en")}
            >
              EN
            </button>
          </div>
          <ThemeMenu />
          {SettingsCenter && (
            <button
              className="settings-trigger"
              onClick={() => setSettingsOpen(true)}
              aria-label={t("settings")}
              title={t("settings")}
            >
              <Settings size={18} />
            </button>
          )}
        </div>
      </header>
      {DemoBar && <DemoBar />}
      <div className="surface">
        <FilesNavigator
          open={navOpen}
          width={navigatorWidth}
          resizing={resizingNavigator}
          onResizeStart={resizeNavigator}
          onResizeKeyDown={resizeNavigatorWithKeyboard}
        />
        <section className="workbench">
          <Tabs />
          <div className="content">
            {!active && (
              <div className="welcome">
                <h1 className="welcome-brand">
                  <span
                    className="welcome-wordmark"
                    role="img"
                    aria-label="ReMe"
                  />
                  <span>Studio</span>
                </h1>
                <p>{t("welcomeDescription")}</p>
                <button onClick={openAgent}>
                  <Sparkles size={16} />
                  {t("startChat")}
                </button>
                <small>{t("localFiles")}</small>
              </div>
            )}
            {active?.type === "markdown" && (
              <TabbedEditor key={active.id} tab={active} />
            )}
            {active?.type === "agent" && <Chat tab={active} />}
            {active?.type === "graph" && (
              <MemoryGraphView key={active.id} root={active.root} />
            )}
          </div>
        </section>
      </div>
      {SettingsCenter && (
        <SettingsCenter
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
        />
      )}
    </main>
  );
}

export function ReMeWorkspace() {
  return <Workspace />;
}
