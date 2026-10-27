"use client";

import { useEffect, useRef, useState } from "react";
import { Bot, FileText, Network, X } from "lucide-react";
import { useI18n } from "./i18n";
import { useWorkspaceStore } from "./store";
import { hasUnsavedChanges, unsavedTabsClosedBy } from "./tab-close";

export default function Tabs() {
  const { t } = useI18n();
  const { tabs, activeTabId, setActiveTab, closeTab, closeOtherTabs } =
    useWorkspaceStore();
  const [contextMenu, setContextMenu] = useState<{
    tabId: string;
    left: number;
    top: number;
  }>();
  const menuRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!contextMenu) return;
    const close = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node))
        setContextMenu(undefined);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setContextMenu(undefined);
    };
    const closeOnViewportChange = () => setContextMenu(undefined);
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", closeOnEscape);
    window.addEventListener("resize", closeOnViewportChange);
    window.addEventListener("scroll", closeOnViewportChange, true);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", closeOnEscape);
      window.removeEventListener("resize", closeOnViewportChange);
      window.removeEventListener("scroll", closeOnViewportChange, true);
    };
  }, [contextMenu]);
  const confirmDiscard = (tabId: string, closeOthers: boolean) => {
    const unsaved = unsavedTabsClosedBy(tabs, tabId, closeOthers);
    return (
      !unsaved.length ||
      window.confirm(
        t("discardUnsavedConfirm", { count: String(unsaved.length) }),
      )
    );
  };
  return (
    <>
      <div className="tabs" role="tablist">
        {tabs.map((tab) => {
          const dirty = hasUnsavedChanges(tab);
          return (
            <button
              key={tab.id}
              className={`tab ${tab.id === activeTabId ? "active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
              onContextMenu={(event) => {
                event.preventDefault();
                setContextMenu({
                  tabId: tab.id,
                  left: Math.max(
                    8,
                    Math.min(event.clientX, window.innerWidth - 176),
                  ),
                  top: Math.max(
                    8,
                    Math.min(event.clientY, window.innerHeight - 82),
                  ),
                });
              }}
              role="tab"
            >
              {tab.type === "agent" ? (
                <Bot size={14} />
              ) : tab.type === "graph" ? (
                <Network size={14} />
              ) : (
                <FileText size={14} />
              )}
              <span>
                {tab.type === "markdown"
                  ? tab.path.split("/").slice(-2).join("/")
                  : tab.type === "graph"
                  ? `${tab.root} · ${t("memoryGraphShort")}`
                  : tab.title || t("newConversation")}
              </span>
              {dirty ? (
                <i className="dirty" />
              ) : (
                <X
                  size={13}
                  onClick={(event) => {
                    event.stopPropagation();
                    closeTab(tab.id);
                  }}
                />
              )}
            </button>
          );
        })}
      </div>
      {contextMenu && (
        <div
          className="tab-context-menu"
          ref={menuRef}
          role="menu"
          style={{ left: contextMenu.left, top: contextMenu.top }}
        >
          <button
            role="menuitem"
            onClick={() => {
              if (!confirmDiscard(contextMenu.tabId, false)) return;
              closeTab(contextMenu.tabId, true);
              setContextMenu(undefined);
            }}
          >
            {t("closeCurrentTab")}
          </button>
          <button
            role="menuitem"
            disabled={tabs.length <= 1}
            onClick={() => {
              if (!confirmDiscard(contextMenu.tabId, true)) return;
              closeOtherTabs(contextMenu.tabId, true);
              setContextMenu(undefined);
            }}
          >
            {t("closeOtherTabs")}
          </button>
        </div>
      )}
    </>
  );
}
