import type { WorkspaceTab } from "./types";

export function hasUnsavedChanges(tab: WorkspaceTab): boolean {
  return tab.type === "markdown" && tab.content !== tab.savedContent;
}

export function unsavedTabsClosedBy(
  tabs: WorkspaceTab[],
  tabId: string,
  closeOthers: boolean,
): WorkspaceTab[] {
  return tabs.filter(
    (tab) =>
      hasUnsavedChanges(tab) &&
      (closeOthers ? tab.id !== tabId : tab.id === tabId),
  );
}
