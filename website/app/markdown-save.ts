import type { WorkspaceTab } from "./types";

type MarkdownTab = Extract<WorkspaceTab, { type: "markdown" }>;

export function markMarkdownContentSaved(
  tab: MarkdownTab,
  savedContent: string,
  mtime?: string,
): MarkdownTab {
  return {
    ...tab,
    savedContent,
    mtime: mtime ?? tab.mtime,
  };
}
