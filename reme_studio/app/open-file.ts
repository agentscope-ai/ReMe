import { readWorkspaceFile } from "./api";
import { useWorkspaceStore } from "./store";

/** Open one workspace file consistently from the navigator, graph, or a link. */
export async function openWorkspaceFile(path: string) {
  const store = useWorkspaceStore.getState();
  const existing = store.tabs.some(
    (tab) => tab.type === "markdown" && tab.path === path,
  );
  const id = store.openMarkdown(path);
  if (existing) return;
  await loadWorkspaceFile(id, path);
}

/** Load both newly opened and restored tabs, respecting the caller's lifetime. */
export async function loadWorkspaceFile(
  id: string,
  path: string,
  shouldApply: () => boolean = () => true,
) {
  const store = useWorkspaceStore.getState();
  try {
    const file = await readWorkspaceFile(path);
    if (shouldApply()) store.hydrateMarkdown(id, file.content, file.stat.mtime);
  } catch (error) {
    if (shouldApply())
      store.failMarkdown(
        id,
        error instanceof Error ? error.message : "Failed to read file",
      );
  }
}
