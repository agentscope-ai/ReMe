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
  try {
    const file = await readWorkspaceFile(path);
    store.hydrateMarkdown(id, file.content, file.stat.mtime);
  } catch (error) {
    store.failMarkdown(
      id,
      error instanceof Error ? error.message : "Failed to read file",
    );
  }
}
