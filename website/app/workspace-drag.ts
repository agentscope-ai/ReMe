export const WORKSPACE_FILE_DRAG_TYPE = "application/x-reme-workspace-file";

export function absoluteWorkspacePath(
  workspaceDir: string,
  relativePath: string,
): string {
  const root = workspaceDir.trim().replace(/[\\/]+$/, "");
  const relative = relativePath.trim().replace(/^[\\/]+/, "");
  const separator = root.includes("\\") && !root.includes("/") ? "\\" : "/";
  return `${root}${separator}${relative.replace(/[\\/]+/g, separator)}`;
}

export function appendWorkspaceFileReference(
  input: string,
  absolutePath: string,
): string {
  const reference = `\`${absolutePath.replace(/`/g, "\\`")}\``;
  if (input.includes(reference)) return input;
  const current = input.trimEnd();
  return `${current}${current ? "\n" : ""}${reference}`;
}
