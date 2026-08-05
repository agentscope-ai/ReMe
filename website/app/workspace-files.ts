import type { TreeNode } from "./types";

const defaults = ["md", "txt"];

export type WorkspaceDirectoryConfig = {
  daily_dir: string;
  digest_dir: string;
};

export type WorkspaceFileSource = "workspace" | "daily" | "digest";

export function parseWorkspaceExtensions(value?: string): Set<string> {
  const extensions = (value || "")
    .split(",")
    .map((item) => item.trim().toLowerCase().replace(/^\./, ""))
    .filter(Boolean);
  return new Set(extensions.length ? extensions : defaults);
}

export function filterWorkspacePaths(
  paths: string[],
  extensions: Set<string>,
): string[] {
  return paths.filter((path) => {
    const parts = path.split("/");
    const extension = parts.at(-1)?.split(".").pop()?.toLowerCase() || "";
    return (
      parts.every((part) => part && !part.startsWith(".")) &&
      extensions.has(extension)
    );
  });
}

/** Build a hierarchy while preserving the API's newest-modified-first path order. */
export function buildTree(
  paths: string[],
  extensions: Set<string>,
  rootDirectory = "",
): TreeNode[] {
  const root: TreeNode[] = [];
  for (const path of filterWorkspacePaths(paths, extensions)) {
    const relativePath =
      rootDirectory && path.startsWith(`${rootDirectory}/`)
        ? path.slice(rootDirectory.length + 1)
        : path;
    const parts = relativePath.split("/").filter(Boolean);
    let level = root;
    parts.forEach((name, index) => {
      const relativeNodePath = parts.slice(0, index + 1).join("/");
      const nodePath = rootDirectory
        ? `${rootDirectory}/${relativeNodePath}`
        : relativeNodePath;
      let node = level.find((item) => item.name === name);
      if (!node) {
        node = {
          name,
          path: nodePath,
          type: index === parts.length - 1 ? "file" : "directory",
          children: [],
        };
        level.push(node);
      }
      level = node.children;
    });
  }
  return root;
}

function cleanDirectory(value: string): string {
  return value.replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
}

export function sourceDirectory(
  source: WorkspaceFileSource,
  config: WorkspaceDirectoryConfig,
): string {
  if (source === "daily") return cleanDirectory(config.daily_dir);
  if (source === "digest") return cleanDirectory(config.digest_dir);
  return "";
}

export function filterPathsBySource(
  paths: string[],
  source: WorkspaceFileSource,
  config: WorkspaceDirectoryConfig,
): string[] {
  const directory = sourceDirectory(source, config);
  if (!directory) return paths;
  return paths.filter(
    (path) => path === directory || path.startsWith(`${directory}/`),
  );
}
