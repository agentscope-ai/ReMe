import { parseDocument } from "yaml";

export interface FileTree {
  name: string;
  path: string;
  children?: FileTree[];
}
export function buildTree(paths: string[], root: string): FileTree[] {
  const tree: FileTree[] = [];
  for (const path of paths) {
    if (path.split("/").some((part) => !part || part.startsWith("."))) continue;
    const relative =
      root && path.startsWith(root + "/") ? path.slice(root.length + 1) : path;
    const parts = relative.split("/");
    let level = tree;
    parts.forEach((name, index) => {
      let node = level.find((entry) => entry.name === name);
      if (!node) {
        node = {
          name,
          path: (root ? root + "/" : "") + parts.slice(0, index + 1).join("/"),
          ...(index < parts.length - 1 ? { children: [] } : {}),
        };
        level.push(node);
      }
      if (node.children) level = node.children;
    });
  }
  return tree;
}

export function parseMarkdown(content: string) {
  const empty = {
    body: content,
    metadata: {} as Record<string, unknown>,
    rawFrontmatter: "",
    frontmatterError: false,
  };
  const opening = /^\uFEFF?---[ \t]*\r?\n/.exec(content);
  if (!opening) return empty;
  const rest = content.slice(opening[0].length);
  const closing = /^---[ \t]*(?:\r?\n|$)/m.exec(rest);
  if (!closing) return empty;
  const rawFrontmatter = rest.slice(0, closing.index);
  const body = rest.slice(closing.index + closing[0].length);
  const invalid = { ...empty, body, rawFrontmatter, frontmatterError: true };
  try {
    const document = parseDocument(rawFrontmatter);
    if (document.errors.length) return invalid;
    const raw: unknown = document.toJS({ maxAliasCount: 50 });
    if (raw !== null && (typeof raw !== "object" || Array.isArray(raw)))
      return invalid;
    const metadata = (raw ?? {}) as Record<string, unknown>;
    return { body, metadata, rawFrontmatter, frontmatterError: false };
  } catch {
    return invalid;
  }
}
