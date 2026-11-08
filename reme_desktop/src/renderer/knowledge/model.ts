import { parseDocument } from "yaml";
import { fromMarkdown } from "mdast-util-from-markdown";

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

export function parseMarkdown(content: string, tagKey = "memory_tags") {
  const empty = {
    body: content,
    tags: [] as string[],
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
    const tags = Array.isArray(metadata[tagKey])
      ? (metadata[tagKey] as unknown[])
          .filter(
            (value) => typeof value === "string" || typeof value === "number",
          )
          .map(String)
      : [];
    return { body, tags, metadata, rawFrontmatter, frontmatterError: false };
  } catch {
    return invalid;
  }
}

/** Promote a leading Markdown title once, respecting code fences and Setext headings. */
export function documentPresentation(
  path: string,
  parsed: ReturnType<typeof parseMarkdown>,
) {
  const explicit = [parsed.metadata.title, parsed.metadata.name]
    .find(
      (value): value is string =>
        typeof value === "string" && Boolean(value.trim()),
    )
    ?.trim();
  const first = fromMarkdown(parsed.body).children[0];
  const titleNode =
    first?.type === "heading" && first.depth === 1 ? first : undefined;
  const text = (node: {
    value?: string;
    alt?: string;
    children?: unknown[];
  }): string =>
    node.value ??
    node.alt ??
    node.children
      ?.map((child) => text(child as Parameters<typeof text>[0]))
      .join("") ??
    "";
  const heading = titleNode ? text(titleNode) : undefined;
  const title =
    explicit ??
    heading ??
    path
      .split("/")
      .at(-1)
      ?.replace(/\.[^.]+$/, "") ??
    path;
  const consumed = heading && (!explicit || explicit === heading);
  const body =
    consumed && titleNode?.position?.end.offset !== undefined
      ? parsed.body
          .slice(titleNode.position.end.offset)
          .replace(/^(?:[ \t]*\r?\n)+/, "")
      : parsed.body;
  return { title, body };
}

export function resolveDocumentLink(
  target: string,
  current: string,
  paths: string[],
): string | undefined {
  if (!target || /^(?:[a-z]+:|\/|~)/i.test(target) || target.includes("\\"))
    return;
  const clean = (value: string) => {
    const result: string[] = [];
    for (const part of value.split("/")) {
      if (part === "..") {
        if (!result.length) return;
        result.pop();
      } else if (part && part !== ".") result.push(part);
    }
    return result.join("/");
  };
  const parent = current.split("/").slice(0, -1).join("/");
  const candidate = clean(parent + "/" + target);
  if (!candidate) return;
  if (paths.includes(candidate)) return candidate;
  if (paths.includes(candidate + ".md")) return candidate + ".md";
  // Ordinary Markdown links resolve from their document, even outside the loaded tree.
  if (current && /\.(mdx?|txt|csv)$/i.test(candidate)) return candidate;
  const matches = paths.filter((path) => {
    const name = path.split("/").at(-1);
    return name === target || name?.replace(/\.mdx?$/i, "") === target;
  });
  if (matches.length === 1) return matches[0];
  if (/\.(mdx?|txt|csv)$/i.test(candidate)) return candidate;
  return target.includes("/") ? candidate + ".md" : undefined;
}
