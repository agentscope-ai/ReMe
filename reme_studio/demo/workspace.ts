import { parseDocument } from "yaml";
import type { FileStat, GraphSnapshot } from "../app/types.ts";
import { parseMarkdownFrontmatter } from "../app/files-workspace/markdown.ts";

export const DEMO_STORAGE_KEY = "reme-demo-files-v1";
export const WELCOME_PATH = "START_HERE.md";
const INITIAL_MTIME = "2026-09-18T09:00:00.000Z";

type SavedFile = { content: string; mtime: string };
type StoragePort = Pick<Storage, "getItem" | "setItem" | "removeItem">;

/** Tags are derived from YAML source files, with ReMe's default list/count limits. */
export function memoryTags(content: string): string[] {
  const block = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(content)?.[1];
  if (!block) return [];
  let value: unknown;
  try {
    const document = parseDocument(block);
    if (document.errors.length) return [];
    value = document.toJS({ maxAliasCount: 50 })?.memory_tags;
  } catch {
    return [];
  }
  if (!Array.isArray(value)) return [];
  const tags = value.flatMap((item: unknown) => {
    if (
      typeof item !== "string" &&
      !(typeof item === "number" && Number.isInteger(item))
    )
      return [];
    const raw = String(item).trim().replace(/\s+/g, "_");
    if (!raw || [...raw].length > 64 || !/[\p{L}\p{N}]/u.test(raw)) return [];
    return [raw.toLowerCase()];
  });
  return [...new Set(tags)].slice(0, 3);
}

export function wikilinks(content: string) {
  return [
    ...content.matchAll(
      /!?\[\[([^[\]|#\n]+?)(?:#([^[\]|\n]+))?(?:\|[^[\]\n]+)?\]\]/g,
    ),
  ].map((match) => ({
    target: match[1].trim(),
    anchor: match[2]?.trim() || null,
  }));
}

export function createDemoWorkspace(
  seeds: Record<string, string>,
  storage?: StoragePort,
  key = DEMO_STORAGE_KEY,
) {
  function savedFiles(): Record<string, SavedFile> {
    try {
      const saved: unknown = JSON.parse(storage?.getItem(key) || "{}");
      if (!saved || typeof saved !== "object" || Array.isArray(saved))
        return {};
      return Object.fromEntries(
        Object.entries(saved).filter(
          ([path, file]) =>
            Object.hasOwn(seeds, path) &&
            file &&
            typeof file.content === "string" &&
            typeof file.mtime === "string" &&
            Number.isFinite(Date.parse(file.mtime)),
        ),
      );
    } catch {
      return {};
    }
  }
  let changes = savedFiles();
  const files = () => {
    if (storage) changes = savedFiles();
    return Object.fromEntries(
      Object.entries(seeds).map(([path, content]) => [
        path,
        changes[path]?.content ?? content,
      ]),
    );
  };
  const stat = (path: string, content: string): FileStat => ({
    path,
    exists: true,
    type: "file",
    mtime: changes[path]?.mtime || INITIAL_MTIME,
    size: new TextEncoder().encode(content).length,
  });
  const read = (path: string) => {
    if (!Object.hasOwn(seeds, path)) throw new Error(`File not found: ${path}`);
    const content = files()[path];
    return { content, stat: stat(path, content) };
  };
  const search = (query = "", tags: string[] = []) => {
    const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    return Object.entries(files()).flatMap(([path, content]) => {
      const parsed = parseMarkdownFrontmatter(content);
      const fileTags = memoryTags(content);
      const title =
        parsed.entries.find((entry) => entry.key === "name")?.value || path;
      const haystack = `${path}\n${content}\n${fileTags.join(
        " ",
      )}`.toLowerCase();
      if (
        (tags.length && !tags.some((tag) => fileTags.includes(tag))) ||
        !terms.every((term) => haystack.includes(term))
      )
        return [];
      const lines = parsed.body
        .split(/\r?\n/)
        .filter((line) => line.trim() && !line.startsWith("#"));
      const snippet =
        lines.find(
          (line) =>
            terms.length &&
            terms.some((term) => line.toLowerCase().includes(term)),
        ) ||
        lines[0] ||
        "";
      return [{ path, title, tags: fileTags, snippet: snippet.slice(0, 180) }];
    });
  };
  return {
    files,
    read,
    search,
    save(path: string, content: string, expectedMtime?: string) {
      const previous = read(path).stat.mtime!;
      if (expectedMtime && expectedMtime !== previous)
        throw new Error(
          "This file changed in another tab. Reopen it before saving. / 文件已在其他标签页修改，请重新打开后保存。",
        );
      if (new TextEncoder().encode(content).length > 512_000)
        throw new Error("Demo files must be smaller than 512 KB.");
      const mtime = new Date(
        Math.max(Date.now(), Date.parse(previous) + 1),
      ).toISOString();
      const next = { ...changes, [path]: { content, mtime } };
      storage?.setItem(key, JSON.stringify(next));
      changes = next;
      return stat(path, content);
    },
    reset() {
      storage?.removeItem(key);
      changes = {};
    },
    tags() {
      const counts = new Map<string, number>();
      search().forEach((file) =>
        file.tags.forEach((tag) => counts.set(tag, (counts.get(tag) || 0) + 1)),
      );
      return [...counts]
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([tag, count]) => ({ tag, count }));
    },
    graph(): GraphSnapshot {
      const entries = Object.entries(files());
      const nodes: GraphSnapshot["nodes"] = entries.map(([path, content]) => {
        const { entries: metadata } = parseMarkdownFrontmatter(content);
        return {
          id: path,
          path,
          name:
            metadata.find((entry) => entry.key === "name")?.value ||
            path.split("/").pop()!,
          description:
            metadata.find((entry) => entry.key === "description")?.value || "",
          indexed: true,
          virtual: false,
        };
      });
      const edges: GraphSnapshot["edges"] = [];
      const known = new Set(nodes.map((node) => node.id));
      const seen = new Set<string>();
      for (const [source, content] of entries)
        for (const { target, anchor } of wikilinks(content)) {
          const key = `${source}\0${target}\0${anchor}`;
          if (!target || seen.has(key)) continue;
          seen.add(key);
          if (!known.has(target)) {
            known.add(target);
            nodes.push({
              id: target,
              path: target,
              name: target,
              description: "",
              indexed: false,
              virtual: false,
            });
          }
          edges.push({ source, target, target_anchor: anchor });
        }
      for (const root of ["wiki", "personal", "procedure"]) {
        const id = `virtual:${root}`;
        nodes.push({
          id,
          path: "",
          name: root,
          description: "",
          indexed: false,
          virtual: true,
        });
        entries
          .filter(([path]) => path.startsWith(`digest/${root}/`))
          .forEach(([path]) =>
            edges.push({ source: id, target: path, target_anchor: null }),
          );
      }
      return { version: 1, nodes, edges };
    },
  };
}
