import { parseDocument, stringify } from "yaml";

export interface FrontmatterEntry {
  key: string;
  value: string;
}

const FRONTMATTER_PATTERN =
  /^\uFEFF?---[ \t]*\r?\n([\s\S]*?)^---[ \t]*(?:\r?\n|$)/m;

/** Invalid YAML stays visible as source instead of silently discarding document data. */
export function parseMarkdownFrontmatter(content: string): {
  body: string;
  entries: FrontmatterEntry[];
} {
  const match = FRONTMATTER_PATTERN.exec(content);
  if (!match || match.index !== 0) return { body: content, entries: [] };
  try {
    const yaml = parseDocument(match[1]);
    if (yaml.errors.length) return { body: content, entries: [] };
    const metadata: unknown = yaml.toJS({ maxAliasCount: 50 });
    if (
      metadata !== null &&
      (typeof metadata !== "object" || Array.isArray(metadata))
    )
      return { body: content, entries: [] };
    const entries = Object.entries(metadata ?? {}).map(([key, value]) => ({
      key,
      value:
        value !== null && typeof value === "object"
          ? stringify(value, { lineWidth: 0 }).trimEnd()
          : String(value ?? ""),
    }));
    return { body: content.slice(match[0].length), entries };
  } catch {
    return { body: content, entries: [] };
  }
}
