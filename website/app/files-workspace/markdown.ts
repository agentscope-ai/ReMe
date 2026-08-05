export interface FrontmatterEntry {
  key: string;
  value: string;
}

const FRONTMATTER_PATTERN = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/;

/** Split a leading YAML frontmatter block from the Markdown body. Copied from QwenPaw. */
export function parseMarkdownFrontmatter(content: string): {
  body: string;
  entries: FrontmatterEntry[];
} {
  const match = FRONTMATTER_PATTERN.exec(content);
  if (!match) return { body: content, entries: [] };
  const entries = match[1]
    .split(/\r?\n/)
    .map((line) => {
      const separator = line.indexOf(":");
      if (separator <= 0 || /^\s/.test(line)) return null;
      return {
        key: line.slice(0, separator).trim(),
        value: line.slice(separator + 1).trim(),
      };
    })
    .filter((entry): entry is FrontmatterEntry => entry !== null);
  return { body: content.slice(match[0].length), entries };
}
