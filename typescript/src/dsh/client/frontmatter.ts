/** One displayable top-level YAML frontmatter field. */
export interface FrontmatterEntry {
  key: string;
  value: string;
}

const FRONTMATTER_PATTERN = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/;

/** Split a leading YAML frontmatter block from its Markdown body. */
export function parseMarkdownFrontmatter(content: string): {
  body: string;
  entries: FrontmatterEntry[];
} {
  const match = FRONTMATTER_PATTERN.exec(content);
  if (!match) return { body: content, entries: [] };
  const entries: FrontmatterEntry[] = [];
  for (const line of (match[1] ?? "").split(/\r?\n/)) {
    const separator = line.indexOf(":");
    if (separator > 0 && !/^\s/.test(line)) {
      entries.push({
        key: line.slice(0, separator).trim(),
        value: line.slice(separator + 1).trim(),
      });
      continue;
    }
    const previous = entries.at(-1);
    const continuation = line.trim();
    if (previous !== undefined && continuation) {
      previous.value = previous.value
        ? `${previous.value}\n${continuation}`
        : continuation;
    }
  }
  return { body: content.slice(match[0].length), entries };
}
