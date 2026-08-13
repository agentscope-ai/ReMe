const FRONTMATTER_PATTERN = /^\uFEFF?---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/;

/** Remove a leading YAML frontmatter block before rendering Markdown. */
export function stripMarkdownFrontmatter(markdown) {
  return markdown.replace(FRONTMATTER_PATTERN, "");
}
