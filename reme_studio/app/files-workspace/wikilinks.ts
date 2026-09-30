interface MarkdownNode {
  type: string;
  value?: string;
  url?: string;
  children?: MarkdownNode[];
}

/** Ignore malformed fragments authored directly in Markdown. */
export function workspaceLinkTarget(href?: string): string | undefined {
  const prefix = "#reme-file=";
  if (!href?.startsWith(prefix)) return;
  try {
    return decodeURIComponent(href.slice(prefix.length)) || undefined;
  } catch {
    return;
  }
}

/** Resolve literal workspace wikilinks without rewriting code or existing links. */
export function remarkWikilinks() {
  return (tree: MarkdownNode) => {
    const visit = (node: MarkdownNode) => {
      if (
        !node.children ||
        ["link", "image", "code", "inlineCode"].includes(node.type)
      )
        return;
      node.children = node.children.flatMap((child) => {
        if (child.type !== "text" || !child.value) {
          visit(child);
          return [child];
        }
        const parts: MarkdownNode[] = [];
        let end = 0;
        for (const match of child.value.matchAll(
          /\[\[([^[\]|#\n]+?)(?:#[^[\]|\n]+)?(?:\|([^[\]\n]+))?\]\]/g,
        )) {
          const target = match[1].trim();
          if (!target) continue;
          parts.push({
            type: "text",
            value: child.value.slice(end, match.index),
          });
          parts.push({
            type: "link",
            url: `#reme-file=${encodeURIComponent(target)}`,
            children: [{ type: "text", value: match[2]?.trim() || target }],
          });
          end = match.index + match[0].length;
        }
        if (!end) return [child];
        parts.push({ type: "text", value: child.value.slice(end) });
        return parts;
      });
    };
    visit(tree);
  };
}
