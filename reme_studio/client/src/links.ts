/** Resolve workspace links without allowing them to escape the configured root. */
export function resolveWorkspaceLink(
  href: string,
  current = "",
  workspace = "",
): string | undefined {
  try {
    const wikiPrefix = "#reme-file=";
    const wiki = href.startsWith(wikiPrefix);
    if (href.startsWith("#") && !wiki) return;
    const root = workspace.replaceAll("\\", "/").replace(/\/$/, "");
    const windows = /^[a-z]:/i.test(root);
    let path = wiki ? href.slice(wikiPrefix.length) : href.split("#")[0];
    if (!wiki && /^file:/i.test(path)) {
      const url = new URL(path);
      if (url.host && url.host !== "localhost") return;
      path = url.pathname;
      if (windows && /^\/[a-z]:\//i.test(path)) path = path.slice(1);
    }
    let target = decodeURIComponent(path).replaceAll("\\", "/");
    const absolute = target.startsWith("/") || /^[a-z]:\//i.test(target);
    if (absolute) {
      const compare = (value: string) =>
        windows ? value.toLowerCase() : value;
      if (!workspace || !compare(target).startsWith(compare(root) + "/"))
        return;
      target = target.slice(root.length + 1);
    } else if (/^[a-z][a-z\d+.-]*:/i.test(target) || target.startsWith("~"))
      return;
    if (!target || target.includes("\0")) return;
    const parts: string[] = [];
    const relative =
      !wiki && !absolute && current
        ? current.split("/").slice(0, -1).join("/") + "/" + target
        : target;
    for (const part of relative.split("/")) {
      if (part === "..") {
        if (!parts.length) return;
        parts.pop();
      } else if (part && part !== ".") parts.push(part);
    }
    if (!parts.length) return;
    const relativePath = parts.join("/");
    return wiki && !/\.[^/]+$/.test(relativePath)
      ? relativePath + ".md"
      : relativePath;
  } catch {
    return;
  }
}
