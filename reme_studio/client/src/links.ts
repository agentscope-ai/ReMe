/** Resolve workspace links without allowing them to escape the configured root. */
export function resolveWorkspaceLink(
  href: string,
  current = "",
  workspace = "",
): string | undefined {
  try {
    const wiki = href.startsWith("#reme-file=");
    if (href.startsWith("#") && !wiki) return;
    let target = decodeURIComponent(wiki ? href.slice(11) : href.split("#")[0]);
    if (target.startsWith("file:")) {
      const url = new URL(target);
      if (url.host && url.host !== "localhost") return;
      target = decodeURIComponent(url.pathname);
    }
    target = target.replaceAll("\\", "/");
    const root = workspace.replaceAll("\\", "/").replace(/\/$/, "");
    const absolute = target.startsWith("/") || /^[a-z]:\//i.test(target);
    if (absolute) {
      const windows = /^[a-z]:/i.test(root);
      const compare = (value: string) =>
        windows ? value.toLowerCase() : value;
      if (!root || !compare(target).startsWith(compare(root) + "/")) return;
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
    const path = parts.join("/");
    return wiki && !/\.[^/]+$/.test(path) ? path + ".md" : path;
  } catch {
    return;
  }
}
