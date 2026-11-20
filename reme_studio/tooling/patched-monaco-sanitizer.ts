import type { Plugin } from "vite";

/**
 * Monaco ships a bundled DOMPurify snapshot. Resolve its internal sanitizer
 * to our security-pinned DOMPurify dependency so app and editor do not ship
 * different sanitizer versions.
 */
export function patchedMonacoSanitizer(): Plugin {
  const virtualId = "\0reme-monaco-patched-dompurify";
  return {
    name: "reme-monaco-patched-dompurify",
    enforce: "pre",
    resolveId(source, importer) {
      if (
        source === "./dompurify/dompurify.js" &&
        importer
          ?.replaceAll("\\", "/")
          .includes("/monaco-editor/esm/vs/base/browser/domSanitize.js")
      ) {
        return virtualId;
      }
    },
    load(id) {
      if (id === virtualId) return 'export { default } from "dompurify";';
    },
  };
}
