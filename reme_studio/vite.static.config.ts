import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv, type Plugin } from "vite";

const staticDir = fileURLToPath(new URL("./static/", import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const demo = mode === "demo";
  const demoApi = fileURLToPath(new URL("./demo/api.ts", import.meta.url));
  const appApi = fileURLToPath(new URL("./app/api", import.meta.url));
  const hiddenPanels = new Set(
    demo
      ? [fileURLToPath(new URL("./app/settings-center", import.meta.url))]
      : ["DemoBar", "DemoTags"].map((name) =>
          fileURLToPath(new URL(`./demo/${name}`, import.meta.url)),
        ),
  );
  const browserApi: Plugin = {
    name: "reme-demo-api",
    enforce: "pre",
    transformIndexHtml(html) {
      return demo
        ? html.replace(
            "<title>ReMe Workspace</title>",
            "<title>Try ReMe Studio</title>",
          )
        : html;
    },
    resolveId(id, importer) {
      if (!importer) return;
      const resolved = path.resolve(path.dirname(importer), id);
      if (hiddenPanels.has(resolved)) return "\0reme-hidden-panel";
      if (demo && resolved === appApi) return demoApi;
    },
    load(id) {
      if (id === "\0reme-hidden-panel") return "export default () => null";
    },
  };

  return {
    base: demo ? "./" : "/",
    plugins: [browserApi, react()],
    resolve: {
      alias: {
        "next/dynamic": path.resolve(staticDir, "next-dynamic.tsx"),
      },
    },
    define: {
      "process.env.NEXT_PUBLIC_REME_DEMO": JSON.stringify(String(demo)),
      "process.env.NEXT_PUBLIC_REME_API_URL": JSON.stringify(
        env.VITE_REME_API_URL || "/",
      ),
      "process.env.NEXT_PUBLIC_REME_WORKSPACE_EXTENSIONS": JSON.stringify(
        demo ? "md" : env.VITE_REME_WORKSPACE_EXTENSIONS ?? "",
      ),
    },
    build: {
      outDir: demo ? "dist-demo" : "dist-static",
      emptyOutDir: true,
      sourcemap: mode !== "production" && !demo,
    },
  };
});
