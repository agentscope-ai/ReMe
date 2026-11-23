import { readFileSync } from "node:fs";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const source = readFileSync(
  new URL("../../reme/_version.py", import.meta.url),
  "utf8",
);
const version = source.match(/^__version__ = "([^"]+)"$/m)?.[1];
if (!version) throw new Error("Missing ReMe version.");

export default defineConfig({
  publicDir: "../workbench/public",
  plugins: [react()],
  define: { __REME_VERSION__: JSON.stringify(version) },
  build: {
    outDir: "dist-static",
    target: "es2023",
    // Keep light-dark() tied to Studio's color-scheme setting.
    cssTarget: ["chrome123", "edge123", "firefox120", "safari17.5"],
  },
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:2333",
        rewrite: (path) => path.slice(4),
      },
    },
  },
});
