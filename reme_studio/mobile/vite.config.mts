import { readFileSync } from "node:fs";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const versionSource = readFileSync(
  new URL("../../reme/_version.py", import.meta.url),
  "utf8",
);
const version = versionSource.match(/^__version__ = "([^"]+)"$/m)?.[1];
if (!version) throw new Error("Missing ReMe version.");

export default defineConfig(({ mode }) => ({
  base: "./",
  publicDir: false,
  plugins: [react()],
  define: { __REME_VERSION__: JSON.stringify(version) },
  build: {
    outDir: mode === "ios" ? "dist-ios" : "dist-android",
    target: mode === "ios" ? "safari18" : "es2023",
  },
}));
