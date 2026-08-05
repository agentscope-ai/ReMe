import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv } from "vite";

const staticDir = fileURLToPath(new URL("./static/", import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return {
    plugins: [react()],
    resolve: {
      alias: {
        "next/dynamic": path.resolve(staticDir, "next-dynamic.tsx"),
      },
    },
    define: {
      "process.env.NEXT_PUBLIC_REME_API_URL": JSON.stringify(
        env.VITE_REME_API_URL ?? "",
      ),
      "process.env.NEXT_PUBLIC_REME_WORKSPACE_EXTENSIONS": JSON.stringify(
        env.VITE_REME_WORKSPACE_EXTENSIONS ?? "",
      ),
    },
    build: {
      outDir: "dist-static",
      emptyOutDir: true,
      sourcemap: mode !== "production",
    },
  };
});
