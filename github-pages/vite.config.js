import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  publicDir: ".generated",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
