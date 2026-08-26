import { mkdir, readFile, unlink, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const output = resolve(root, "dist/dsh/client.js");
const temporary = resolve(root, "dist/dsh/client.bundle.cjs");

await mkdir(dirname(output), { recursive: true });
await build({
  entryPoints: [resolve(root, "src/dsh/client/index.tsx")],
  outfile: temporary,
  bundle: true,
  format: "cjs",
  platform: "browser",
  target: "es2022",
  jsx: "automatic",
  external: [
    "react",
    "react/jsx-runtime",
    "@deepseek-ai/dsh-client-ui-primitives",
  ],
  sourcemap: false,
  logLevel: "info",
});
const body = await readFile(temporary, "utf8");
const wrapped = `window.__ModuleLoader__.load({\n  id: "@agentscope-ai/reme",\n  factory: (require) => {\n    var module = { exports: {} };\n    var exports = module.exports;\n${body
  .split("\n")
  .map((line) => `    ${line}`)
  .join("\n")}\n    return module.exports;\n  }\n});\n`;
await writeFile(output, wrapped);
await unlink(temporary);
