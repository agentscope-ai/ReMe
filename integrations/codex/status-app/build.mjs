import { build } from "esbuild";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";

const result = await build({
  entryPoints: [new URL("./src/status.js", import.meta.url).pathname],
  bundle: true,
  write: false,
  format: "iife",
  target: "es2022",
  legalComments: "inline",
});
const template = await readFile(new URL("./src/status.html", import.meta.url), "utf8");
const script = result.outputFiles[0].text.replaceAll("</script", "<\\/script");
const output = new URL("../plugins/reme/ui/", import.meta.url);
await mkdir(output, { recursive: true });
await writeFile(new URL("status.html", output), template.replace("/* STATUS_APP */", () => script));
await copyFile(new URL("./node_modules/@openai/mcp-extensions/LICENSE", import.meta.url), new URL("LICENSE-mcp-extensions.txt", output));
console.log("Built plugins/reme/ui/status.html (self-contained; no runtime Node dependency)");
