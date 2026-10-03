import { execFileSync } from "node:child_process";
import { cp, rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const studioDir = fileURLToPath(new URL("../../reme_studio/", import.meta.url));
const destination = new URL("../.generated/site/public/studio/", import.meta.url);
execFileSync(process.platform === "win32" ? "npm.cmd" : "npm", ["run", "build:demo"], {
  cwd: studioDir,
  stdio: "inherit",
});
await rm(destination, { recursive: true, force: true });
await cp(new URL("../../reme_studio/dist-demo/", import.meta.url), destination, { recursive: true });
