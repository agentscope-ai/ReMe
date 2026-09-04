import { execFile } from "node:child_process";
import {
  copyFile,
  cp,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const packageDirectory = fileURLToPath(new URL("..", import.meta.url));
const destinationDirectory = path.resolve(process.argv[2] ?? packageDirectory);
const temporaryDirectory = await mkdtemp(
  path.join(tmpdir(), "reme-clawhub-pack-"),
);
const stagingDirectory = path.join(temporaryDirectory, "package");

try {
  await mkdir(stagingDirectory, { recursive: true });
  await mkdir(destinationDirectory, { recursive: true });

  const packageManifest = JSON.parse(
    await readFile(path.join(packageDirectory, "package.json"), "utf8"),
  );
  delete packageManifest.scripts?.prepare;

  await Promise.all([
    cp(
      path.join(packageDirectory, "dist"),
      path.join(stagingDirectory, "dist"),
      {
        recursive: true,
      },
    ),
    cp(path.join(packageDirectory, "dsh"), path.join(stagingDirectory, "dsh"), {
      recursive: true,
    }),
    writeFile(
      path.join(stagingDirectory, "package.json"),
      `${JSON.stringify(packageManifest, null, 2)}\n`,
    ),
    copyFile(
      path.join(packageDirectory, "openclaw.plugin.json"),
      path.join(stagingDirectory, "openclaw.plugin.json"),
    ),
    writeFile(
      path.join(stagingDirectory, "README.md"),
      (
        await readFile(path.join(packageDirectory, "docs/openclaw.md"), "utf8")
      ).replace("(./openclaw.zh-CN.md)", "(./README_ZH.md)"),
    ),
    writeFile(
      path.join(stagingDirectory, "README_ZH.md"),
      (
        await readFile(
          path.join(packageDirectory, "docs/openclaw.zh-CN.md"),
          "utf8",
        )
      ).replace("(./openclaw.md)", "(./README.md)"),
    ),
  ]);

  const { stdout } = await execFileAsync(
    "npm",
    [
      "pack",
      "--json",
      "--ignore-scripts",
      "--pack-destination",
      destinationDirectory,
    ],
    { cwd: stagingDirectory },
  );
  const [{ filename }] = JSON.parse(stdout);
  console.log(
    JSON.stringify({
      filename,
      tarball: path.join(destinationDirectory, filename),
    }),
  );
} finally {
  await rm(temporaryDirectory, { force: true, recursive: true });
}
