import { execFile } from "node:child_process";
import assert from "node:assert/strict";
import {
  cp,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const root = fileURLToPath(new URL("..", import.meta.url));
const manifest = JSON.parse(
  await readFile(path.join(root, "package.json"), "utf8"),
);
const temporaryDirectory = await mkdtemp(
  path.join(tmpdir(), "reme-dsh-package-"),
);
const releases = [
  { dsh: "0.1.7-rc.2", cordis: "4.0.4", loader: "1.0.5", schema: "3.18.4" },
  { dsh: "0.2.0-rc.2", cordis: "4.0.4", loader: "1.0.5", schema: "3.18.4" },
  {
    dsh: "0.2.1-alpha.2",
    cordis: "4.0.5-alpha.1",
    loader: "1.0.6-alpha.1",
    schema: "3.18.5-alpha.1",
  },
  {
    dsh: "0.2.1-alpha.4",
    cordis: "4.0.5-alpha.2",
    loader: "1.0.6-alpha.2",
    schema: "3.18.5-alpha.2",
  },
  { dsh: "0.2.1-rc.2", cordis: "4.0.5", loader: "1.0.6", schema: "3.18.5" },
];

async function run(command, args, cwd) {
  try {
    return await execFileAsync(command, args, {
      cwd,
      maxBuffer: 10 * 1024 * 1024,
    });
  } catch (error) {
    process.stderr.write(error.stdout || "");
    process.stderr.write(error.stderr || "");
    throw error;
  }
}

try {
  const { stdout } = await run(
    "npm",
    [
      "pack",
      "--json",
      "--ignore-scripts",
      "--pack-destination",
      temporaryDirectory,
    ],
    root,
  );
  const [result] = JSON.parse(stdout);
  const files = new Set(result.files.map(({ path: file }) => file));
  for (const file of [
    "dist/index.js",
    "dist/client.js",
    "cordis.patch.yml",
    "README.md",
    "figures/reme-status-overview.png",
  ]) {
    assert.ok(files.has(file), `missing ${file}`);
  }
  assert.ok(![...files].some((file) => file.includes("openclaw")));

  for (const release of releases) {
    const consumer = path.join(temporaryDirectory, release.dsh);
    await mkdir(consumer);
    const dependencies = Object.fromEntries(
      Object.entries(manifest.devDependencies)
        .filter(
          ([name]) =>
            name.startsWith("@deepseek-ai/") ||
            [
              "typescript",
              "esbuild",
              "react",
              "react-test-renderer",
              "semver",
              "@types/react",
              "@types/node",
            ].includes(name),
        )
        .map(([name, version]) => [
          name,
          name.startsWith("@deepseek-ai/dsh-") ? release.dsh : version,
        ]),
    );
    Object.assign(dependencies, {
      "@deepseek-ai/cordis": release.cordis,
      "@deepseek-ai/cordis-plugin-loader": release.loader,
      "@deepseek-ai/schemastery": release.schema,
      [manifest.name]: path.join(temporaryDirectory, result.filename),
    });
    await writeFile(
      path.join(consumer, "package.json"),
      JSON.stringify({ private: true, type: "module", dependencies }),
    );
    // Install as a consumer with normal peer resolution, then check the shipped artifact.
    await run(
      "npm",
      [
        "install",
        "--ignore-scripts",
        "--no-audit",
        "--no-fund",
        "--fetch-retries=0",
        "--fetch-timeout=30000",
      ],
      consumer,
    );
    await run(
      process.execPath,
      [
        "--input-type=module",
        "--eval",
        `
      const plugin = await import(${JSON.stringify(manifest.name)});
      if (typeof plugin.apply !== "function" || !Array.isArray(plugin.inject) || plugin.Config === undefined)
        throw new Error("missing Host plugin exports");
    `,
      ],
      consumer,
    );

    const installed = path.join(consumer, "node_modules", manifest.name);
    for (const file of [
      "src",
      "tests",
      "tsconfig.json",
      "scripts/build-client.mjs",
    ]) {
      await cp(path.join(root, file), path.join(installed, file), {
        recursive: true,
      });
    }
    const testFiles = (await readdir(path.join(installed, "tests")))
      .filter((file) => file.endsWith(".test.mjs"))
      .map((file) => path.join("tests", file));
    const tested = await run(
      process.execPath,
      ["--test", ...testFiles],
      installed,
    );
    await run("npm", ["run", "build"], installed);
    const count = /# tests (\d+)/.exec(tested.stdout)?.[1];
    process.stdout.write(
      `DSH ${release.dsh}: packed import, source build, and ${count} tests passed\n`,
    );
  }
} finally {
  await rm(temporaryDirectory, { force: true, recursive: true });
}
