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
  {
    dsh: "0.2.2-alpha.2",
    cordis: "4.0.6-alpha.1",
    loader: "1.0.7-alpha.1",
    schema: "3.18.6-alpha.1",
  },
];

// `npm` is `npm.cmd` on Windows, which execFile cannot launch directly; the
// script runs the npm CLI through node instead, using the path npm exported to
// this process.
const npmCommand = process.env.npm_execpath
  ? [process.execPath, [process.env.npm_execpath]]
  : ["npm", []];

async function run(command, args, cwd, label = "command") {
  try {
    return await execFileAsync(command, args, {
      cwd,
      maxBuffer: 10 * 1024 * 1024,
    });
  } catch (error) {
    const stdout = String(error.stdout ?? "");
    const stderr = String(error.stderr ?? "");
    const combined = `${stdout}\n${stderr}`;
    // The raw TAP buffer is far too large for a CI log and hides which step
    // failed. Report the failing cases and a short tail instead.
    const failures = combined
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => /^not ok \d+ - /.test(line) || line.startsWith("✖ "))
      .slice(0, 20);
    const tail = combined.trim().split("\n").slice(-40).join("\n");
    process.stderr.write(
      `\n[test-package] ${label} failed (exit ${error.code ?? "error"})\n`,
    );
    if (failures.length > 0)
      process.stderr.write(
        `[test-package] failing tests:\n${failures.join("\n")}\n`,
      );
    if (tail) process.stderr.write(`[test-package] last output:\n${tail}\n`);
    // A short error keeps Node's unhandled-rejection dump readable; the large
    // stdout/stderr buffers stay out of it.
    const failure = new Error(`${label} failed: ${error.message}`);
    failure.code = error.code;
    throw failure;
  }
}

/** Run the npm CLI on the platform this check runs on. */
function runNpm(args, cwd, label = "npm") {
  return run(npmCommand[0], [...npmCommand[1], ...args], cwd, label);
}

try {
  const { stdout } = await runNpm(
    [
      "pack",
      "--json",
      "--ignore-scripts",
      "--pack-destination",
      temporaryDirectory,
    ],
    root,
    "npm pack",
  );
  const [result] = JSON.parse(stdout);
  const files = new Set(result.files.map(({ path: file }) => file));
  for (const file of [
    "dist/index.js",
    "dist/client.js",
    "cordis.patch.yml",
    "README.md",
    "docs/environment.md",
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
    await runNpm(
      [
        "install",
        "--ignore-scripts",
        "--no-audit",
        "--no-fund",
        "--fetch-retries=0",
        "--fetch-timeout=30000",
      ],
      consumer,
      `install DSH ${release.dsh}`,
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
      `import DSH ${release.dsh}`,
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
      `tests DSH ${release.dsh}`,
    );
    await runNpm(["run", "build"], installed, `build DSH ${release.dsh}`);
    const count = /# tests (\d+)/.exec(tested.stdout)?.[1];
    process.stdout.write(
      `DSH ${release.dsh}: packed import, source build, and ${count} tests passed\n`,
    );
  }
} finally {
  await rm(temporaryDirectory, { force: true, recursive: true });
}
