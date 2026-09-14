import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("Studio reuses the canonical ReMe brand assets", async () => {
  const pairs = [
    ["../../docs/figure/reme-icon.svg", "../public/reme-icon.svg"],
    ["../../docs/figure/reme_logo.png", "../public/reme_logo.png"],
  ];

  for (const [canonical, studio] of pairs) {
    assert.deepEqual(
      await readFile(new URL(canonical, import.meta.url)),
      await readFile(new URL(studio, import.meta.url)),
    );
  }
});

test("Studio theme uses the documentation site's canonical palette", async () => {
  const css = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );

  for (const color of [
    "#087f6a",
    "#086554",
    "#19a98f",
    "#3156d9",
    "#ffffff",
    "#f4f7f5",
    "#17221d",
    "#57dfc3",
    "#0d1512",
    "#edf7f3",
  ]) {
    assert.match(css, new RegExp(color));
  }
});

test("Studio package versions stay aligned", async () => {
  const packageJson = JSON.parse(
    await readFile(new URL("../package.json", import.meta.url), "utf8"),
  );
  const packageLock = JSON.parse(
    await readFile(new URL("../package-lock.json", import.meta.url), "utf8"),
  );
  const pyproject = await readFile(
    new URL("../pyproject.toml", import.meta.url),
    "utf8",
  );

  assert.equal(packageJson.version, "0.1.2");
  assert.equal(packageLock.version, packageJson.version);
  assert.equal(packageLock.packages[""].version, packageJson.version);
  assert.match(
    pyproject,
    new RegExp(`^version = "${packageJson.version}"$`, "m"),
  );
});
