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

test("solid theme actions use foreground colors with sufficient contrast", async () => {
  const css = await readFile(
    new URL("../app/globals.css", import.meta.url),
    "utf8",
  );
  const graphCss = await readFile(
    new URL("../app/files-workspace/memory-graph.module.css", import.meta.url),
    "utf8",
  );

  const tokens = [
    ...css.matchAll(/:root(?:\[data-theme="dark"\])?\s*\{([^}]+)\}/g),
  ].map(([, declarations]) =>
    Object.fromEntries(
      [...declarations.matchAll(/--([\w-]+):\s*(#[\da-f]{6});/gi)].map(
        ([, name, value]) => [name, value],
      ),
    ),
  );
  const luminance = (color) => {
    const channels = [1, 3, 5]
      .map(
        (offset) => Number.parseInt(color.slice(offset, offset + 2), 16) / 255,
      )
      .map((channel) =>
        channel <= 0.04045
          ? channel / 12.92
          : ((channel + 0.055) / 1.055) ** 2.4,
      );
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
  };
  const contrast = (first, second) => {
    const values = [luminance(first), luminance(second)].sort(
      (left, right) => right - left,
    );
    return (values[0] + 0.05) / (values[1] + 0.05);
  };

  assert.equal(tokens.length, 2);
  for (const theme of tokens) {
    assert.ok(contrast(theme.accent, theme["accent-foreground"]) >= 4.5);
    assert.ok(contrast(theme.danger, theme["danger-foreground"]) >= 4.5);
  }
  for (const selector of [
    ".welcome > button",
    ".composer button",
    ".settings-sidebar button.active",
    ".primary-action",
  ]) {
    const rule =
      css.match(
        new RegExp(`${selector.replaceAll(".", "\\.")} \\{[^}]+\\}`),
      )?.[0] || "";
    assert.match(rule, /color: var\(--accent-foreground\);/, selector);
  }
  assert.match(graphCss, /fill: var\(--accent-foreground\);/);
  assert.match(graphCss, /color: var\(--accent-foreground\);/);
  assert.match(css, /color: var\(--danger-foreground\);/);
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
