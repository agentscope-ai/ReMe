import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

test("the browser demo builds for a subdirectory with no service API or live settings", async () => {
  const directory = new URL("../dist-demo/", import.meta.url);
  const html = await readFile(new URL("index.html", directory), "utf8");
  assert.match(html, /<title>Try ReMe Studio<\/title>/);
  assert.match(html, /src="\.\/assets\//);
  for (const match of html.matchAll(/(?:src|href)="(\.\/[^"?#]+)"/g))
    await access(new URL(match[1], directory));
  const assets = await readdir(new URL("assets/", directory));
  assert.ok(
    !assets.some(
      (name) => name.startsWith("settings-center") || name.endsWith(".map"),
    ),
  );
  const scripts = (
    await Promise.all(
      assets
        .filter((name) => name.endsWith(".js"))
        .map((name) => readFile(new URL(`assets/${name}`, directory), "utf8")),
    )
  ).join("\n");
  assert.doesNotMatch(scripts, /127\.0\.0\.1:2333|expected_mtime|health_check/);
});
