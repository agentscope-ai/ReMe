import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";
import { fileURLToPath } from "node:url";

test("static build contains the ReMe workspace entry and assets", async () => {
  const output = fileURLToPath(new URL("../dist-static/", import.meta.url));
  const html = await readFile(
    new URL("../dist-static/index.html", import.meta.url),
    "utf8",
  );
  const assets = await readdir(
    new URL("../dist-static/assets/", import.meta.url),
  );

  assert.match(html, /<title>ReMe Workspace<\/title>/i);
  assert.match(html, /<div id="root"><\/div>/i);
  assert.match(html, /<script type="module"[^>]+src="\/assets\//i);
  assert.ok(
    assets.some((name) => name.endsWith(".js")),
    output,
  );
  assert.ok(
    assets.some((name) => name.endsWith(".css")),
    output,
  );
});
