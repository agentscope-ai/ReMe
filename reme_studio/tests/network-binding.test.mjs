import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("Studio development servers require an explicit remote command", async () => {
  const packageJson = JSON.parse(
    await readFile(new URL("../package.json", import.meta.url), "utf8"),
  );
  const { scripts } = packageJson;

  assert.doesNotMatch(scripts.dev, /0\.0\.0\.0/);
  assert.doesNotMatch(scripts["dev:static"], /0\.0\.0\.0/);
  assert.match(scripts["dev:remote"], /--hostname 0\.0\.0\.0/);
  assert.match(scripts["dev:static:remote"], /--host 0\.0\.0\.0/);
});
