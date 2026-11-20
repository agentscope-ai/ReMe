import assert from "node:assert/strict";
import { open, readFile, readdir } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = new URL("../../", import.meta.url);
const studioPackage = JSON.parse(
  await readFile(new URL("../../package.json", import.meta.url), "utf8"),
);
const expectedVersion = studioPackage.overrides?.dompurify;
assert.equal(
  typeof expectedVersion,
  "string",
  "reme_studio/package.json must pin overrides.dompurify",
);

const installedVersion = JSON.parse(
  await readFile(
    new URL("../../node_modules/dompurify/package.json", import.meta.url),
    "utf8",
  ),
).version;
assert.equal(
  installedVersion,
  expectedVersion,
  "installed DOMPurify must match the package.json override",
);

const MAX_ASSET_BYTES = 8 * 1024 * 1024;
const SANITIZER_MARKER = "TRUSTED_TYPES_POLICY callback";

// The Monaco editor bundles its own DOMPurify source unless the Vite plugin
// redirects that module. Verify the actual shipped JS, not only npm lockfiles.
async function javascriptFiles(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...(await javascriptFiles(filename)));
    if (entry.isFile() && /\.(?:m?js)$/.test(entry.name)) files.push(filename);
  }
  return files;
}

async function readBoundedText(file) {
  const handle = await open(file, "r");
  try {
    const { size } = await handle.stat();
    assert.ok(
      size <= MAX_ASSET_BYTES,
      `${file}: compiled asset exceeds the ${MAX_ASSET_BYTES}-byte scan limit`,
    );
    const buffer = Buffer.alloc(size);
    let offset = 0;
    while (offset < size) {
      const { bytesRead } = await handle.read(
        buffer,
        offset,
        size - offset,
        offset,
      );
      if (bytesRead === 0) break;
      offset += bytesRead;
    }
    assert.equal(
      offset,
      size,
      `${file}: compiled asset could not be read completely`,
    );
    return buffer.toString("utf8");
  } finally {
    await handle.close();
  }
}

test("Studio bundles Monaco with the pinned DOMPurify sanitizer", async () => {
  for (const relativeDir of [
    "dist-static/assets",
    "dist-demo/assets",
    "dist/client",
    "dist/server",
  ]) {
    const directory = fileURLToPath(new URL(relativeDir + "/", root));
    const files = await javascriptFiles(directory);
    let foundSanitizer = false;
    for (const file of files) {
      const content = await readBoundedText(file);
      if (!content.includes(SANITIZER_MARKER)) continue;
      foundSanitizer = true;
      const versions = [
        ...content.matchAll(/\bversion\s*=\s*([`'"])([^`'"]+)\1/g),
      ].map((match) => match[2]);
      assert.ok(
        versions.length > 0,
        `${relativeDir}/${path.basename(
          file,
        )}: sanitizer marker must include a version literal`,
      );
      assert.deepEqual(
        [...new Set(versions)],
        [expectedVersion],
        `${relativeDir}/${path.basename(
          file,
        )}: every sanitizer version must match pinned DOMPurify ${expectedVersion}`,
      );
    }
    assert.ok(
      foundSanitizer,
      `${relativeDir}: Monaco sanitizer must be present in the built output`,
    );
  }
});
