import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { normalizeEndpoint, Storage } from "../src/main/storage.ts";

test("settings and scoped receipts survive serialized atomic writes without persisting chat content", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-desktop-storage-"));
  try {
    const storage = new Storage(directory);
    await storage.load();
    storage.settings.scale = 1.2;
    storage.presentation("workspace-a:session-1").receipts.a = {
      read: false,
      presented: true,
      snoozeUntil: 1000,
    };
    storage.changed();
    const first = storage.flush();
    storage.settings.scale = 1.4;
    storage.changed();
    await Promise.all([first, storage.flush()]);
    const restored = new Storage(directory);
    await restored.load();
    assert.equal(restored.settings.scale, 1.4);
    assert.equal(
      restored.presentation("workspace-a:session-1").receipts.a.snoozeUntil,
      1000,
    );
    assert.deepEqual(
      restored.presentation("workspace-b:session-1").receipts,
      {},
    );
    assert.equal(
      (
        await readFile(path.join(directory, "presentation.json"), "utf8")
      ).includes("content"),
      false,
    );
  } finally {
    await rm(directory, { recursive: true });
  }
});

test("malformed JSON is preserved for repair", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "reme-desktop-invalid-"));
  try {
    await writeFile(path.join(directory, "settings.json"), "user-invalid-json");
    await assert.rejects(new Storage(directory).load(), /不是有效 JSON/);
    assert.equal(
      await readFile(path.join(directory, "settings.json"), "utf8"),
      "user-invalid-json",
    );
  } finally {
    await rm(directory, { recursive: true });
  }
});

test("endpoints keep reverse-proxy prefixes and reject embedded credentials", () => {
  assert.equal(
    normalizeEndpoint(" http://127.0.0.1:2333/reme/ "),
    "http://127.0.0.1:2333/reme",
  );
  assert.throws(() => normalizeEndpoint("file:///etc/passwd"));
  assert.throws(() => normalizeEndpoint("http://user:secret@localhost:2333"));
  assert.throws(() => normalizeEndpoint("http://localhost:2333?token=secret"));
});
