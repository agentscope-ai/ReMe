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
    assert.equal(storage.settings.language, "zh");
    assert.equal(storage.settings.activityLevel, "normal");
    assert.equal(storage.settings.sleepMinutes, 15);
    storage.settings.activityLevel = "lively";
    storage.settings.sleepMinutes = 0;
    storage.settings.language = "en";
    storage.settings.scale = 1.2;
    assert.equal(storage.settings.voiceAutoSend, true);
    assert.equal(storage.settings.voiceEnabled, true);
    assert.equal(storage.settings.voiceReminders, false);
    storage.settings.voiceAutoSend = false;
    storage.settings.voiceEnabled = true;
    storage.settings.voiceReminders = true;
    storage.settings.voiceVolume = 0.35;
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
    assert.equal(restored.settings.language, "en");
    assert.equal(restored.settings.activityLevel, "lively");
    assert.equal(restored.settings.sleepMinutes, 0);
    assert.equal(restored.settings.voiceEnabled, true);
    assert.equal(restored.settings.voiceAutoSend, false);
    assert.equal(restored.settings.voiceReminders, true);
    assert.equal(restored.settings.voiceVolume, 0.35);
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

test("old or invalid animation preferences use natural activity and fifteen-minute sleep", async () => {
  const directory = await mkdtemp(
    path.join(tmpdir(), "reme-animation-settings-"),
  );
  try {
    for (const values of [
      {},
      { activityLevel: "invalid", sleepMinutes: -1 },
      { activityLevel: null, sleepMinutes: true },
    ]) {
      await writeFile(
        path.join(directory, "settings.json"),
        JSON.stringify(values),
      );
      const storage = new Storage(directory);
      await storage.load();
      assert.equal(storage.settings.activityLevel, "normal");
      assert.equal(storage.settings.sleepMinutes, 15);
    }
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
