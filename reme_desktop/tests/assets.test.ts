import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { loadAssets } from "../src/main/assets.ts";

test("runtime frames match the source timing and shared transparent canvas", async () => {
  const directory = fileURLToPath(
    new URL("../../assets/remy/anime/animations/v1/", import.meta.url),
  );
  const timeline = JSON.parse(
    await readFile(path.join(directory, "timeline.json"), "utf8"),
  );
  const { manifest, files } = await loadAssets(directory);
  assert.equal(manifest.animations.length, 12);
  assert.equal(files.size, 96);
  assert.equal(manifest.output_fps, timeline.fps);
  assert.deepEqual(manifest.ambient, timeline.ambient);
  assert.deepEqual(manifest.ambient_weights, timeline.ambient_weights);
  for (const clip of manifest.animations) {
    const source = timeline.animations.find(
      (entry: { id: string }) => entry.id === clip.id,
    );
    assert.deepEqual(clip.ticks, source.ticks);
    assert.deepEqual(clip.segments, source.segments);
    assert.equal(clip.frame_files.length, 8);
  }
  for (const file of files.values()) {
    const png = await readFile(file);
    assert.equal(png.subarray(1, 4).toString(), "PNG");
    assert.equal(png.readUInt32BE(16), manifest.canvas[0]);
    assert.equal(png.readUInt32BE(20), manifest.canvas[1]);
    assert.equal(png[25], 6, "frames need RGBA transparency");
  }
});
