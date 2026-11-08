import { animationLabels, UiError } from "../shared/i18n.ts";
import { readFile } from "node:fs/promises";
import path from "node:path";
import type { AnimationManifest } from "../shared/protocol.ts";

export async function loadAssets(
  directory: string,
): Promise<{ manifest: AnimationManifest; files: Map<string, string> }> {
  const manifest = JSON.parse(
    await readFile(path.join(directory, "manifest.json"), "utf8"),
  ) as AnimationManifest;
  if (
    manifest.version !== 2 ||
    !Array.isArray(manifest.ambient) ||
    new Set(manifest.ambient).size < 2 ||
    !Array.isArray(manifest.canvas) ||
    manifest.canvas.length !== 2 ||
    !manifest.canvas.every((n) => Number.isFinite(n) && n > 0 && n <= 2048) ||
    !Number.isFinite(manifest.output_fps) ||
    manifest.output_fps <= 0 ||
    !Array.isArray(manifest.animations)
  )
    throw new UiError("invalidManifest");
  if (
    manifest.ambient_weights !== undefined &&
    (!manifest.ambient_weights ||
      typeof manifest.ambient_weights !== "object" ||
      Array.isArray(manifest.ambient_weights) ||
      Object.entries(manifest.ambient_weights).some(
        ([id, weight]) =>
          !manifest.ambient.includes(id) ||
          !Number.isFinite(weight) ||
          weight <= 0,
      ))
  )
    throw new UiError("invalidManifest");
  const files = new Map<string, string>();
  for (const clip of manifest.animations) {
    if (
      !Object.hasOwn(animationLabels, clip.id) ||
      !clip.frame_files?.length ||
      clip.frame_files.length !== clip.ticks?.length ||
      !clip.ticks.every((tick) => Number.isFinite(tick) && tick > 0)
    )
      throw new UiError("invalidTiming", { id: clip.id });
    if (
      clip.segments &&
      ["enter", "hold", "exit"].some((name) => {
        const frames = clip.segments![name as keyof typeof clip.segments];
        return (
          !Array.isArray(frames) ||
          !frames.length ||
          frames.some(
            (frame) =>
              !Number.isInteger(frame) ||
              frame < 0 ||
              frame >= clip.frame_files.length,
          )
        );
      })
    )
      throw new UiError("invalidSegments", { id: clip.id });
    for (const frame of clip.frame_files) {
      if (!/^frames\/[a-z_]+\/\d+\.png$/.test(frame))
        throw new UiError("invalidFrame", { frame });
      files.set("/" + frame, path.join(directory, frame));
    }
  }
  if (
    [
      "idle",
      "thinking",
      "speaking",
      "remembering",
      "reminding",
      "happy",
      "confused",
      "sleeping",
      "dragged",
      "listening",
      "walking",
      ...manifest.ambient,
    ].some((id) => !manifest.animations.some((clip) => clip.id === id)) ||
    !manifest.animations.find((clip) => clip.id === "reminding")?.segments
  )
    throw new UiError("missingAnimation");
  return { manifest, files };
}
