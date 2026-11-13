import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const base = dirname(fileURLToPath(import.meta.url));
const config = JSON.parse(readFileSync(join(base, "timeline.json"), "utf8"));
const args = process.argv.slice(2);
const metadataOnly = args.includes("--metadata-only");
const requested = args.filter((arg) => arg !== "--metadata-only");
const animations = requested.length
  ? config.animations.filter((item) => requested.includes(item.id))
  : config.animations;
if (
  !animations.length ||
  requested.some((id) => !config.animations.some((item) => item.id === id))
) {
  throw new Error(
    "Unknown animation. Available: " +
      config.animations.map((item) => item.id).join(", "),
  );
}

function ffmpeg(args) {
  execFileSync(
    "ffmpeg",
    ["-hide_banner", "-loglevel", "error", "-y", ...args],
    { stdio: "pipe", maxBuffer: 8 * 1024 * 1024 },
  );
}
function probe(path) {
  return JSON.parse(
    execFileSync(
      "ffprobe",
      ["-v", "error", "-show_streams", "-show_format", "-of", "json", path],
      { encoding: "utf8" },
    ),
  );
}

// Scale premultiplied colors so hidden RGB in transparent source pixels cannot
// bleed into the visible outline. This preserves the actual character colors.
const clearInvisible =
  "geq=r='if(lte(alpha(X,Y),8),0,r(X,Y))':g='if(lte(alpha(X,Y),8),0,g(X,Y))':b='if(lte(alpha(X,Y),8),0,b(X,Y))':a='if(lte(alpha(X,Y),8),0,alpha(X,Y))'";
const [width, height] = config.canvas;

for (const animation of metadataOnly ? [] : animations) {
  const sheet = join(base, "sheets", animation.id + ".png");
  if (!existsSync(sheet)) throw new Error("Missing source sheet: " + sheet);
  const info = probe(sheet).streams[0];
  const { columns, rows } = config.grid;
  const renderScale = animation.render_scale ?? 1;
  if (!(renderScale > 0 && renderScale <= 1))
    throw new Error("Invalid render scale: " + animation.id);
  const renderWidth = Math.max(2, Math.round((width * renderScale) / 2) * 2);
  const renderHeight = Math.max(2, Math.round((height * renderScale) / 2) * 2);
  if (
    animation.ticks.length !== columns * rows ||
    animation.ticks.some((tick) => !Number.isInteger(tick) || tick < 1)
  ) {
    throw new Error("Invalid frame timing: " + animation.id);
  }
  const output = join(base, "frames", animation.id);
  mkdirSync(output, { recursive: true });
  for (let index = 0; index < animation.ticks.length; index++) {
    const column = index % columns,
      row = Math.floor(index / columns);
    const x = Math.floor((column * info.width) / columns),
      y = Math.floor((row * info.height) / rows);
    const cellWidth = Math.floor(((column + 1) * info.width) / columns) - x;
    const cellHeight = Math.floor(((row + 1) * info.height) / rows) - y;
    const filter = [
      "crop=" + cellWidth + ":" + cellHeight + ":" + x + ":" + y + ":exact=1",
      "format=gbrap",
      "premultiply=planes=7:inplace=1",
      "scale=" + renderWidth + ":" + renderHeight + ":flags=lanczos",
      "pad=" +
        width +
        ":" +
        height +
        ":" +
        Math.floor((width - renderWidth) / 2) +
        ":" +
        (height - renderHeight) +
        ":color=black@0",
      "unpremultiply=planes=7:inplace=1",
      clearInvisible,
      "format=rgba",
    ].join(",");
    ffmpeg([
      "-i",
      sheet,
      "-vf",
      filter,
      "-frames:v",
      "1",
      "-update",
      "1",
      join(output, String(index).padStart(2, "0") + ".png"),
    ]);
  }
  console.log(animation.id + ": " + animation.ticks.length + " keyframes");
}

// The desktop consumes PNG frames directly; timing remains authoritative here.
const manifest = {
  name: "Remy",
  version: config.version,
  ambient: config.ambient,
  ambient_weights: config.ambient_weights,
  canvas: config.canvas,
  output_fps: config.fps,
  animations: config.animations.map((item) => ({
    id: item.id,
    ticks: item.ticks,
    ...(item.segments ? { segments: item.segments } : {}),
    frame_files: item.ticks.map((_, index) => {
      const file =
        "frames/" + item.id + "/" + String(index).padStart(2, "0") + ".png";
      if (!existsSync(join(base, file)))
        throw new Error("Missing frame: " + file);
      return file;
    }),
  })),
};
writeFileSync(
  join(base, "manifest.json"),
  JSON.stringify(manifest, null, 2) + "\n",
);
