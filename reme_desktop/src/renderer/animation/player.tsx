import { useEffect, useRef, useState } from "react";
import { translator, UiError } from "../../shared/i18n.ts";
import type { Language } from "../../shared/i18n.ts";
import type { RefObject } from "react";
import type {
  AnimationIntent,
  AnimationManifest,
  PlaybackStatus,
} from "../../shared/protocol.ts";
import { frameTimeline, timelineFrame } from "./timing.ts";

const cache = new Map<string, Promise<HTMLImageElement[]>>();

export function Character({
  manifest,
  language,
  animation,
  canvasRef,
  onPlayback,
  active = true,
}: {
  manifest: AnimationManifest;
  language: Language;
  animation: AnimationIntent;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  onPlayback: (id: number, status: PlaybackStatus) => void;
  active?: boolean;
}) {
  const t = translator(language);
  const [failed, setFailed] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(
    () => matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  const report = useRef(onPlayback);
  const previousClip = useRef("");
  report.current = onPlayback;
  useEffect(() => {
    const query = matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReducedMotion(query.matches);
    query.addEventListener("change", update);
    update();
    return () => query.removeEventListener("change", update);
  }, []);
  useEffect(() => {
    setFailed(false);
    const clip = manifest.animations.find(
      (value) => value.id === animation.clip,
    )!;
    const frames = animation.segment
      ? clip.segments![animation.segment]
      : clip.ticks.map((_, index) => index);
    const still =
      animation.segment || animation.mode === "hold"
        ? frames.length - 1
        : clip.id === "idle"
          ? 0
          : Math.floor(frames.length / 2);
    const canvas = canvasRef.current;
    let outgoing: HTMLCanvasElement | null = null;
    if (
      active &&
      !reducedMotion &&
      previousClip.current &&
      previousClip.current !== clip.id &&
      canvas
    ) {
      outgoing = document.createElement("canvas");
      outgoing.width = canvas.width;
      outgoing.height = canvas.height;
      outgoing.getContext("2d")?.drawImage(canvas, 0, 0);
    }
    previousClip.current = clip.id;
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    if (!cache.has(clip.id))
      cache.set(
        clip.id,
        Promise.all(
          clip.frame_files.map(
            (file) =>
              new Promise<HTMLImageElement>((resolve, reject) => {
                const image = new Image();
                image.crossOrigin = "anonymous";
                image.onload = () => resolve(image);
                image.onerror = () =>
                  reject(new UiError("assetLoadFailed", { file }));
                image.src = `reme-assets://remy/${file}`;
              }),
          ),
        ),
      );
    void cache
      .get(clip.id)!
      .then((images) => {
        if (!live) return;
        const context = canvasRef.current?.getContext("2d", {
          willReadFrequently: true,
        });
        if (!context) return;
        const start = animation.startedAt ?? Date.now();
        const timeline = frameTimeline(
          frames.map((index) => clip.ticks[index]),
          manifest.output_fps,
          clip.id === "idle" && animation.mode === "loop",
          start,
        );
        const transitionStart = Date.now();
        let last = -1;
        const paint = () => {
          if (!live) return;
          const elapsed = Math.max(0, Date.now() - start);
          const frame = timelineFrame(timeline, elapsed, animation.mode);
          const index = reducedMotion ? still : frame.index;
          if (index !== last || outgoing) {
            context.clearRect(0, 0, manifest.canvas[0], manifest.canvas[1]);
            const blend = outgoing
              ? Math.min(1, (Date.now() - transitionStart) / 120)
              : 1;
            if (outgoing) {
              context.globalAlpha = 1 - blend;
              context.drawImage(outgoing, 0, 0);
              context.globalCompositeOperation = "lighter";
            }
            context.globalAlpha = blend;
            context.drawImage(
              images[frames[index]],
              0,
              0,
              manifest.canvas[0],
              manifest.canvas[1],
            );
            context.globalAlpha = 1;
            context.globalCompositeOperation = "source-over";
            if (blend === 1) outgoing = null;
            if (last === -1 && active) report.current(animation.id, "started");
            last = index;
          }
          if (frame.ended && active) report.current(animation.id, "ended");
          else if (
            active &&
            (outgoing ||
              (animation.mode !== "hold" &&
                (!reducedMotion || animation.mode === "once")))
          )
            timer = setTimeout(
              paint,
              Math.max(
                10,
                reducedMotion
                  ? timeline.duration - elapsed
                  : outgoing
                    ? Math.min(16, frame.delay)
                    : frame.delay,
              ),
            );
        };
        paint();
      })
      .catch((error) => {
        cache.delete(clip.id);
        if (live && canvasRef.current) {
          canvasRef.current.dataset.error = String(error);
          setFailed(true);
          if (active) report.current(animation.id, "failed");
        }
      });
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [manifest, animation.id, canvasRef, active, reducedMotion]);
  return (
    <canvas
      ref={canvasRef}
      width={manifest.canvas[0]}
      height={manifest.canvas[1]}
      aria-label={t(failed ? "assetFailed" : "characterAnimation")}
      data-animation={animation.clip}
      data-segment={animation.segment}
    />
  );
}

export function hitCharacter(
  canvas: HTMLCanvasElement | null,
  x: number,
  y: number,
): boolean {
  if (!canvas) return false;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return false;
  const px = Math.floor(((x - rect.left) / rect.width) * canvas.width);
  const py = Math.floor(((y - rect.top) / rect.height) * canvas.height);
  if (px < 0 || py < 0 || px >= canvas.width || py >= canvas.height)
    return false;
  // A small tolerance keeps edges and gaps in animated hair easy to click.
  const radius = Math.ceil((4 * canvas.width) / rect.width);
  const left = Math.max(0, px - radius);
  const top = Math.max(0, py - radius);
  const pixels = canvas
    .getContext("2d", { willReadFrequently: true })
    ?.getImageData(
      left,
      top,
      Math.min(canvas.width, px + radius + 1) - left,
      Math.min(canvas.height, py + radius + 1) - top,
    ).data;
  if (pixels)
    for (let i = 3; i < pixels.length; i += 4) if (pixels[i] > 20) return true;
  return false;
}
