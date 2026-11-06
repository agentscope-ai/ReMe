import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import type {
  AnimationIntent,
  AnimationManifest,
} from "../../shared/protocol.ts";

const cache = new Map<string, Promise<HTMLImageElement[]>>();

export function Character({
  manifest,
  animation,
  canvasRef,
  onEnded,
}: {
  manifest: AnimationManifest;
  animation: AnimationIntent;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  onEnded: (id: number) => void;
}) {
  const ended = useRef(onEnded);
  ended.current = onEnded;
  useEffect(() => {
    const clip = manifest.animations.find(
      (value) => value.id === animation.clip,
    )!;
    const frames = animation.segment
      ? clip.segments![animation.segment]
      : clip.ticks.map((_, index) => index);
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
                  reject(new Error(`角色素材加载失败：${file}`));
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
        const holds = frames.map(
          (index) => (clip.ticks[index] / manifest.output_fps) * 1000,
        );
        const duration = holds.reduce((a, b) => a + b, 0);
        const start = performance.now();
        let last = -1;
        const paint = () => {
          if (!live) return;
          const elapsed = performance.now() - start;
          if (animation.mode === "once" && elapsed >= duration) {
            ended.current(animation.id);
            return;
          }
          let position = elapsed % duration;
          let index = 0;
          while (index < holds.length - 1 && position >= holds[index])
            position -= holds[index++];
          if (index !== last) {
            context.clearRect(0, 0, manifest.canvas[0], manifest.canvas[1]);
            context.drawImage(
              images[
                frames[animation.mode === "hold" ? frames.length - 1 : index]
              ],
              0,
              0,
              manifest.canvas[0],
              manifest.canvas[1],
            );
            last = index;
          }
          if (animation.mode !== "hold")
            timer = setTimeout(paint, Math.max(10, holds[index] - position));
        };
        paint();
      })
      .catch((error) => {
        cache.delete(clip.id);
        if (live && canvasRef.current) {
          canvasRef.current.dataset.error = String(error);
          canvasRef.current.setAttribute(
            "aria-label",
            "角色素材加载失败，请检查安装文件",
          );
        }
      });
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [manifest, animation.id, canvasRef]);
  return (
    <canvas
      ref={canvasRef}
      width={manifest.canvas[0]}
      height={manifest.canvas[1]}
      aria-label="Remy 角色动画"
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
