export interface FrameTimeline {
  frames: { index: number; end: number }[];
  duration: number;
}

/** A shared seed keeps idle blinks consistent when the character changes windows. */
export function frameTimeline(
  ticks: number[],
  fps: number,
  idle = false,
  seed = 0,
): FrameTimeline {
  const frames: FrameTimeline["frames"] = [];
  let duration = 0;
  let random = seed >>> 0;
  for (let cycle = 0; cycle < (idle ? 16 : 1); cycle++) {
    random = (Math.imul(random, 1664525) + 1013904223) >>> 0;
    const rest = idle ? 1800 + (random / 2 ** 32) * 4200 : 0;
    ticks.forEach((tick, index) => {
      duration += (tick / fps) * 1000 + (index === 0 ? rest : 0);
      frames.push({ index, end: duration });
    });
  }
  return { frames, duration };
}

export function timelineFrame(
  timeline: FrameTimeline,
  elapsed: number,
  mode: "once" | "loop" | "hold",
): { index: number; delay: number; ended: boolean } {
  const ended = mode === "once" && elapsed >= timeline.duration;
  const last = timeline.frames.at(-1)!;
  if (mode === "hold" || ended)
    return { index: last.index, delay: Infinity, ended };
  const position = Math.max(0, elapsed) % timeline.duration;
  const frame = timeline.frames.find((frame) => position < frame.end)!;
  return { index: frame.index, delay: frame.end - position, ended };
}
