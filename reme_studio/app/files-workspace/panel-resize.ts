export const MIN_NAVIGATOR_WIDTH = 220;
export const MIN_WORKBENCH_WIDTH = 420;

export function clampNavigatorWidth(
  width: number,
  containerWidth: number,
): number {
  const maximum = Math.max(
    MIN_NAVIGATOR_WIDTH,
    containerWidth - MIN_WORKBENCH_WIDTH,
  );
  return Math.min(Math.max(MIN_NAVIGATOR_WIDTH, width), maximum);
}
