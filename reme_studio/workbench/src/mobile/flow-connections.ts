export type FlowBox = {
  top: number;
  bottom: number;
  left: number;
  right: number;
};

/** The connector stays in the gap even when a card expands to show its content. */
export function verticalConnection(from: FlowBox, to: FlowBox) {
  const x = (from.left + from.right) / 2;
  const gap = (from.bottom + to.top) / 2;
  return {
    path: `M ${x} ${from.bottom - 1} L ${(to.left + to.right) / 2} ${to.top + 1}`,
    chevrons: [0, 1].map(
      (index) => `M ${x - 4} ${gap - 5 + index * 6} l 4 4 4 -4`,
    ),
  };
}
