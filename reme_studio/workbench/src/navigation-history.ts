import type { StudioPage } from "../../client/src/protocol.ts";

export type NavigationValues = Record<string, unknown>;
export type NavigationLocation = { page: StudioPage; values: NavigationValues };
export type NavigationHistory = {
  entries: NavigationLocation[];
  index: number;
};

export function sameLocation(a: NavigationLocation, b: NavigationLocation) {
  const keys = Object.keys(a.values);
  return (
    a.page === b.page &&
    keys.length === Object.keys(b.values).length &&
    keys.every(
      (key) => JSON.stringify(a.values[key]) === JSON.stringify(b.values[key]),
    )
  );
}

export function visit(
  history: NavigationHistory,
  location: NavigationLocation,
): NavigationHistory {
  if (sameLocation(history.entries[history.index], location)) return history;
  const entries = [
    ...history.entries.slice(0, history.index + 1),
    location,
  ].slice(-100);
  return { entries, index: entries.length - 1 };
}

export function travel(
  history: NavigationHistory,
  delta: -1 | 1,
): NavigationHistory {
  const index = history.index + delta;
  return index < 0 || index >= history.entries.length
    ? history
    : { ...history, index };
}
