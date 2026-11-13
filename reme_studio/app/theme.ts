"use client";

import { create } from "zustand";
import { browserStorage } from "./browser-storage";

export type ThemePreference = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "reme-theme";
let listeningForSystemTheme = false;

const resolveTheme = (preference: ThemePreference): ResolvedTheme =>
  preference === "system" && typeof window !== "undefined"
    ? window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light"
    : preference === "dark"
    ? "dark"
    : "light";

const applyTheme = (preference: ThemePreference) => {
  const resolved = resolveTheme(preference);
  if (typeof document !== "undefined")
    document.documentElement.dataset.theme = resolved;
  return resolved;
};

interface ThemeState {
  preference: ThemePreference;
  resolved: ResolvedTheme;
  hydrate: () => void;
  setPreference: (preference: ThemePreference) => void;
}

export const useThemeStore = create<ThemeState>((set, get) => ({
  preference: "system",
  resolved: "light",
  hydrate: () => {
    const saved = browserStorage.getItem(STORAGE_KEY);
    const preference: ThemePreference =
      saved === "light" || saved === "dark" || saved === "system"
        ? saved
        : "system";
    set({ preference, resolved: applyTheme(preference) });
    if (listeningForSystemTheme) return;
    listeningForSystemTheme = true;
    window
      .matchMedia("(prefers-color-scheme: dark)")
      .addEventListener("change", () => {
        if (get().preference === "system")
          set({ resolved: applyTheme("system") });
      });
  },
  setPreference: (preference) => {
    browserStorage.setItem(STORAGE_KEY, preference);
    set({ preference, resolved: applyTheme(preference) });
  },
}));
