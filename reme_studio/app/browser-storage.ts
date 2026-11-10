/** Browser preferences are best-effort; blocked or full storage must not break editing. */
export const browserStorage = {
  getItem(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  setItem(key: string, value: string): void {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* The current workspace and preferences remain in memory. */
    }
  },
  removeItem(key: string): void {
    try {
      localStorage.removeItem(key);
    } catch {
      /* Unavailable browser storage has nothing to remove. */
    }
  },
};
