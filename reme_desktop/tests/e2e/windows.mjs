import { expect } from "playwright/test";

export async function desktopWindows(app) {
  await expect.poll(() => app.windows().length).toBe(2);
  const pages = app.windows();
  await Promise.all(
    pages.map((page) =>
      page.waitForURL((url) =>
        ["chat", "companion"].includes(url.searchParams.get("view")),
      ),
    ),
  );
  return {
    chat: pages.find((page) => page.url().includes("view=chat")),
    companion: pages.find((page) => page.url().includes("view=companion")),
  };
}
