import { expect } from "playwright/test";

export async function openCompanionMenu(companion) {
  await companion.evaluate(() =>
    window.desktop.action({ type: "companion.menu" }),
  );
  await expect(companion.getByRole("menu")).toBeVisible();
}

export async function chooseCompanionMenu(companion, id) {
  await openCompanionMenu(companion);
  if (id.includes(":")) {
    const group = id.startsWith("animation:") ? "animations" : id.split(":")[0];
    await companion.locator(`[data-menu-id="${group}"]`).click();
  }
  await companion.locator(`[data-menu-id="${id}"]`).click();
}
