import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { _electron as electron } from "playwright";
import { expect } from "playwright/test";
import { desktopWindows } from "./windows.mjs";
import { mockReMe } from "./mock-reme.mjs";

const project = path.resolve(fileURLToPath(new URL("../../", import.meta.url)));

test(
  "context menus restore keyboard focus and fit the pet layout on small displays",
  { timeout: 30_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-menu-layout-"));
    const backend = await mockReMe();
    let app;
    try {
      app = await electron.launch({
        args: [project],
        env: {
          ...process.env,
          REME_DESKTOP_DATA_DIR: directory,
          REME_DESKTOP_API_URL: backend.url,
        },
      });
      const { chat, companion } = await desktopWindows(app);
      const action = (value) =>
        chat.evaluate((value) => window.desktop.action(value), value);
      const state = () =>
        chat.evaluate(() =>
          window.desktop.bootstrap().then((value) => value.state),
        );
      await expect
        .poll(async () => (await state()).connection)
        .toBe("connected");
      await action({
        type: "settings.update",
        quietUntil: Date.now() + 60_000,
      });
      backend.push("keyboard", "菜单键盘操作回归");
      await action({ type: "chat.open" });
      const card = chat
        .locator(".inbox-card")
        .filter({ hasText: "菜单键盘操作回归" });
      await expect(card).toBeVisible();
      // Control native focus only; use the real renderer, menu, IPC, and window placement.
      await chat.evaluate(() =>
        Object.defineProperty(document, "hasFocus", { value: () => true }),
      );
      const input = chat.getByRole("textbox", { name: "消息", exact: true });
      await input.focus();
      await card.dispatchEvent("contextmenu", { clientX: 150, clientY: 300 });
      await expect(chat.getByRole("menu")).toBeVisible();
      await chat.getByRole("menu").press("Escape");
      await expect(input).toBeFocused();

      await action({ type: "chat.close" });
      await app.evaluate(({ screen }) => {
        const nearest = screen.getDisplayNearestPoint.bind(screen);
        screen.getDisplayNearestPoint = (point) => ({
          ...nearest(point),
          workArea: { x: 0, y: 0, width: 800, height: 400 },
        });
      });
      await action({ type: "companion.reset" });
      await companion.evaluate(() =>
        window.desktop.action({ type: "companion.menu" }),
      );
      const menu = companion.getByRole("menu");
      await expect(menu).toBeVisible();
      const layout = await companion.evaluate(() =>
        window.desktop.bootstrap().then((value) => value.layout),
      );
      const rect = await menu.boundingBox();
      assert.ok(
        rect.width <= layout.bubble.width &&
          rect.height <= layout.bubble.height,
      );
      assert.ok(
        rect.x + rect.width <= layout.body.left ||
          rect.x >= layout.body.left + layout.body.size ||
          rect.y + rect.height <= layout.body.top ||
          rect.y >= layout.body.top + layout.body.size,
      );
      await expect(menu).toBeInViewport({ ratio: 1 });
      const language = companion.locator('[data-menu-id="language"]');
      await language.focus();
      await language.press("ArrowRight");
      await expect(
        companion.locator('[data-menu-id="language:en"]'),
      ).toBeVisible();
      await menu.press("ArrowLeft");
      await expect(language).toBeVisible();
      await menu.press("Escape");
      await expect(menu).toHaveCount(0);
    } finally {
      await app?.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
