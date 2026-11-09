import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
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
  "companion glass surfaces follow text, theme and screen bounds",
  { timeout: 30_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-surface-e2e-"));
    const backend = await mockReMe();
    const app = await electron.launch({
      args: [project],
      env: {
        ...process.env,
        REME_DESKTOP_DATA_DIR: directory,
        REME_DESKTOP_API_URL: backend.url,
      },
    });
    try {
      const { companion } = await desktopWindows(app);
      await expect
        .poll(() =>
          companion.evaluate(() =>
            window.desktop.bootstrap().then(({ state }) => state.connection),
          ),
        )
        .toBe("connected");
      const settled = async () => {
        await expect
          .poll(async () =>
            companion.evaluate(() => {
              const rect = document
                .querySelector(".status-notice")
                ?.getBoundingClientRect();
              return window.desktop
                .bootstrap()
                .then(({ layout }) =>
                  Boolean(
                    rect &&
                    layout.bubble.width === Math.ceil(rect.width) &&
                    layout.bubble.height === Math.ceil(rect.height),
                  ),
                );
            }),
          )
          .toBe(true);
      };
      const position = () =>
        app.evaluate(({ BrowserWindow }) => {
          const win = BrowserWindow.getAllWindows().find((win) =>
            win.webContents.getURL().includes("view=companion"),
          );
          return win.webContents
            .executeJavaScript("window.desktop.bootstrap()")
            .then(({ layout }) => ({
              x: win.getBounds().x + layout.body.left,
              y: win.getBounds().y + layout.body.top,
            }));
        });
      backend.activity("answering", "chat", "surface-size");
      await expect(companion.getByRole("status")).toHaveText("正在回复…");
      await settled();
      const short = await companion.locator(".status-notice").boundingBox();
      const character = await companion.locator("canvas").boundingBox();
      assert.ok(
        Math.abs(
          short.x + short.width / 2 - character.x - character.width / 2,
        ) <= 1,
      );
      assert.equal(character.y - short.y - short.height, 6);
      const originalPosition = await position();
      assert.ok(short.width < 160, String(short.width));
      assert.ok(short.height <= 38, String(short.height));
      await mkdir(path.join(project, "reports"), { recursive: true });
      await companion.screenshot({
        path: path.join(project, "reports/companion-status-light.png"),
        omitBackground: true,
      });
      const oldKey = JSON.stringify([
        "status",
        "chat:surface-size:answering",
        null,
        "正在回复…",
      ]);
      backend.activity("working", "chat", "surface-size");
      await expect(companion.getByRole("status")).toHaveText("正在查找和处理…");
      await settled();
      const wider = await companion.locator(".status-notice").boundingBox();
      assert.ok(wider.width > short.width);
      assert.deepEqual(await position(), originalPosition);
      const before = await companion.evaluate(() =>
        window.desktop.bootstrap().then(({ layout }) => layout),
      );
      await companion.evaluate(
        (key) =>
          window.desktop.measureSurface({ key, width: 600, height: 300 }),
        oldKey,
      );
      await expect
        .poll(() =>
          companion.evaluate(() =>
            window.desktop.bootstrap().then(({ layout }) => layout),
          ),
        )
        .toEqual(before);
      await companion.evaluate(() =>
        window.desktop.action({
          type: "settings.update",
          theme: "dark",
          language: "en",
        }),
      );
      await expect(companion.locator("html")).toHaveAttribute(
        "data-theme",
        "dark",
      );
      await expect(companion.getByRole("status")).toHaveText(
        "Searching and working…",
      );
      await settled();
      await companion.screenshot({
        path: path.join(project, "reports/companion-status-dark.png"),
        omitBackground: true,
      });
      // A larger font and long localized line exercise measurement and wrapping together.
      await companion.locator(".status-text").evaluate((element) => {
        element.textContent =
          "Searching your connected memories and organizing the information for this conversation.";
        element.style.fontSize = "16px";
      });
      await settled();
      const wrapped = await companion.locator(".status-notice").boundingBox();
      assert.ok(wrapped.height > short.height);
      assert.ok(wrapped.width <= 280);
      assert.deepEqual(await position(), originalPosition);
      backend.activity("completed", "chat", "surface-size");
      await companion.evaluate(() =>
        window.desktop.action({
          type: "settings.update",
          theme: "light",
          language: "zh",
        }),
      );
      await expect(companion.getByRole("status")).toHaveText("完成啦");
      await settled();
      assert.ok(
        (await companion.locator(".status-notice").boundingBox()).width <
          short.width,
      );
      const clip = await companion
        .locator(".status-notice")
        .evaluate((element) => {
          const rect = element.getBoundingClientRect();
          return (
            rect.x >= 0 &&
            rect.y >= 0 &&
            rect.right <= innerWidth &&
            rect.bottom <= innerHeight
          );
        });
      assert.equal(clip, true);
      backend.push("compact-reminder", "记得留一点时间休息。 ");
      await expect(
        companion.getByRole("region", { name: "主动提醒", exact: true }),
      ).toBeVisible({ timeout: 15_000 });
      await expect
        .poll(() =>
          companion.evaluate(() => {
            const rect = document
              .querySelector(".message-bubble")
              .getBoundingClientRect();
            return window.desktop
              .bootstrap()
              .then(
                ({ layout }) =>
                  layout.bubble.width === Math.ceil(rect.width) &&
                  layout.bubble.height === Math.ceil(rect.height),
              );
          }),
        )
        .toBe(true);
      assert.ok(
        (await companion.locator(".message-bubble").boundingBox()).width < 280,
      );
      await companion.screenshot({
        path: path.join(project, "reports/companion-reminder-light.png"),
        omitBackground: true,
      });
    } finally {
      await app.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
