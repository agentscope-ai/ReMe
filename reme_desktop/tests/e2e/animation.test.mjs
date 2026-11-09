import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm } from "node:fs/promises";
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
  "animation preferences and reduced motion preserve meaningful activity and playback completion",
  { timeout: 40_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-animation-e2e-"));
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
      const { companion, chat } = await desktopWindows(app);
      const errors = [];
      companion.on("pageerror", (error) => errors.push(String(error)));
      chat.on("pageerror", (error) => errors.push(String(error)));
      const state = () =>
        companion.evaluate(() =>
          window.desktop.bootstrap().then(({ state }) => state),
        );
      await expect
        .poll(async () => (await state()).connection)
        .toBe("connected");
      await chat.evaluate(() => window.desktop.action({ type: "chat.open" }));
      await chat.getByRole("button", { name: "设置", exact: true }).click();
      await chat.getByRole("button", { name: "Remy", exact: true }).click();
      await chat.getByLabel("空闲动作", { exact: true }).selectOption("calm");
      await expect
        .poll(async () => (await state()).settings.activityLevel)
        .toBe("calm");
      await chat.getByLabel("多久后打盹", { exact: true }).selectOption("0");
      await expect
        .poll(async () => (await state()).settings.sleepMinutes)
        .toBe(0);
      const stored = JSON.parse(
        await readFile(path.join(directory, "desktop/settings.json"), "utf8"),
      );
      assert.equal(stored.activityLevel, "calm");
      assert.equal(stored.sleepMinutes, 0);
      await mkdir(path.join(project, "reports"), { recursive: true });
      await chat.screenshot({
        path: path.join(project, "reports/animation-settings.png"),
      });
      for (const update of [
        { activityLevel: "invalid" },
        { sleepMinutes: -1 },
        { sleepMinutes: true },
      ]) {
        const error = await companion.evaluate(
          (update) =>
            window.desktop.action({ type: "settings.update", ...update }),
          update,
        );
        assert.equal(error?.key, "invalidAnimationSettings");
      }
      await chat.evaluate(() => window.desktop.action({ type: "chat.close" }));
      await companion.evaluate(() =>
        window.desktop.action({ type: "companion.visibility", visible: false }),
      );
      await companion.waitForTimeout(5500);
      assert.equal((await state()).behavior.error, "");
      await companion.evaluate(() =>
        window.desktop.action({ type: "companion.visibility", visible: true }),
      );
      await expect(companion.locator("canvas")).toBeVisible();
      assert.equal((await state()).behavior.error, "");
      await companion.emulateMedia({ reducedMotion: "reduce" });
      await companion.evaluate(() => {
        window.animationDraws = 0;
        const draw = CanvasRenderingContext2D.prototype.drawImage;
        CanvasRenderingContext2D.prototype.drawImage = function (...args) {
          if (this.canvas === document.querySelector("canvas"))
            window.animationDraws++;
          return draw.apply(this, args);
        };
      });
      backend.activity("thinking", "chat", "animation-turn");
      await expect(companion.locator("canvas")).toHaveAttribute(
        "data-animation",
        "thinking",
      );
      await expect
        .poll(() => companion.evaluate(() => window.animationDraws))
        .toBeGreaterThan(0);
      const animation = (await state()).behavior.animation;
      const draws = await companion.evaluate(() => window.animationDraws);
      await companion.waitForTimeout(700);
      assert.equal(
        await companion.evaluate(() => window.animationDraws),
        draws,
      );
      backend.activity("working", "chat", "animation-turn");
      await expect
        .poll(async () => (await state()).behavior.label)
        .toBe("正在查找和处理…");
      assert.deepEqual((await state()).behavior.animation, animation);
      backend.activity("completed", "chat", "animation-turn");
      await expect(companion.locator("canvas")).toHaveAttribute(
        "data-animation",
        "happy",
      );
      await expect(companion.locator("canvas")).toHaveAttribute(
        "data-animation",
        "idle",
        { timeout: 4000 },
      );
      const awake = await companion.evaluate(async () => {
        const image = new Image();
        image.crossOrigin = "anonymous";
        image.src = "reme-assets://remy/frames/idle/00.png";
        await image.decode();
        const canvas = document.createElement("canvas");
        const actual = document.querySelector("canvas");
        canvas.width = actual.width;
        canvas.height = actual.height;
        canvas.getContext("2d").drawImage(image, 0, 0);
        return actual.toDataURL() === canvas.toDataURL();
      });
      assert.equal(awake, true, "static idle must use the awake pose");
      await companion.emulateMedia({ reducedMotion: "no-preference" });
      backend.activity("thinking", "chat", "animation-next");
      await expect(companion.locator("canvas")).toHaveAttribute(
        "data-animation",
        "thinking",
      );
      const before = await companion.evaluate(() => window.animationDraws);
      await expect
        .poll(() => companion.evaluate(() => window.animationDraws))
        .toBeGreaterThan(before + 2);
      assert.deepEqual(errors, []);
    } finally {
      if (app) await app.close().catch(() => {});
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
