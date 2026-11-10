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
  "companion frames stay fixed through streaming, hover, completion and theme changes",
  { timeout: 30_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-surface-e2e-"));
    const backend = await mockReMe({ speech: true });
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
      const state = () =>
        chat.evaluate(() =>
          window.desktop.bootstrap().then(({ state }) => state),
        );
      await expect
        .poll(async () => (await state()).connection)
        .toBe("connected");
      await chat.evaluate(() =>
        window.desktop.action({ type: "settings.update", voiceReplies: false }),
      );
      backend.activity("answering", "chat", "fixed-frame");
      await expect(companion.getByRole("status")).toHaveText("正在回复…");
      const status = await companion.locator(".status-notice").boundingBox();
      await app.evaluate(({ BrowserWindow }) => {
        const win = BrowserWindow.getAllWindows().find((win) =>
          win.webContents.getURL().includes("view=companion"),
        );
        globalThis.boundsCalls = 0;
        const setBounds = win.setBounds.bind(win);
        win.setBounds = (...args) => {
          globalThis.boundsCalls++;
          return setBounds(...args);
        };
      });
      backend.activity("working", "chat", "fixed-frame");
      await expect(companion.getByRole("status")).toHaveText("正在查找和处理…");
      assert.deepEqual(
        await companion.locator(".status-notice").boundingBox(),
        status,
      );
      assert.equal(await app.evaluate(() => globalThis.boundsCalls), 0);
      backend.activity("answering", "chat", "fixed-frame");
      const message = {
        id: "fixed-answer",
        role: "assistant",
        source: "chat",
        content: "先说结论。",
        delivery: "idle",
        notification_eligible: true,
        completed: false,
        turn_id: "fixed-frame",
        chunks: [],
      };
      backend.upsert(message);
      const reply = companion.locator('.message-bubble[data-source="chat"]');
      await expect(reply).toBeVisible();
      const original = await reply.boundingBox();
      const bounds = await app.evaluate(({ BrowserWindow }) => {
        globalThis.boundsCalls = 0;
        return BrowserWindow.getAllWindows()
          .find((win) => win.webContents.getURL().includes("view=companion"))
          .getBounds();
      });
      for (const content of [
        "先说结论，你的判断基本正确。",
        "先说结论，**尺寸保持固定**。\n\n- 文字继续增加\n- 正文区域保持稳定",
        "先说结论。".repeat(60),
      ]) {
        backend.upsert({ ...message, content });
        await expect(reply.locator(".message-body")).toContainText("先说结论");
        await expect
          .poll(
            async () =>
              (await state()).messages.find((item) => item.id === message.id)
                ?.content,
          )
          .toBe(content);
        assert.deepEqual(await reply.boundingBox(), original);
      }
      await reply.hover();
      await expect.poll(async () => (await state()).surface?.paused).toBe(true);
      backend.upsert({
        ...message,
        content: "完成后的最终回答。",
        completed: true,
      });
      backend.activity("completed", "chat", "fixed-frame");
      await expect(reply.locator(".message-body")).toHaveText(
        "完成后的最终回答。",
      );
      await expect(
        reply.getByRole("button", { name: "朗读这条回复", exact: true }),
      ).toBeVisible();
      assert.deepEqual(await reply.boundingBox(), original);
      assert.equal(await app.evaluate(() => globalThis.boundsCalls), 0);
      assert.deepEqual(
        await app.evaluate(({ BrowserWindow }) =>
          BrowserWindow.getAllWindows()
            .find((win) => win.webContents.getURL().includes("view=companion"))
            .getBounds(),
        ),
        bounds,
      );
      const typography = await reply
        .locator(".message-body")
        .evaluate((element) => getComputedStyle(element).fontSize);
      assert.equal(typography, "13px");
      await mkdir(path.join(project, "reports"), { recursive: true });
      await companion.screenshot({
        path: path.join(project, "reports/companion-fixed-light.png"),
        omitBackground: true,
      });
      await chat.evaluate(() =>
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
      assert.deepEqual(await reply.boundingBox(), original);
      assert.equal(await app.evaluate(() => globalThis.boundsCalls), 0);
      await companion.screenshot({
        path: path.join(project, "reports/companion-fixed-dark.png"),
        omitBackground: true,
      });
      await companion.evaluate(() => {
        document.documentElement.dataset.platform = "win32";
      });
      const fill = await reply.evaluate(
        (element) => getComputedStyle(element).backgroundColor,
      );
      assert.match(fill, /^rgb\(/, "Windows uses an opaque reading surface");
      assert.deepEqual(await reply.boundingBox(), original);
    } finally {
      await app?.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
