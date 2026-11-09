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
  "language is shared by native controls and renderers, and survives restarting the desktop",
  { timeout: 60_000 },
  async (t) => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-language-e2e-"));
    const backend = await mockReMe();
    let app;
    let companion;
    let chat;
    const launch = async () => {
      app = await electron.launch({
        args: [project],
        env: {
          ...process.env,
          REME_DESKTOP_DATA_DIR: directory,
          REME_DESKTOP_API_URL: backend.url,
        },
      });
      ({ companion, chat } = await desktopWindows(app));
      await expect
        .poll(() =>
          chat.evaluate(() =>
            window.desktop.bootstrap().then(({ state }) => state.connection),
          ),
        )
        .toBe("connected");
      await app.evaluate(({ Menu }) => {
        Menu.prototype.popup = function () {
          globalThis.languageMenu = this;
        };
      });
    };
    const openMenu = async () => {
      await companion.evaluate(() =>
        window.desktop.action({ type: "companion.menu" }),
      );
    };
    const choose = async (id) => {
      await openMenu();
      await app.evaluate((_, id) => {
        const item = globalThis.languageMenu.getMenuItemById(id);
        if (!item) throw new Error(`Missing menu item ${id}`);
        item.click(item, undefined, {});
      }, id);
    };
    try {
      await launch();
      await t.test(
        "Chinese is the default and English switches every desktop surface in place",
        async () => {
          await expect(companion.locator("html")).toHaveAttribute(
            "lang",
            "zh-CN",
          );
          await openMenu();
          assert.equal(
            await app.evaluate(
              () =>
                globalThis.languageMenu.getMenuItemById("language:zh").checked,
            ),
            true,
          );
          await choose("chat");
          const draft = "保留草稿 / keep this draft";
          await chat
            .getByRole("textbox", { name: "消息", exact: true })
            .fill(draft);
          backend.activity("thinking", "chat", "language-turn");
          await expect(companion.getByRole("status")).toHaveText("正在认真想…");
          const animation = await companion.evaluate(() =>
            window.desktop
              .bootstrap()
              .then(({ state }) => state.behavior.animation.id),
          );
          const subscriptions = backend.cursors.length;
          await choose("language:en");
          await expect(companion.locator("html")).toHaveAttribute("lang", "en");
          await expect(chat.locator("html")).toHaveAttribute("lang", "en");
          await expect(companion.getByRole("status")).toHaveText(
            "Thinking it through…",
          );
          await expect(
            chat.getByRole("textbox", { name: "Message", exact: true }),
          ).toHaveValue(draft);
          assert.equal(
            await companion.evaluate(() =>
              window.desktop
                .bootstrap()
                .then(({ state }) => state.behavior.animation.id),
            ),
            animation,
          );
          assert.equal(backend.cursors.length, subscriptions);
          await openMenu();
          const labels = await app.evaluate(() => ({
            settings: globalThis.languageMenu.getMenuItemById("settings").label,
            animation:
              globalThis.languageMenu.getMenuItemById("animation:thinking")
                .label,
            checked:
              globalThis.languageMenu.getMenuItemById("language:en").checked,
          }));
          assert.deepEqual(labels, {
            settings: "Settings…",
            animation: "Thinking",
            checked: true,
          });
          await chat
            .getByRole("button", { name: "Settings", exact: true })
            .click();
          await expect(
            chat.getByRole("region", { name: "Desktop settings" }),
          ).toBeVisible();
          await mkdir(path.join(project, "reports"), { recursive: true });
          await chat.screenshot({
            path: path.join(project, "reports/chat-en.png"),
          });
          await companion.screenshot({
            path: path.join(project, "reports/companion-en.png"),
            omitBackground: true,
          });
        },
      );

      await t.test(
        "an IPC validation error is translated again when the menu changes language",
        async () => {
          await chat
            .getByRole("textbox", { name: "ReMe service URL", exact: true })
            .fill("file:///invalid");
          await chat
            .getByRole("button", { name: "Connect", exact: true })
            .click();
          await expect(
            chat.getByText(
              "Use an HTTP/HTTPS service URL without credentials, query parameters, or fragments",
              { exact: true },
            ),
          ).toBeVisible();
          await choose("language:zh");
          await expect(
            chat.getByText(
              "服务地址需要是 HTTP/HTTPS 地址，不包含账号、查询参数或片段",
              { exact: true },
            ),
          ).toBeVisible();
          await chat.getByRole("tab", { name: "Remy", exact: true }).click();
          await expect(
            chat.getByRole("textbox", { name: "消息", exact: true }),
          ).toHaveValue("保留草稿 / keep this draft");
          await choose("language:en");
          await expect(chat.locator("html")).toHaveAttribute("lang", "en");
        },
      );

      await t.test(
        "restart restores the saved language and notifications use it",
        async () => {
          await app.close();
          app = undefined;
          const settings = JSON.parse(
            await readFile(
              path.join(directory, "desktop/settings.json"),
              "utf8",
            ),
          );
          assert.equal(settings.language, "en");
          await launch();
          await choose("chat");
          await expect(
            chat.getByRole("textbox", { name: "Message", exact: true }),
          ).toBeVisible();
          await expect(
            chat
              .locator(".chat-history")
              .getByText("历史提醒不会在启动时补弹。", { exact: true }),
          ).toBeVisible();
          await app.evaluate(({ Notification }) => {
            Notification.isSupported = () => true;
            globalThis.localizedNotices = [];
            Notification.prototype.show = function () {
              globalThis.localizedNotices.push({
                title: this.title,
                body: this.body,
              });
              this.emit("show");
            };
            Notification.prototype.close = function () {
              this.emit("close");
            };
          });
          await chat.evaluate(() =>
            window.desktop.action({ type: "chat.close" }),
          );
          await companion.evaluate(() =>
            window.desktop.action({
              type: "settings.update",
              notifications: true,
            }),
          );
          backend.activity("waiting", "chat", "notify-language");
          await expect
            .poll(() => app.evaluate(() => globalThis.localizedNotices))
            .toEqual([
              {
                title: "Remy needs your attention",
                body: "Click to return to the conversation",
              },
            ]);
        },
      );
    } finally {
      await app?.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
