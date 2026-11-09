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
  "workspace navigation preserves conversation, reads tagged documents and restores appearance",
  { timeout: 90_000 },
  async (t) => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-workspace-e2e-"));
    const backend = await mockReMe({ workspace: true });
    let app;
    let chat;
    const errors = [];
    const launch = async () => {
      app = await electron.launch({
        args: [project],
        env: {
          ...process.env,
          REME_DESKTOP_DATA_DIR: directory,
          REME_DESKTOP_API_URL: backend.url,
        },
      });
      ({ chat } = await desktopWindows(app));
      chat.on("pageerror", (error) => errors.push(String(error)));
      await expect
        .poll(() =>
          chat.evaluate(() =>
            window.desktop.bootstrap().then(({ state }) => state.connection),
          ),
        )
        .toBe("connected");
      await app.evaluate(({ BrowserWindow }) => {
        const win = BrowserWindow.getAllWindows().find((win) =>
          win.webContents.getURL().includes("view=chat"),
        );
        win.show();
        win.focus();
      });
    };
    const state = () =>
      chat.evaluate(() =>
        window.desktop.bootstrap().then(({ state }) => state),
      );
    await mkdir(path.join(project, "reports"), { recursive: true });
    try {
      await launch();
      await t.test(
        "wide native window and accessible top-level tabs",
        async () => {
          const bounds = await app.evaluate(({ BrowserWindow }) =>
            BrowserWindow.getAllWindows()
              .find((win) => win.webContents.getURL().includes("view=chat"))
              .getBounds(),
          );
          assert.ok(bounds.width > bounds.height);
          await expect(chat.getByRole("tab")).toHaveCount(2);
          await chat
            .getByRole("textbox", { name: "消息", exact: true })
            .fill("切换页面后保留这份草稿");
          await chat.screenshot({
            path: path.join(project, "reports/remy-light.png"),
          });
        },
      );
      await t.test(
        "documents, directory views, YAML tags and wikilinks",
        async () => {
          await chat.getByRole("tab", { name: "知识库", exact: true }).click();
          backend.push(
            "workspace-notice",
            "发现了与你最近研究相关的新线索。点击通知，我们可以继续聊聊。",
          );
          await expect.poll(async () => (await state()).unread).toBe(1);
          assert.ok(
            !(await state()).readMessageIds.includes("workspace-notice"),
          );
          await chat
            .locator(".source-tabs")
            .getByRole("button", { name: "知识库", exact: true })
            .click();
          await chat
            .getByRole("button", { name: "agent-memory.md", exact: true })
            .click();
          await expect(
            chat.getByRole("heading", {
              name: "Agent 记忆，从记录到连接",
              exact: true,
            }),
          ).toBeVisible();
          await expect(chat.locator(".document-tag")).toHaveCount(2);
          await chat.screenshot({
            path: path.join(project, "reports/knowledge-light.png"),
          });
          await chat
            .locator(".document-tag")
            .filter({ hasText: "agent" })
            .click();
          await expect(
            chat.locator(".tag-cloud button[aria-pressed=true]"),
          ).toHaveCount(1);
          await expect(
            chat.getByRole("button", { name: "preferences.md", exact: true }),
          ).toHaveCount(0);
          await chat
            .getByRole("button", { name: "清除筛选", exact: true })
            .click();
          await chat
            .getByRole("link", { name: "今天的研究笔记", exact: true })
            .click();
          await expect(
            chat.getByRole("heading", { name: "今天的研究笔记", exact: true }),
          ).toBeVisible();
          await chat.getByRole("tab", { name: "Remy", exact: true }).click();
          await expect(
            chat.getByRole("textbox", { name: "消息", exact: true }),
          ).toHaveValue("切换页面后保留这份草稿");
          await chat
            .locator(".inbox-card")
            .filter({ hasText: "发现了与你最近研究相关的新线索" })
            .click();
          await expect(chat.locator(".message-focused")).toContainText(
            "发现了与你最近研究相关的新线索",
          );
          await expect
            .poll(async () =>
              (await state()).readMessageIds.includes("workspace-notice"),
            )
            .toBe(true);
          await chat.screenshot({
            path: path.join(project, "reports/remy-notifications.png"),
          });
        },
      );
      await t.test(
        "theme and language settings preserve both page state and preferences",
        async () => {
          await chat.getByRole("button", { name: "设置", exact: true }).click();
          await chat
            .locator(".settings-nav")
            .getByRole("button", { name: "外观", exact: true })
            .click();
          await chat.getByRole("button", { name: "深色", exact: true }).click();
          await expect(chat.locator("html")).toHaveAttribute(
            "data-theme",
            "dark",
          );
          await chat.screenshot({
            path: path.join(project, "reports/settings-dark.png"),
          });
          await chat
            .getByRole("combobox", { name: "语言", exact: true })
            .selectOption("en");
          await expect(
            chat.getByRole("tab", { name: "Knowledge", exact: true }),
          ).toBeVisible();
          await chat
            .getByRole("tab", { name: "Knowledge", exact: true })
            .click();
          await expect(
            chat.getByRole("heading", { name: "今天的研究笔记", exact: true }),
          ).toBeVisible();
          await chat.screenshot({
            path: path.join(project, "reports/knowledge-dark.png"),
          });
          const overflow = await chat.evaluate(
            () => document.documentElement.scrollWidth > innerWidth,
          );
          assert.equal(overflow, false);
          await app.evaluate(({ BrowserWindow }) =>
            BrowserWindow.getAllWindows()
              .find((win) => win.webContents.getURL().includes("view=chat"))
              .setBounds({ width: 1080, height: 680 }),
          );
          await app.close();
          const settings = JSON.parse(
            await readFile(
              path.join(directory, "desktop/settings.json"),
              "utf8",
            ),
          );
          assert.equal(settings.theme, "dark");
          assert.equal(settings.language, "en");
          assert.equal(settings.windowBounds.width, 1080);
          await launch();
          await expect(chat.locator("html")).toHaveAttribute(
            "data-theme",
            "dark",
          );
          await expect(
            chat.getByRole("button", { name: "Settings", exact: true }),
          ).toBeVisible();
          const restored = await app.evaluate(({ BrowserWindow, screen }) => {
            const bounds = BrowserWindow.getAllWindows()
              .find((win) => win.webContents.getURL().includes("view=chat"))
              .getBounds();
            return { bounds, area: screen.getDisplayMatching(bounds).workArea };
          });
          assert.equal(
            restored.bounds.width,
            Math.min(settings.windowBounds.width, restored.area.width),
          );
          assert.equal(
            restored.bounds.height,
            Math.min(settings.windowBounds.height, restored.area.height),
          );
        },
      );
      assert.deepEqual(errors, []);
    } finally {
      if (app) await app.close().catch(() => {});
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
