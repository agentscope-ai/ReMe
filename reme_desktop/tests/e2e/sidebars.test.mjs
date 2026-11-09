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
  "all pages collapse their sidebar independently, preserve content and restore preferences",
  { timeout: 60_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-sidebars-e2e-"));
    const backend = await mockReMe({ workspace: true });
    let app;
    let page;
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
      ({ chat: page } = await desktopWindows(app));
      page.on("pageerror", (error) => errors.push(String(error)));
      await app.evaluate(({ BrowserWindow }) => {
        const win = BrowserWindow.getAllWindows().find((entry) =>
          entry.webContents.getURL().includes("view=chat"),
        );
        win.setBounds({ width: 1200, height: 780 });
        win.show();
        win.focus();
      });
      await expect
        .poll(() =>
          page.evaluate(() =>
            window.desktop.bootstrap().then(({ state }) => state.connection),
          ),
        )
        .toBe("connected");
    };
    const toggle = () => page.locator(".sidebar-toggle");
    const collapse = async (name) => {
      await expect(toggle()).toHaveAttribute(
        "aria-controls",
        `sidebar-${name}`,
      );
      await expect(toggle()).toHaveAttribute("aria-expanded", "true");
      await toggle().click();
      await expect(page.locator(`#sidebar-${name}`)).toBeHidden();
      await expect(
        page.locator(".workspace-page:visible .split-handle"),
      ).toBeHidden();
      await expect(toggle()).toHaveAttribute("aria-expanded", "false");
      await expect
        .poll(() =>
          page.locator(".workspace-page:visible").evaluate((element) => {
            const available = element.getBoundingClientRect();
            const main = element
              .querySelector(".workspace-main")
              .getBoundingClientRect();
            return (
              Math.abs(main.x - available.x) < 1 &&
              Math.abs(main.width - available.width) < 1
            );
          }),
        )
        .toBe(true);
    };
    try {
      await launch();
      const input = () =>
        page.getByRole("textbox", { name: "消息", exact: true });
      await input().fill("折叠后保留的草稿");
      const separator = page.getByRole("separator");
      await separator.press("ArrowRight");
      const width = await page
        .locator("#sidebar-remy")
        .evaluate((node) => node.getBoundingClientRect().width);
      await collapse("remy");
      await expect(input()).toHaveValue("折叠后保留的草稿");
      await toggle().click();
      assert.equal(
        await page
          .locator("#sidebar-remy")
          .evaluate((node) => node.getBoundingClientRect().width),
        width,
      );
      await collapse("remy");
      await page.getByRole("tab", { name: "知识库", exact: true }).click();
      await page
        .getByRole("button", { name: "long-title.md", exact: true })
        .click();
      await expect(page.locator(".document-title")).toContainText("研究笔记");
      await page
        .locator(".document-scroll")
        .evaluate((node) => (node.scrollTop = 400));
      await collapse("knowledge");
      await expect(page.locator(".document-title")).toContainText("研究笔记");
      assert.ok(
        (await page
          .locator(".document-scroll")
          .evaluate((node) => node.scrollTop)) > 0,
      );
      await page.getByRole("button", { name: "设置", exact: true }).click();
      await page
        .locator(".settings-nav")
        .getByRole("button", { name: "外观", exact: true })
        .click();
      await collapse("settings");
      await expect(
        page.getByRole("heading", { name: "外观", exact: true }),
      ).toBeVisible();
      await mkdir(path.join(project, "reports"), { recursive: true });
      await page.screenshot({
        path: path.join(project, "reports/settings-sidebar-collapsed.png"),
      });
      await toggle().click();
      await expect(
        page.locator(".settings-nav [aria-current=page]"),
      ).toHaveText("外观");
      await collapse("settings");
      await page.getByRole("tab", { name: "Remy", exact: true }).click();
      await expect(toggle()).toHaveAttribute("aria-expanded", "false");
      await expect(input()).toHaveValue("折叠后保留的草稿");
      await page.screenshot({
        path: path.join(project, "reports/remy-sidebar-collapsed.png"),
      });
      await app.close();
      await launch();
      await expect(toggle()).toHaveAttribute("aria-expanded", "false");
      await expect(page.locator("#sidebar-remy")).toBeHidden();
      await toggle().click();
      assert.equal(
        await page
          .locator("#sidebar-remy")
          .evaluate((node) => node.getBoundingClientRect().width),
        width,
      );
      await page.getByRole("tab", { name: "知识库", exact: true }).click();
      await expect(page.locator("#sidebar-knowledge")).toBeHidden();
      await expect(toggle()).toHaveAttribute("aria-expanded", "false");
      await page.getByRole("button", { name: "设置", exact: true }).click();
      await expect(page.locator("#sidebar-settings")).toBeHidden();
      await expect(toggle()).toHaveAttribute("aria-expanded", "false");
      assert.deepEqual(errors, []);
    } finally {
      if (app) await app.close().catch(() => {});
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
