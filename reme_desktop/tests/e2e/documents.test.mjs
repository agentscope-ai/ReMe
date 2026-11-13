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
  "document properties, long Markdown titles and the settings shortcut",
  { timeout: 60_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-documents-e2e-"));
    const backend = await mockReMe({ workspace: true });
    let app;
    const errors = [];
    try {
      app = await electron.launch({
        args: [project],
        env: {
          ...process.env,
          REME_DESKTOP_DATA_DIR: directory,
          REME_DESKTOP_API_URL: backend.url,
        },
      });
      const { chat: page } = await desktopWindows(app);
      page.on("pageerror", (error) => errors.push(String(error)));
      await app.evaluate(({ BrowserWindow }) => {
        const win = BrowserWindow.getAllWindows().find((entry) =>
          entry.webContents.getURL().includes("view=chat"),
        );
        win.show();
        win.focus();
        win.setBounds({ width: 1200, height: 780 });
      });
      const tabs = page.locator(".workspace-tabs");
      await expect(tabs.getByRole("tab")).toHaveCount(2);
      await expect(tabs.getByText("设置", { exact: true })).toHaveCount(0);
      await page.getByRole("tab", { name: "知识库", exact: true }).click();
      await page
        .locator(".source-tabs")
        .getByRole("button", { name: "每日记忆", exact: true })
        .click();
      await page
        .getByRole("button", { name: "long-title.md", exact: true })
        .click();
      const title = page.locator(".document-title");
      await expect(title).toHaveAttribute("data-long", "true");
      await expect(title).toContainText("研究笔记：把零散记录连接成知识");
      await expect(page.locator(".document-content h1")).toHaveCount(1);
      assert.equal(
        await title.evaluate((element) => getComputedStyle(element).fontSize),
        "20px",
      );
      const properties = page.locator(".document-properties");
      await expect(properties.locator("dt")).toHaveText([
        "created_at",
        "status",
        "target_session_id",
        "memory_tags",
        "context",
      ]);
      await expect(properties).toContainText("2026-11-08T10:53:47+08:00");
      await expect(properties).toContainText("unsent");
      await expect(properties).toContainText("example-session-id");
      await expect(properties).toContainText("reviewed: true");
      await expect(page.locator(".document-markdown table")).toHaveCount(1);
      await expect(page.locator(".document-markdown pre code")).toHaveText(
        "source: local\n",
      );
      await mkdir(path.join(project, "reports"), { recursive: true });
      await page.screenshot({
        path: path.join(project, "reports/knowledge-frontmatter-light.png"),
      });
      await properties.locator("summary").click();
      await expect(properties).not.toHaveAttribute("open");
      const scroll = page.locator(".document-scroll");
      await scroll.evaluate((element) => (element.scrollTop = 480));
      const position = await scroll.evaluate((element) => element.scrollTop);
      assert.ok(position > 0);
      const settings = page.getByRole("button", { name: "设置", exact: true });
      await settings.click();
      await expect(settings).toHaveAttribute("aria-pressed", "true");
      await expect(title).toBeHidden();
      await settings.click();
      await expect(title).toBeVisible();
      await expect(properties).not.toHaveAttribute("open");
      assert.equal(
        await scroll.evaluate((element) => element.scrollTop),
        position,
      );
      // The already-selected Knowledge tab must also close Settings.
      await settings.click();
      await page.getByRole("tab", { name: "知识库", exact: true }).click();
      await expect(settings).toHaveAttribute("aria-pressed", "false");
      await expect(title).toBeVisible();
      await settings.click();
      await page
        .locator(".settings-nav")
        .getByRole("button", { name: "外观", exact: true })
        .click();
      await page.getByRole("button", { name: "深色", exact: true }).click();
      await settings.click();
      await scroll.evaluate((element) => (element.scrollTop = 0));
      await properties.locator("summary").click();
      await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
      await page.screenshot({
        path: path.join(project, "reports/knowledge-frontmatter-dark.png"),
      });
      await page
        .getByRole("button", { name: "invalid-yaml.md", exact: true })
        .click();
      await expect(
        page.getByRole("heading", { name: "属性格式有误的笔记", exact: true }),
      ).toBeVisible();
      await expect(page.locator(".frontmatter-error pre")).toHaveText(
        "memory_tags: [unclosed\n",
      );
      await expect(page.locator(".document-markdown")).toContainText(
        "正文依然可以阅读。",
      );
      backend.setDocument(
        "daily/2026-11-08/research.md",
        "# 链接测试\n\n[[digest/wiki/late.md|跨目录笔记]]\n\n[空格笔记](research%20notes.md)",
      );
      backend.setDocument(
        "digest/wiki/late.md",
        "# 尚未出现在目录中的笔记\n\n直接读取工作区文件。",
      );
      backend.setDocument(
        "daily/2026-11-08/research notes.md",
        "# 包含空格的笔记\n\n相对路径保持原目录。",
      );
      await page
        .getByRole("button", { name: "research.md", exact: true })
        .click();
      await page.getByRole("link", { name: "跨目录笔记", exact: true }).click();
      await expect(page.locator(".document-title")).toHaveText(
        "尚未出现在目录中的笔记",
      );
      await page
        .getByRole("button", { name: "research.md", exact: true })
        .click();
      await page.getByRole("link", { name: "空格笔记", exact: true }).click();
      await expect(page.locator(".document-title")).toHaveText(
        "包含空格的笔记",
      );
      const tags = Array.from({ length: 70 }, (_, i) => [
        `topic-${String(i).padStart(2, "0")}`,
        1,
      ]);
      backend.setTags(tags);
      await page
        .getByRole("button", { name: "刷新知识库", exact: true })
        .click();
      await expect(page.locator(".tag-cloud button")).toHaveCount(60);
      await page.locator(".more-tags").click();
      await expect(page.locator(".tag-cloud button")).toHaveCount(70);
      backend.setTags([["new-topic", 2], ...tags.slice(1)]);
      await page
        .getByRole("button", { name: "刷新知识库", exact: true })
        .click();
      await expect(page.locator(".tag-cloud")).toContainText("new-topic");
      await expect(page.locator(".tag-cloud")).not.toContainText("topic-00");
      assert.equal(
        await page.evaluate(
          () => document.documentElement.scrollWidth > innerWidth,
        ),
        false,
      );
      assert.deepEqual(errors, []);
    } finally {
      if (app) await app.close().catch(() => {});
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
