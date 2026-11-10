import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
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
  "reminder details show one item and explain it in the current chat without losing drafts",
  { timeout: 60_000 },
  async (t) => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-reminder-e2e-"));
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
      const { chat } = await desktopWindows(app);
      await chat.emulateMedia({ reducedMotion: "reduce" });
      const errors = [];
      chat.on("pageerror", (error) => errors.push(String(error)));
      const state = () =>
        chat.evaluate(() =>
          window.desktop.bootstrap().then(({ state }) => state),
        );
      const action = (value) =>
        chat.evaluate((value) => window.desktop.action(value), value);
      const focus = async () => {
        await app.evaluate(({ app, BrowserWindow }) => {
          app.focus({ steal: true });
          const window = BrowserWindow.getAllWindows().find((window) =>
            window.webContents.getURL().includes("view=chat"),
          );
          window.focus();
          window.webContents.focus();
        });
        await expect
          .poll(() => chat.evaluate(() => document.hasFocus()))
          .toBe(true);
        await expect.poll(async () => (await state()).chatFocused).toBe(true);
      };
      const expectRead = async (id) => {
        await expect
          .poll(async () => {
            if (
              !(await chat.evaluate(() => document.hasFocus())) ||
              !(await state()).chatFocused
            )
              await focus();
            return (await state()).readMessageIds;
          })
          .toContain(id);
      };
      await expect
        .poll(async () => (await state()).connection)
        .toBe("connected");
      await app.evaluate(({ app, BrowserWindow }) => {
        app.focus({ steal: true });
        const window = BrowserWindow.getAllWindows().find((window) =>
          window.webContents.getURL().includes("view=chat"),
        );
        window.setBounds({ width: 1180, height: 800 });
        window.show();
        window.focus();
      });
      await action({ type: "chat.open" });
      const input = chat.getByRole("textbox", { name: "消息", exact: true });
      await input.fill("保留这份未发送的草稿");
      await action({ type: "chat.close" });
      const oldContent =
        "上一会话的提醒。[原始资料](https://example.com/source?item=old)";
      backend.push("old-reminder", oldContent);
      await expect
        .poll(async () => (await state()).inboxMessages.length)
        .toBe(2);
      await action({ type: "session.new" });
      backend.push("short-reminder", "当前会话的另一条提醒");
      const longContent =
        "长提醒的开头\n\n" +
        Array.from(
          { length: 20 },
          (_, index) =>
            `${index + 1}. **完整事实**：${"保留足够长的正文。".repeat(12)}`,
        ).join("\n\n") +
        "\n\n长提醒的结尾";
      backend.push("long-reminder", longContent);
      await expect
        .poll(async () => (await state()).inboxMessages.length)
        .toBe(3);
      await action({
        type: "chat.open",
        target: { kind: "message", id: "long-reminder" },
      });
      const pane = chat.locator(".remy-detail > .document-pane");
      const cards = chat.locator("#sidebar-remy .inbox-card");
      await mkdir(path.join(project, "reports"), { recursive: true });

      await t.test(
        "long reminders start at the beginning and acknowledge only the displayed item",
        async () => {
          await expect(pane).toBeVisible();
          await expect(pane.locator(".message-body")).toHaveCount(1);
          await expect(pane).toContainText("长提醒的开头");
          await expect(pane).toContainText("长提醒的结尾");
          await expect(pane).not.toContainText("当前会话的另一条提醒");
          await expect(pane).not.toContainText("上一会话的提醒");
          await expect(
            pane.locator(".message-body p").first(),
          ).toBeInViewport();
          assert.equal(await pane.evaluate((element) => element.scrollTop), 0);
          await expect(chat.locator(".result-toolbar button")).toHaveText([
            "返回当前对话",
          ]);
          await expect(input).toBeHidden();
          await expect(
            pane.getByRole("button", { name: "详细解读", exact: true }),
          ).toBeEnabled();
          await expectRead("long-reminder");
          const read = (await state()).readMessageIds;
          assert.ok(
            !read.includes("short-reminder") && !read.includes("old-reminder"),
          );
          await chat.screenshot({
            path: path.join(project, "reports/reminder-detail-long.png"),
          });
          await pane.evaluate((element) => {
            element.scrollTop = 400;
          });
          await cards.filter({ hasText: "长提醒的开头" }).click();
          await expect
            .poll(() => pane.evaluate((element) => element.scrollTop))
            .toBe(0);
        },
      );

      await t.test(
        "details on an inactive page do not acknowledge a reminder",
        async () => {
          await cards.filter({ hasText: "当前会话的另一条提醒" }).click();
          await chat.getByRole("tab", { name: "知识库", exact: true }).click();
          await chat.waitForTimeout(800);
          assert.ok(!(await state()).readMessageIds.includes("short-reminder"));
          await chat.getByRole("tab", { name: "Remy", exact: true }).click();
          await expectRead("short-reminder");
        },
      );

      await t.test(
        "historical reminders use the same detail view and return to the current draft",
        async () => {
          await cards.filter({ hasText: "上一会话的提醒" }).click();
          await expect(pane.locator(".message-body")).toHaveCount(1);
          await expect(pane).toContainText("上一会话的提醒");
          await expect(pane).not.toContainText("长提醒的开头");
          await expect(
            pane.getByRole("button", { name: "详细解读", exact: true }),
          ).toBeEnabled();
          await expectRead("old-reminder");
          assert.equal((await state()).sessionId, "session-2");
          backend.push("arriving-reminder", "新提醒不应替换正在阅读的内容");
          await expect(cards.filter({ hasText: "新提醒不应替换" })).toHaveCount(
            1,
          );
          await expect(pane).toContainText("上一会话的提醒");
          assert.ok(
            !(await state()).readMessageIds.includes("arriving-reminder"),
          );
          for (const theme of ["light", "dark"]) {
            await action({ type: "settings.update", theme });
            await expect(chat.locator("html")).toHaveAttribute(
              "data-theme",
              theme,
            );
            await chat.screenshot({
              path: path.join(project, `reports/reminder-detail-${theme}.png`),
            });
          }
          await chat
            .getByRole("button", { name: "返回当前对话", exact: true })
            .click();
          await expect(input).toHaveValue("保留这份未发送的草稿");
          await expect(
            chat.locator("#sidebar-remy .inbox-card.selected"),
          ).toHaveCount(0);
        },
      );
      await t.test(
        "incomplete reminders stay out of the list and empty details cannot be explained",
        async () => {
          backend.push("incomplete-reminder", "尚未完成的提醒", false);
          await expect
            .poll(async () =>
              (await state()).messages.some(
                (item) => item.id === "incomplete-reminder",
              ),
            )
            .toBe(true);
          await expect(cards.filter({ hasText: "尚未完成的提醒" })).toHaveCount(
            0,
          );
          assert.ok(
            !(await state()).readMessageIds.includes("incomplete-reminder"),
          );
          backend.push("incomplete-reminder", "已经完成的提醒");
          await cards.filter({ hasText: "已经完成的提醒" }).click();
          await expect(pane.locator(".explain-reminder")).toBeEnabled();
          await expectRead("incomplete-reminder");
          backend.push("empty-reminder", "");
          await expect
            .poll(async () =>
              (await state()).inboxMessages.some(
                (item) => item.id === "empty-reminder",
              ),
            )
            .toBe(true);
          await action({
            type: "chat.open",
            target: { kind: "message", id: "empty-reminder" },
          });
          await expect(pane.locator(".explain-reminder")).toHaveCount(0);
          await chat.waitForTimeout(800);
          assert.ok(!(await state()).readMessageIds.includes("empty-reminder"));
        },
      );

      await t.test(
        "explanations send only the selected reminder to the main chat and preserve its draft",
        async () => {
          await chat
            .getByRole("button", { name: "显示更多", exact: true })
            .click();
          await cards.filter({ hasText: "上一会话的提醒" }).click();
          const explain = pane.getByRole("button", {
            name: "详细解读",
            exact: true,
          });
          backend.activity("thinking");
          await expect(explain).toBeDisabled();
          backend.activity("completed");
          await expect(explain).toBeEnabled();
          await explain.click();
          await expect(input).toHaveValue("保留这份未发送的草稿");
          await expect(pane).toHaveCount(0);
          await expect.poll(async () => (await state()).pending).toBeNull();
          const current = await state();
          const queries = current.messages.filter(
            (message) => message.role === "user",
          );
          assert.equal(queries.length, 1);
          assert.ok(queries[0].content.startsWith("请详细解读"));
          assert.ok(queries[0].content.includes(oldContent));
          assert.ok(!queries[0].content.includes("保留这份未发送的草稿"));
          assert.ok(!queries[0].content.includes("长提醒的开头"));
          assert.equal(current.sessionId, "session-2");
          await expect(
            chat.locator(".chat-history .message.user").last(),
          ).toBeInViewport();
          await cards.filter({ hasText: "上一会话的提醒" }).click();
          await action({ type: "settings.update", language: "en" });
          await pane
            .getByRole("button", { name: "Explain in detail", exact: true })
            .click();
          await expect(
            chat.getByRole("textbox", { name: "Message", exact: true }),
          ).toHaveValue("保留这份未发送的草稿");
          await expect.poll(async () => (await state()).pending).toBeNull();
          const english = (await state()).messages.filter(
            (message) => message.role === "user",
          );
          assert.equal(english.length, 2);
          assert.ok(english[1].content.startsWith("Explain this content"));
          assert.ok(english[1].content.includes(oldContent));
        },
      );
      assert.deepEqual(errors, []);
    } finally {
      await app?.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
