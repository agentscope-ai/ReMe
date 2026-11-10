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
  "read receipts follow the displayed content through visibility changes and transient errors",
  { timeout: 60_000 },
  async (t) => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-read-receipts-"));
    const tasks = [0, 1, 2, 3].map((index) => ({
      id: `task:results/receipt-${index}.md`,
      kind: "task_result",
      title: `已读回归任务 ${index}`,
      preview: `任务摘要 ${index}`,
      content: index === 2 ? "   " : `完整任务正文 ${index}`,
      status: "received",
      created_at: new Date().toISOString(),
      task_id: "receipts",
      path: `results/receipt-${index}.md`,
      user_read_at: null,
      source_modified: false,
      source_present: true,
      receipt_status: null,
    }));
    const backend = await mockReMe({ taskResults: tasks });
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
      const state = () =>
        chat.evaluate(() =>
          window.desktop.bootstrap().then(({ state }) => state),
        );
      const action = (value) =>
        chat.evaluate((value) => window.desktop.action(value), value);
      await expect
        .poll(async () => (await state()).connection)
        .toBe("connected");
      await action({ type: "companion.visibility", visible: false });
      await action({ type: "chat.open" });
      // Control the OS/browser focus boundary; the real render, IPC and receipt stores remain in use.
      await chat.evaluate(() => {
        window.receiptFocused = false;
        window.receiptVisible = true;
        Object.defineProperty(document, "hasFocus", {
          value: () => window.receiptFocused,
        });
        Object.defineProperty(document, "visibilityState", {
          get: () => (window.receiptVisible ? "visible" : "hidden"),
        });
      });
      await app.evaluate(({ BrowserWindow }) => {
        const window = BrowserWindow.getAllWindows().find((window) =>
          window.webContents.getURL().includes("view=chat"),
        );
        window.emit("focus");
        // Keep the mocked native foreground state independent of the host window manager.
        window.removeAllListeners("focus");
        window.removeAllListeners("blur");
      });
      const focus = (focused) =>
        chat.evaluate((focused) => {
          window.receiptFocused = focused;
          window.dispatchEvent(new Event(focused ? "focus" : "blur"));
        }, focused);
      const visibility = (visible) =>
        chat.evaluate((visible) => {
          window.receiptVisible = visible;
          document.dispatchEvent(new Event("visibilitychange"));
        }, visible);
      const cards = chat.locator("#sidebar-remy .inbox-card");
      const read = (id) =>
        expect
          .poll(async () => (await state()).readMessageIds, { timeout: 2500 })
          .toContain(id);
      backend.push("receipt-first", "点击未读提醒正文");
      backend.push("receipt-second", "另一条提醒正文");

      await t.test(
        "clicking an unread reminder reads only the selected item after it is visible",
        async () => {
          await cards.filter({ hasText: "点击未读提醒正文" }).click();
          assert.ok(!(await state()).readMessageIds.includes("receipt-first"));
          await focus(true);
          await read("receipt-first");
          assert.ok(!(await state()).readMessageIds.includes("receipt-second"));
          await expect(
            cards.filter({ hasText: "点击未读提醒正文" }),
          ).toContainText("已读");
        },
      );

      await t.test(
        "a reminder resumes acknowledgement when document visibility returns without a focus event",
        async () => {
          await visibility(false);
          await cards.filter({ hasText: "另一条提醒正文" }).click();
          await chat.waitForTimeout(750);
          assert.ok(!(await state()).readMessageIds.includes("receipt-second"));
          await visibility(true);
          await read("receipt-second");
        },
      );

      await t.test(
        "inactive details and rapid selections do not read the previous item",
        async () => {
          await focus(false);
          backend.push("receipt-abandoned", "快速切换前的提醒");
          backend.push("receipt-returned", "返回后阅读的提醒");
          await cards.filter({ hasText: "快速切换前的提醒" }).click();
          await focus(true);
          await cards.filter({ hasText: "返回后阅读的提醒" }).click();
          await action({ type: "page.select", page: "knowledge" });
          await chat.waitForTimeout(750);
          assert.ok(
            !(await state()).readMessageIds.includes("receipt-abandoned"),
          );
          assert.ok(
            !(await state()).readMessageIds.includes("receipt-returned"),
          );
          await action({ type: "page.select", page: "remy" });
          await read("receipt-returned");
          assert.ok(
            !(await state()).readMessageIds.includes("receipt-abandoned"),
          );
        },
      );

      await t.test(
        "task read failures retry while the same result stays visible",
        async () => {
          await chat
            .getByRole("button", { name: "收件箱", exact: true })
            .click();
          backend.setTaskReadError(tasks[0].path);
          backend.setInboxReadDelay(1000);
          await cards.filter({ hasText: tasks[0].title }).click();
          await expect
            .poll(() => backend.taskReads.includes(tasks[0].path))
            .toBe(true);
          assert.equal(tasks[0].user_read_at, null);
          backend.setTaskReadError("");
          await expect
            .poll(() => tasks[0].user_read_at, { timeout: 2500 })
            .not.toBeNull();
          await expect(cards.filter({ hasText: tasks[0].title })).toContainText(
            "已读",
          );
          try {
            await expect(chat.locator(".inbox-detail-meta")).toContainText(
              "已读",
              { timeout: 2500 },
            );
          } finally {
            backend.setInboxReadDelay(0);
          }
        },
      );

      await t.test(
        "task receipts also resume after visibility changes without reselecting the result",
        async () => {
          await visibility(false);
          await cards.filter({ hasText: tasks[1].title }).click();
          await expect(chat.locator(".inbox-detail")).toContainText(
            tasks[1].content,
          );
          await chat.waitForTimeout(750);
          assert.equal(tasks[1].user_read_at, null);
          await visibility(true);
          await expect
            .poll(() => tasks[1].user_read_at, { timeout: 2500 })
            .not.toBeNull();
        },
      );

      await t.test(
        "an unfocused task detail becomes read after focus returns without another click",
        async () => {
          await focus(false);
          await cards.filter({ hasText: tasks[3].title }).click();
          await expect(chat.locator(".inbox-detail")).toContainText(
            tasks[3].content,
          );
          await chat.waitForTimeout(750);
          assert.equal(tasks[3].user_read_at, null);
          await focus(true);
          await expect
            .poll(() => tasks[3].user_read_at, { timeout: 2500 })
            .not.toBeNull();
        },
      );

      await t.test(
        "empty task content is not acknowledged as read",
        async () => {
          await cards.filter({ hasText: tasks[2].title }).click();
          await chat.waitForTimeout(750);
          assert.equal(tasks[2].user_read_at, null);
          assert.ok(!backend.taskReads.includes(tasks[2].path));
        },
      );

      await t.test(
        "main chat also resumes receipts for visible messages after visibility returns",
        async () => {
          await focus(false);
          await action({ type: "chat.open" });
          backend.push("receipt-empty", "   ");
          backend.push("receipt-partial", "尚未完成的主对话提醒", false);
          backend.push("receipt-main", "主对话恢复可见后的提醒");
          await expect(
            chat.locator(
              '.chat-history article[data-message-id="receipt-main"]',
            ),
          ).toBeVisible();
          const heading = chat.locator(
            '.chat-history .proactive-label[data-message-id="receipt-main"]',
          );
          await heading.scrollIntoViewIfNeeded();
          await expect(heading).toBeInViewport();
          await visibility(false);
          await focus(true);
          await chat.waitForTimeout(750);
          assert.ok(!(await state()).readMessageIds.includes("receipt-main"));
          await visibility(true);
          await read("receipt-main");
          const ids = (await state()).readMessageIds;
          assert.ok(!ids.includes("receipt-empty"));
          assert.ok(!ids.includes("receipt-partial"));
          backend.push("receipt-partial", "现在已完成的主对话提醒");
          await chat
            .locator(
              '.chat-history .proactive-label[data-message-id="receipt-partial"]',
            )
            .scrollIntoViewIfNeeded();
          await read("receipt-partial");
        },
      );
    } finally {
      await app?.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
