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
  "chat and companion share phases, streamed answers, outcomes and reminder messages",
  { timeout: 45_000 },
  async () => {
    const directory = await mkdtemp(
      path.join(tmpdir(), "reme-presentation-e2e-"),
    );
    const backend = await mockReMe();
    backend.setReplyDelay(1500);
    const app = await electron.launch({
      args: [project],
      env: {
        ...process.env,
        REME_DESKTOP_DATA_DIR: directory,
        REME_DESKTOP_API_URL: backend.url,
      },
    });
    try {
      const { companion, chat } = await desktopWindows(app);
      const state = () =>
        chat.evaluate(() =>
          window.desktop.bootstrap().then(({ state }) => state),
        );
      const action = (value) =>
        chat.evaluate((value) => window.desktop.action(value), value);
      await expect
        .poll(async () => (await state()).connection)
        .toBe("connected");
      await action({ type: "chat.open" });

      for (const [phase, text] of [
        ["thinking", "正在认真想…"],
        ["working", "正在查找和处理…"],
        ["remembering", "正在回忆…"],
        ["waiting", "等你确认…"],
      ]) {
        backend.activity(phase);
        await expect(companion.getByRole("status")).toHaveText(text);
        await expect(
          chat.locator(".message.assistant .message-status").last(),
        ).toContainText(text);
      }
      backend.activity("completed");
      await expect.poll(async () => (await state()).status).toBe("idle");
      await action({ type: "session.new" });
      await expect
        .poll(async () => (await state()).sessionId)
        .toBe("session-2");
      await expect(companion.locator(".status-notice")).toHaveCount(0);

      const start = (query) =>
        chat.evaluate((query) => {
          globalThis.reply = window.desktop.action({
            type: "chat.send",
            query,
          });
        }, query);
      await start("hello");
      const reply = companion.locator('.message-bubble[data-source="chat"]');
      await expect(reply).toBeVisible();
      await expect(reply.locator(".message-body")).toContainText("旧的增量");
      await reply.getByRole("button", { name: "展开聊聊" }).click();
      await expect(chat.locator(".message.assistant").last()).toContainText(
        "旧的增量",
      );
      await chat.evaluate(() => globalThis.reply);
      const final = "我记住了。我们可以继续聊聊你的研究。";
      await expect(reply.locator(".message-body")).toHaveText(final);
      await expect(
        chat.locator(".message.assistant .message-body").last(),
      ).toHaveText(final);
      assert.equal(
        (await state()).messages.filter((message) => message.content === final)
          .length,
        1,
      );
      await mkdir(path.join(project, "reports"), { recursive: true });
      await chat.screenshot({
        path: path.join(project, "reports/presentation-chat.png"),
      });
      await companion.screenshot({
        path: path.join(project, "reports/presentation-companion.png"),
        omitBackground: true,
      });
      await reply.getByRole("button", { name: "展开聊聊" }).click();
      await expect
        .poll(async () => (await state()).focusMessageId)
        .toBe("assistant-1");

      await start("请停止");
      await expect(reply.locator(".message-body")).toContainText("旧的增量");
      await action({ type: "chat.stop" });
      await chat.evaluate(() => globalThis.reply);
      await expect(reply.locator(".message-status")).toHaveText("回复已停止");
      await expect(
        chat.locator(".message.assistant .message-status").last(),
      ).toHaveText("回复已停止");
      await expect.poll(async () => (await state()).status).toBe("idle");
      await action({ type: "session.new" });
      await expect
        .poll(async () => (await state()).sessionId)
        .toBe("session-3");
      await expect(reply).toHaveCount(0);

      backend.upsert({
        id: "failed-answer",
        role: "assistant",
        source: "chat",
        content: "保留的部分回答",
        delivery: "idle",
        notification_eligible: true,
        completed: true,
        turn_id: "failed-turn",
        chunks: [],
      });
      backend.activity("failed", "chat", "failed-turn");
      await expect(reply.locator(".message-status")).toHaveText("这次没有完成");
      await expect(
        chat.locator(".message.assistant .message-status").last(),
      ).toHaveText("这次没有完成");
      await reply.getByRole("button", { name: "收起气泡" }).click();

      // A rejected receipt during a focus transition must be retried, not optimistically marked read.
      await app.evaluate(({ ipcMain }) => {
        const original = ipcMain._invokeHandlers.get("reme:action");
        globalThis.readAttempts = 0;
        ipcMain.removeHandler("reme:action");
        ipcMain.handle("reme:action", (event, value) => {
          if (value.type === "messages.read" && ++globalThis.readAttempts === 1)
            return null;
          return original(event, value);
        });
      });
      backend.push("visible-reminder", "聊天打开时，两边也是同一条提醒。");
      const reminder = companion.locator(
        '.message-bubble[data-source="proactive"]',
      );
      await expect(reminder).toBeVisible();
      await expect(
        chat.locator('[data-message-id="visible-reminder"] .message-body'),
      ).toHaveText("聊天打开时，两边也是同一条提醒。");
      await reminder.getByRole("button", { name: "展开聊聊" }).click();
      await expect
        .poll(async () => (await state()).focusMessageId)
        .toBe("visible-reminder");
      await app.evaluate(({ app, BrowserWindow }) => {
        app.focus({ steal: true });
        const chat = BrowserWindow.getAllWindows().find((window) =>
          window.webContents.getURL().includes("view=chat"),
        );
        chat.focus();
        chat.webContents.focus();
      });
      await expect
        .poll(() => chat.evaluate(() => document.hasFocus()))
        .toBe(true);
      await expect.poll(async () => (await state()).chatFocused).toBe(true);
      await expect.poll(async () => (await state()).unread).toBe(0);
      assert.ok(await app.evaluate(() => globalThis.readAttempts >= 2));
      await expect(reminder).toBeVisible();
      await reminder.getByRole("button", { name: "收起气泡" }).click();
      await expect(reminder).toHaveCount(0);
    } finally {
      await app.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
