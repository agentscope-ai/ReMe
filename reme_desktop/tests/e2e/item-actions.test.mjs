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
import { openCompanionMenu } from "./menus.mjs";

const project = path.resolve(fileURLToPath(new URL("../../", import.meta.url)));

test(
  "reminder and task menus share replay, navigation, copy, explanation and bulk reading",
  { timeout: 60_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-item-actions-"));
    const tasks = Array.from({ length: 8 }, (_, index) => ({
      id: `task:results/${index}.md`,
      kind: "task_result",
      title: `研究结果 ${index}`,
      preview: `任务摘要 ${index}`,
      content: `任务完整正文 ${index}。\n\n[资料](https://example.com/${index})\n\n${"完整证据。".repeat(100)}`,
      status: "received",
      created_at: new Date().toISOString(),
      task_id: "research",
      path: `results/${index}.md`,
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
      const { chat, companion } = await desktopWindows(app);
      const errors = [];
      chat.on("pageerror", (error) => errors.push(String(error)));
      companion.on("pageerror", (error) => errors.push(String(error)));
      const state = () =>
        chat.evaluate(() =>
          window.desktop.bootstrap().then(({ state }) => state),
        );
      const action = (value) =>
        chat.evaluate((value) => window.desktop.action(value), value);
      await expect
        .poll(async () => (await state()).connection)
        .toBe("connected");
      await app.evaluate(({ Menu, clipboard }) => {
        globalThis.nativeMenus = 0;
        Menu.prototype.popup = function () {
          globalThis.nativeMenus++;
        };
        clipboard.writeText = (text) => {
          globalThis.copiedText = text;
        };
      });
      let menuPage = chat;
      const choose = (id) => menuPage.locator(`[data-menu-id="${id}"]`).click();
      const openMenu = async (card) => {
        menuPage = chat;
        await card.click({ button: "right" });
        await expect(chat.getByRole("menu")).toBeVisible();
        return chat
          .locator(".context-menu [data-menu-id]")
          .evaluateAll((items) =>
            items.map((item) => ({
              id: item.dataset.menuId,
              label: item.textContent,
              enabled: !item.disabled,
            })),
          );
      };
      backend.notificationWindow({
        enabled: false,
        starts_at: new Date(Date.now() - 3600_000).toISOString(),
        ends_at: new Date(Date.now() + 3600_000).toISOString(),
      });
      backend.push("old", "历史提醒正文。[来源](https://example.com/old)");
      await expect
        .poll(async () =>
          (await state()).inboxMessages.some((item) => item.id === "old"),
        )
        .toBe(true);
      await action({ type: "session.new" });
      await action({
        type: "settings.update",
        quietUntil: Date.now() + 60_000,
      });
      await action({ type: "chat.open" });
      const input = chat.getByRole("textbox", { name: "消息", exact: true });
      await input.fill("保留主对话草稿");
      const cards = chat.locator("#sidebar-remy .inbox-card");
      const old = cards.filter({ hasText: "历史提醒正文" });
      const menu = await openMenu(old);
      assert.deepEqual(
        menu.map((item) => item.label),
        ["重新提醒", "查看详情", "详细解读", "复制内容"],
      );
      assert.equal((await state()).focusTarget, null);
      await choose("copy");
      await expect
        .poll(() => app.evaluate(() => globalThis.copiedText))
        .toBe("历史提醒正文。[来源](https://example.com/old)");
      assert.ok(!(await state()).readMessageIds.includes("old"));
      await mkdir(path.join(project, "reports"), { recursive: true });
      for (const theme of ["light", "dark"]) {
        await action({ type: "settings.update", theme });
        await openMenu(old);
        await expect(chat.getByRole("menu")).toBeInViewport({ ratio: 1 });
        await chat.getByRole("menu").screenshot({
          path: path.join(project, `reports/item-menu-${theme}.png`),
        });
        await chat.getByRole("menu").press("Escape");
        await expect(chat.getByRole("menu")).toHaveCount(0);
      }
      await action({ type: "companion.visibility", visible: false });
      await openMenu(old);
      await choose("remind");
      const bubble = companion.locator(
        '.message-bubble[data-message-id="old"]',
      );
      await expect(bubble).toBeVisible();
      await expect(bubble).toContainText("历史提醒正文");
      await expect(
        bubble.getByRole("button", { name: "10 分钟后", exact: true }),
      ).toHaveCount(0);
      assert.ok(!(await state()).readMessageIds.includes("old"));
      assert.equal((await state()).sessionId, "session-2");
      assert.equal((await state()).pending, null);
      const first = (await state()).surface.presentationId;
      await openMenu(old);
      await choose("remind");
      await expect
        .poll(async () => (await state()).surface.presentationId)
        .not.toBe(first);
      await expect(companion.locator(".message-bubble")).toHaveCount(1);
      await bubble
        .getByRole("button", { name: "展开聊聊", exact: true })
        .click();
      await expect(chat.locator(".document-pane")).toContainText(
        "历史提醒正文",
      );
      assert.deepEqual((await state()).focusTarget, {
        kind: "message",
        id: "old",
      });
      await chat
        .getByRole("button", { name: "返回当前对话", exact: true })
        .click();
      await expect(input).toHaveValue("保留主对话草稿");

      await chat.getByRole("button", { name: "收件箱", exact: true }).click();
      await expect(cards).toHaveCount(5);
      const task = cards.filter({ hasText: "研究结果 0" });
      await openMenu(task);
      await choose("copy");
      await expect
        .poll(() => app.evaluate(() => globalThis.copiedText))
        .toBe(tasks[0].content);
      assert.equal(tasks[0].user_read_at, null);
      backend.activity("thinking");
      await expect.poll(async () => (await state()).status).toBe("active");
      const busyMenu = await openMenu(task);
      assert.equal(
        busyMenu.find((item) => item.id === "explain").enabled,
        false,
      );
      await choose("remind");
      const taskBubble = companion.locator(
        `.message-bubble[data-message-id="${tasks[0].id}"]`,
      );
      await expect(taskBubble).toBeVisible();
      assert.equal((await state()).surface.message.content, tasks[0].content);
      assert.equal(tasks[0].user_read_at, null);
      await mkdir(path.join(project, "reports"), { recursive: true });
      await companion.screenshot({
        path: path.join(project, "reports/item-replay.png"),
        omitBackground: true,
      });
      await taskBubble
        .getByRole("button", { name: "展开聊聊", exact: true })
        .click();
      await expect(chat.locator(".inbox-detail")).toContainText(
        "任务完整正文 0",
      );
      assert.deepEqual((await state()).focusTarget, {
        kind: "task_result",
        id: tasks[0].id,
      });
      backend.activity("completed");
      await chat
        .getByRole("button", { name: "返回当前对话", exact: true })
        .click();
      await expect(input).toHaveValue("保留主对话草稿");
      await openMenu(task);
      await choose("explain");
      await expect
        .poll(
          async () =>
            (await state()).messages.filter((item) => item.role === "user")
              .length,
        )
        .toBe(1);
      const query = (await state()).messages.find(
        (item) => item.role === "user",
      ).content;
      assert.ok(query.includes(tasks[0].content));
      assert.ok(!query.includes("任务摘要"));
      await expect(input).toHaveValue("保留主对话草稿");
      await expect.poll(async () => (await state()).pending).toBeNull();
      await chat
        .getByRole("button", { name: "全部标为已读", exact: true })
        .click();
      await expect
        .poll(() => tasks.every((item) => item.user_read_at))
        .toBe(true);
      await expect.poll(async () => (await state()).unread).toBe(0);
      await expect(cards).toHaveCount(5);
      await expect(cards.locator(".inbox-read-state")).toHaveText(
        Array(5).fill("已读"),
      );

      tasks[7].user_read_at = null;
      await action({ type: "chat.close" });
      backend.push("new-unread", "萌宠菜单操作前的新提醒");
      await expect.poll(async () => (await state()).unread).toBe(1);
      menuPage = companion;
      await openCompanionMenu(companion);
      for (const theme of ["light", "dark"]) {
        await action({ type: "settings.update", theme });
        await expect(companion.getByRole("menu")).toBeInViewport({ ratio: 1 });
        await companion.getByRole("menu").screenshot({
          path: path.join(project, `reports/pet-menu-${theme}.png`),
        });
      }
      await choose("read-all");
      await expect
        .poll(() => tasks.every((item) => item.user_read_at))
        .toBe(true);
      await expect.poll(async () => (await state()).unread).toBe(0);
      assert.equal((await state()).chatVisible, false);
      assert.equal(await app.evaluate(() => globalThis.nativeMenus), 0);
      assert.deepEqual(errors, []);
    } finally {
      await app?.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
