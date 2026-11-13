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
  "Remy queries and task inbox remain readable and keyboard accessible",
  { timeout: 60_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-inbox-e2e-"));
    const taskResults = Array.from({ length: 15 }, (_, index) => ({
      id: `task:results/run-${index}.md`,
      kind: "task_result",
      title: `任务结果 ${index + 1}`,
      preview:
        "市场变化与值得关注的信息。\n\n保留必要时间、来源，只说确认的事情。",
      status: "received",
      created_at: new Date().toISOString(),
      task_id: index < 7 ? "market" : `task-${index}`,
      path: `results/run-${index}.md`,
      user_read_at: index === 2 ? new Date().toISOString() : null,
      source_modified: false,
      source_present: true,
      receipt_status: null,
    }));
    const backend = await mockReMe({ taskResults, speech: true });
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
      const errors = [];
      chat.on("pageerror", (error) => errors.push(String(error)));
      await expect
        .poll(() =>
          chat.evaluate(() =>
            window.desktop.bootstrap().then(({ state }) => state.connection),
          ),
        )
        .toBe("connected");
      await app.evaluate(({ BrowserWindow }) => {
        const window = BrowserWindow.getAllWindows().find((window) =>
          window.webContents.getURL().includes("view=chat"),
        );
        window.setBounds({ width: 1180, height: 800 });
        window.show();
        window.focus();
      });
      await chat.getByRole("button", { name: "新建对话", exact: true }).click();
      const queries = [
        "每天早上 8 点，帮我总结昨晚美股表现。",
        "5 分钟后提醒我起来活动一下。",
        "帮我关注黄金价格，有重要变化时提醒我。",
      ];
      await expect(chat.locator(".conversation-prompts button")).toHaveText(
        queries,
      );
      const checkLayout = async () => {
        const layout = await chat.evaluate(() => {
          const buttons = [
            ...document.querySelectorAll(".conversation-prompts button"),
          ];
          const hint = document.querySelector(".empty-chat > p");
          return {
            tops: buttons.map((button) => button.getBoundingClientRect().top),
            singleLines: buttons.every(
              (button) =>
                button.querySelector("span").getBoundingClientRect().height <=
                parseFloat(getComputedStyle(button).lineHeight) + 1,
            ),
            hintLines:
              hint.getBoundingClientRect().height /
              parseFloat(getComputedStyle(hint).lineHeight),
            overflow: document.documentElement.scrollWidth > innerWidth,
          };
        });
        assert.equal(new Set(layout.tops).size, 1);
        assert.ok(layout.singleLines && layout.hintLines <= 1.01);
        assert.equal(layout.overflow, false);
      };
      await checkLayout();
      const input = chat.getByRole("textbox", { name: "消息", exact: true });
      for (const query of queries) {
        await chat.getByRole("button", { name: query, exact: true }).click();
        await expect(input).toHaveValue(query);
        await expect(input).toBeFocused();
      }
      await expect(chat.locator(".keyboard-hint")).toContainText(
        "按住 Ctrl + R 语音输入",
      );
      await chat.getByRole("button", { name: "收件箱", exact: true }).click();
      const cards = chat.locator("#sidebar-remy .inbox-card");
      await expect(cards).toHaveCount(5);
      await expect(chat.locator("#sidebar-remy .inbox-filters")).toHaveCount(0);
      await expect(cards.nth(0).locator(".inbox-read-state")).toHaveText(
        "未读",
      );
      await expect(cards.nth(2).locator(".inbox-read-state")).toHaveText(
        "已读",
      );
      const bounds = await cards.evaluateAll((elements) =>
        elements.map((element) => ({
          top: element.getBoundingClientRect().top,
          bottom: element.getBoundingClientRect().bottom,
          height: element.getBoundingClientRect().height,
          content: element.scrollHeight,
        })),
      );
      assert.ok(
        bounds.every(
          (card) => card.height > 100 && card.content <= card.height + 1,
        ),
      );
      assert.ok(
        bounds
          .slice(1)
          .every((card, index) => card.top >= bounds[index].bottom),
      );
      const combo = chat.getByRole("combobox", { name: "按任务筛选" });
      await combo.press("End");
      const last = chat.getByRole("option", { name: "task-14", exact: true });
      await expect(last).toBeInViewport();
      await combo.press("Enter");
      await expect(combo).toHaveText("task-14");
      await expect(cards).toHaveCount(1);
      await combo.press("Home");
      await combo.press("Enter");
      await expect(combo).toHaveText("全部任务");
      await expect(cards).toHaveCount(5);
      await combo.click();
      await chat.getByRole("option", { name: "market", exact: true }).click();
      await expect(combo).toHaveText("market");
      await chat.getByRole("button", { name: "下一页", exact: true }).click();
      await expect(cards).toHaveCount(2);
      await chat.getByRole("button", { name: "上一页", exact: true }).click();
      await expect(cards).toHaveCount(5);
      await combo.click();
      await combo.press("Escape");
      await expect(chat.getByRole("listbox")).toHaveCount(0);
      await combo.click();
      await chat.locator("#sidebar-remy .sidebar-heading h1").click();
      await expect(chat.getByRole("listbox")).toHaveCount(0);
      await cards.first().click();
      await expect(chat.locator(".inbox-detail")).toBeVisible();
      await expect.poll(() => taskResults[0].user_read_at).not.toBeNull();
      await expect(chat.locator(".inbox-detail-meta")).toContainText("已读");
      await chat
        .getByRole("button", { name: "返回当前对话", exact: true })
        .click();
      await expect(input).toHaveValue(queries[2]);
      await chat.evaluate(() =>
        window.desktop.action({
          type: "settings.update",
          language: "en",
          theme: "dark",
        }),
      );
      await expect(chat.locator("html")).toHaveAttribute("lang", "en");
      await checkLayout();
      await expect(chat.locator(".keyboard-hint")).toContainText(
        "Hold Ctrl + R for voice input",
      );
      await chat
        .getByRole("button", { name: "Reminders", exact: true })
        .click();
      await expect(chat.locator("#sidebar-remy .inbox-filters")).toHaveCount(0);
      assert.deepEqual(errors, []);
    } finally {
      await app?.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
