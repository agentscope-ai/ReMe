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
  "proactive reminder explanations keep the selected context and composer draft",
  { timeout: 45_000 },
  async (t) => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-explain-e2e-"));
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
      const errors = [];
      chat.on("pageerror", (error) => errors.push(String(error)));
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

      const content =
        "芯片指数本周波动明显，数据仍需核实。 [原始行情](https://example.com/quote?symbol=SOX&range=1mo)，“涨跌”与 {风险} 应分开说明。";
      const otherContent = "另一条提醒：周末留一点时间休息。";
      backend.push("explain-target", content);
      backend.push("other-reminder", otherContent);
      backend.push("unfinished-reminder", "仍在整理", false);
      backend.push("empty-reminder", "");
      backend.upsert({
        id: "ordinary-reply",
        role: "assistant",
        source: "chat",
        content: "一条普通的对话回复。",
        delivery: "idle",
        notification_eligible: true,
        completed: true,
        chunks: [],
      });
      const card = chat.locator('.message[data-message-id="explain-target"]');
      const explain = card.getByRole("button", {
        name: "详细解读",
        exact: true,
      });
      const draft = chat.getByRole("textbox", { name: "消息", exact: true });

      await t.test(
        "the action is right-aligned and only offered on complete reminders",
        async () => {
          await expect(explain).toBeEnabled();
          for (const id of [
            "unfinished-reminder",
            "empty-reminder",
            "ordinary-reply",
          ])
            await expect(
              chat.locator(
                `.message[data-message-id="${id}"] .explain-reminder`,
              ),
            ).toHaveCount(0);
          backend.activity("thinking");
          await expect(explain).toBeDisabled();
          backend.activity("interrupted");
          await expect(explain).toBeEnabled();
          const placement = await card.evaluate((element) => {
            const bounds = element.getBoundingClientRect();
            const button = element
              .querySelector(".explain-reminder")
              .getBoundingClientRect();
            const narration = element
              .querySelector(".read-aloud")
              .getBoundingClientRect();
            return {
              rightAligned:
                button.right > bounds.right - 30 &&
                button.right <= bounds.right,
              sameRow:
                Math.abs(
                  button.top +
                    button.height / 2 -
                    narration.top -
                    narration.height / 2,
                ) < 2,
            };
          });
          assert.deepEqual(placement, { rightAligned: true, sameRow: true });
          await mkdir(path.join(project, "reports"), { recursive: true });
          for (const theme of ["light", "dark"]) {
            await action({ type: "settings.update", theme });
            await expect(chat.locator("html")).toHaveAttribute(
              "data-theme",
              theme,
            );
            await card.screenshot({
              path: path.join(project, `reports/reminder-explain-${theme}.png`),
            });
          }
          await action({ type: "settings.update", theme: "light" });
        },
      );

      await t.test(
        "one click sends the selected reminder with its source and preserves the draft",
        async () => {
          await draft.fill("这是还没有发送的草稿。");
          backend.setReplyDelay(1200);
          await explain.click();
          await expect(explain).toBeDisabled();
          await expect(draft).toHaveValue("这是还没有发送的草稿。");
          await expect
            .poll(
              async () =>
                (await state()).messages.filter(
                  (message) => message.role === "user",
                ).length,
            )
            .toBe(1);
          const query = (await state()).messages.find(
            (message) => message.role === "user",
          ).content;
          assert.ok(query.startsWith("请详细解读"));
          assert.ok(query.includes(content));
          assert.ok(!query.includes(otherContent));
          await expect(explain).toBeEnabled();
          await expect(draft).toHaveValue("这是还没有发送的草稿。");
          await expect(
            chat.locator(
              '.message.assistant[data-proactive="false"] .explain-reminder',
            ),
          ).toHaveCount(0);
        },
      );

      await t.test(
        "English uses a matching action and request without sending the draft",
        async () => {
          await action({ type: "settings.update", language: "en" });
          const english = card.getByRole("button", {
            name: "Explain in detail",
            exact: true,
          });
          await expect(english).toBeEnabled();
          await english.click();
          await expect
            .poll(
              async () =>
                (await state()).messages.filter(
                  (message) => message.role === "user",
                ).length,
            )
            .toBe(2);
          const queries = (await state()).messages.filter(
            (message) => message.role === "user",
          );
          assert.ok(
            queries[1].content.startsWith("Explain this proactive reminder"),
          );
          assert.ok(queries[1].content.includes(content));
          assert.ok(!queries[1].content.includes("这是还没有发送的草稿。"));
          await expect(english).toBeEnabled();
          await expect(
            chat.getByRole("textbox", { name: "Message", exact: true }),
          ).toHaveValue("这是还没有发送的草稿。");
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
