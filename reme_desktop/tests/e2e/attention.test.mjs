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
  "attention, native delivery and renderer failures use the same session-scoped state",
  { timeout: 60_000 },
  async (t) => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-attention-e2e-"));
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
      const { companion, chat } = await desktopWindows(app);
      await expect
        .poll(() =>
          chat.evaluate(() =>
            window.desktop.bootstrap().then(({ state }) => state.connection),
          ),
        )
        .toBe("connected");
      await app.evaluate(({ Notification }) => {
        // Exercise Electron's adapter without displaying real OS notifications.
        Notification.isSupported = () => true;
        globalThis.notices = [];
        globalThis.noticeClicks = [];
        Notification.prototype.show = function () {
          globalThis.notices.push(this);
          // Capture before native focus can legitimately close and remove listeners.
          globalThis.noticeClicks.push(this.listeners("click")[0]);
          this.emit("show");
        };
        Notification.prototype.close = function () {
          this.emit("close");
        };
      });
      const action = (value) =>
        chat.evaluate((value) => window.desktop.action(value), value);
      const notices = () => app.evaluate(() => globalThis.notices.length);
      const status = (text) =>
        expect(companion.getByRole("status")).toHaveText(text);

      await t.test(
        "hidden reminders are opt-in, acknowledged, rate-limited, and clickable",
        async () => {
          await action({ type: "chat.close" });
          await action({ type: "companion.visibility", visible: false });
          assert.equal(
            await app.evaluate(({ BrowserWindow }) =>
              BrowserWindow.getAllWindows()
                .find((win) =>
                  win.webContents.getURL().includes("view=companion"),
                )
                .isVisible(),
            ),
            false,
          );
          await expect
            .poll(() =>
              chat.evaluate(() =>
                window.desktop
                  .bootstrap()
                  .then(({ state }) => state.chatVisible),
              ),
            )
            .toBe(false);
          backend.push("native-one", "这是一条系统提醒");
          await expect
            .poll(() =>
              chat.evaluate(() =>
                window.desktop.bootstrap().then(({ state }) => state.unread),
              ),
            )
            .toBe(1);
          assert.equal(await notices(), 0);
          await action({ type: "settings.update", notifications: true });
          await expect.poll(notices).toBe(1);
          backend.push("native-two", "同一冷却期里的第二条消息");
          await expect
            .poll(() =>
              chat.evaluate(() =>
                window.desktop.bootstrap().then(({ state }) => state.unread),
              ),
            )
            .toBe(2);
          assert.equal(await notices(), 1);
          await app.evaluate(() => globalThis.noticeClicks[0]());
          await expect(
            chat
              .locator(".chat-history")
              .getByText("这是一条系统提醒", { exact: true }),
          ).toBeVisible();
        },
      );

      await t.test(
        "focused and quiet outcomes stay quiet; waiting can return to its own conversation",
        async () => {
          await action({ type: "companion.visibility", visible: true });
          backend.activity("thinking", "chat", "focused");
          backend.activity("waiting", "chat", "focused");
          await status("等你确认…");
          assert.equal(await notices(), 1);
          await action({
            type: "settings.update",
            quietUntil: Date.now() + 60_000,
          });
          await action({ type: "chat.close" });
          backend.activity("thinking", "chat", "quiet");
          backend.activity("waiting", "chat", "quiet");
          await expect
            .poll(() =>
              chat.evaluate(() =>
                window.desktop
                  .bootstrap()
                  .then(({ state }) => state.surface?.id),
              ),
            )
            .toContain("quiet");
          assert.equal(await notices(), 1);
          await action({ type: "settings.update", quietUntil: 0 });
          backend.activity("waiting", "chat", "quiet");
          backend.activity("thinking", "chat", "attention");
          backend.activity("waiting", "chat", "attention");
          await expect.poll(notices).toBe(2);
          await app.evaluate(() => {
            globalThis.staleClick = globalThis.noticeClicks[1];
          });
          backend.activity("completed", "chat", "attention");
          await expect
            .poll(() =>
              chat.evaluate(() =>
                window.desktop.bootstrap().then(({ state }) => state.status),
              ),
            )
            .toBe("idle");
          await action({ type: "session.new" });
          await expect
            .poll(() =>
              chat.evaluate(() =>
                window.desktop.bootstrap().then(({ state }) => state.sessionId),
              ),
            )
            .toBe("session-2");
          await app.evaluate(() => globalThis.staleClick());
          assert.equal(
            await app.evaluate(({ BrowserWindow }) =>
              BrowserWindow.getAllWindows()
                .find((win) => win.webContents.getURL().includes("view=chat"))
                .isVisible(),
            ),
            false,
          );
          await action({ type: "settings.update", notifications: false });
        },
      );

      await t.test(
        "phase text resizes the surface, while repeated phases do not resize or rebroadcast history",
        async () => {
          backend.activity("thinking", "chat", "work");
          await status("正在认真想…");
          await app.evaluate(({ BrowserWindow }) => {
            const win = BrowserWindow.getAllWindows().find((win) =>
              win.webContents.getURL().includes("view=companion"),
            );
            globalThis.boundsCalls = 0;
            globalThis.statePatches = [];
            const setBounds = win.setBounds.bind(win);
            const send = win.webContents.send.bind(win.webContents);
            win.setBounds = (...args) => {
              globalThis.boundsCalls++;
              return setBounds(...args);
            };
            win.webContents.send = (channel, patch) => {
              if (channel === "reme:state") globalThis.statePatches.push(patch);
              return send(channel, patch);
            };
          });
          backend.activity("working", "chat", "work");
          await status("正在查找和处理…");
          backend.activity("answering", "chat", "work");
          await status("正在回复…");
          const updates = await app.evaluate(() => ({
            bounds: globalThis.boundsCalls,
            patches: globalThis.statePatches,
          }));
          assert.ok(updates.bounds > 0);
          assert.ok(updates.patches.length >= 2);
          assert.ok(
            updates.patches.every(
              (patch) => !("messages" in patch) && !("pending" in patch),
            ),
          );
          await expect
            .poll(() =>
              companion.evaluate(() => {
                const rect = document
                  .querySelector(".status-notice")
                  .getBoundingClientRect();
                return window.desktop
                  .bootstrap()
                  .then(
                    ({ layout }) =>
                      layout.bubble.width === Math.ceil(rect.width),
                  );
              }),
            )
            .toBe(true);
          const count = await app.evaluate(() => globalThis.boundsCalls);
          const revision = await companion.evaluate(() =>
            window.desktop.bootstrap().then(({ state }) => state.revision),
          );
          backend.activity("answering", "chat", "work");
          await expect
            .poll(() =>
              companion.evaluate(() =>
                window.desktop.bootstrap().then(({ state }) => state.revision),
              ),
            )
            .toBeGreaterThan(revision);
          assert.equal(await app.evaluate(() => globalThis.boundsCalls), count);
          await mkdir(path.join(project, "reports"), { recursive: true });
          await companion.screenshot({
            path: path.join(project, "reports/status.png"),
            omitBackground: true,
          });
        },
      );

      await t.test(
        "a decode failure recovers to idle while preserving the task and rejecting stale reports",
        async () => {
          const stale = await companion.evaluate(() =>
            window.desktop
              .bootstrap()
              .then(({ state }) => state.behavior.animation.id),
          );
          await companion.evaluate(() => {
            const Original = window.Image;
            const setter = Object.getOwnPropertyDescriptor(
              HTMLImageElement.prototype,
              "src",
            ).set;
            window.Image = function (...args) {
              const image = new Original(...args);
              Object.defineProperty(image, "src", {
                set(value) {
                  if (value.includes("/confused/"))
                    queueMicrotask(() =>
                      image.dispatchEvent(new Event("error")),
                    );
                  else setter.call(image, value);
                },
              });
              return image;
            };
          });
          backend.activity("failed", "chat", "work");
          await expect(
            companion.getByText("部分角色动画无法加载，已切换为待机动作"),
          ).toBeVisible();
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "idle",
          );
          backend.activity("thinking", "chat", "next");
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "thinking",
          );
          await companion.evaluate(
            (id) =>
              window.desktop.action({
                type: "playback.report",
                id,
                status: "failed",
              }),
            stale,
          );
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "thinking",
          );
        },
      );
    } finally {
      await app?.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
