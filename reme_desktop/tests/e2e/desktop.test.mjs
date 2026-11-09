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
  "desktop conversation and subscription lifecycle against an isolated ReMe boundary",
  { timeout: 90_000 },
  async (t) => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-desktop-e2e-"));
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
        timeout: 30_000,
      });
      const errors = [];
      const logs = [];
      const { companion, chat } = await desktopWindows(app);
      for (const page of [companion, chat]) {
        page.on("pageerror", (error) => errors.push(String(error)));
        page.on("console", (message) => {
          if (message.type() === "error") logs.push(message.text());
        });
      }
      await companion.waitForSelector("canvas");
      await expect
        .poll(() =>
          companion.evaluate(() =>
            window.desktop.bootstrap().then((value) => value.state.connection),
          ),
        )
        .toBe("connected");
      await expect
        .poll(() =>
          companion
            .locator("canvas")
            .evaluate(
              (canvas) =>
                canvas.getContext("2d").getImageData(256, 170, 1, 1).data[3],
            ),
        )
        .toBeGreaterThan(0);

      await t.test(
        "native transparent companion loads existing art without historical popup",
        async () => {
          await expect(
            companion.getByRole("region", { name: "主动提醒" }),
          ).toHaveCount(0);
          const info = await app.evaluate(({ BrowserWindow }) =>
            BrowserWindow.getAllWindows().map((window) => ({
              visible: window.isVisible(),
              focusable: window.isFocusable(),
              alwaysOnTop: window.isAlwaysOnTop(),
              url: window.webContents.getURL(),
            })),
          );
          assert.equal(
            info.find((entry) => entry.url.includes("view=companion"))
              .focusable,
            false,
          );
          assert.equal(
            info.find((entry) => entry.url.includes("view=companion"))
              .alwaysOnTop,
            true,
          );
          assert.equal(
            info.find((entry) => entry.url.includes("view=chat")).visible,
            false,
          );
          assert.equal(
            await companion.evaluate(() => typeof window.require),
            "undefined",
          );
        },
      );

      await t.test(
        "reopening a minimized conversation restores its native window",
        async () => {
          await companion.evaluate(() =>
            window.desktop.action({ type: "chat.open" }),
          );
          const windowState = () =>
            app.evaluate(({ BrowserWindow }) => {
              const window = BrowserWindow.getAllWindows().find((entry) =>
                entry.webContents.getURL().includes("view=chat"),
              );
              return {
                minimized: window.isMinimized(),
                visible: window.isVisible(),
                menuVisible:
                  process.platform !== "darwin" && window.isMenuBarVisible(),
              };
            });
          assert.equal((await windowState()).menuVisible, false);
          await app.evaluate(({ BrowserWindow }) =>
            BrowserWindow.getAllWindows()
              .find((window) =>
                window.webContents.getURL().includes("view=chat"),
              )
              .minimize(),
          );
          await expect
            .poll(async () => (await windowState()).minimized)
            .toBe(true);
          await companion.evaluate(() =>
            window.desktop.action({ type: "chat.open" }),
          );
          await expect
            .poll(async () => (await windowState()).minimized)
            .toBe(false);
          await expect
            .poll(async () => (await windowState()).visible)
            .toBe(true);
          await companion.evaluate(() =>
            window.desktop.action({ type: "chat.close" }),
          );
          await expect
            .poll(async () => (await windowState()).visible)
            .toBe(false);
        },
      );

      await t.test(
        "right-click opens native controls without starting a drag or opening chat",
        async () => {
          await app.evaluate(({ Menu }) => {
            globalThis.originalMenuPopup = Menu.prototype.popup;
            Menu.prototype.popup = function (options) {
              globalThis.characterMenu = this;
              globalThis.characterMenuWindow = options.window.id;
            };
          });
          const openMenu = async () => {
            await app.evaluate(() => {
              globalThis.characterMenu = undefined;
            });
            const rect = await companion.locator("canvas").boundingBox();
            await companion.mouse.click(
              rect.x + rect.width / 2,
              rect.y + rect.height * 0.35,
              { button: "right" },
            );
            await expect
              .poll(() => app.evaluate(() => Boolean(globalThis.characterMenu)))
              .toBe(true);
          };
          const choose = (id) =>
            app.evaluate((_, id) => {
              const item = globalThis.characterMenu.getMenuItemById(id);
              if (!item) throw new Error(`Missing menu item: ${id}`);
              item.click(item, undefined, {});
            }, id);
          try {
            await openMenu();
            const menu = await app.evaluate(({ BrowserWindow }) => ({
              labels: globalThis.characterMenu.items.map((item) => item.label),
              animations:
                globalThis.characterMenu.getMenuItemById("animations").submenu
                  .items.length,
              unreadEnabled:
                globalThis.characterMenu.getMenuItemById("unread").enabled,
              chatVisible: BrowserWindow.getAllWindows()
                .find((window) =>
                  window.webContents.getURL().includes("view=chat"),
                )
                .isVisible(),
              companionId: BrowserWindow.getAllWindows().find((window) =>
                window.webContents.getURL().includes("view=companion"),
              ).id,
              popupWindow: globalThis.characterMenuWindow,
            }));
            assert.ok(menu.labels.includes("和 Remy 聊聊"));
            assert.ok(menu.labels.includes("播放动作"));
            assert.ok(menu.labels.includes("设置…"));
            assert.equal(menu.animations, 12);
            assert.equal(menu.unreadEnabled, false);
            assert.equal(menu.chatVisible, false);
            assert.equal(menu.popupWindow, menu.companionId);
            await expect(companion.locator(".character")).not.toHaveClass(
              /dragging/,
            );

            await choose("animation:petting_fish");
            await expect(companion.locator("canvas")).toHaveAttribute(
              "data-animation",
              "petting_fish",
            );
            await expect
              .poll(() =>
                companion.locator("canvas").evaluate((canvas) => {
                  const pixels = canvas
                    .getContext("2d")
                    .getImageData(0, 0, canvas.width, canvas.height).data;
                  let orange = 0;
                  for (let i = 0; i < pixels.length; i += 4)
                    if (
                      pixels[i] > 180 &&
                      pixels[i + 1] > 60 &&
                      pixels[i + 1] < 190 &&
                      pixels[i + 2] < 100 &&
                      pixels[i + 3] > 150
                    )
                      orange++;
                  return orange;
                }),
              )
              .toBeGreaterThan(800);
            await companion.locator("canvas").screenshot({
              path: path.join(project, "reports/petting-fish.png"),
            });
            await expect(companion.locator("canvas")).toHaveAttribute(
              "data-animation",
              "idle",
            );

            await choose("animation:remembering");
            await expect(companion.locator("canvas")).toHaveAttribute(
              "data-animation",
              "remembering",
            );
            await expect(companion.locator("canvas")).toHaveAttribute(
              "data-animation",
              "idle",
              { timeout: 10_000 },
            );

            await choose("quiet");
            await expect
              .poll(() =>
                companion.evaluate(() =>
                  window.desktop
                    .bootstrap()
                    .then((value) => value.state.settings.quietUntil),
                ),
              )
              .toBeGreaterThan(Date.now());
            await openMenu();
            assert.equal(
              await app.evaluate(
                () => globalThis.characterMenu.getMenuItemById("quiet").label,
              ),
              "恢复主动提醒",
            );
            await choose("quiet");
            await expect
              .poll(() =>
                companion.evaluate(() =>
                  window.desktop
                    .bootstrap()
                    .then((value) => value.state.settings.quietUntil),
                ),
              )
              .toBe(0);

            await choose("scale:1.25");
            await expect
              .poll(() =>
                companion.evaluate(() =>
                  window.desktop
                    .bootstrap()
                    .then((value) => value.state.settings.scale),
                ),
              )
              .toBe(1.25);
            await choose("scale:1");
            await expect
              .poll(() =>
                companion.evaluate(() =>
                  window.desktop
                    .bootstrap()
                    .then((value) => value.state.settings.scale),
                ),
              )
              .toBe(1);

            await companion.evaluate(() =>
              window.desktop.action({ type: "settings.update", scale: 0.9 }),
            );
            await openMenu();
            assert.deepEqual(
              await app.evaluate(() =>
                globalThis.characterMenu
                  .getMenuItemById("scale")
                  .submenu.items.filter((item) => item.checked)
                  .map((item) => item.label),
              ),
              ["90%"],
            );
            await choose("scale:1");
            await expect
              .poll(() =>
                companion.evaluate(() =>
                  window.desktop
                    .bootstrap()
                    .then((value) => value.state.settings.scale),
                ),
              )
              .toBe(1);

            await choose("settings");
            await expect(
              chat.getByRole("region", { name: "桌面设置" }),
            ).toBeVisible();
            await chat.getByRole("tab", { name: "Remy", exact: true }).click();
            await chat.evaluate(() =>
              window.desktop.action({ type: "chat.close" }),
            );
            await expect
              .poll(() =>
                companion.evaluate(() =>
                  window.desktop
                    .bootstrap()
                    .then((value) => value.state.chatVisible),
                ),
              )
              .toBe(false);

            await choose("visibility");
            assert.equal(
              await app.evaluate(({ BrowserWindow }) =>
                BrowserWindow.getAllWindows()
                  .find((window) =>
                    window.webContents.getURL().includes("view=companion"),
                  )
                  .isVisible(),
              ),
              false,
            );
            await choose("visibility");
            assert.equal(
              await app.evaluate(({ BrowserWindow }) =>
                BrowserWindow.getAllWindows()
                  .find((window) =>
                    window.webContents.getURL().includes("view=companion"),
                  )
                  .isVisible(),
              ),
              true,
            );

            await expect
              .poll(() =>
                companion.evaluate(() =>
                  window.desktop
                    .bootstrap()
                    .then((value) => value.state.chatVisible),
                ),
              )
              .toBe(false);
            const rect = await companion.locator("canvas").boundingBox();
            await companion.mouse.click(rect.x + 1, rect.y + 1);
            assert.equal(
              await companion.evaluate(() =>
                window.desktop
                  .bootstrap()
                  .then((value) => value.state.chatVisible),
              ),
              false,
            );
          } finally {
            await app.evaluate(({ Menu }) => {
              Menu.prototype.popup = globalThis.originalMenuPopup;
            });
          }
        },
      );

      await t.test(
        "drag messages move the native window using a mocked OS cursor",
        async () => {
          const before = await app.evaluate(({ BrowserWindow }) =>
            BrowserWindow.getAllWindows()
              .find((window) =>
                window.webContents.getURL().includes("view=companion"),
              )
              .getBounds(),
          );
          await app.evaluate(({ screen }) => {
            globalThis.originalCursor = screen.getCursorScreenPoint;
            globalThis.testCursor = { x: 300, y: 300 };
            screen.getCursorScreenPoint = () => globalThis.testCursor;
          });
          const rect = await companion.locator("canvas").boundingBox();
          await companion.mouse.move(
            rect.x + rect.width / 2,
            rect.y + rect.height * 0.35,
          );
          await companion.mouse.down();
          await app.evaluate(() => {
            globalThis.testCursor = { x: 260, y: 270 };
          });
          await companion.mouse.move(
            rect.x + rect.width / 2 - 40,
            rect.y + rect.height * 0.35 - 30,
          );
          await companion.mouse.up();
          const after = await app.evaluate(({ BrowserWindow, screen }) => {
            screen.getCursorScreenPoint = globalThis.originalCursor;
            return BrowserWindow.getAllWindows()
              .find((window) =>
                window.webContents.getURL().includes("view=companion"),
              )
              .getBounds();
          });
          assert.equal(after.x, before.x - 40);
          assert.equal(after.y, before.y - 30);
          assert.equal(
            await app.evaluate(({ BrowserWindow }) =>
              BrowserWindow.getAllWindows()
                .find((window) =>
                  window.webContents.getURL().includes("view=chat"),
                )
                .isVisible(),
            ),
            false,
          );
          await app.evaluate(() => {
            const item = globalThis.characterMenu.getMenuItemById("reset");
            item.click(item, undefined, {});
          });
          const reset = await app.evaluate(({ BrowserWindow, screen }) => {
            const bounds = BrowserWindow.getAllWindows()
              .find((window) =>
                window.webContents.getURL().includes("view=companion"),
              )
              .getBounds();
            return { bounds, area: screen.getDisplayMatching(bounds).workArea };
          });
          const layout = await companion.evaluate(() =>
            window.desktop.bootstrap().then((value) => value.layout),
          );
          assert.equal(
            reset.bounds.x + layout.body.left + layout.body.size / 2,
            reset.area.x + reset.area.width - 110,
          );
          assert.equal(
            reset.bounds.y + layout.body.top + layout.body.size,
            reset.area.y + reset.area.height - 24,
          );
        },
      );

      await t.test(
        "a small click movement opens chat without moving the character",
        async () => {
          const before = await app.evaluate(({ BrowserWindow, screen }) => {
            globalThis.originalCursor = screen.getCursorScreenPoint;
            globalThis.testCursor = { x: 300, y: 300 };
            screen.getCursorScreenPoint = () => globalThis.testCursor;
            return BrowserWindow.getAllWindows()
              .find((window) =>
                window.webContents.getURL().includes("view=companion"),
              )
              .getBounds();
          });
          try {
            const rect = await companion.locator("canvas").boundingBox();
            await companion.mouse.move(
              rect.x + rect.width / 2,
              rect.y + rect.height * 0.35,
            );
            await companion.mouse.down();
            await app.evaluate(() => {
              globalThis.testCursor = { x: 303, y: 302 };
            });
            await companion.mouse.move(
              rect.x + rect.width / 2 + 3,
              rect.y + rect.height * 0.35 + 2,
            );
            await companion.mouse.up();
            await expect
              .poll(() =>
                companion.evaluate(() =>
                  window.desktop
                    .bootstrap()
                    .then((value) => value.state.chatVisible),
                ),
              )
              .toBe(true);
            assert.deepEqual(
              await app.evaluate(({ BrowserWindow }) =>
                BrowserWindow.getAllWindows()
                  .find((window) =>
                    window.webContents.getURL().includes("view=companion"),
                  )
                  .getBounds(),
              ),
              before,
            );
            await chat.evaluate(() =>
              window.desktop.action({ type: "chat.close" }),
            );
            await expect
              .poll(() =>
                companion.evaluate(() =>
                  window.desktop
                    .bootstrap()
                    .then((value) => value.state.chatVisible),
                ),
              )
              .toBe(false);
          } finally {
            await app.evaluate(({ screen }) => {
              screen.getCursorScreenPoint = globalThis.originalCursor;
            });
          }
        },
      );

      await t.test(
        "clicking the character always raises chat, including activation during a press",
        async () => {
          const windowState = () =>
            app.evaluate(({ BrowserWindow }) => {
              const window = BrowserWindow.getAllWindows().find((entry) =>
                entry.webContents.getURL().includes("view=chat"),
              );
              return {
                visible: window.isVisible(),
                minimized: window.isMinimized(),
                focused: window.isFocused(),
              };
            });
          const clickCharacter = async (activateOnPress = false) => {
            const canvas = companion.locator(".character canvas");
            await expect(canvas).toBeVisible();
            const rect = await canvas.boundingBox();
            await companion.mouse.move(
              rect.x + rect.width / 2,
              rect.y + rect.height * 0.35,
            );
            await companion.mouse.down();
            if (activateOnPress)
              await app.evaluate(({ app }) => app.emit("activate"));
            await companion.mouse.up();
          };
          const expectVisible = async (visible) => {
            await expect
              .poll(async () => (await windowState()).visible)
              .toBe(visible);
            await expect
              .poll(() =>
                companion.evaluate(() =>
                  window.desktop
                    .bootstrap()
                    .then(({ state }) => state.chatVisible),
                ),
              )
              .toBe(visible);
            if (visible)
              await expect
                .poll(async () => (await windowState()).focused)
                .toBe(true);
          };

          await clickCharacter();
          await expectVisible(true);
          await clickCharacter();
          await expectVisible(true);
          await clickCharacter();
          await expectVisible(true);

          await app.evaluate(({ BrowserWindow }) =>
            BrowserWindow.getAllWindows()
              .find((window) =>
                window.webContents.getURL().includes("view=chat"),
              )
              .blur(),
          );
          await expect
            .poll(async () => (await windowState()).focused)
            .toBe(false);
          await companion
            .locator(".character")
            .evaluate((button) => button.click());
          await expectVisible(true);

          // OS activation may raise chat before the mouse-up reaches the character.
          await app.evaluate(({ BrowserWindow }) =>
            BrowserWindow.getAllWindows()
              .find((window) =>
                window.webContents.getURL().includes("view=chat"),
              )
              .blur(),
          );
          await expect
            .poll(async () => (await windowState()).focused)
            .toBe(false);
          await clickCharacter(true);
          await expectVisible(true);

          await chat.evaluate(() =>
            window.desktop.action({ type: "window.minimize" }),
          );
          await expect
            .poll(async () => (await windowState()).minimized)
            .toBe(true);
          await clickCharacter();
          await expectVisible(true);
          assert.equal((await windowState()).minimized, false);

          // Keyboard and assistive activation have no pointer gesture.
          await chat.evaluate(() =>
            window.desktop.action({ type: "chat.close" }),
          );
          await expectVisible(false);
          await companion
            .locator(".character")
            .evaluate((button) => button.click());
          await expectVisible(true);
          // Opening a message or using another open entry point must not toggle.
          await companion.evaluate(() =>
            window.desktop.action({ type: "chat.open" }),
          );
          await expectVisible(true);
          await clickCharacter();
          await expectVisible(true);
          await chat.evaluate(() =>
            window.desktop.action({ type: "chat.close" }),
          );
          await expectVisible(false);
        },
      );

      await t.test(
        "completed proactive upserts show once, dismissing preserves unread",
        async () => {
          backend.push("new-reminder", "未完成内容", false);
          await new Promise((resolve) => setTimeout(resolve, 150));
          await expect(
            companion.getByRole("region", { name: "主动提醒" }),
          ).toHaveCount(0);
          backend.push("new-reminder");
          backend.push("new-reminder");
          await expect(
            companion.getByRole("region", { name: "主动提醒" }),
          ).toBeVisible();
          await expect(
            companion.getByRole("button", { name: "1 条未读提醒" }),
          ).toBeVisible();
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-segment",
            "hold",
          );
          assert.equal(
            await companion.evaluate(() =>
              window.desktop
                .bootstrap()
                .then(({ state }) => state.behavior.animation.mode),
            ),
            "hold",
          );
          await mkdir(path.join(project, "reports"), { recursive: true });
          await companion.screenshot({
            path: path.join(project, "reports/companion.png"),
            omitBackground: true,
          });
          await companion
            .getByRole("button", { name: "收起气泡", exact: true })
            .click();
          await expect(
            companion.getByRole("region", { name: "主动提醒" }),
          ).toHaveCount(0);
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-segment",
            "exit",
          );
          await expect(
            companion.getByRole("button", { name: "1 条未读提醒" }),
          ).toBeVisible();
        },
      );

      await t.test(
        "opening chat reads visible reminders and streams one reconciled answer",
        async () => {
          const rect = await companion.locator("canvas").boundingBox();
          await companion.mouse.click(
            rect.x + rect.width / 2,
            rect.y + rect.height * 0.35,
          );
          await expect(
            chat.getByRole("textbox", { name: "消息", exact: true }),
          ).toBeVisible();
          await expect
            .poll(() =>
              app.evaluate(({ BrowserWindow }) =>
                BrowserWindow.getAllWindows()
                  .find((window) =>
                    window.webContents.getURL().includes("view=chat"),
                  )
                  .isVisible(),
              ),
            )
            .toBe(true);
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "happy",
          );
          await app.evaluate(({ app, BrowserWindow }) => {
            app.focus({ steal: true });
            BrowserWindow.getAllWindows()
              .find((window) =>
                window.webContents.getURL().includes("view=chat"),
              )
              .focus();
          });
          await expect
            .poll(() => chat.evaluate(() => document.hasFocus()))
            .toBe(true);
          await expect
            .poll(() =>
              companion.evaluate(() =>
                window.desktop.bootstrap().then((value) => value.state.unread),
              ),
            )
            .toBe(0);
          await chat
            .getByRole("textbox", { name: "消息", exact: true })
            .fill("你好 Remy");
          await chat
            .getByRole("button", { name: "发送消息", exact: true })
            .click();
          await expect(
            chat.getByText("我记住了。我们可以继续聊聊你的研究。", {
              exact: true,
            }),
          ).toHaveCount(1);
          await expect
            .poll(() =>
              chat.evaluate(() =>
                window.desktop.bootstrap().then((value) => value.state.pending),
              ),
            )
            .toBe(null);
          await expect(
            chat.locator(".message.user .message-body").last(),
          ).toHaveText("你好 Remy");
          await expect(chat.getByText("旧的增量", { exact: true })).toHaveCount(
            0,
          );
          await chat.screenshot({
            path: path.join(project, "reports/chat.png"),
          });
        },
      );

      await t.test(
        "an explicit stop cancels generation, closing the panel keeps subscription alive",
        async () => {
          await chat
            .getByRole("textbox", { name: "消息", exact: true })
            .fill("请等待我停止");
          await chat
            .getByRole("button", { name: "发送消息", exact: true })
            .click();
          await chat
            .getByRole("button", { name: "停止回复", exact: true })
            .click();
          await expect(
            chat.locator(".message.assistant .message-status").last(),
          ).toHaveText("回复已停止");
          await chat.evaluate(() =>
            window.desktop.action({ type: "chat.close" }),
          );
          backend.disconnect();
          backend.push("missed-during-reconnect", "重连后补齐的消息");
          await expect
            .poll(
              () =>
                chat.evaluate(() =>
                  window.desktop
                    .bootstrap()
                    .then(
                      (value) =>
                        value.state.messages.filter(
                          (entry) => entry.id === "missed-during-reconnect",
                        ).length,
                    ),
                ),
              { timeout: 15_000 },
            )
            .toBe(1);
          assert.ok(
            backend.cursors.some(
              (cursor) =>
                typeof cursor === "string" && cursor.startsWith("test-epoch:"),
            ),
          );
          await expect
            .poll(() =>
              app.evaluate(({ BrowserWindow }) =>
                BrowserWindow.getAllWindows()
                  .find((window) =>
                    window.webContents.getURL().includes("view=chat"),
                  )
                  .isVisible(),
              ),
            )
            .toBe(false);
        },
      );

      await t.test(
        "remote agent phases drive animations and dragging restores the latest phase",
        async () => {
          await chat.evaluate(() =>
            window.desktop.action({ type: "chat.close" }),
          );
          for (const [phase, clip] of [
            ["thinking", "thinking"],
            ["answering", "speaking"],
            ["working", "thinking"],
            ["remembering", "remembering"],
            ["waiting", "listening"],
          ]) {
            backend.activity(phase);
            await expect(companion.locator("canvas")).toHaveAttribute(
              "data-animation",
              clip,
            );
          }
          backend.activity("answering");
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "speaking",
          );
          await companion.evaluate(() => {
            window.desktop.drag("start");
            window.desktop.drag("move");
          });
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "dragged",
          );
          backend.activity("remembering");
          await companion.evaluate(() => window.desktop.drag("end"));
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "remembering",
          );
          backend.activity("completed");
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "happy",
          );
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "idle",
          );
          backend.activity("completed");
          await expect
            .poll(() =>
              companion.evaluate(() =>
                window.desktop
                  .bootstrap()
                  .then(({ state }) => state.behavior.animation.clip),
              ),
            )
            .toBe("idle");
        },
      );

      await t.test(
        "concurrent task animations yield to foreground work and resume the remaining task",
        async () => {
          backend.taskActivity("remembering", "research", "one");
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "remembering",
          );
          backend.taskActivity("working", "watch", "two");
          backend.taskActivity("completed", "research", "one");
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "thinking",
          );
          await expect
            .poll(() =>
              companion.evaluate(() =>
                window.desktop
                  .bootstrap()
                  .then(({ state }) => state.behavior.label),
              ),
            )
            .toContain("watch");
          backend.activity("answering", "chat", "foreground-priority");
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "speaking",
          );
          backend.activity("interrupted", "chat", "foreground-priority");
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "thinking",
          );
          backend.taskActivity("completed", "watch", "two");
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "idle",
          );
        },
      );

      await t.test(
        "a new PA session clears the previous conversation",
        async () => {
          await companion.evaluate(() =>
            window.desktop.action({ type: "chat.open" }),
          );
          await chat
            .getByRole("button", { name: "新建对话", exact: true })
            .click();
          await expect
            .poll(() =>
              chat.evaluate(() =>
                window.desktop
                  .bootstrap()
                  .then((value) => value.state.sessionId),
              ),
            )
            .toBe("session-2");
          await expect(
            chat.locator(".empty-chat").getByRole("heading", { name: "Remy" }),
          ).toBeVisible();
          assert.deepEqual(errors, []);
          assert.deepEqual(logs, []);
        },
      );
    } finally {
      await app?.close();
      await backend.close();
      const receipts = JSON.parse(
        await readFile(
          path.join(directory, "desktop/presentation.json"),
          "utf8",
        ),
      );
      assert.ok(
        Object.values(receipts).some(
          (entry) => entry.receipts["new-reminder"]?.read,
        ),
      );
      await rm(directory, { recursive: true, force: true });
    }
  },
);
