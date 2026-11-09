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
  "one companion moves between the desktop and free workspace space",
  { timeout: 60_000 },
  async (t) => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-habitat-e2e-"));
    const backend = await mockReMe();
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
      const errors = [];
      for (const page of [companion, chat])
        page.on("pageerror", (error) => errors.push(String(error)));
      await expect
        .poll(() =>
          companion.evaluate(() =>
            window.desktop.bootstrap().then(({ state }) => state.connection),
          ),
        )
        .toBe("connected");
      const presence = () =>
        companion.evaluate(() =>
          window.desktop.bootstrap().then(({ layout }) => layout.presence),
        );
      const setBounds = (bounds) =>
        app.evaluate(({ BrowserWindow }, bounds) => {
          const win = BrowserWindow.getAllWindows().find((win) =>
            win.webContents.getURL().includes("view=chat"),
          );
          if (win.isMinimized()) win.restore();
          win.setBounds(bounds);
          win.show();
          win.focus();
        }, bounds);
      const displays = await app.evaluate(({ screen }) =>
        screen.getAllDisplays().map(({ id, workArea, scaleFactor }) => ({
          id,
          workArea,
          scaleFactor,
        })),
      );
      const wide = displays.find((display) => display.workArea.width >= 1400);
      if (!wide) {
        t.skip(
          "needs a display wide enough for the natural content rail and pet",
        );
        return;
      }
      const openWide = async (display = wide) => {
        await chat.evaluate(() => window.desktop.action({ type: "chat.open" }));
        await setBounds(display.workArea);
        const toggle = chat.getByRole("button", {
          name: "隐藏侧栏",
          exact: true,
        });
        if (await toggle.count()) await toggle.click();
      };

      await t.test(
        "an empty conversation admits a walking companion without covering its welcome or composer",
        async () => {
          const error = await companion.evaluate(() =>
            window.desktop.action({ type: "session.new" }),
          );
          assert.equal(error, null);
          await expect
            .poll(() =>
              chat.evaluate(() =>
                window.desktop
                  .bootstrap()
                  .then(({ state }) => state.messages.length),
              ),
            )
            .toBe(0);
          await openWide();
          await expect(chat.locator(".empty-chat")).toBeVisible();
          await expect(chat.locator(".message")).toHaveCount(0);
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "walking",
          );
          await expect.poll(async () => (await presence()).host).toBe("app");
          await expect(
            chat.locator(".embedded-companion canvas"),
          ).toBeVisible();
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
          const safe = await chat.evaluate(() => {
            const body = document
              .querySelector(".embedded-companion canvas")
              .getBoundingClientRect();
            const form = document
              .querySelector(".chat-composer form")
              .getBoundingClientRect();
            const history = document.querySelector(".chat-history");
            const edge =
              history.getBoundingClientRect().right -
              parseFloat(getComputedStyle(history).paddingRight);
            return (
              body.left >= Math.max(form.right, edge) + 12 &&
              body.right < innerWidth &&
              body.bottom < innerHeight
            );
          });
          assert.equal(safe, true);
          await mkdir(path.join(project, "reports"), { recursive: true });
          await chat.screenshot({
            path: path.join(project, "reports/companion-empty-chat.png"),
          });
          backend.activity("remembering", "chat", "habitat-active");
          await expect(
            chat.locator(".embedded-companion canvas"),
          ).toHaveAttribute("data-animation", "remembering");
          backend.activity("interrupted", "chat", "habitat-active");
        },
      );

      await t.test(
        "sending a message and starting a new empty conversation keeps one companion in the app",
        async () => {
          await chat
            .getByRole("textbox", { name: "消息", exact: true })
            .fill("验证萌宠在新建空对话中的位置");
          await chat
            .getByRole("textbox", { name: "消息", exact: true })
            .press("Enter");
          await expect(chat.locator(".message.assistant")).toContainText(
            "我记住了",
          );
          await expect
            .poll(() =>
              chat.evaluate(() =>
                window.desktop
                  .bootstrap()
                  .then(({ state }) => Boolean(state.pending?.streaming)),
              ),
            )
            .toBe(false);
          await expect.poll(async () => (await presence()).host).toBe("app");
          await chat
            .getByRole("button", { name: "新建对话", exact: true })
            .click();
          await expect(chat.locator(".empty-chat")).toBeVisible();
          await expect(chat.locator(".message")).toHaveCount(0);
          await expect.poll(async () => (await presence()).host).toBe("app");
          await expect(
            chat.locator(".embedded-companion canvas"),
          ).toBeVisible();
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
          await chat.screenshot({
            path: path.join(project, "reports/companion-new-empty-chat.png"),
          });
        },
      );

      await t.test(
        "a delayed hide event cannot hide an already reopened workspace",
        async () => {
          await app.evaluate(({ BrowserWindow }) =>
            BrowserWindow.getAllWindows()
              .find((win) => win.webContents.getURL().includes("view=chat"))
              .emit("hide"),
          );
          const state = await chat.evaluate(() =>
            window.desktop.bootstrap().then(({ state }) => state),
          );
          assert.equal(state.chatVisible, true);
          await expect.poll(async () => (await presence()).host).toBe("app");
        },
      );

      await t.test(
        "clicking the embedded character keeps the workspace in front",
        async () => {
          const canvas = chat.locator(".embedded-companion canvas");
          await expect(canvas).toBeVisible();
          const rect = await canvas.boundingBox();
          await chat.mouse.click(
            rect.x + rect.width / 2,
            rect.y + rect.height * 0.35,
          );
          await expect
            .poll(() =>
              app.evaluate(({ BrowserWindow }) => {
                const window = BrowserWindow.getAllWindows().find((win) =>
                  win.webContents.getURL().includes("view=chat"),
                );
                return window.isVisible() && window.isFocused();
              }),
            )
            .toBe(true);
          await expect.poll(async () => (await presence()).host).toBe("app");
          await expect(canvas).toBeVisible();
          await expect(companion.locator("canvas")).not.toBeVisible();
        },
      );

      await t.test(
        "a narrow workspace releases the pet, reopening admits it again",
        async () => {
          await setBounds({
            x: wide.workArea.x + 20,
            y: wide.workArea.y + 20,
            width: 900,
            height: 620,
          });
          await expect
            .poll(async () => (await presence()).host)
            .toBe("desktop");
          await expect
            .poll(async () => Boolean((await presence()).motion))
            .toBe(false);
          await expect(companion.locator("canvas")).toBeVisible();
          await expect(chat.locator(".embedded-companion")).not.toBeVisible();
          await openWide();
          await expect.poll(async () => (await presence()).host).toBe("app");
        },
      );

      await t.test(
        "dragging out pins the role until the next window opening",
        async () => {
          await app.evaluate(({ screen }) => {
            globalThis.habitatCursor = screen.getCursorScreenPoint;
            screen.getCursorScreenPoint = () => ({ x: 500, y: 500 });
          });
          await chat.evaluate(() => window.desktop.drag("start"));
          await app.evaluate(({ screen }) => {
            screen.getCursorScreenPoint = () => ({ x: 350, y: 500 });
          });
          await chat.evaluate(() => {
            window.desktop.drag("move");
            window.desktop.drag("end");
          });
          await expect
            .poll(async () => (await presence()).host)
            .toBe("desktop");
          await expect
            .poll(async () => Boolean((await presence()).motion))
            .toBe(false);
          await new Promise((resolve) => setTimeout(resolve, 600));
          assert.equal((await presence()).host, "desktop");
          await app.evaluate(({ screen }) => {
            screen.getCursorScreenPoint = globalThis.habitatCursor;
          });
          await chat.evaluate(() =>
            window.desktop.action({ type: "chat.close" }),
          );
          await openWide();
          await expect.poll(async () => (await presence()).host).toBe("app");
        },
      );

      await t.test(
        "hide/show applies to both hosts and minimizing returns to the desktop",
        async () => {
          await chat.evaluate(() =>
            window.desktop.action({
              type: "companion.visibility",
              visible: false,
            }),
          );
          await expect(chat.locator(".embedded-companion")).not.toBeVisible();
          assert.equal((await presence()).visible, false);
          await chat.evaluate(() =>
            window.desktop.action({
              type: "companion.visibility",
              visible: true,
            }),
          );
          await expect.poll(async () => (await presence()).host).toBe("app");
          await chat.evaluate(() =>
            window.desktop.action({ type: "window.minimize" }),
          );
          await expect
            .poll(async () => (await presence()).host)
            .toBe("desktop");
          await expect
            .poll(async () => Boolean((await presence()).motion))
            .toBe(false);
        },
      );

      await t.test(
        "real pointer capture survives the handoff while dragging out",
        async () => {
          await openWide();
          await expect.poll(async () => (await presence()).host).toBe("app");
          const body = await chat
            .locator(".embedded-companion canvas")
            .boundingBox();
          const start = await app.evaluate(({ BrowserWindow }) => {
            const win = BrowserWindow.getAllWindows().find((win) =>
              win.webContents.getURL().includes("view=chat"),
            );
            return win.webContents
              .executeJavaScript("window.desktop.bootstrap()")
              .then(
                ({ layout }) =>
                  win.getContentBounds().x +
                  layout.body.left +
                  layout.body.size / 2,
              );
          });
          await app.evaluate(({ screen }) => {
            globalThis.habitatCursor = screen.getCursorScreenPoint;
            screen.getCursorScreenPoint = () => ({ x: 500, y: 500 });
          });
          try {
            await chat.mouse.move(
              body.x + body.width / 2,
              body.y + body.height * 0.35,
            );
            await chat.mouse.down();
            await app.evaluate(({ screen }) => {
              screen.getCursorScreenPoint = () => ({ x: 400, y: 500 });
            });
            await chat.mouse.move(
              body.x + body.width / 2 - 100,
              body.y + body.height * 0.35,
            );
            await expect
              .poll(async () => (await presence()).host)
              .toBe("desktop");
            await app.evaluate(({ screen }) => {
              screen.getCursorScreenPoint = () => ({ x: 280, y: 500 });
            });
            await chat.mouse.move(
              body.x + body.width / 2 - 220,
              body.y + body.height * 0.35,
            );
            await chat.mouse.up();
            const end = await app.evaluate(({ BrowserWindow }) => {
              const win = BrowserWindow.getAllWindows().find((win) =>
                win.webContents.getURL().includes("view=companion"),
              );
              return win.webContents
                .executeJavaScript("window.desktop.bootstrap()")
                .then(
                  ({ layout }) =>
                    win.getBounds().x + layout.body.left + layout.body.size / 2,
                );
            });
            assert.ok(Math.abs(end - (start - 220)) <= 1, `${start} -> ${end}`);
            await expect(companion.locator("canvas")).not.toHaveAttribute(
              "data-animation",
              "dragged",
            );
          } finally {
            await chat.mouse.up();
            await app.evaluate(({ screen }) => {
              screen.getCursorScreenPoint = globalThis.habitatCursor;
            });
          }
          await app.evaluate(({ BrowserWindow }) =>
            BrowserWindow.getAllWindows()
              .find((win) => win.webContents.getURL().includes("view=chat"))
              .close(),
          );
        },
      );

      const second = displays.find(
        (display) => display.id !== wide.id && display.workArea.width >= 1400,
      );
      if (second)
        await t.test(
          "a real second display receives the pet and keeps it after closing",
          async () => {
            await openWide(second);
            await expect
              .poll(async () => (await presence()).host, { timeout: 10_000 })
              .toBe("app");
            await chat.screenshot({
              path: path.join(project, "reports/companion-second-display.png"),
            });
            await chat.evaluate(() =>
              window.desktop.action({ type: "chat.close" }),
            );
            await expect
              .poll(async () => Boolean((await presence()).motion))
              .toBe(false);
            const landed = await app.evaluate(({ BrowserWindow, screen }) => {
              const win = BrowserWindow.getAllWindows().find((win) =>
                win.webContents.getURL().includes("view=companion"),
              );
              return screen.getDisplayMatching(win.getBounds()).id;
            });
            assert.equal(landed, second.id);
          },
        );
      assert.deepEqual(errors, []);
    } finally {
      await app.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
