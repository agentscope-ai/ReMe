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
  "stale and foreign one-way IPC is ignored through renderer reload and shutdown",
  { timeout: 30_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-ipc-e2e-"));
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
      const { companion } = await desktopWindows(app);
      await companion.waitForSelector("canvas");
      const check = () =>
        app.evaluate(({ BrowserWindow, ipcMain }) => {
          const pet = BrowserWindow.getAllWindows().find((win) =>
            win.webContents.getURL().includes("view=companion"),
          );
          const chat = BrowserWindow.getAllWindows().find((win) =>
            win.webContents.getURL().includes("view=chat"),
          );
          const other = new BrowserWindow({
            show: false,
            webPreferences: { sandbox: true },
          });
          const foreign = other.webContents;
          const events = [
            { sender: pet.webContents, senderFrame: null },
            { sender: pet.webContents, senderFrame: { detached: true } },
            { sender: foreign, senderFrame: foreign.mainFrame },
            {
              sender: chat.webContents,
              senderFrame: chat.webContents.mainFrame,
            },
          ];
          const before = pet.getBounds();
          const send = (event) => {
            ipcMain.emit("reme:pointer", event, true);
            ipcMain.emit("reme:surface-size", event, {
              key: "stale",
              width: 100,
              height: 50,
            });
            ipcMain.emit("reme:drag", event, "start");
            ipcMain.emit("reme:drag", event, "move");
          };
          events.forEach(send);
          other.destroy();
          send({ sender: foreign, senderFrame: null });
          return { before, after: pet.getBounds() };
        });
      let bounds = await check();
      assert.deepEqual(bounds.before, bounds.after);
      await companion.reload();
      await expect(companion.locator("canvas")).toBeVisible();
      bounds = await check();
      assert.deepEqual(bounds.before, bounds.after);
      await expect
        .poll(() =>
          companion.evaluate(() =>
            window.desktop.bootstrap().then(({ state }) => state.connection),
          ),
        )
        .toBe("connected");
      await app.close();
      app = undefined;
    } finally {
      if (app) await app.close().catch(() => {});
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
