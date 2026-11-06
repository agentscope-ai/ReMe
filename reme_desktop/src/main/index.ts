import {
  app,
  dialog,
  globalShortcut,
  ipcMain,
  nativeImage,
  protocol,
  shell,
  Tray,
} from "electron";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DesktopController } from "./controller.ts";
import { loadAssets } from "./assets.ts";
import { Storage, normalizeEndpoint } from "./storage.ts";
import { DesktopWindows } from "./windows.ts";
import { desktopMenu } from "./menu.ts";
import { channels } from "../shared/protocol.ts";
import type { DesktopAction } from "../shared/protocol.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
app.setName("ReMe Desktop");
if (process.env.REME_DESKTOP_DATA_DIR)
  app.setPath("userData", path.resolve(process.env.REME_DESKTOP_DATA_DIR));
protocol.registerSchemesAsPrivileged([
  {
    scheme: "reme-assets",
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
    },
  },
]);

if (!app.requestSingleInstanceLock()) app.quit();
else {
  let windows: DesktopWindows | undefined;
  let controller: DesktopController | undefined;
  let tray: Tray | undefined;
  let refreshTray = () => {};
  let openMenu = () => {};
  let quitting = false;
  app.on("second-instance", () => windows?.openChat());
  app.on("activate", () => windows?.openChat());
  app.on("window-all-closed", () => {});
  app.on("before-quit", (event) => {
    if (quitting || !controller) return;
    event.preventDefault();
    quitting = true;
    globalShortcut.unregisterAll();
    void controller
      .close()
      .catch((error) => console.error(error))
      .finally(() => {
        windows?.destroy();
        tray?.destroy();
        app.quit();
      });
  });

  void app
    .whenReady()
    .then(async () => {
      const directory = app.isPackaged
        ? path.join(process.resourcesPath, "remy")
        : path.resolve(here, "../../../assets/remy/anime/animations/v1");
      const assets = await loadAssets(directory);
      protocol.handle("reme-assets", async (request) => {
        const url = new URL(request.url);
        const file =
          url.hostname === "remy" ? assets.files.get(url.pathname) : undefined;
        if (!file) return new Response("Not found", { status: 404 });
        try {
          return new Response(new Uint8Array(await readFile(file)), {
            headers: {
              "Content-Type": "image/png",
              "Access-Control-Allow-Origin": "*",
              "Cache-Control": "public, max-age=3600",
            },
          });
        } catch {
          return new Response("Not found", { status: 404 });
        }
      });
      const storage = new Storage(
        path.join(app.getPath("userData"), "desktop"),
      );
      await storage.load();
      if (process.env.REME_DESKTOP_API_URL)
        storage.settings.endpoint = normalizeEndpoint(
          process.env.REME_DESKTOP_API_URL,
        );
      controller = new DesktopController(
        storage,
        (state) => {
          windows?.update(state);
          refreshTray();
        },
        assets.manifest,
      );
      windows = new DesktopWindows(
        storage,
        {
          preload: path.join(here, "../preload/index.cjs"),
          renderer: path.join(here, "../renderer/index.html"),
        },
        (visible) => controller!.setChatVisible(visible),
        (focused) => controller!.setChatFocused(focused),
      );

      const trusted = (
        event: Electron.IpcMainEvent | Electron.IpcMainInvokeEvent,
      ) => {
        if (
          !windows!.isSender(event.sender.id) ||
          event.senderFrame !== event.sender.mainFrame
        )
          throw new Error("无效的桌面消息来源");
      };
      ipcMain.handle(channels.bootstrap, (event) => {
        trusted(event);
        return {
          state: controller!.state,
          manifest: assets.manifest,
          layout: windows!.layout,
        };
      });
      ipcMain.handle(channels.action, async (event, action: DesktopAction) => {
        trusted(event);
        if (!action || typeof action.type !== "string")
          throw new Error("无效的桌面操作");
        switch (action.type) {
          case "animation.ended":
            if (
              event.sender.id === windows!.companion.webContents.id &&
              Number.isSafeInteger(action.id)
            )
              controller!.animationEnded(action.id);
            return;
          case "interaction":
            controller!.interact();
            return;
          case "companion.menu":
            if (event.sender.id !== windows!.companion.webContents.id)
              throw new Error("请从角色打开菜单");
            openMenu();
            return;
          case "chat.open":
            if (
              event.sender.id === windows!.companion.webContents.id &&
              !action.messageId
            )
              controller!.preview("happy");
            windows!.openChat();
            controller!.setChatVisible(true, action.messageId);
            return;
          case "chat.close":
            windows!.closeChat();
            return;
          case "chat.send":
            if (typeof action.query !== "string")
              throw new Error("消息需要是文本");
            return controller!.send(action.query);
          case "chat.stop":
            controller!.stopReply();
            return;
          case "session.new":
            return controller!.newSession();
          case "messages.read":
            if (
              typeof action.sessionId !== "string" ||
              !Array.isArray(action.ids) ||
              !action.ids.every((id) => typeof id === "string")
            )
              throw new Error("无效的已读消息");
            controller!.markRead(action.sessionId, action.ids);
            return;
          case "bubble.dismiss":
            controller!.bubbleAction(action.id, "dismiss");
            return;
          case "bubble.snooze":
            controller!.bubbleAction(action.id, "snooze");
            return;
          case "bubble.pause":
            controller!.bubbleAction(
              action.id,
              "pause",
              action.paused === true,
            );
            return;
          case "settings.update":
            return controller!.settings({
              ...(action.endpoint !== undefined
                ? { endpoint: action.endpoint }
                : {}),
              ...(action.scale !== undefined ? { scale: action.scale } : {}),
              ...(action.quietUntil !== undefined
                ? { quietUntil: action.quietUntil }
                : {}),
            });
          case "link.open": {
            const url = new URL(action.url);
            if (
              ["http:", "https:"].includes(url.protocol) &&
              !url.username &&
              !url.password
            )
              await shell.openExternal(url.toString());
            return;
          }
          default:
            throw new Error("未知的桌面操作");
        }
      });
      ipcMain.on(channels.pointer, (event, interactive) => {
        trusted(event);
        if (event.sender.id === windows!.companion.webContents.id)
          windows!.pointer(interactive === true);
      });
      ipcMain.on(channels.drag, (event, phase) => {
        trusted(event);
        if (
          event.sender.id === windows!.companion.webContents.id &&
          ["start", "move", "end"].includes(phase)
        ) {
          windows!.drag(phase);
          controller!.drag(phase);
        }
      });

      const characterIcon = nativeImage.createFromPath(
        path.join(directory, "frames/idle/00.png"),
      );
      app.dock?.setIcon(characterIcon);
      const icon = characterIcon.resize({ width: 20, height: 20 });
      tray = new Tray(icon);
      tray.setToolTip("Remy · ReMe 桌面助手");
      const menu = () =>
        desktopMenu(
          controller!,
          windows!,
          assets.manifest,
          () => app.quit(),
          refreshTray,
        );
      let menuState = "";
      refreshTray = () => {
        if (
          quitting ||
          !windows ||
          windows.companion.isDestroyed() ||
          !tray ||
          tray.isDestroyed()
        )
          return;
        const state = controller!.state;
        const next = JSON.stringify([
          state.unread,
          state.settings.scale,
          state.settings.quietUntil > Date.now(),
          state.status,
          Boolean(state.pending?.streaming),
          windows!.companion.isVisible(),
        ]);
        if (next !== menuState) {
          menuState = next;
          tray!.setContextMenu(menu());
        }
      };
      openMenu = () => menu().popup({ window: windows!.companion });
      windows.companion.on("show", () => {
        if (quitting) return;
        controller!.setCompanionVisible(true);
        refreshTray();
      });
      windows.companion.on("hide", () => {
        if (quitting) return;
        controller!.setCompanionVisible(false);
        refreshTray();
      });
      refreshTray();
      tray.on("click", () => windows!.openChat());
      globalShortcut.register("CommandOrControl+Shift+Space", () =>
        windows!.openChat(),
      );
      controller.start();
    })
    .catch((error) => {
      dialog.showErrorBox(
        "ReMe Desktop 启动失败",
        error instanceof Error ? error.message : String(error),
      );
      app.quit();
    });
}
