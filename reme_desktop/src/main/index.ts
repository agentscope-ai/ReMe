import {
  defaultLanguage,
  errorText,
  serializeError,
  translate,
  UiError,
} from "../shared/i18n.ts";
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
import { SystemNotifications } from "./system-notifications.ts";
import { viewState } from "./view-state.ts";
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
  let notifications: SystemNotifications | undefined;
  let refreshTray = () => {};
  let openMenu = () => {};
  let quitting = false;
  let language = defaultLanguage;
  app.on("second-instance", () => windows?.openChat());
  app.on("activate", () => windows?.openChat());
  app.on("window-all-closed", () => {});
  app.on("before-quit", (event) => {
    if (quitting || !controller) return;
    event.preventDefault();
    quitting = true;
    notifications?.close();
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
      try {
        await storage.load();
      } finally {
        language = storage.settings.language;
      }
      if (process.env.REME_DESKTOP_API_URL)
        storage.settings.endpoint = normalizeEndpoint(
          process.env.REME_DESKTOP_API_URL,
        );
      controller = new DesktopController(
        storage,
        (state) => {
          language = state.settings.language;
          windows?.update(state);
          notifications?.update(state);
          refreshTray();
        },
        assets.manifest,
        (notice, shown) => notifications?.show(notice, shown),
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
      notifications = new SystemNotifications((notice) => {
        if (!controller!.canOpenNotice(notice)) return;
        windows!.openChat();
        controller!.setChatVisible(true, notice.messageId);
      });
      windows.companion.webContents.on("did-start-loading", () =>
        controller!.setCompanionReady(false),
      );
      windows.companion.webContents.on("did-finish-load", () =>
        controller!.setCompanionReady(true),
      );
      windows.companion.webContents.on("render-process-gone", () =>
        controller!.setCompanionReady(false),
      );

      const trusted = (
        event: Electron.IpcMainEvent | Electron.IpcMainInvokeEvent,
      ) => {
        if (
          !windows!.isSender(event.sender.id) ||
          event.senderFrame !== event.sender.mainFrame
        )
          throw new UiError("invalidSender");
      };
      ipcMain.handle(channels.bootstrap, (event) => {
        trusted(event);
        return {
          state: viewState(
            controller!.state,
            event.sender.id === windows!.companion.webContents.id,
          ),
          manifest: assets.manifest,
          layout: windows!.layout,
        };
      });
      const dispatch = async (
        event: Electron.IpcMainInvokeEvent,
        action: DesktopAction,
      ): Promise<void> => {
        trusted(event);
        if (!action || typeof action.type !== "string")
          throw new UiError("invalidAction");
        switch (action.type) {
          case "playback.report":
            if (
              event.sender.id === windows!.companion.webContents.id &&
              Number.isSafeInteger(action.id) &&
              ["started", "ended", "failed"].includes(action.status)
            )
              controller!.reportPlayback(action.id, action.status);
            return;
          case "bubble.shown":
            if (
              event.sender.id === windows!.companion.webContents.id &&
              windows!.companion.isVisible() &&
              typeof action.sessionId === "string" &&
              typeof action.id === "string" &&
              Number.isSafeInteger(action.presentationId)
            )
              controller!.bubbleShown(
                action.sessionId,
                action.id,
                action.presentationId,
              );
            return;
          case "interaction":
            controller!.interact();
            return;
          case "companion.menu":
            if (event.sender.id !== windows!.companion.webContents.id)
              throw new UiError("menuFromCharacter");
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
              throw new UiError("textRequired");
            return controller!.send(action.query);
          case "chat.stop":
            controller!.stopReply();
            return;
          case "session.new":
            return controller!.newSession();
          case "messages.read":
            if (
              event.sender.id !== windows!.chat.webContents.id ||
              typeof action.sessionId !== "string" ||
              !Array.isArray(action.ids) ||
              !action.ids.every((id) => typeof id === "string")
            )
              throw new UiError("invalidRead");
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
              ...(action.language !== undefined
                ? { language: action.language }
                : {}),
              ...(action.endpoint !== undefined
                ? { endpoint: action.endpoint }
                : {}),
              ...(action.scale !== undefined ? { scale: action.scale } : {}),
              ...(action.quietUntil !== undefined
                ? { quietUntil: action.quietUntil }
                : {}),
              ...(action.notifications !== undefined
                ? { notifications: action.notifications }
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
            throw new UiError("unknownAction");
        }
      };
      ipcMain.handle(channels.action, async (event, action: DesktopAction) => {
        try {
          await dispatch(event, action);
          return null;
        } catch (error) {
          return serializeError(error);
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
          state.settings.language,
          state.settings.scale,
          state.settings.notifications,
          state.settings.quietUntil > Date.now(),
          state.status,
          Boolean(state.pending?.streaming),
          windows!.companion.isVisible(),
        ]);
        if (next !== menuState) {
          menuState = next;
          tray!.setContextMenu(menu());
          tray!.setToolTip(translate(state.settings.language, "trayTooltip"));
        }
      };
      openMenu = () => menu().popup({ window: windows!.companion });
      windows.companion.on("show", () => {
        if (quitting) return;
        notifications?.close();
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
        translate(language, "startupFailed"),
        errorText(error, language),
      );
      app.quit();
    });
}
