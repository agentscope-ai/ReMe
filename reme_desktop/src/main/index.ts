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
  nativeTheme,
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
import { isTrustedSender } from "./ipc.ts";
import { viewState } from "./view-state.ts";
import { channels } from "../shared/protocol.ts";
import type { DesktopAction } from "../shared/protocol.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
app.setName("ReMe Desktop");
if (process.platform === "win32")
  app.setAppUserModelId("io.agentscope.reme.desktop");
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
      nativeTheme.themeSource = storage.settings.theme;
      if (process.env.REME_DESKTOP_API_URL)
        storage.settings.endpoint = normalizeEndpoint(
          process.env.REME_DESKTOP_API_URL,
        );
      controller = new DesktopController(
        storage,
        (state) => {
          language = state.settings.language;
          if (nativeTheme.themeSource !== state.settings.theme)
            nativeTheme.themeSource = state.settings.theme;
          windows?.update(state);
          notifications?.update(state);
          refreshTray();
        },
        assets.manifest,
        (notice, shown) => notifications?.show(notice, shown),
        (command) => windows?.command(command),
      );
      windows = new DesktopWindows(
        storage,
        {
          preload: path.join(here, "../preload/index.cjs"),
          renderer: path.join(here, "../renderer/index.html"),
          icon: path.join(directory, "frames/idle/00.png"),
        },
        (visible) => controller!.setChatVisible(visible),
        (focused) => controller!.setChatFocused(focused),
        (visible) => {
          if (quitting) return;
          if (visible) notifications?.close();
          controller!.setCompanionVisible(visible);
          refreshTray();
        },
      );
      const canCapture = () =>
        !quitting &&
        windows!.chat.isVisible() &&
        controller!.state.speech.capture?.phase === "starting";
      const permissionSession = windows.chat.webContents.session;
      permissionSession.setPermissionCheckHandler(
        (contents, permission, _origin, details) =>
          permission === "media" &&
          contents?.id === windows!.chat.webContents.id &&
          details.isMainFrame === true &&
          details.mediaType === "audio" &&
          canCapture(),
      );
      permissionSession.setPermissionRequestHandler(
        (contents, permission, callback, details) => {
          const media = details as Electron.MediaAccessPermissionRequest;
          callback(
            permission === "media" &&
              contents.id === windows!.chat.webContents.id &&
              media.isMainFrame === true &&
              media.requestingUrl === contents.getURL() &&
              media.mediaTypes?.length === 1 &&
              media.mediaTypes[0] === "audio" &&
              canCapture(),
          );
        },
      );
      windows.chat.webContents.on("did-start-loading", () =>
        controller!.speech.rendererReady(false),
      );
      windows.chat.webContents.on("render-process-gone", () =>
        controller!.speech.rendererReady(false),
      );
      notifications = new SystemNotifications(
        (notice) => {
          if (!controller!.canOpenNotice(notice)) return;
          windows!.openChat();
          controller!.setChatVisible(true, notice.messageId);
        },
        (notice) => controller!.canNotify(notice),
      );
      windows.companion.webContents.on("did-start-loading", () =>
        controller!.setCompanionReady(false),
      );
      windows.companion.webContents.on("did-finish-load", () =>
        controller!.setCompanionReady(true),
      );
      windows.companion.webContents.on("render-process-gone", () =>
        controller!.setCompanionReady(false),
      );

      const validSender = (
        event: Electron.IpcMainEvent | Electron.IpcMainInvokeEvent,
      ) =>
        !quitting &&
        isTrustedSender(event, (id) => windows?.isSender(id) === true);
      const trusted = (
        event: Electron.IpcMainEvent | Electron.IpcMainInvokeEvent,
      ) => {
        if (!validSender(event)) throw new UiError("invalidSender");
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
          platform: process.platform,
          version: app.getVersion(),
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
          case "speech.ready":
          case "speech.capture":
          case "speech.recording":
          case "speech.cancel":
          case "speech.report":
            if (event.sender.id !== windows!.chat.webContents.id)
              throw new UiError("invalidSender");
            if (action.type === "speech.ready") {
              if (
                typeof action.token !== "string" ||
                !action.token ||
                action.token.length > 80
              )
                throw new UiError("invalidAction");
              controller!.speech.rendererReady(
                action.ready === true,
                action.token,
              );
            } else if (action.type === "speech.capture")
              controller!.speech.beginCapture(action.id);
            else if (action.type === "speech.recording")
              controller!.speech.recording(action.id);
            else if (action.type === "speech.cancel")
              controller!.speech.cancelCapture(action.id);
            else if (
              Number.isSafeInteger(action.id) &&
              ["started", "ended", "failed"].includes(action.status)
            )
              controller!.speech.report(action.id, action.status);
            return;
          case "speech.speak": {
            if (action.sessionId !== controller!.model.sessionId)
              throw new UiError("sessionUnavailable");
            const message = controller!.model.messages.find(
              (message) =>
                message.id === action.messageId &&
                message.role === "assistant" &&
                message.completed,
            );
            if (!message) throw new UiError("speechNoText");
            controller!.speech.speak(message);
            return;
          }
          case "speech.stop":
            controller!.speech.cancel();
            return;
          case "speech.retry":
            controller!.speech.retry();
            return;
          case "speech.clearError":
            controller!.speech.clearError();
            return;
          case "playback.report":
            if (
              windows!.isCompanionSender(event.sender.id) &&
              Number.isSafeInteger(action.id) &&
              ["started", "ended", "failed"].includes(action.status)
            )
              controller!.reportPlayback(action.id, action.status);
            return;
          case "bubble.shown":
            if (
              windows!.isCompanionSender(event.sender.id) &&
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
            if (!windows!.isCompanionSender(event.sender.id))
              throw new UiError("menuFromCharacter");
            openMenu();
            return;
          case "chat.open":
          case "chat.close": {
            const visible = action.type === "chat.open";
            const messageId =
              action.type === "chat.open" ? action.messageId : undefined;
            if (visible) {
              if (
                event.sender.id === windows!.companion.webContents.id &&
                !messageId
              )
                controller!.preview("happy");
              windows!.openChat();
            } else windows!.closeChat();
            controller!.setChatVisible(visible, messageId);
            return;
          }
          case "chat.send":
            if (typeof action.query !== "string")
              throw new UiError("textRequired");
            return controller!.send(action.query);
          case "chat.stop":
            controller!.stopReply();
            return;
          case "page.select":
            if (
              event.sender.id !== windows!.chat.webContents.id ||
              !["remy", "knowledge", "settings"].includes(action.page)
            )
              throw new UiError("invalidAction");
            controller!.selectPage(action.page);
            return;
          case "window.minimize":
            windows!.chat.minimize();
            return;
          case "window.maximize":
            if (windows!.chat.isMaximized()) windows!.chat.unmaximize();
            else windows!.chat.maximize();
            return;
          case "companion.reset":
            windows!.resetPosition();
            return;
          case "companion.visibility":
            if (action.visible) windows!.showCompanion();
            else windows!.hideCompanion();
            return;
          case "session.new":
            return controller!.newSession();
          case "session.select":
            if (
              event.sender.id !== windows!.chat.webContents.id ||
              typeof action.sessionId !== "string"
            )
              throw new UiError("invalidAction");
            return controller!.selectSession(action.sessionId);
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
              ...(action.theme !== undefined ? { theme: action.theme } : {}),
              ...(action.language !== undefined
                ? { language: action.language }
                : {}),
              ...(action.endpoint !== undefined
                ? { endpoint: action.endpoint }
                : {}),
              ...(action.scale !== undefined ? { scale: action.scale } : {}),
              ...(action.activityLevel !== undefined
                ? { activityLevel: action.activityLevel }
                : {}),
              ...(action.sleepMinutes !== undefined
                ? { sleepMinutes: action.sleepMinutes }
                : {}),
              ...(action.quietUntil !== undefined
                ? { quietUntil: action.quietUntil }
                : {}),
              ...(action.notifications !== undefined
                ? { notifications: action.notifications }
                : {}),
              ...(action.voiceAutoSend !== undefined
                ? { voiceAutoSend: action.voiceAutoSend }
                : {}),
              ...(action.voiceEnabled !== undefined
                ? { voiceEnabled: action.voiceEnabled }
                : {}),
              ...(action.voiceReplies !== undefined
                ? { voiceReplies: action.voiceReplies }
                : {}),
              ...(action.voiceReminders !== undefined
                ? { voiceReminders: action.voiceReminders }
                : {}),
              ...(action.voiceVolume !== undefined
                ? { voiceVolume: action.voiceVolume }
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
      ipcMain.handle(channels.workspace, async (event, request) => {
        try {
          trusted(event);
          if (event.sender.id !== windows!.chat.webContents.id)
            throw new UiError("invalidSender");
          return await controller!.workspace(request);
        } catch (error) {
          return { error: serializeError(error) };
        }
      });
      ipcMain.handle(channels.transcribe, async (event, request) => {
        try {
          trusted(event);
          if (
            event.sender.id !== windows!.chat.webContents.id ||
            !request ||
            typeof request.id !== "string"
          )
            throw new UiError("invalidSender");
          return {
            text: await controller!.speech.transcribe(
              request.id,
              request.audio,
            ),
          };
        } catch (error) {
          return { error: serializeError(error) };
        }
      });
      ipcMain.on(channels.pointer, (event, interactive) => {
        if (!validSender(event)) return;
        if (event.sender.id === windows!.companion.webContents.id)
          windows!.pointer(interactive === true);
      });
      ipcMain.on(channels.surfaceSize, (event, size) => {
        if (!validSender(event)) return;
        if (windows!.isCompanionSender(event.sender.id))
          windows!.measureSurface(size);
      });
      ipcMain.on(channels.drag, (event, phase) => {
        if (!validSender(event)) return;
        if (
          ["start", "move", "end"].includes(phase) &&
          windows!.drag(phase, event.sender.id)
        ) {
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
          state.settings.voiceEnabled,
          state.speech.tts,
          Boolean(state.speech.playback || state.speech.capture),
          state.settings.quietUntil > Date.now(),
          state.status,
          Boolean(state.pending?.streaming),
          windows!.companionVisible,
        ]);
        if (next !== menuState) {
          menuState = next;
          tray!.setContextMenu(menu());
          tray!.setToolTip(translate(state.settings.language, "trayTooltip"));
        }
      };
      openMenu = () =>
        menu().popup({
          window: windows!.companion,
        });
      refreshTray();
      tray.on("click", () => windows!.openChat());
      globalShortcut.register("CommandOrControl+Shift+Space", () =>
        windows!.openChat(),
      );
      controller.start();
    })
    .catch((error) => {
      console.error(error);
      dialog.showErrorBox(
        translate(language, "startupFailed"),
        errorText(error, language),
      );
      app.quit();
    });
}
