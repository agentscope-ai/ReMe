/// <reference types="electron-vite/node" />

import { BehaviorController } from "./behavior.ts";
import {
  defaultLanguage,
  errorText,
  serializeError,
  translate,
  UiError,
} from "../../../client/src/i18n.ts";
import {
  app,
  autoUpdater as nativeUpdater,
  clipboard,
  dialog,
  globalShortcut,
  ipcMain,
  nativeImage,
  nativeTheme,
  protocol,
  shell,
  Tray,
} from "electron";
import { readFile, realpath, stat } from "node:fs/promises";
import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import electronUpdater from "electron-updater";
import { fileURLToPath } from "node:url";
import { resolveWorkspaceLink } from "../../../client/src/links.ts";
import {
  ClientController,
  validateItemTarget,
} from "../../../client/src/controller.ts";
import { loadAssets } from "./assets.ts";
import { Storage } from "./storage.ts";
import { normalizeEndpoint } from "../../../client/src/settings.ts";
import { DesktopWindows } from "./windows.ts";
import { desktopMenu } from "./menu.ts";
import { SystemNotifications } from "./system-notifications.ts";
import { isTrustedSender } from "./ipc.ts";
import { viewState } from "./view-state.ts";
import { DesktopUpdater } from "./updater.ts";
import { WORKBENCH_PAGES } from "../../../client/src/protocol.ts";
import { channels } from "../protocol.ts";
import type { DesktopAction } from "../protocol.ts";
import studioIconPath from "../../../workbench/public/reme-app-icon.png?asset";

const here = path.dirname(fileURLToPath(import.meta.url));
app.setName("ReMe Studio");
if (process.platform === "win32")
  app.setAppUserModelId("io.agentscope.reme.desktop");
if (process.env.REME_DESKTOP_DATA_DIR) {
  const userData = path.resolve(process.env.REME_DESKTOP_DATA_DIR);
  mkdirSync(userData, { recursive: true });
  app.setPath("userData", userData);
}
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
  let controller: ClientController | undefined;
  let tray: Tray | undefined;
  let notifications: SystemNotifications | undefined;
  let companionMenu: Electron.Menu | undefined;
  let refreshTray = () => {};
  let quitting = false;
  let language = defaultLanguage;
  const closeCompanionMenu = () => {
    const menu = companionMenu;
    if (!menu) return;
    companionMenu = undefined;
    if (!windows!.companion.isDestroyed()) menu.closePopup(windows!.companion);
    controller!.setMenuOpen(false);
  };
  app.on("second-instance", () => windows?.openChat());
  app.on("activate", () => {
    if (controller?.state.surface?.kind !== "menu") windows?.openChat();
  });
  app.on("window-all-closed", () => {});
  app.on("before-quit", (event) => {
    if (quitting || !controller) return;
    event.preventDefault();
    quitting = true;
    closeCompanionMenu();
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
        : path.resolve(here, "../../assets/remy");
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
      controller = new ClientController(
        storage,
        (state) => {
          language = state.settings.language;
          if (nativeTheme.themeSource !== state.settings.theme)
            nativeTheme.themeSource = state.settings.theme;
          windows?.update(state);
          notifications?.update(state);
          refreshTray();
        },
        new BehaviorController(assets.manifest),
        (notice, shown) => notifications?.show(notice, shown),
        (command) => windows?.command(command),
      );
      windows = new DesktopWindows(
        storage,
        {
          preload: path.join(here, "../preload/index.cjs"),
          renderer: path.join(here, "../renderer/index.html"),
          icon: studioIconPath,
        },
        (visible) => controller!.setChatVisible(visible),
        (focused) => controller!.setChatFocused(focused),
        (visible) => {
          if (quitting) return;
          if (visible) notifications?.close();
          else closeCompanionMenu();
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
          controller!.setChatVisible(
            true,
            notice.messageId
              ? { kind: "message", id: notice.messageId }
              : undefined,
          );
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
      const updateAvailable =
        app.isPackaged &&
        (process.platform !== "linux" || Boolean(process.env.APPIMAGE)) &&
        existsSync(path.join(process.resourcesPath, "app-update.yml"));
      if (updateAvailable) {
        nativeUpdater.once("before-quit-for-update", () => {
          // Window close normally hides the workbench and companion. Let the updater exit them.
          closeCompanionMenu();
          windows?.destroy();
        });
      }
      const updater = new DesktopUpdater(
        updateAvailable ? electronUpdater.autoUpdater : null,
        (state) => {
          const chat = windows!.chat;
          if (!chat.isDestroyed())
            chat.webContents.send(channels.updateState, state);
        },
      );
      ipcMain.handle(channels.update, (event, command: unknown) => {
        trusted(event);
        if (event.sender.id !== windows!.chat.webContents.id)
          throw new UiError("invalidSender");
        if (
          command !== "check" &&
          command !== "download" &&
          command !== "install"
        )
          throw new UiError("invalidAction");
        return updater.run(command);
      });
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
            if (
              action.sessionId === controller!.model.sessionId &&
              controller!.state.selectedAgentName !== "Remy"
            )
              controller!.selectAgent("Remy");
            if (action.sessionId !== controller!.state.sessionId)
              throw new UiError("sessionUnavailable");
            controller!.speakMessage(action.messageId);
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
          case "companion.menu": {
            if (!windows!.isCompanionSender(event.sender.id))
              throw new UiError("menuFromCharacter");
            closeCompanionMenu();
            companionMenu = desktopMenu(
              controller!,
              windows!,
              assets.manifest,
              () => app.quit(),
              refreshTray,
            );
            controller!.setMenuOpen(true);
            const menu = companionMenu;
            try {
              menu.popup({
                window: windows!.companion,
                callback: () => {
                  if (companionMenu !== menu) return;
                  companionMenu = undefined;
                  controller!.setMenuOpen(false);
                },
              });
            } catch (error) {
              closeCompanionMenu();
              throw error;
            }
            return;
          }
          case "companion.menu.close":
            if (event.sender.id !== windows!.companion.webContents.id)
              throw new UiError("invalidSender");
            closeCompanionMenu();
            return;
          case "settings.open":
            windows!.openChat();
            controller!.selectPage("settings");
            return;
          case "animation.play":
            if (
              !assets.manifest.animations.some(
                (clip) => clip.id === action.clip,
              )
            )
              throw new UiError("invalidAction");
            if (
              controller!.state.status === "active" ||
              controller!.state.pending?.streaming
            )
              throw new UiError("busy");
            controller!.preview(action.clip);
            return;
          case "app.quit":
            app.quit();
            return;
          case "chat.open":
          case "chat.close": {
            const visible = action.type === "chat.open";
            const target =
              action.type === "chat.open" ? action.target : undefined;
            if (visible) {
              if (
                event.sender.id === windows!.companion.webContents.id &&
                !target
              )
                controller!.preview("happy");
              windows!.openChat();
            } else windows!.closeChat();
            controller!.setChatVisible(visible, target);
            return;
          }
          case "item.action": {
            validateItemTarget(action.target);
            if (
              event.sender.id !== windows!.chat.webContents.id &&
              action.action !== "open"
            )
              throw new UiError("invalidSender");
            switch (action.action) {
              case "open":
                windows!.openChat();
                controller!.setChatVisible(true, action.target);
                return;
              case "remind":
                if (await controller!.remind(action.target))
                  windows!.showCompanion();
                return;
              case "explain":
                windows!.openChat();
                return controller!.explain(action.target);
              case "copy":
                clipboard.writeText(
                  (await controller!.item(action.target)).content,
                );
                return;
              default:
                throw new UiError("invalidAction");
            }
          }
          case "chat.send":
            if (event.sender.id !== windows!.chat.webContents.id)
              throw new UiError("invalidSender");
            if (typeof action.query !== "string")
              throw new UiError("textRequired");
            return controller!.send(action.query);
          case "chat.stop":
            if (event.sender.id !== windows!.chat.webContents.id)
              throw new UiError("invalidSender");
            controller!.stopReply();
            return;
          case "page.select":
            if (
              event.sender.id !== windows!.chat.webContents.id ||
              !WORKBENCH_PAGES.includes(action.page)
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
          case "agent.select":
            if (
              event.sender.id !== windows!.chat.webContents.id ||
              typeof action.agentName !== "string"
            )
              throw new UiError("invalidAction");
            controller!.selectAgent(action.agentName);
            return;
          case "agent.reveal": {
            if (event.sender.id !== windows!.chat.webContents.id)
              throw new UiError("invalidSender");
            const agent = controller!.state.agents?.find(
              (item) => item.agent_name === action.agentName,
            );
            const workspace = controller!.state.clientInfo?.workspace_dir;
            if (!agent || agent.agent_type === "chat" || !workspace)
              throw new UiError("invalidAction");
            const root = await realpath(workspace);
            const directory = await realpath(
              path.resolve(root, agent.directory),
            );
            const relative = path.relative(root, directory);
            if (
              !relative ||
              relative === ".." ||
              relative.startsWith(`..${path.sep}`) ||
              path.isAbsolute(relative) ||
              !(await stat(directory)).isDirectory()
            )
              throw new UiError("invalidWorkspaceRequest");
            const error = await shell.openPath(directory);
            if (error) throw new Error(error);
            return;
          }
          case "session.new":
            if (event.sender.id !== windows!.chat.webContents.id)
              throw new UiError("invalidSender");
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
          case "items.read":
            if (
              event.sender.id !== windows!.chat.webContents.id &&
              !windows!.isCompanionSender(event.sender.id)
            )
              throw new UiError("invalidSender");
            return controller!.markAllRead();
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
            if (event.sender.id !== windows!.chat.webContents.id)
              throw new UiError("invalidSender");
            return controller!.settings({
              ...(action.theme !== undefined ? { theme: action.theme } : {}),
              ...(action.transparentBackground !== undefined
                ? { transparentBackground: action.transparentBackground }
                : {}),
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
          case "document.open":
          case "link.open": {
            const href =
              action.type === "document.open" ? action.path : action.url;
            if (typeof href !== "string" || href.length > 8192)
              throw new UiError("invalidAction");
            if (action.type === "link.open" && /^https?:\/\//i.test(href)) {
              const url = new URL(href);
              if (url.username || url.password)
                throw new UiError("invalidAction");
              await shell.openExternal(url.toString());
            } else {
              const target = resolveWorkspaceLink(
                href,
                action.type === "link.open" ? action.sourcePath : "",
                controller!.state.clientInfo?.workspace_dir,
              );
              if (!target) throw new UiError("invalidWorkspaceRequest");
              windows!.openChat();
              controller!.setChatVisible(true);
              controller!.openDocument(target);
            }
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
          if (!(error instanceof UiError && error.key === "speechCancelled"))
            console.warn("Voice transcription failed:", error);
          return { error: serializeError(error) };
        }
      });
      ipcMain.on(channels.pointer, (event, interactive) => {
        if (!validSender(event)) return;
        if (event.sender.id === windows!.companion.webContents.id)
          windows!.pointer(interactive === true);
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

      app.dock?.setIcon(studioIconPath);
      const trayIcon = nativeImage.createFromPath(studioIconPath);
      tray = new Tray(trayIcon.resize({ width: 20, height: 20 }));
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
