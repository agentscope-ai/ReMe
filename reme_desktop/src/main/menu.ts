import { dialog, Menu } from "electron";
import type { AnimationManifest } from "../shared/protocol.ts";
import type { DesktopController } from "./controller.ts";
import type { DesktopWindows } from "./windows.ts";

/** The character and tray expose the same native controls. */
export function desktopMenu(
  controller: DesktopController,
  windows: DesktopWindows,
  manifest: AnimationManifest,
  quit: () => void,
  refresh: () => void,
): Menu {
  const { state } = controller;
  const quiet = state.settings.quietUntil > Date.now();
  const busy = Boolean(state.pending?.streaming) || state.status === "active";
  const settings = (update: Parameters<DesktopController["settings"]>[0]) => {
    void controller.settings(update).catch((error: unknown) => {
      dialog.showErrorBox(
        "桌面操作失败",
        error instanceof Error ? error.message : String(error),
      );
    });
  };
  return Menu.buildFromTemplate([
    { id: "chat", label: "和 Remy 聊聊", click: () => windows.openChat() },
    {
      id: "unread",
      label: state.unread ? `未读提醒（${state.unread}）` : "暂无未读提醒",
      enabled: state.unread > 0,
      click: () => windows.openChat(),
    },
    { type: "separator" },
    {
      id: "animations",
      label: "播放动作",
      enabled: !busy,
      submenu: manifest.animations.map((clip) => ({
        id: `animation:${clip.id}`,
        label: clip.label,
        click: () => windows.command({ type: "animation.play", id: clip.id }),
      })),
    },
    {
      id: "quiet",
      label: quiet ? "恢复主动提醒" : "安静 30 分钟",
      click: () =>
        settings({ quietUntil: quiet ? 0 : Date.now() + 30 * 60_000 }),
    },
    {
      id: "scale",
      label: "角色大小",
      submenu: [...new Set([0.75, 1, 1.25, 1.5, state.settings.scale])]
        .sort((a, b) => a - b)
        .map((scale) => ({
          id: `scale:${scale}`,
          label: `${scale * 100}%`,
          type: "radio" as const,
          checked: state.settings.scale === scale,
          click: () => settings({ scale }),
        })),
    },
    {
      id: "settings",
      label: "设置…",
      click: () => {
        windows.openChat();
        windows.command({ type: "settings.open" });
      },
    },
    { type: "separator" },
    {
      id: "reset",
      label: "回到屏幕右下角",
      click: () => windows.resetPosition(),
    },
    {
      id: "visibility",
      label: windows.companion.isVisible() ? "隐藏角色" : "显示角色",
      click: () => {
        if (windows.companion.isVisible()) windows.companion.hide();
        else windows.showCompanion();
        refresh();
      },
    },
    { type: "separator" },
    { id: "quit", label: "退出 ReMe Desktop", click: quit },
  ]);
}
