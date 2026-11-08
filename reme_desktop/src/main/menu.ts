import { dialog, Menu } from "electron";
import type { AnimationManifest } from "../shared/protocol.ts";
import type { DesktopController } from "./controller.ts";
import type { DesktopWindows } from "./windows.ts";
import {
  animationLabels,
  errorText,
  languages,
  translator,
} from "../shared/i18n.ts";

/** The character and tray expose the same native controls. */
export function desktopMenu(
  controller: DesktopController,
  windows: DesktopWindows,
  manifest: AnimationManifest,
  quit: () => void,
  refresh: () => void,
): Menu {
  const { state } = controller;
  const t = translator(state.settings.language);
  const quiet = state.settings.quietUntil > Date.now();
  const busy = Boolean(state.pending?.streaming) || state.status === "active";
  const settings = (update: Parameters<DesktopController["settings"]>[0]) => {
    void controller.settings(update).catch((error: unknown) => {
      dialog.showErrorBox(
        t("operationFailed"),
        errorText(error, state.settings.language),
      );
    });
  };
  return Menu.buildFromTemplate([
    { id: "chat", label: t("openChat"), click: () => windows.openChat() },
    {
      id: "unread",
      label: state.unread
        ? t("unreadReminders", { count: state.unread })
        : t("noUnread"),
      enabled: state.unread > 0,
      click: () => windows.openChat(),
    },
    { type: "separator" },
    {
      id: "animations",
      label: t("playAnimation"),
      enabled: !busy,
      submenu: manifest.animations.map((clip) => ({
        id: `animation:${clip.id}`,
        label: t(animationLabels[clip.id]),
        click: () => controller.preview(clip.id),
      })),
    },
    {
      id: "quiet",
      label: t(quiet ? "resumeReminders" : "quiet"),
      click: () =>
        settings({ quietUntil: quiet ? 0 : Date.now() + 30 * 60_000 }),
    },
    {
      id: "scale",
      label: t("characterSize"),
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
      id: "notifications",
      label: t("notifications"),
      type: "checkbox",
      checked: state.settings.notifications,
      click: () => settings({ notifications: !state.settings.notifications }),
    },
    {
      id: "settings",
      label: t("settingsMenu"),
      click: () => {
        windows.openChat();
        windows.command({ type: "settings.open" });
      },
    },
    {
      id: "voice",
      label: t("voiceEnabled"),
      type: "checkbox",
      checked: state.settings.voiceEnabled,
      enabled: state.speech.tts,
      click: () => settings({ voiceEnabled: !state.settings.voiceEnabled }),
    },
    {
      id: "speech-stop",
      label: t("stopSpeech"),
      enabled: Boolean(state.speech.playback || state.speech.capture),
      click: () => controller.speech.cancel(),
    },
    {
      id: "language",
      label: t("language"),
      submenu: languages.map((language) => ({
        id: `language:${language}`,
        label: t(language === "zh" ? "chinese" : "english"),
        type: "radio" as const,
        checked: state.settings.language === language,
        click: () => settings({ language }),
      })),
    },
    { type: "separator" },
    {
      id: "reset",
      label: t("resetPosition"),
      click: () => windows.resetPosition(),
    },
    {
      id: "visibility",
      label: t(windows.companionVisible ? "hideCharacter" : "showCharacter"),
      click: () => {
        if (windows.companionVisible) windows.hideCompanion();
        else windows.showCompanion();
        refresh();
      },
    },
    { type: "separator" },
    { id: "quit", label: t("quit"), click: quit },
  ]);
}
