import type {
  AnimationManifest,
  DesktopAction,
  DesktopState,
} from "../../shared/protocol.ts";
import { translator } from "../../shared/i18n.ts";
import { animationLabels, languages } from "../../shared/i18n.ts";
import type { MenuItem } from "../context-menu.tsx";

export function companionMenu(
  state: DesktopState,
  manifest: AnimationManifest,
  run: (action: DesktopAction) => void,
): MenuItem[] {
  const t = translator(state.settings.language);
  const busy = state.status === "active" || Boolean(state.pending?.streaming);
  const quiet = state.settings.quietUntil > Date.now();
  return [
    {
      id: "chat",
      label: t("openChat"),
      select: () => run({ type: "chat.open" }),
    },
    {
      id: "unread",
      label: state.unread
        ? t("unreadReminders", { count: state.unread })
        : t("noUnread"),
      disabled: !state.unread,
      select: () => run({ type: "chat.open" }),
    },
    {
      id: "read-all",
      label: t("markAllRead"),
      disabled: state.connection !== "connected",
      select: () => run({ type: "items.read" }),
    },
    {
      id: "animations",
      label: t("playAnimation"),
      disabled: busy,
      children: manifest.animations.map((clip) => ({
        id: `animation:${clip.id}`,
        label: t(animationLabels[clip.id]),
        select: () => run({ type: "animation.play", clip: clip.id }),
      })),
    },
    {
      id: "quiet",
      label: t(quiet ? "resumeReminders" : "quiet"),
      select: () =>
        run({
          type: "settings.update",
          quietUntil: quiet ? 0 : Date.now() + 30 * 60_000,
        }),
    },
    {
      id: "scale",
      label: t("characterSize"),
      children: [...new Set([0.75, 1, 1.25, 1.5, state.settings.scale])]
        .sort((a, b) => a - b)
        .map((scale) => ({
          id: `scale:${scale}`,
          label: `${scale * 100}%`,
          checked: scale === state.settings.scale,
          select: () => run({ type: "settings.update", scale }),
        })),
    },
    {
      id: "notifications",
      label: t("notifications"),
      checked: state.settings.notifications,
      select: () =>
        run({
          type: "settings.update",
          notifications: !state.settings.notifications,
        }),
    },
    {
      id: "settings",
      label: t("settingsMenu"),
      select: () => run({ type: "settings.open" }),
    },
    {
      id: "voice",
      label: t("voiceEnabled"),
      checked: state.settings.voiceEnabled,
      disabled: !state.speech.tts,
      select: () =>
        run({
          type: "settings.update",
          voiceEnabled: !state.settings.voiceEnabled,
        }),
    },
    {
      id: "speech-stop",
      label: t("stopSpeech"),
      disabled: !state.speech.capture && !state.speech.playback,
      select: () => run({ type: "speech.stop" }),
    },
    {
      id: "language",
      label: t("language"),
      children: languages.map((language) => ({
        id: `language:${language}`,
        label: t(language === "zh" ? "chinese" : "english"),
        checked: state.settings.language === language,
        select: () => run({ type: "settings.update", language }),
      })),
    },
    {
      id: "reset",
      label: t("resetPosition"),
      select: () => run({ type: "companion.reset" }),
    },
    {
      id: "visibility",
      label: t("hideCharacter"),
      select: () => run({ type: "companion.visibility", visible: false }),
    },
    { id: "quit", label: t("quit"), select: () => run({ type: "app.quit" }) },
  ];
}
