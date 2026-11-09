import { defaultLanguage, UiError } from "../shared/i18n.ts";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import type { Presentation, Settings } from "../shared/protocol.ts";
import { activityLevels, sleepDurations } from "../shared/protocol.ts";

export const defaultSettings: Settings = {
  language: defaultLanguage,
  theme: "system",
  endpoint: "http://127.0.0.1:2333",
  scale: 1,
  activityLevel: "normal",
  sleepMinutes: 15,
  quietUntil: 0,
  notifications: false,
  voiceAutoSend: true,
  voiceEnabled: true,
  voiceReplies: true,
  voiceReminders: false,
  voiceVolume: 0.8,
};

export function normalizeEndpoint(value: string): string {
  let url: URL;
  try {
    url = new URL(value.trim());
  } catch {
    throw new UiError("invalidEndpoint");
  }
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    throw new UiError("invalidEndpoint");
  }
  return url.toString().replace(/\/+$/, "");
}

async function load<T>(file: string, fallback: T): Promise<T> {
  let content: string;
  try {
    content = await readFile(file, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return fallback;
    throw error;
  }
  try {
    return JSON.parse(content) as T;
  } catch {
    throw new UiError("invalidSettingsJson", { file });
  }
}

/** Writes are serialized and atomically renamed; conversation content stays on ReMe. */
export class Storage {
  settings: Settings = { ...defaultSettings };
  presentations: Record<string, Presentation> = {};
  private writing: Promise<void> = Promise.resolve();
  private dirty = false;
  constructor(readonly directory: string) {}

  async load(): Promise<void> {
    const settings = await load<Partial<Settings>>(
      path.join(this.directory, "settings.json"),
      {},
    );
    this.settings = {
      language: settings.language === "en" ? "en" : defaultLanguage,
      theme:
        settings.theme === "light" || settings.theme === "dark"
          ? settings.theme
          : "system",
      endpoint: normalizeEndpoint(
        settings.endpoint ?? defaultSettings.endpoint,
      ),
      scale: Number.isFinite(settings.scale)
        ? Math.max(0.7, Math.min(1.5, settings.scale!))
        : 1,
      quietUntil: Number.isFinite(settings.quietUntil)
        ? settings.quietUntil!
        : 0,
      activityLevel:
        settings.activityLevel &&
        activityLevels.includes(settings.activityLevel)
          ? settings.activityLevel
          : defaultSettings.activityLevel,
      sleepMinutes: sleepDurations.includes(settings.sleepMinutes!)
        ? settings.sleepMinutes!
        : defaultSettings.sleepMinutes,
      notifications: settings.notifications === true,
      voiceAutoSend: settings.voiceAutoSend !== false,
      voiceEnabled: settings.voiceEnabled !== false,
      voiceReplies: settings.voiceReplies !== false,
      voiceReminders: settings.voiceReminders === true,
      voiceVolume: Number.isFinite(settings.voiceVolume)
        ? Math.min(1, Math.max(0, settings.voiceVolume!))
        : defaultSettings.voiceVolume,
      ...(settings.anchor &&
      Number.isFinite(settings.anchor.x) &&
      Number.isFinite(settings.anchor.y)
        ? { anchor: settings.anchor }
        : {}),
      ...(settings.windowBounds &&
      Object.values(settings.windowBounds).length === 4 &&
      [
        settings.windowBounds.x,
        settings.windowBounds.y,
        settings.windowBounds.width,
        settings.windowBounds.height,
      ].every(Number.isFinite) &&
      settings.windowBounds.width >= 500 &&
      settings.windowBounds.height >= 400
        ? { windowBounds: settings.windowBounds }
        : {}),
    };
    this.presentations = await load(
      path.join(this.directory, "presentation.json"),
      {},
    );
    if (
      !this.presentations ||
      typeof this.presentations !== "object" ||
      Array.isArray(this.presentations)
    ) {
      throw new UiError("invalidPresentation");
    }
    for (const value of Object.values(this.presentations)) {
      if (
        !value ||
        typeof value.initialized !== "boolean" ||
        !value.receipts ||
        typeof value.receipts !== "object"
      ) {
        throw new UiError("invalidReceipts");
      }
      for (const receipt of Object.values(value.receipts)) {
        if (
          !receipt ||
          typeof receipt.read !== "boolean" ||
          typeof receipt.presented !== "boolean" ||
          (receipt.snoozeUntil !== undefined &&
            (!Number.isFinite(receipt.snoozeUntil) || receipt.snoozeUntil < 0))
        ) {
          throw new UiError("invalidReceipts");
        }
      }
    }
  }

  presentation(key: string): Presentation {
    return (this.presentations[key] ??= { initialized: false, receipts: {} });
  }

  changed(): void {
    this.dirty = true;
  }

  async flush(): Promise<void> {
    if (!this.dirty) return this.writing;
    this.dirty = false;
    const files = [
      ["settings.json", JSON.stringify(this.settings, null, 2) + "\n"],
      ["presentation.json", JSON.stringify(this.presentations, null, 2) + "\n"],
    ];
    const write = async () => {
      await mkdir(this.directory, { recursive: true });
      for (const [name, content] of files) {
        const file = path.join(this.directory, name);
        await writeFile(file + ".tmp", content, { mode: 0o600 });
        await rename(file + ".tmp", file);
      }
    };
    const result = this.writing.then(write);
    this.writing = result.catch(() => {
      this.dirty = true;
    });
    return result;
  }
}
