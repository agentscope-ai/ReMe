import { defaultLanguage, UiError } from "./i18n.ts";
import type { Presentation, Settings } from "./protocol.ts";

export interface ClientStorage {
  settings: Settings;
  scope(endpoint: string, workspace: string): string;
  presentation(key: string): Presentation;
  changed(): void;
  flush(): Promise<void>;
}

export const defaultSettings: Settings = {
  language: defaultLanguage,
  theme: "system",
  transparentBackground: false,
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
