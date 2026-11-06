import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import type { Presentation, Settings } from "../shared/protocol.ts";

export const defaultSettings: Settings = {
  endpoint: "http://127.0.0.1:2333",
  scale: 1,
  quietUntil: 0,
  notifications: false,
};

export function normalizeEndpoint(value: string): string {
  const url = new URL(value.trim());
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    throw new Error(
      "服务地址需要是 HTTP/HTTPS 地址，不包含账号、查询参数或片段",
    );
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
    throw new Error(`配置文件不是有效 JSON，请修复后重试：${file}`);
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
      endpoint: normalizeEndpoint(
        settings.endpoint ?? defaultSettings.endpoint,
      ),
      scale: Number.isFinite(settings.scale)
        ? Math.max(0.7, Math.min(1.5, settings.scale!))
        : 1,
      quietUntil: Number.isFinite(settings.quietUntil)
        ? settings.quietUntil!
        : 0,
      notifications: settings.notifications === true,
      ...(settings.anchor &&
      Number.isFinite(settings.anchor.x) &&
      Number.isFinite(settings.anchor.y)
        ? { anchor: settings.anchor }
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
      throw new Error("presentation.json 需要是对象；原文件未修改");
    }
    for (const value of Object.values(this.presentations)) {
      if (
        !value ||
        typeof value.initialized !== "boolean" ||
        !value.receipts ||
        typeof value.receipts !== "object"
      ) {
        throw new Error("presentation.json 的展示记录无效；原文件未修改");
      }
      for (const receipt of Object.values(value.receipts)) {
        if (
          !receipt ||
          typeof receipt.read !== "boolean" ||
          typeof receipt.presented !== "boolean" ||
          (receipt.snoozeUntil !== undefined &&
            (!Number.isFinite(receipt.snoozeUntil) || receipt.snoozeUntil < 0))
        ) {
          throw new Error("presentation.json 的消息回执无效；原文件未修改");
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
