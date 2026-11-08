import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { createServer } from "vite";
import { _electron as electron } from "playwright";
import { expect } from "playwright/test";
import config from "../electron.vite.config.ts";
import { mockReMe } from "./e2e/mock-reme.mjs";

const project = path.resolve(fileURLToPath(new URL("../", import.meta.url)));

test(
  "Vite development renderer starts with React refresh and sandboxed preload",
  { timeout: 30_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-desktop-dev-"));
    const backend = await mockReMe();
    const server = await createServer({
      ...config.renderer,
      configFile: false,
      root: path.join(project, "src/renderer"),
      logLevel: "error",
      server: { host: "127.0.0.1", port: 0 },
    });
    let app;
    try {
      await server.listen();
      app = await electron.launch({
        args: [project],
        env: {
          ...process.env,
          REME_DESKTOP_DATA_DIR: directory,
          REME_DESKTOP_API_URL: backend.url,
          ELECTRON_RENDERER_URL: server.resolvedUrls.local[0],
        },
      });
      if (app.windows().length < 2)
        await app.waitForEvent("window", {
          predicate: () => app.windows().length >= 2,
        });
      const companion = app
        .windows()
        .find((page) => page.url().includes("view=companion"));
      const chat = app
        .windows()
        .find((page) => page.url().includes("view=chat"));
      await companion.waitForSelector("canvas", { timeout: 8_000 });
      await companion.evaluate(() =>
        window.desktop.action({ type: "chat.open" }),
      );
      await expect(
        chat.getByRole("textbox", { name: "消息", exact: true }),
      ).toBeVisible();
      await expect
        .poll(() =>
          chat.evaluate(() =>
            window.desktop.bootstrap().then((value) => value.state.connection),
          ),
        )
        .toBe("connected");
    } finally {
      await app?.close();
      await server.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
