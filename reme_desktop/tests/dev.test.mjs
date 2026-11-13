import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { createServer, normalizePath } from "vite";
import { _electron as electron } from "playwright";
import { expect } from "playwright/test";
import config from "../electron.vite.config.ts";
import { desktopWindows } from "./e2e/windows.mjs";
import { mockReMe } from "./e2e/mock-reme.mjs";

const project = path.resolve(fileURLToPath(new URL("../", import.meta.url)));

test(
  "Vite development renderer starts with React refresh and sandboxed preload",
  { timeout: 30_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-desktop-dev-"));
    const backend = await mockReMe({ speech: true });
    const server = await createServer({
      ...config.renderer,
      plugins: [
        ...config.renderer.plugins,
        {
          name: "entry-load-counter",
          enforce: "pre",
          transform(code, id) {
            if (
              normalizePath(id.split("?")[0]) ===
              normalizePath(path.join(project, "src/renderer/src.tsx"))
            )
              return (
                code +
                "\nglobalThis.__entryLoads = (globalThis.__entryLoads || 0) + 1;"
              );
          },
        },
      ],
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
      const { companion, chat } = await desktopWindows(app);
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

      const errors = [];
      chat.on("console", (message) => {
        if (message.type() === "error" && message.text().includes("createRoot"))
          errors.push(message.text());
      });
      const ready = () =>
        chat.evaluate(() =>
          window.desktop.bootstrap().then(({ state }) => state.speech.ready),
        );
      await expect.poll(ready).toBe(true);
      const marker = await chat.evaluate(
        () => (window.hmrMarker = crypto.randomUUID()),
      );
      for (let index = 1; index <= 2; index++) {
        server.moduleGraph.invalidateAll();
        server.ws.send({
          type: "update",
          updates: [
            {
              type: "js-update",
              path: "/src.tsx",
              acceptedPath: "/src.tsx",
              timestamp: Date.now() + index,
            },
          ],
        });
        await expect
          .poll(() => chat.evaluate(() => globalThis.__entryLoads))
          .toBe(index + 1);
        await expect.poll(ready).toBe(true);
        assert.equal(await chat.evaluate(() => window.hmrMarker), marker);
      }
      assert.deepEqual(errors, []);
      await chat.reload();
      await expect.poll(ready).toBe(true);
    } finally {
      await app?.close();
      await server.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
