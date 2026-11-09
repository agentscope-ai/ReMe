import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { _electron as electron } from "playwright";
import { expect } from "playwright/test";
import { desktopWindows } from "./e2e/windows.mjs";
import { mockReMe } from "./e2e/mock-reme.mjs";

const project = fileURLToPath(new URL("../", import.meta.url));

test(
  "packaged application loads its bundled frames and subscription",
  {
    timeout: 30_000,
    skip: !["darwin", "win32"].includes(process.platform),
  },
  async () => {
    const directory = await mkdtemp(
      path.join(tmpdir(), "reme-desktop-packaged-"),
    );
    const backend = await mockReMe();
    let app;
    try {
      const executable =
        process.env.REME_DESKTOP_EXECUTABLE ??
        path.join(
          project,
          process.platform === "win32"
            ? "dist/win-unpacked/ReMe Desktop.exe"
            : `dist/${process.arch === "arm64" ? "mac-arm64" : "mac"}/ReMe Desktop.app/Contents/MacOS/ReMe Desktop`,
        );
      const resources = path.resolve(
        path.dirname(executable),
        process.platform === "darwin" ? "../Resources" : "resources",
      );
      const manifest = JSON.parse(
        await readFile(path.join(resources, "remy/manifest.json"), "utf8"),
      );
      if (process.platform === "darwin") {
        const plist = await readFile(
          path.join(resources, "../Info.plist"),
          "utf8",
        );
        assert.match(plist, /NSMicrophoneUsageDescription/);
        assert.match(plist, /Microphone access is used only when you record/);
      }
      for (const clip of manifest.animations)
        for (const frame of clip.frame_files)
          assert.ok(
            (await readFile(path.join(resources, "remy", frame))).length > 0,
          );
      app = await electron.launch({
        executablePath: executable,
        args: [],
        env: {
          ...process.env,
          REME_DESKTOP_DATA_DIR: directory,
          REME_DESKTOP_API_URL: backend.url,
        },
      });
      const { companion, chat } = await desktopWindows(app);
      await expect
        .poll(() =>
          companion.evaluate(() =>
            window.desktop.bootstrap().then((value) => value.state.connection),
          ),
        )
        .toBe("connected");
      await expect
        .poll(() =>
          companion
            .locator("canvas")
            .evaluate(
              (canvas) =>
                canvas.getContext("2d").getImageData(256, 170, 1, 1).data[3],
            ),
        )
        .toBeGreaterThan(0);
      backend.push("packaged-reminder");
      await expect(
        companion.getByRole("region", { name: "主动提醒" }),
      ).toBeVisible();
      await companion.getByRole("button", { name: "展开聊聊" }).click();
      await expect(
        chat
          .locator(".chat-history")
          .getByText("你关注的 Agent 记忆论文有更新，想看看重点吗？", {
            exact: true,
          }),
      ).toBeVisible();
      await chat.evaluate(() =>
        window.desktop.action({ type: "settings.update", theme: "dark" }),
      );
      await expect(chat.locator("html")).toHaveAttribute("data-theme", "dark");
      const background = await chat.evaluate(() => ({
        color: getComputedStyle(document.querySelector(".workspace-main"))
          .backgroundColor,
        reducedTransparency: matchMedia(
          "(prefers-reduced-transparency: reduce)",
        ).matches,
      }));
      assert.equal(
        background.color,
        background.reducedTransparency
          ? "rgb(29, 37, 33)"
          : "rgba(29, 37, 33, 0.96)",
      );
      await expect(chat.locator(".workspace-shell")).toBeVisible();
      await mkdir(path.join(project, "reports"), { recursive: true });
      await chat.screenshot({
        path: path.join(project, "reports/chat-dark.png"),
      });
      const info = await app.evaluate(({ app }) => ({
        packaged: app.isPackaged,
        name: app.getName(),
        metrics: app
          .getAppMetrics()
          .map(({ type, memory }) => ({ type, memory })),
      }));
      assert.equal(info.packaged, true);
      assert.equal(info.name, "ReMe Desktop");
      if (process.platform === "win32") {
        assert.equal(
          await app.evaluate(({ BrowserWindow }) =>
            BrowserWindow.getAllWindows()
              .find((window) =>
                window.webContents.getURL().includes("view=chat"),
              )
              .isMenuBarVisible(),
          ),
          false,
        );
      }
      await writeFile(
        path.join(project, "reports/packaged.json"),
        JSON.stringify(info, null, 2),
      );
      const url = chat.url();
      await chat.evaluate(() => {
        const link = document.createElement("a");
        link.href = "https://example.com/";
        document.body.append(link);
        link.click();
        link.remove();
      });
      // A cancelled navigation leaves Playwright waiting for a load that never
      // starts. Check the native WebContents after dispatch instead.
      await new Promise((resolve) => setTimeout(resolve, 100));
      const retained = await app.evaluate(async ({ BrowserWindow }) => {
        const window = BrowserWindow.getAllWindows().find((entry) =>
          entry.webContents.getURL().includes("view=chat"),
        );
        return {
          url: window?.webContents.getURL(),
          body: await window?.webContents.executeJavaScript(
            'Boolean(document.querySelector(".workspace-shell"))',
          ),
        };
      });
      assert.equal(retained.url, url);
      assert.equal(retained.body, true);
    } finally {
      await app?.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
