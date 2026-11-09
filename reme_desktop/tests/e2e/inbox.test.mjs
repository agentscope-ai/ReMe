import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { _electron as electron } from "playwright";
import { expect } from "playwright/test";
import { desktopWindows } from "./windows.mjs";
import { mockReMe } from "./mock-reme.mjs";

const project = path.resolve(fileURLToPath(new URL("../../", import.meta.url)));

test(
  "a removed task filter can still be cleared after an inbox refresh",
  { timeout: 30_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-inbox-e2e-"));
    const backend = await mockReMe();
    backend.setTaskIds(["gold", "stocks"]);
    let app;
    try {
      app = await electron.launch({
        args: [project],
        env: {
          ...process.env,
          REME_DESKTOP_DATA_DIR: directory,
          REME_DESKTOP_API_URL: backend.url,
        },
      });
      const { chat } = await desktopWindows(app);
      await chat.evaluate(() => window.desktop.action({ type: "chat.open" }));
      await chat.getByRole("button", { name: "收件箱", exact: true }).click();
      const filter = chat.getByRole("combobox", {
        name: "按任务筛选",
        exact: true,
      });
      await filter.click();
      await chat.getByRole("option", { name: "stocks", exact: true }).click();
      backend.setTaskIds([]);
      await filter.click();
      await expect(chat.getByRole("option")).toHaveCount(2, {
        timeout: 10_000,
      });
      await expect(filter).toBeVisible();
      await expect(filter).toHaveText("stocks");
      await chat.getByRole("option", { name: "全部任务", exact: true }).click();
      await expect(filter).toHaveCount(0);
      backend.setTaskIds(["gold"]);
      await expect(filter).toBeVisible({ timeout: 10_000 });
      await filter.click();
      await chat.getByRole("option", { name: "gold", exact: true }).click();
      await expect(filter).toHaveText("gold");
    } finally {
      if (app) await app.close().catch(() => {});
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
