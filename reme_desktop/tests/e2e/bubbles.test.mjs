import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
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
  "switching reminder queues gives each painted message a unique presentation ID",
  { timeout: 30_000 },
  async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-bubble-e2e-"));
    const backend = await mockReMe();
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
      const { companion, chat } = await desktopWindows(app);
      const state = () =>
        chat.evaluate(() =>
          window.desktop.bootstrap().then(({ state }) => state),
        );
      await expect
        .poll(async () => (await state()).connection)
        .toBe("connected");
      backend.push("ordinary", "Ordinary reminder.");
      await expect(
        companion.locator('[data-message-id="ordinary"]'),
      ).toBeVisible();
      await expect
        .poll(async () => {
          try {
            const presentations = JSON.parse(
              await readFile(
                path.join(directory, "desktop/presentation.json"),
                "utf8",
              ),
            );
            return Object.values(presentations).some(
              (entry) => entry.receipts.ordinary?.presented,
            );
          } catch {
            return false;
          }
        })
        .toBe(true);
      const ordinary = (await state()).surface;
      backend.upsert({
        id: "immediate",
        role: "assistant",
        source: "proactive",
        content: "Immediate task result.",
        completed: true,
        created_at: new Date().toISOString(),
        delivery: "immediate",
        origin_session_id: "session-1",
        notification_eligible: true,
        chunks: [],
      });
      await expect(
        companion.locator('[data-message-id="immediate"]'),
      ).toBeVisible();
      const immediate = (await state()).surface;
      assert.notEqual(immediate.presentationId, ordinary.presentationId);
      await expect
        .poll(async () => (await state()).readMessageIds.includes("immediate"))
        .toBe(true);
      await companion
        .locator('[data-message-id="immediate"]')
        .getByRole("button", { name: "收起气泡" })
        .click();
      await expect(
        companion.locator('[data-message-id="ordinary"]'),
      ).toBeVisible();
      assert.equal((await state()).readMessageIds.includes("ordinary"), false);
    } finally {
      await app?.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
