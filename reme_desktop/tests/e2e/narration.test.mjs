import assert from "node:assert/strict";
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
const longText = (prefix) =>
  Array.from(
    { length: 6 },
    (_, i) =>
      `${prefix}第${i + 1}段：` +
      `这一段用于验证长文朗读队列的完整性，版本4.7，变化12.25%。`.repeat(8),
  ).join("\n\n") + `\n\n${prefix}最后一句，必须完整朗读到这里。`;
const compact = (text) => text.replace(/\s+/g, "");

test(
  "long narration reaches the end, resumes failures and cancels pending audio in real Electron",
  { timeout: 60_000 },
  async (t) => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-narration-e2e-"));
    const backend = await mockReMe({ speech: true, ttsSeconds: 0.25 });
    const app = await electron.launch({
      args: [project],
      env: {
        ...process.env,
        REME_DESKTOP_DATA_DIR: directory,
        REME_DESKTOP_API_URL: backend.url,
      },
    });
    try {
      const { chat, companion } = await desktopWindows(app);
      const errors = [];
      chat.on("pageerror", (error) => errors.push(error.message));
      const state = () =>
        chat.evaluate(() =>
          window.desktop.bootstrap().then(({ state }) => state),
        );
      await expect.poll(async () => (await state()).speech.tts).toBe(true);
      await app.evaluate(({ app }) => app.focus({ steal: true }));
      await companion.evaluate(() =>
        window.desktop.action({ type: "chat.open" }),
      );
      await chat.evaluate(() => {
        window.narrationPlayers = [];
        window.narrationStates = [];
        const NativeAudio = window.Audio;
        window.Audio = function (url) {
          const audio = new NativeAudio(url);
          const record = { audio, ended: false };
          audio.addEventListener("ended", () => {
            record.ended = true;
          });
          window.narrationPlayers.push(record);
          return audio;
        };
        window.desktop.onState((state) => {
          if (state.speech?.playback)
            window.narrationStates.push(state.speech.playback);
        });
      });

      await t.test(
        "automatic speech keeps every block beyond 600 characters",
        async () => {
          const source = longText("自动朗读样本");
          backend.upsert({
            id: "long-auto",
            turn_id: "long-turn",
            role: "assistant",
            source: "chat",
            content: source,
            delivery: "idle",
            notification_eligible: true,
            completed: true,
            chunks: [],
          });
          backend.activity("completed", "chat", "long-turn");
          await expect
            .poll(() => backend.ttsRequests.length)
            .toBeGreaterThan(2);
          await expect
            .poll(async () => (await state()).speech.playback, {
              timeout: 15_000,
            })
            .toBe(null);
          assert.equal(
            compact(backend.ttsRequests.map(({ text }) => text).join("")),
            compact(source),
          );
          assert.ok(
            backend.ttsRequests.every(({ text }) => [...text].length <= 400),
          );
          const playback = await chat.evaluate(() => ({
            ended: window.narrationPlayers.filter((item) => item.ended).length,
            states: window.narrationStates,
          }));
          assert.equal(playback.ended, backend.ttsRequests.length);
          assert.equal(
            new Set(
              playback.states
                .filter((item) => item.phase === "playing")
                .map((item) => item.part),
            ).size,
            backend.ttsRequests.length,
          );
          assert.ok(
            backend.ttsRequests
              .at(-1)
              .text.endsWith("最后一句，必须完整朗读到这里。"),
          );
        },
      );

      await t.test(
        "a failed block pauses visibly and resumes without replaying the prefix",
        async () => {
          const before = backend.ttsRequests.length;
          const source = longText("继续朗读样本").replace(
            "继续朗读样本第3段",
            "失败标记第3段",
          );
          backend.setSpeechErrorText("失败标记");
          backend.push("long-retry", source);
          await chat
            .locator('[data-message-id="long-retry"] .read-aloud')
            .click();
          await expect
            .poll(async () => (await state()).speech.playback?.phase)
            .toBe("error");
          const stoppedAt = (await state()).speech.playback.part;
          assert.ok(stoppedAt >= 3);
          await expect(chat.locator(".voice-status")).toContainText(
            "朗读已暂停",
          );
          assert.equal(
            backend.ttsRequests
              .slice(before)
              .filter(({ text }) => text.includes("失败标记")).length,
            2,
          );
          backend.setSpeechErrorText("");
          await chat
            .locator(".voice-status")
            .getByRole("button", { name: "继续朗读", exact: true })
            .click();
          await expect
            .poll(async () => (await state()).speech.playback, {
              timeout: 15_000,
            })
            .toBe(null);
          const requested = backend.ttsRequests.slice(before);
          assert.equal(
            requested.filter(({ text }) => text.startsWith("继续朗读样本第1段"))
              .length,
            1,
          );
          assert.ok(
            requested.at(-1).text.endsWith("最后一句，必须完整朗读到这里。"),
          );
        },
      );

      await t.test(
        "stopping during priming discards queued and late audio",
        async () => {
          backend.setSpeechDelay(500);
          const before = await chat.evaluate(
            () => window.narrationPlayers.length,
          );
          backend.push("long-cancel", longText("取消样本"));
          await chat
            .locator('[data-message-id="long-cancel"] .read-aloud')
            .click();
          await expect
            .poll(async () => (await state()).speech.playback?.preparing)
            .toBe(2);
          await chat
            .locator(".voice-status")
            .getByRole("button", { name: "停止朗读", exact: true })
            .click();
          await new Promise((resolve) => setTimeout(resolve, 700));
          assert.equal((await state()).speech.playback, null);
          assert.equal(
            await chat.evaluate(() => window.narrationPlayers.length),
            before,
          );
        },
      );
      assert.deepEqual(errors, []);
    } finally {
      await app.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
