import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
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
  "voice input, narration, reminders and cancellation in real sandboxed Electron windows",
  { timeout: 100_000 },
  async (t) => {
    const directory = await mkdtemp(path.join(tmpdir(), "reme-speech-e2e-"));
    const backend = await mockReMe({ speech: true });
    let app;
    try {
      app = await electron.launch({
        args: ["--mute-audio", project],
        env: {
          ...process.env,
          REME_DESKTOP_DATA_DIR: directory,
          REME_DESKTOP_API_URL: backend.url,
        },
      });
      const errors = [];
      const { chat, companion } = await desktopWindows(app);
      chat.on("pageerror", (error) => errors.push(error.message));
      const state = () =>
        chat.evaluate(() =>
          window.desktop.bootstrap().then((value) => value.state),
        );
      const open = async () => {
        await app.evaluate(({ app }) => app.focus({ steal: true }));
        await companion.evaluate(() =>
          window.desktop.action({ type: "chat.open" }),
        );
        await expect(
          chat.getByRole("textbox", { name: "消息", exact: true }),
        ).toBeVisible();
      };
      await expect
        .poll(async () => (await state()).connection)
        .toBe("connected");
      await open();
      await chat.evaluate(() => {
        window.voicePlayers = [];
        const NativeAudio = window.Audio;
        window.Audio = function (src) {
          const audio = new NativeAudio(src);
          window.voicePlayers.push(audio);
          return audio;
        };
        window.voiceStreams = [];
        window.voiceAmplitude = 1;
        window.voiceDelay = 0;
        // Synthetic streams guarantee tests never open a physical microphone.
        navigator.mediaDevices.getUserMedia = async () => {
          if (window.voiceDelay)
            await new Promise((resolve) =>
              setTimeout(resolve, window.voiceDelay),
            );
          const context = new AudioContext();
          const oscillator = context.createOscillator();
          const destination = context.createMediaStreamDestination();
          oscillator.frequency.value = 440;
          const gain = context.createGain();
          gain.gain.value = window.voiceAmplitude;
          oscillator.connect(gain).connect(destination);
          oscillator.start();
          await context.resume();
          const stream = destination.stream;
          const track = stream.getAudioTracks()[0];
          const stop = track.stop.bind(track);
          track.stop = () => {
            stop();
            oscillator.stop();
            void context.close();
          };
          window.voiceStreams.push(stream);
          return stream;
        };
      });
      const historical = chat.locator(
        '[data-message-id="historical"] .read-aloud',
      );

      await t.test(
        "manual playback lasts until audio ends, can stop, and reuses synthesis",
        async () => {
          assert.equal(backend.ttsRequests.length, 0);
          await historical.click();
          await expect
            .poll(async () => (await state()).speech.playback?.phase)
            .toBe("playing");
          await expect(companion.locator("canvas")).toHaveAttribute(
            "data-animation",
            "speaking",
          );
          await expect
            .poll(async () => (await state()).speech.playback)
            .toBe(null);
          assert.equal(backend.ttsRequests.length, 1);
          await historical.click();
          await expect
            .poll(async () => (await state()).speech.playback?.phase)
            .toBe("playing");
          assert.equal(backend.ttsRequests.length, 1);
          await historical.click();
          await expect
            .poll(async () => (await state()).speech.playback)
            .toBe(null);
          assert.equal(
            await chat.evaluate(() =>
              window.voicePlayers.every((audio) => audio.paused),
            ),
            true,
          );
        },
      );

      await t.test(
        "settings persist and new completed replies narrate once across reconnect",
        async () => {
          await chat.getByRole("button", { name: "设置", exact: true }).click();
          await chat
            .locator(".settings-nav")
            .getByRole("button", { name: "语音", exact: true })
            .click();
          const narration = chat.getByRole("switch", {
            name: "自动语音播报",
            exact: true,
          });
          await narration.click();
          await expect(narration).not.toBeChecked();
          await narration.click();
          await expect(narration).toBeChecked();
          const reminders = chat.getByRole("switch", {
            name: "朗读主动提醒",
            exact: true,
          });
          await reminders.click();
          await expect(reminders).toBeChecked();
          const volume = chat.getByRole("slider", {
            name: "语音音量",
            exact: true,
          });
          await volume.focus();
          await volume.press("Home");
          for (let i = 0; i < 6; i++) await volume.press("ArrowRight");
          await expect
            .poll(async () => (await state()).settings.voiceVolume)
            .toBe(0.3);
          await mkdir(path.join(project, "reports"), { recursive: true });
          await chat.screenshot({
            path: path.join(project, "reports/voice-settings.png"),
          });
          await chat.getByRole("tab", { name: "Remy", exact: true }).click();
          await chat
            .getByRole("textbox", { name: "消息", exact: true })
            .fill("请回复我的研究计划");
          await chat
            .getByRole("button", { name: "发送消息", exact: true })
            .click();
          await expect
            .poll(async () => (await state()).speech.playback?.kind)
            .toBe("reply");
          await expect
            .poll(async () => (await state()).speech.playback?.phase)
            .toBe("playing");
          assert.equal(
            await chat.evaluate(() => window.voicePlayers.at(-1).volume),
            0.3,
          );
          await expect
            .poll(async () => (await state()).speech.playback)
            .toBe(null);
          const calls = backend.ttsRequests.length;
          backend.disconnect();
          await expect.poll(() => backend.cursors.length).toBeGreaterThan(1);
          await expect
            .poll(async () => (await state()).connection)
            .toBe("connected");
          assert.equal(backend.ttsRequests.length, calls);
        },
      );

      await t.test(
        "only painted reminders narrate; quiet mode interrupts without replay",
        async () => {
          const surface = (await state()).surface;
          if (surface?.kind === "message" && surface.message.source === "chat")
            await companion.evaluate(
              (id) => window.desktop.action({ type: "bubble.dismiss", id }),
              surface.message.id,
            );
          await chat.evaluate(() =>
            window.desktop.action({ type: "chat.close" }),
          );
          await expect
            .poll(async () => (await state()).chatVisible)
            .toBe(false);
          backend.push("spoken-reminder", "新的记忆研究值得看一看。");
          await expect
            .poll(async () => (await state()).speech.playback?.kind)
            .toBe("reminder");
          await expect
            .poll(async () => (await state()).speech.playback?.phase)
            .toBe("playing");
          await expect(companion.locator(".message-bubble")).toBeVisible();
          await companion.evaluate(() =>
            window.desktop.action({
              type: "settings.update",
              quietUntil: Date.now() + 60_000,
            }),
          );
          await expect
            .poll(async () => (await state()).speech.playback)
            .toBe(null);
          const calls = backend.ttsRequests.length;
          await companion.evaluate(() =>
            window.desktop.action({ type: "settings.update", quietUntil: 0 }),
          );
          await expect(companion.locator(".message-bubble")).toBeVisible();
          assert.equal(backend.ttsRequests.length, calls);
          await companion
            .getByRole("button", { name: "朗读这条回复", exact: true })
            .click();
          await expect
            .poll(async () => (await state()).speech.playback?.kind)
            .toBe("manual");
          await companion
            .getByRole("button", { name: "收起气泡", exact: true })
            .click();
          await expect
            .poll(async () => (await state()).speech.playback)
            .toBe(null);
          await open();
        },
      );

      await t.test(
        "hold-to-talk records WAV through AudioWorklet and inserts into the current draft",
        async () => {
          await open();
          await chat.evaluate(() =>
            window.desktop.action({
              type: "settings.update",
              voiceAutoSend: false,
            }),
          );
          const input = chat.getByRole("textbox", {
            name: "消息",
            exact: true,
          });
          await input.fill("已有草稿：");
          await input.press("End");
          const mic = chat.locator(".mic-button");
          await mic.hover();
          await chat.mouse.down();
          await expect(mic).toHaveAttribute("aria-pressed", "true");
          await expect
            .poll(() => chat.locator("meter").evaluate((node) => node.value))
            .toBeGreaterThan(0);
          await expect(chat.locator(".voice-status")).toContainText("录音中 1");
          await chat.screenshot({
            path: path.join(project, "reports/voice-recording.png"),
          });
          await chat.mouse.up();
          await expect(input).toHaveValue("已有草稿：语音输入的研究计划。");
          assert.equal(backend.asrRequests.length, 1);
          const recorded = Buffer.from(backend.asrRequests[0].audio, "base64");
          assert.equal(recorded.toString("ascii", 0, 4), "RIFF");
          assert.equal(recorded.readUInt32LE(24), 16000);
          assert.equal(recorded.readUInt16LE(22), 1);
          assert.equal(recorded.readUInt16LE(34), 16);
          assert.ok(recorded.length > 16000);
          assert.equal(
            await chat.evaluate(() =>
              window.voiceStreams.every((stream) =>
                stream
                  .getTracks()
                  .every((track) => track.readyState === "ended"),
              ),
            ),
            true,
          );
        },
      );

      await t.test(
        "shortcut cancellation and hiding chat release the microphone without sending audio",
        async () => {
          const input = chat.getByRole("textbox", {
            name: "消息",
            exact: true,
          });
          await input.focus();
          const shortcut =
            process.platform === "darwin" ? "Meta+Shift+M" : "Control+Shift+M";
          await chat.keyboard.press(shortcut);
          await expect(chat.locator(".mic-button")).toHaveAttribute(
            "aria-pressed",
            "true",
          );
          await chat.keyboard.press("Escape");
          await expect
            .poll(async () => (await state()).speech.capture)
            .toBe(null);
          await chat.keyboard.press(shortcut);
          await expect(chat.locator(".mic-button")).toHaveAttribute(
            "aria-pressed",
            "true",
          );
          await chat.evaluate(() =>
            window.desktop.action({ type: "chat.close" }),
          );
          await expect
            .poll(() =>
              chat.evaluate(() =>
                window.voiceStreams.every((stream) =>
                  stream
                    .getTracks()
                    .every((track) => track.readyState === "ended"),
                ),
              ),
            )
            .toBe(true);
          assert.equal(backend.asrRequests.length, 1);
          await open();
        },
      );

      await t.test(
        "releasing during microphone permission waits discards the late stream",
        async () => {
          await chat.evaluate(() => {
            window.savedGetUserMedia = navigator.mediaDevices.getUserMedia;
            navigator.mediaDevices.getUserMedia = async (options) => {
              const stream = await window.savedGetUserMedia(options);
              return new Promise((resolve) => {
                window.releaseVoiceGrant = () => resolve(stream);
              });
            };
          });
          await chat.locator(".mic-button").hover();
          await chat.mouse.down();
          await chat.waitForFunction(
            () => typeof window.releaseVoiceGrant === "function",
          );
          await chat.mouse.up();
          await chat.evaluate(() => {
            window.releaseVoiceGrant();
            navigator.mediaDevices.getUserMedia = window.savedGetUserMedia;
          });
          await expect
            .poll(async () => (await state()).speech.capture)
            .toBe(null);
          await expect
            .poll(() =>
              chat.evaluate(() =>
                window.voiceStreams
                  .at(-1)
                  .getTracks()
                  .every((track) => track.readyState === "ended"),
              ),
            )
            .toBe(true);
          assert.equal(backend.asrRequests.length, 1);
        },
      );

      await t.test(
        "microphone denial and provider failure preserve drafts and allow retry",
        async () => {
          await chat.evaluate(() => {
            window.savedGetUserMedia = navigator.mediaDevices.getUserMedia;
            navigator.mediaDevices.getUserMedia = async () => {
              throw new DOMException("Denied", "NotAllowedError");
            };
          });
          await chat.locator(".mic-button").hover();
          await chat.mouse.down();
          await expect(chat.getByRole("alert")).toContainText("系统设置");
          await chat.mouse.up();
          await chat.evaluate(() => {
            navigator.mediaDevices.getUserMedia = window.savedGetUserMedia;
          });
          await expect(
            chat.getByRole("textbox", { name: "消息", exact: true }),
          ).toHaveValue("已有草稿：语音输入的研究计划。");
          await chat
            .getByRole("button", { name: "关闭提示", exact: true })
            .click();
          backend.push("retry-speech", "用一句新的文字测试错误恢复。");
          backend.setSpeechError(true);
          const speaker = chat.locator(
            '[data-message-id="retry-speech"] .read-aloud',
          );
          await speaker.click();
          await expect(chat.getByRole("alert")).toContainText(
            "Mock speech provider unavailable",
          );
          backend.setSpeechError(false);
          await speaker.click();
          await expect
            .poll(async () => (await state()).speech.playback?.phase)
            .toBe("playing");
          await expect(chat.getByRole("alert")).toHaveCount(0);
          await speaker.click();
        },
      );

      await t.test(
        "switching sessions during transcription preserves the draft and discards late text",
        async () => {
          backend.setSpeechDelay(1200);
          const input = chat.getByRole("textbox", {
            name: "消息",
            exact: true,
          });
          const draft = await input.inputValue();
          const before = backend.asrRequests.length;
          const shortcut =
            process.platform === "darwin" ? "Meta+Shift+M" : "Control+Shift+M";
          await input.focus();
          await chat.keyboard.press(shortcut);
          await expect(chat.locator(".voice-status")).toContainText("录音中 1");
          await chat.keyboard.press(shortcut);
          await expect.poll(() => backend.asrRequests.length).toBe(before + 1);
          await expect
            .poll(async () => (await state()).speech.capture?.phase)
            .toBe("transcribing");
          await chat
            .getByRole("button", { name: "新建对话", exact: true })
            .click();
          await expect
            .poll(async () => (await state()).sessionId)
            .toBe("session-2");
          await expect
            .poll(async () => (await state()).speech.capture)
            .toBe(null);
          await expect(input).toHaveValue(draft);
          backend.setSpeechDelay(0);
        },
      );

      await t.test(
        "switching sessions cancels pending synthesis and rejects untrusted microphone access",
        async () => {
          backend.setSpeechDelay(2000);
          backend.push("late-speech", "这条旧会话的语音不应该迟到播放。");
          await chat
            .locator('[data-message-id="late-speech"] .read-aloud')
            .click();
          await expect
            .poll(async () => (await state()).speech.playback?.phase)
            .toBe("loading");
          await chat
            .getByRole("button", { name: "新建对话", exact: true })
            .click();
          await expect
            .poll(async () => (await state()).sessionId)
            .toBe("session-3");
          await expect
            .poll(async () => (await state()).speech.playback)
            .toBe(null);
          assert.equal(
            await companion.evaluate(async () => {
              try {
                const stream = await navigator.mediaDevices.getUserMedia({
                  audio: true,
                });
                stream.getTracks().forEach((track) => track.stop());
                return "allowed";
              } catch {
                return "denied";
              }
            }),
            "denied",
          );
          const stale = await companion.evaluate(() =>
            window.desktop.action({
              type: "speech.speak",
              sessionId: "superseded-session",
              messageId: "historical",
            }),
          );
          assert.equal(stale?.key, "sessionUnavailable");
        },
      );

      await t.test(
        "voice auto-send submits the composed draft and narrates the reply",
        async () => {
          backend.setSpeechDelay(0);
          await chat.evaluate(() =>
            window.desktop.action({
              type: "settings.update",
              voiceAutoSend: true,
              voiceEnabled: true,
              voiceReplies: true,
              quietUntil: 0,
            }),
          );
          const input = chat.getByRole("textbox", {
            name: "消息",
            exact: true,
          });
          await input.fill("语音：");
          await input.press("End");
          await chat.locator(".mic-button").hover();
          await chat.mouse.down();
          await expect(chat.locator(".voice-status")).toContainText("录音中 1");
          await chat.mouse.up();
          await expect
            .poll(async () =>
              (await state()).messages.some(
                (message) =>
                  message.role === "user" &&
                  message.content === "语音：语音输入的研究计划。",
              ),
            )
            .toBe(true);
          await expect(input).toHaveValue("");
          await expect
            .poll(async () => (await state()).speech.capture)
            .toBe(null);
          await expect
            .poll(async () => (await state()).speech.playback?.kind)
            .toBe("reply");
          await expect
            .poll(async () => (await state()).speech.playback?.phase)
            .toBe("playing");
        },
      );
      const input = chat.getByRole("textbox", { name: "消息", exact: true });
      await open();
      await chat.getByRole("tab", { name: "Remy", exact: true }).click();
      const modifier = process.platform === "darwin" ? "meta" : "control";
      const modifierKey = process.platform === "darwin" ? "Command" : "Control";
      // CDP keyboard events bypass Electron's before-input-event/menu pipeline.
      const nativeKey = (type, keyCode, modifiers = []) =>
        app.evaluate(
          ({ BrowserWindow }, input) => {
            BrowserWindow.getAllWindows()
              .find((win) => win.webContents.getURL().includes("view=chat"))
              .webContents.sendInputEvent(input);
          },
          { type, keyCode, modifiers },
        );
      const hold = async () => {
        await expect(input).toBeVisible();
        await input.focus();
        await nativeKey("keyDown", modifierKey, [modifier]);
        await nativeKey("keyDown", "R", [modifier]);
        await expect(chat.locator(".voice-status")).toContainText("录音中 1");
      };
      const release = async () => {
        await nativeKey("keyUp", "R", [modifier]);
        await nativeKey("keyUp", modifierKey);
      };
      await chat.evaluate(() =>
        window.desktop.action({
          type: "settings.update",
          voiceAutoSend: false,
        }),
      );
      await t.test(
        "holding Cmd/Ctrl R captures once and releasing the modifier first still stops without reload",
        async () => {
          await input.fill("快捷语音：");
          await input.press("End");
          const before = backend.asrRequests.length;
          const marker = await chat.evaluate(
            () => (window.voiceTestMarker = crypto.randomUUID()),
          );
          await hold();
          const capture = (await state()).speech.capture.id;
          await nativeKey("keyDown", "R", [modifier, "isautorepeat"]);
          assert.equal((await state()).speech.capture.id, capture);
          await nativeKey("keyUp", modifierKey);
          await expect(input).toHaveValue("快捷语音：语音输入的研究计划。");
          await nativeKey("keyUp", "R");
          assert.equal(backend.asrRequests.length, before + 1);
          assert.equal(
            await chat.evaluate(() => window.voiceTestMarker),
            marker,
          );
        },
      );
      if (process.platform === "darwin")
        await t.test(
          "literal Control R also records on macOS without reloading",
          async () => {
            const before = backend.asrRequests.length;
            const marker = await chat.evaluate(() => window.voiceTestMarker);
            await input.fill("Control 快捷语音：");
            await input.press("End");
            await nativeKey("keyDown", "Control", ["control"]);
            await nativeKey("keyDown", "R", ["control"]);
            await expect(chat.locator(".voice-status")).toContainText(
              "录音中 1",
            );
            await nativeKey("keyUp", "Control");
            await nativeKey("keyUp", "R");
            await expect(input).toHaveValue(
              "Control 快捷语音：语音输入的研究计划。",
            );
            assert.equal(backend.asrRequests.length, before + 1);
            assert.equal(
              await chat.evaluate(() => window.voiceTestMarker),
              marker,
            );
          },
        );
      await t.test(
        "an initializing audio host explains readiness instead of blaming the conversation",
        async () => {
          const before = backend.asrRequests.length;
          await chat.evaluate(async () => {
            await window.desktop.action({
              type: "speech.ready",
              ready: true,
              token: "readiness-probe",
            });
            await window.desktop.action({
              type: "speech.ready",
              ready: false,
              token: "readiness-probe",
            });
          });
          await expect
            .poll(async () => (await state()).speech.ready)
            .toBe(false);
          await nativeKey("keyDown", modifierKey, [modifier]);
          await nativeKey("keyDown", "R", [modifier]);
          await expect(chat.getByRole("alert")).toContainText(
            "语音功能正在准备",
          );
          await release();
          assert.equal(backend.asrRequests.length, before);
          await chat.evaluate(() =>
            window.desktop.action({
              type: "speech.ready",
              ready: true,
              token: "readiness-recovered",
            }),
          );
          await expect
            .poll(async () => (await state()).speech.ready)
            .toBe(true);
        },
      );
      await t.test(
        "silence is discarded locally and empty recognition never auto-sends the existing draft",
        async () => {
          const draft = await input.inputValue();
          const before = backend.asrRequests.length;
          await chat.evaluate(() => (window.voiceAmplitude = 0));
          await hold();
          await release();
          await expect(chat.locator(".voice-notice")).toHaveText(
            "未检测到语音",
          );
          await expect(chat.getByRole("alert")).toHaveCount(0);
          await expect
            .poll(async () => (await state()).speech.capture)
            .toBe(null);
          assert.equal(backend.asrRequests.length, before);
          await chat.evaluate(() => (window.voiceAmplitude = 0.003));
          backend.setAsrText("");
          await chat.evaluate(() =>
            window.desktop.action({
              type: "settings.update",
              voiceAutoSend: true,
            }),
          );
          const users = (await state()).messages.filter(
            (message) => message.role === "user",
          ).length;
          await hold();
          await release();
          await expect.poll(() => backend.asrRequests.length).toBe(before + 1);
          await expect(chat.locator(".voice-notice")).toHaveText(
            "未检测到语音",
          );
          await expect(input).toHaveValue(draft);
          assert.equal(
            (await state()).messages.filter(
              (message) => message.role === "user",
            ).length,
            users,
          );
          await expect(chat.getByRole("alert")).toHaveCount(0);
          await chat.screenshot({
            path: path.join(project, "reports/voice-no-speech.png"),
          });
          await expect(chat.locator(".voice-notice")).toHaveCount(0);
          backend.setAsrText("语音输入的研究计划。");
          await chat.evaluate(() => {
            window.voiceAmplitude = 1;
          });
        },
      );
      await t.test(
        "early key release, Escape and switching pages cancel held capture and release tracks",
        async () => {
          const before = backend.asrRequests.length;
          await chat.evaluate(() => (window.voiceDelay = 500));
          await nativeKey("keyDown", modifierKey, [modifier]);
          await nativeKey("keyDown", "R", [modifier]);
          await expect
            .poll(async () => (await state()).speech.capture?.phase)
            .toBe("starting");
          await release();
          await expect
            .poll(async () => (await state()).speech.capture)
            .toBe(null);
          await chat.evaluate(() => (window.voiceDelay = 0));
          await hold();
          await chat.keyboard.press("Escape");
          await release();
          await expect
            .poll(async () => (await state()).speech.capture)
            .toBe(null);
          await hold();
          await chat.getByRole("tab", { name: "知识库", exact: true }).click();
          await release();
          await expect
            .poll(async () => (await state()).speech.capture)
            .toBe(null);
          await expect
            .poll(() =>
              chat.evaluate(() =>
                window.voiceStreams.every((stream) =>
                  stream
                    .getTracks()
                    .every((track) => track.readyState === "ended"),
                ),
              ),
            )
            .toBe(true);
          assert.equal(backend.asrRequests.length, before);
          await chat.getByRole("tab", { name: "Remy", exact: true }).click();
          await app.evaluate(({ app, BrowserWindow }) => {
            app.focus({ steal: true });
            BrowserWindow.getAllWindows()
              .find((win) => win.webContents.getURL().includes("view=chat"))
              .focus();
          });
          await expect.poll(async () => (await state()).chatFocused).toBe(true);
          await hold();
          await app.evaluate(({ BrowserWindow }) =>
            BrowserWindow.getAllWindows()
              .find((win) => win.webContents.getURL().includes("view=chat"))
              .blur(),
          );
          await expect
            .poll(async () => (await state()).speech.capture)
            .toBe(null);
          await release();
          assert.equal(backend.asrRequests.length, before);
        },
      );
      assert.deepEqual(errors, []);
    } finally {
      await app?.close();
      await backend.close();
      await rm(directory, { recursive: true, force: true });
    }
  },
);
