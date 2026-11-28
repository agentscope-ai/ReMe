import assert from "node:assert/strict";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import type { BackendSettings } from "../../client/src/backend.ts";
import type { StudioState } from "../../client/src/protocol.ts";
import { defaultSettings } from "../../client/src/settings.ts";

const config: BackendSettings = {
  workspace_dir: "/tmp/speech-settings-test",
  timezone: "UTC",
  language: "zh",
  harness: "codex",
  base_url: "https://model.example",
  model: "model",
  effort: null,
  asr_enabled: false,
  tts_enabled: false,
  speech_base_url: "https://speech.example/api/v1",
  asr_model: "recognition-model",
  tts_model: "synthesis-model",
  tts_voice: "voice-1",
  revision: 1,
  initialized: true,
  settings_error: null,
  model_editable: true,
  available_harnesses: ["codex"],
  api_key_set: true,
  dashscope_api_key_set: true,
  dashscope_editable: true,
  asr_available: true,
  tts_available: true,
  missing_fields: [],
  defaults: {
    speech_base_url: "https://speech.example/api/v1",
    asr_model: "recognition-model",
    tts_model: "synthesis-model",
    tts_voice: "voice-1",
  },
};

test("speech settings separate service capabilities from device playback preferences", async (t) => {
  const server = await createServer({
    configFile: false,
    root: fileURLToPath(new URL("../../", import.meta.url)),
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  t.after(() => server.close());
  const { default: SpeechServiceSettings } = await server.ssrLoadModule(
    "/workbench/src/speech/service-settings.tsx",
  );
  const { default: VoiceSettings } = await server.ssrLoadModule(
    "/workbench/src/speech/settings.tsx",
  );
  const { HostContext } = await server.ssrLoadModule("/workbench/src/host.ts");
  const service = (changes: Partial<BackendSettings>) =>
    renderToStaticMarkup(
      createElement(
        "fieldset",
        {},
        createElement(SpeechServiceSettings, {
          language: "zh",
          config: { ...config, ...changes },
          draft: { ...config, ...changes },
          apiKey: "",
          onKeyChange: () => {},
          onChange: () => {},
          onReset: () => {},
        }),
      ),
    );
  const preferences = (tts: boolean, automatic: boolean, disabled = false) =>
    renderToStaticMarkup(
      createElement(
        HostContext.Provider,
        { value: {} },
        createElement(VoiceSettings, {
          state: {
            settings: {
              ...defaultSettings,
              language: "zh",
              voiceEnabled: automatic,
            },
            speech: { asr: true, tts },
          } as StudioState,
          run: async () => {},
          disabled,
        }),
      ),
    );

  const input = (html: string, value: string) => {
    const tag = html
      .match(/<input\b[^>]*>/g)
      ?.find((tag) => tag.includes(`value="${value}"`));
    assert.ok(tag, `Missing input for ${value}`);
    return tag;
  };

  await t.test(
    "disabled services keep all configuration visible and optional",
    () => {
      const html = service({});
      assert.equal((html.match(/role="switch"/g) ?? []).length, 2);
      for (const value of [
        "recognition-model",
        "synthesis-model",
        "voice-1",
        config.speech_base_url!,
      ])
        assert.doesNotMatch(input(html, value), /required=/);
      assert.match(html, /type="password"/);
      assert.doesNotMatch(html, /<details\b|<summary\b/);
    },
  );
  await t.test(
    "each service requires only its enabled model and voice fields",
    () => {
      const recognition = service({ asr_enabled: true });
      assert.match(input(recognition, "recognition-model"), /required=/);
      assert.doesNotMatch(input(recognition, "synthesis-model"), /required=/);
      assert.doesNotMatch(input(recognition, "voice-1"), /required=/);
      const synthesis = service({ tts_enabled: true });
      assert.match(input(synthesis, "synthesis-model"), /required=/);
      assert.match(input(synthesis, "voice-1"), /required=/);
      assert.doesNotMatch(input(synthesis, "recognition-model"), /required=/);
    },
  );
  await t.test(
    "saved credentials remain blank and the key hint appears once",
    () => {
      const html = service({ asr_enabled: true, tts_enabled: true });
      const key = html.match(/<input[^>]*type="password"[^>]*>/)?.[0];
      assert.ok(key);
      assert.match(key, /value=""/);
      assert.doesNotMatch(key, /required=/);
      assert.equal((html.match(/已设置，留空以保留/g) ?? []).length, 1);
      assert.match(input(html, config.speech_base_url!), /required=/);
      assert.doesNotMatch(html, /<details\b|<summary\b/);
      const missingKey = service({
        asr_enabled: true,
        dashscope_api_key_set: false,
      });
      assert.match(
        missingKey.match(/<input[^>]*type="password"[^>]*>/)![0],
        /required=/,
      );
    },
  );
  await t.test(
    "manual playback volume remains available with automatic narration off",
    () => {
      const html = preferences(true, false);
      const range = html.match(/<input[^>]*type="range"[^>]*>/)?.[0];
      assert.ok(range);
      assert.doesNotMatch(range, /disabled=/);
      for (const label of ["朗读新的对话回复", "朗读主动提醒"]) {
        const control = html
          .match(/<button\b[^>]*>/g)
          ?.find((tag) => tag.includes(`aria-label="${label}"`));
        assert.ok(control);
        assert.match(control, /disabled=/);
      }
      assert.match(html, /手动点击朗读/);
      assert.match(html, /录音方式与快捷键/);
      assert.doesNotMatch(html, /<details\b|<summary\b/);
    },
  );
  await t.test(
    "automatic narration exposes its reply and reminder choices",
    () => {
      const html = preferences(true, true);
      assert.match(html, /朗读新的对话回复/);
      assert.match(html, /朗读主动提醒/);
      assert.equal((html.match(/role="switch"/g) ?? []).length, 4);
    },
  );
  await t.test(
    "unavailable synthesis and backend restarts disable playback changes",
    () => {
      for (const html of [
        preferences(false, false),
        preferences(true, true, true),
      ]) {
        const range = html.match(/<input[^>]*type="range"[^>]*>/)?.[0];
        assert.ok(range);
        assert.match(range, /disabled=/);
      }
      assert.match(preferences(false, false), /语音合成暂不可用/);
    },
  );
});
