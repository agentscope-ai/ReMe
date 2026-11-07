import assert from "node:assert/strict";
import test from "node:test";
import { SpeechController } from "../src/main/speech.ts";
import { ReMeClient } from "../src/main/reme-client.ts";
import { defaultSettings } from "../src/main/storage.ts";
import {
  encodeWave,
  narrationParts,
  narrationText,
  validRecording,
} from "../src/shared/speech.ts";
import type {
  DesktopCommand,
  SessionMessage,
  SpeechState,
} from "../src/shared/protocol.ts";

const message = (id = "m", content = "A complete reply."): SessionMessage => ({
  id,
  content,
  role: "assistant",
  completed: true,
  source: "chat",
  turn_id: "turn",
  chunks: [],
});
const audio = {
  audio: Buffer.from("audio").toString("base64"),
  format: "mp3",
  sample_rate: 22050,
  is_last: true,
};
const completed = {
  session_id: "s",
  turn_id: "turn",
  source: "chat",
  phase: "completed",
} as const;
const flush = async () => {
  for (let i = 0; i < 5; i++)
    await new Promise((resolve) => setImmediate(resolve));
};

function harness(call: ReMeClient["call"] = async () => audio as never) {
  const context = {
    sessionId: "s",
    connected: true,
    chatVisible: true,
    visible: true,
    settings: { ...defaultSettings, voiceEnabled: true },
  };
  const commands: DesktopCommand[] = [];
  const client = new ReMeClient("http://test.invalid");
  client.call = call;
  const speech = new SpeechController(
    () => context,
    () => client,
    () => {},
    (command) => commands.push(command),
  );
  speech.configure(true, true, "provider");
  speech.rendererReady(true);
  return { speech, context, commands };
}

test("narration removes markup and code, preserves Unicode, and bounds automatic speech", () => {
  assert.equal(
    narrationText(
      "## Hello **friend** [paper](https://example.com)\n```js\nsecret()\n```",
    ),
    "Hello friend paper",
  );
  assert.deepEqual(narrationParts("```js\nonlyCode()\n```"), []);
  const long = "你好。🙂".repeat(400);
  assert.equal(narrationParts(long).join(""), long);
  assert.ok(narrationParts(long).every((part) => [...part].length <= 400));
  assert.equal([...narrationParts(long, true).join("")].length, 601);
});

test("recording validation rejects escapes in format, lengths, channels, rates and size", () => {
  const wave = encodeWave(new Int16Array(16000));
  assert.equal(validRecording(wave), true);
  assert.equal(validRecording(wave.subarray(0, 100)), false);
  for (const offset of [0, 4, 8, 12, 16, 20, 22, 24, 28, 32, 34, 36, 40]) {
    const invalid = wave.slice();
    invalid[offset] ^= 1;
    assert.equal(validRecording(invalid), false, String(offset));
  }
  assert.equal(validRecording(encodeWave(new Int16Array(500_000))), false);
});

test("only new successful turns narrate once; snapshots, failed turns, quiet and disabled updates stay silent", async () => {
  let calls = 0;
  const { speech, context, commands } = harness(async () => {
    calls++;
    return audio as never;
  });
  speech.baseline([message("old")]);
  speech.observe(message("old"));
  speech.finish(completed);
  speech.observe(message("failed"));
  speech.finish({ ...completed, phase: "failed" });
  context.settings.quietUntil = Date.now() + 60_000;
  speech.observe(message("quiet"));
  speech.finish(completed);
  context.settings.quietUntil = 0;
  speech.observe(message("quiet"));
  speech.finish(completed);
  context.settings.voiceEnabled = false;
  speech.observe(message("disabled"));
  speech.finish(completed);
  context.settings.voiceEnabled = true;
  speech.observe(message("live"));
  assert.equal(
    calls,
    0,
    "a completed message still waits for successful turn completion",
  );
  speech.finish(completed);
  await flush();
  assert.equal(calls, 1);
  assert.equal(
    commands.filter((command) => command.type === "speech.play").length,
    1,
  );
  const id = speech.state.playback!.id;
  speech.report(id, "started");
  speech.report(id, "ended");
  speech.observe(message("live"));
  speech.finish(completed);
  await flush();
  assert.equal(speech.state.playback, null);
  await speech.close();
});

test("long manual narration advances only on ended, ignores stale reports, and reuses bounded cache", async () => {
  let calls = 0;
  const { speech } = harness(async () => {
    calls++;
    return audio as never;
  });
  const long = message("long", "Sentence. ".repeat(100));
  speech.speak(long);
  await flush();
  let id = speech.state.playback!.id;
  assert.equal(speech.state.playback!.phase, "loading");
  speech.report(id, "ended");
  assert.equal(calls, 1);
  speech.report(id, "started");
  assert.equal(speech.state.playback!.phase, "playing");
  speech.report(id, "ended");
  await flush();
  assert.equal(speech.state.playback!.part, 2);
  speech.report(id, "failed");
  assert.equal(speech.state.error, null);
  while (speech.state.playback) {
    id = speech.state.playback.id;
    speech.report(id, "started");
    speech.report(id, "ended");
    await flush();
  }
  const before = calls;
  speech.speak(long);
  await flush();
  assert.equal(calls, before);
  speech.stop();
  speech.configure(true, true, "changed-voice");
  speech.speak(long);
  await flush();
  assert.equal(calls, before + 1);
  await speech.close();
});

test("replaced or cancelled synthesis never plays a late result", async () => {
  let resolve!: (value: never) => void;
  const { speech, commands } = harness(
    async () =>
      new Promise((done) => {
        resolve = done;
      }),
  );
  speech.speak(message());
  speech.stop();
  resolve(audio as never);
  await flush();
  assert.equal(
    commands.some((command) => command.type === "speech.play"),
    false,
  );
  assert.equal(speech.state.playback, null);
  await speech.close();
});

test("manual playback bypasses quiet mode; automatic playback stops immediately when quiet or hidden", async () => {
  const { speech, context } = harness();
  context.settings.quietUntil = Date.now() + 60_000;
  context.settings.voiceEnabled = false;
  speech.speak(message());
  await flush();
  speech.tick();
  assert.equal(
    (speech.state.playback as SpeechState["playback"])?.kind,
    "manual",
  );
  speech.stop();
  context.settings.quietUntil = 0;
  context.settings.voiceEnabled = true;
  speech.speak(message(), "reply");
  await flush();
  context.settings.quietUntil = Date.now() + 60_000;
  speech.tick();
  assert.equal(speech.state.playback, null);
  context.settings.quietUntil = 0;
  speech.speak(message(), "reply");
  await flush();
  context.visible = false;
  speech.tick();
  assert.equal(speech.state.playback, null);
  await speech.close();
});

test("transcription accepts only the active recording and discards late results after session reset", async () => {
  let resolve!: (value: never) => void;
  let sent: object | undefined;
  const { speech } = harness(async (_job, body) => {
    sent = body;
    return new Promise((done) => {
      resolve = done;
    });
  });
  speech.beginCapture("capture");
  speech.recording("capture");
  await assert.rejects(
    speech.transcribe("other", encodeWave(new Int16Array(20))),
    /取消/,
  );
  const result = speech.transcribe("capture", encodeWave(new Int16Array(20)));
  assert.equal(speech.state.capture?.phase, "transcribing");
  assert.equal((sent as { audio_format: string }).audio_format, "wav");
  speech.reset();
  resolve("late transcript" as never);
  await assert.rejects(result, /取消/);
  assert.equal(speech.state.capture, null);
  await speech.close();
});

test("missing capabilities degrade gracefully and invalid synthesis remains retryable", async () => {
  const { speech } = harness(
    async () => ({ ...audio, is_last: false }) as never,
  );
  speech.speak(message());
  await flush();
  assert.equal(speech.state.playback, null);
  assert.ok(
    speech.state.error &&
      "key" in speech.state.error &&
      speech.state.error.key === "speechInvalidAudio",
  );
  speech.configure(false, false, "none");
  assert.throws(() => speech.beginCapture("a"), /未启用语音识别/);
  assert.throws(() => speech.speak(message()), /未启用语音合成/);
  speech.speak(message(), "reply");
  await speech.close();
});

test("unacknowledged audio and stalled capture time out instead of holding the UI forever", async () => {
  const { speech } = harness();
  speech.speak(message());
  await flush();
  speech.tick(Date.now() + 11_000);
  assert.equal(speech.state.playback, null);
  assert.ok(speech.state.error);
  speech.beginCapture("a");
  speech.tick(Date.now() + 61_000);
  assert.equal(speech.state.capture, null);
  await speech.close();
});

test("recording failure frees capture, while a new remote turn interrupts obsolete automatic audio", async () => {
  const { speech } = harness();
  speech.beginCapture("invalid");
  speech.recording("invalid");
  await assert.rejects(
    speech.transcribe("invalid", new Uint8Array([0])),
    /录音格式/,
  );
  assert.equal(speech.state.capture, null);
  speech.speak(message(), "reply");
  await flush();
  speech.finish({ ...completed, turn_id: "next", phase: "thinking" });
  assert.equal(speech.state.playback, null);
  speech.speak(message());
  await flush();
  speech.finish({ ...completed, turn_id: "next", phase: "thinking" });
  assert.equal(
    (speech.state.playback as SpeechState["playback"])?.kind,
    "manual",
  );
  await speech.close();
});

test("oversized speech responses are cancelled before JSON decoding", async (t) => {
  let cancelled = false;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new Uint8Array(101));
    },
    cancel() {
      cancelled = true;
    },
  });
  t.mock.method(globalThis, "fetch", async () => new Response(stream));
  const client = new ReMeClient("http://test.invalid");
  await assert.rejects(
    client.call("tts_demo", {}, new AbortController().signal, {
      maxBytes: 100,
    }),
    /语音结果过大/,
  );
  assert.equal(cancelled, true);
});
