import assert from "node:assert/strict";
import test from "node:test";
import { SpeechController } from "../src/main/speech.ts";
import { ReMeClient } from "../src/main/reme-client.ts";
import { defaultSettings } from "../src/main/storage.ts";
import {
  encodeWave,
  hasAudibleSignal,
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
  delivery: "idle",
  notification_eligible: true,
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

test("narration removes markup and code and preserves full Unicode text in bounded parts", () => {
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
  assert.ok([...narrationParts(long).join("")].length > 600);
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

test("voice readiness has a dedicated error and stale renderer cleanup cannot disable the current host", async () => {
  const { speech } = harness();
  speech.rendererReady(false);
  assert.equal(speech.state.ready, false);
  assert.throws(
    () => speech.beginCapture("not-ready"),
    (error: unknown) =>
      error instanceof Error &&
      "key" in error &&
      error.key === "speechNotReady",
  );
  speech.rendererReady(true, "old-host");
  speech.rendererReady(true, "new-host");
  speech.beginCapture("current");
  speech.rendererReady(false, "old-host");
  assert.equal(speech.state.ready, true);
  assert.equal(speech.state.capture?.id, "current");
  speech.rendererReady(false, "new-host");
  assert.equal(speech.state.ready, false);
  assert.equal(speech.state.capture, null);
  await speech.close();
});

test("the local energy gate rejects silence and a click but retains sustained quiet audio", () => {
  assert.equal(hasAudibleSignal(new Int16Array()), false);
  const silence = new Int16Array(16000);
  assert.equal(hasAudibleSignal(silence), false);
  silence[8000] = 32767;
  assert.equal(hasAudibleSignal(silence), false);
  const tone = (amplitude: number) =>
    Int16Array.from(
      { length: 16000 },
      (_, i) => Math.sin((i * Math.PI * 2 * 440) / 16000) * amplitude * 32767,
    );
  assert.equal(hasAudibleSignal(tone(0.0001)), false);
  assert.equal(hasAudibleSignal(tone(0.003)), true);
  assert.equal(hasAudibleSignal(tone(0.5)), true);
});

test("an empty transcript finishes capture normally instead of raising a service error", async () => {
  for (const text of ["", "  \n"]) {
    const { speech } = harness(async () => text as never);
    speech.beginCapture("empty");
    speech.recording("empty");
    assert.equal(
      await speech.transcribe("empty", encodeWave(new Int16Array(16000))),
      "",
    );
    assert.equal(speech.state.capture, null);
    assert.equal(speech.state.error, null);
    await speech.close();
  }
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

const longReply = () =>
  Array.from(
    { length: 8 },
    (_, i) =>
      `第${i + 1}段。` +
      `这个段落保留自己的顺序和完整内容，数字是${i + 1}。`.repeat(12),
  ).join("\n\n") + "\n\n最后一句必须完整读到这里。";
const noSpace = (text: string) => text.replace(/\s+/g, "");
const finishPart = (speech: SpeechController) => {
  const id = speech.state.playback!.id;
  speech.report(id, "started");
  speech.report(id, "ended");
};

test("paragraphs, lists, decimals, versions and a long Unicode sentence survive narration", () => {
  const source =
    `# 研究结果\n\n- Sonnet 4.6，跌幅 12.25%，样本 1,000。\n- Opus 5.5。\n\n` +
    "长句🙂".repeat(300);
  const parts = narrationParts(source);
  assert.ok(parts.every((part) => [...part].length <= 400));
  assert.equal(noSpace(parts.join("")), noSpace(narrationText(source)));
  for (const token of ["Sonnet 4.6", "12.25%", "1,000", "Opus 5.5"])
    assert.ok(
      parts.some((part) => part.includes(token)),
      token,
    );
  const boundary = narrationParts(
    "字".repeat(398) + "12.25% 版本4.7仍然完整。" + "尾".repeat(100),
  );
  assert.ok(boundary.some((part) => part.includes("12.25%")));
  assert.ok(boundary.some((part) => part.includes("4.7")));
});

test("automatic long narration reaches the final sentence and cache stays reusable", async () => {
  const requested: string[] = [];
  const { speech, commands } = harness(async (_job, body) => {
    requested.push((body as { text: string }).text);
    return audio as never;
  });
  const source = longReply();
  const expected = narrationParts(source);
  speech.speak(message("long", source), "reply");
  await flush();
  assert.equal(speech.state.playback!.total, expected.length);
  assert.equal(
    requested.length,
    3,
    "one current block and two prefetched blocks",
  );
  for (let part = 1; part <= expected.length; part++) {
    assert.equal(speech.state.playback!.part, part);
    const id = speech.state.playback!.id;
    speech.report(id, "ended");
    assert.equal(
      speech.state.playback!.part,
      part,
      "ended before started is ignored",
    );
    finishPart(speech);
    speech.report(id, "failed");
    assert.equal(speech.state.error, null);
    await flush();
  }
  assert.equal(speech.state.playback, null);
  assert.deepEqual(requested, expected);
  assert.ok(requested.at(-1)!.endsWith("最后一句必须完整读到这里。"));
  assert.equal(
    commands.filter((item) => item.type === "speech.play").length,
    expected.length,
  );
  const before = requested.length;
  speech.speak(message("long", source));
  await flush();
  assert.equal(requested.length, before);
  speech.stop();
  speech.configure(true, true, "changed-voice");
  speech.speak(message("long", source));
  await flush();
  assert.equal(requested.length, before + 3);
  await speech.close();
});

test("synthesis is serial, primes two blocks, bounds the buffer and never plays past an unready block", async () => {
  const pending: Array<() => void> = [];
  let active = 0,
    peak = 0,
    calls = 0;
  const { speech, commands } = harness(async (_job, _body, signal) => {
    calls++;
    active++;
    peak = Math.max(peak, active);
    return await new Promise<never>((resolve, reject) => {
      pending.push(() => {
        active--;
        resolve(audio as never);
      });
      signal.addEventListener(
        "abort",
        () => {
          active--;
          reject(new Error("cancelled"));
        },
        { once: true },
      );
    });
  });
  speech.speak(message("long", longReply()));
  await flush();
  assert.equal(calls, 1);
  assert.equal(commands.length, 0);
  pending.shift()!();
  await flush();
  assert.equal(calls, 2);
  assert.equal(commands.length, 0);
  pending.shift()!();
  await flush();
  assert.equal(calls, 3);
  assert.equal(
    commands.filter((item) => item.type === "speech.play").length,
    1,
  );
  assert.equal(peak, 1);
  finishPart(speech);
  assert.equal(speech.state.playback!.part, 2);
  finishPart(speech);
  const waitingId = speech.state.playback!.id;
  assert.equal(speech.state.playback!.part, 3);
  speech.report(waitingId, "started");
  assert.equal(
    speech.state.playback!.phase,
    "loading",
    "unsent blocks cannot acknowledge playback",
  );
  pending.shift()!();
  await flush();
  assert.equal(speech.state.playback!.part, 3);
  assert.equal(peak, 1);
  pending.shift()!();
  await flush();
  pending.shift()!();
  await flush();
  assert.equal(calls, 5, "the producer stops after two ready future blocks");
  await speech.close();
});

test("prefetch failure retries once, preserves playable audio and resumes at the failed block", async () => {
  const parts = narrationParts(longReply());
  const requested: string[] = [];
  let failing = true;
  const { speech, commands } = harness(async (_job, body) => {
    const text = (body as { text: string }).text;
    requested.push(text);
    if (text === parts[2] && failing)
      throw new Error("temporary synthesis failure");
    return audio as never;
  });
  speech.speak(message("long", longReply()));
  await flush();
  assert.equal(requested.filter((text) => text === parts[2]).length, 2);
  assert.equal(
    speech.state.error,
    null,
    "future failure does not interrupt the current block",
  );
  finishPart(speech);
  finishPart(speech);
  assert.equal(speech.state.playback!.phase, "error");
  assert.equal(speech.state.playback!.part, 3);
  assert.ok(speech.state.error);
  const oldId = speech.state.playback!.id;
  failing = false;
  speech.retry();
  await flush();
  assert.equal(speech.state.playback!.part, 3);
  speech.report(oldId, "failed");
  assert.equal(speech.state.error, null);
  for (let i = 2; i < parts.length; i++) {
    finishPart(speech);
    await flush();
  }
  assert.equal(speech.state.playback, null);
  assert.equal(requested.filter((text) => text === parts[0]).length, 1);
  assert.equal(
    commands.filter((item) => item.type === "speech.play").length,
    parts.length,
  );
  await speech.close();
});

test("a decoded-audio failure invalidates its cache and retries the same block", async () => {
  let calls = 0;
  const { speech } = harness(async () => {
    calls++;
    return audio as never;
  });
  speech.speak(message());
  await flush();
  const id = speech.state.playback!.id;
  speech.report(id, "failed");
  assert.equal(speech.state.playback!.phase, "error");
  assert.equal(speech.state.playback!.part, 1);
  speech.retry();
  await flush();
  assert.equal(calls, 2);
  assert.notEqual(speech.state.playback!.id, id);
  finishPart(speech);
  assert.equal(speech.state.playback, null);
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

test("later automatic blocks preserve their kind and respect interruption policies", async () => {
  for (const kind of ["reply", "reminder"] as const) {
    for (const interruption of ["quiet", "hidden", "disabled", "turn"]) {
      const { speech, context } = harness();
      context.settings.voiceReminders = true;
      speech.speak(message("long", longReply()), kind);
      await flush();
      finishPart(speech);
      await flush();
      assert.equal(speech.state.playback?.part, 2);
      assert.equal(speech.state.playback?.kind, kind);
      if (interruption === "quiet")
        context.settings.quietUntil = Date.now() + 60_000;
      else if (interruption === "hidden") context.visible = false;
      else if (interruption === "disabled") {
        if (kind === "reply") context.settings.voiceReplies = false;
        else context.settings.voiceReminders = false;
      } else
        speech.finish({ ...completed, turn_id: "next", phase: "thinking" });
      speech.tick();
      assert.equal(speech.state.playback, null, `${kind}: ${interruption}`);
      await speech.close();
    }
  }
});

test("explicitly resuming automatic narration becomes manual playback", async () => {
  const { speech, context } = harness();
  speech.speak(message("long", longReply()), "reply");
  await flush();
  speech.report(speech.state.playback!.id, "failed");
  context.settings.quietUntil = Date.now() + 60_000;
  speech.retry();
  await flush();
  speech.tick();
  assert.equal(speech.state.playback?.kind, "manual");
  assert.equal(speech.state.playback?.part, 1);
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
  assert.equal(speech.state.playback?.phase, "error");
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
  assert.equal(speech.state.playback?.phase, "error");
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

test("immediate reminder narration bypasses quiet mode but preserves explicit voice controls", async () => {
  const { speech, context } = harness();
  context.settings.quietUntil = Date.now() + 60_000;
  context.settings.voiceReminders = true;
  const immediate = {
    ...message("urgent"),
    source: "proactive" as const,
    delivery: "immediate" as const,
  };
  speech.speak(immediate, "reminder");
  await flush();
  assert.equal(speech.state.playback?.messageId, "urgent");
  speech.tick();
  assert.equal(speech.state.playback?.messageId, "urgent");
  context.settings.voiceReminders = false;
  speech.tick();
  assert.equal(speech.state.playback, null);
  await speech.close();
});
