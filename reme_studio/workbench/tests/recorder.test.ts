import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "vite";
import type { Recorder } from "../src/speech/recorder.ts";

class Track extends EventTarget {
  readyState = "live";
  enabled = true;
  muted = false;
  stops = 0;
  stop() {
    this.readyState = "ended";
    this.stops++;
  }
}

const turn = () => new Promise<void>((resolve) => setImmediate(resolve));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

test("microphone readiness and recording resource lifetime", async (t) => {
  const server = await createServer({
    configFile: false,
    root: fileURLToPath(new URL("../../", import.meta.url)),
    server: { middlewareMode: true, watch: null, hmr: false, ws: false },
  });
  t.after(() => server.close());
  const { openRecorder } = (await server.ssrLoadModule(
    "/workbench/src/speech/recorder.ts",
  )) as {
    openRecorder(
      signal: AbortSignal,
      progress: (seconds: number, level: number) => void,
    ): Promise<Recorder>;
  };

  function setup(
    options: {
      permission?: Promise<unknown>;
      module?: Promise<void>;
      muted?: boolean;
      resume?: Promise<void>;
    } = {},
  ) {
    const track = new Track();
    track.muted = options.muted ?? false;
    const stream = { getTracks: () => [track], getAudioTracks: () => [track] };
    const order: string[] = [];
    let node: Node;
    let context: Context;
    class Node {
      disconnected = false;
      port = {
        onmessage: null as ((event: { data: Float32Array }) => void) | null,
        closed: false,
        close() {
          this.closed = true;
        },
      };
      constructor() {
        node = this;
      }
      connect(value: unknown) {
        return value;
      }
      disconnect() {
        this.disconnected = true;
      }
      emit(samples: Float32Array) {
        this.port.onmessage?.({ data: samples });
      }
    }
    class Context {
      state = "suspended";
      closed = false;
      destination = {};
      audioWorklet = {
        addModule: async () => {
          order.push("module");
          await options.module;
        },
      };
      constructor() {
        context = this;
      }
      createMediaStreamSource() {
        return { connect: (value: unknown) => value };
      }
      createGain() {
        return { gain: { value: 1 }, connect: () => {} };
      }
      async resume() {
        await options.resume;
        this.state = "running";
      }
      async close() {
        this.closed = true;
        this.state = "closed";
      }
    }
    const globals = {
      navigator: {
        mediaDevices: {
          getUserMedia: async () => {
            order.push("microphone");
            await options.permission;
            return stream;
          },
        },
      },
      AudioContext: Context,
      AudioWorkletNode: Node,
    };
    const descriptors = new Map(
      Object.keys(globals).map((name) => [
        name,
        Object.getOwnPropertyDescriptor(globalThis, name),
      ]),
    );
    for (const [name, value] of Object.entries(globals))
      Object.defineProperty(globalThis, name, { configurable: true, value });
    const controller = new AbortController();
    let ready = false;
    const opening = openRecorder(controller.signal, () => {}).then((value) => {
      ready = true;
      return value;
    });
    return {
      track,
      order,
      controller,
      opening,
      get node() {
        return node!;
      },
      get context() {
        return context!;
      },
      get ready() {
        return ready;
      },
      restore() {
        for (const [name, descriptor] of descriptors)
          if (descriptor) Object.defineProperty(globalThis, name, descriptor);
          else Reflect.deleteProperty(globalThis, name);
      },
    };
  }

  await t.test(
    "waits for PCM and preserves the very first batch",
    async (t) => {
      const recording = setup();
      t.after(() => recording.restore());
      await turn();
      assert.deepEqual(recording.order, ["module", "microphone"]);
      assert.equal(recording.context.state, "running");
      assert.equal(recording.ready, false);
      recording.node.emit(new Float32Array());
      await turn();
      assert.equal(recording.ready, false);
      recording.node.emit(new Float32Array(512).fill(0.5));
      const recorder = await recording.opening;
      recording.node.emit(new Float32Array(512).fill(-0.25));
      const clip = await recorder.stop();
      const pcm = new DataView(clip.audio.buffer, clip.audio.byteOffset);
      assert.equal(pcm.getInt16(44, true), 16383);
      assert.equal(pcm.getInt16(44 + 512 * 2, true), -8192);
      assert.equal(clip.duration, 64);
      assert.equal(clip.audible, true);
      assert.equal(recording.track.stops, 1);
      assert.equal(recording.context.closed, true);
      assert.equal(recording.node.port.closed, true);
    },
  );

  await t.test(
    "silent samples are ready without waiting for the user to speak",
    async (t) => {
      const recording = setup();
      t.after(() => recording.restore());
      await turn();
      recording.node.emit(new Float32Array(512));
      const recorder = await recording.opening;
      recorder.cancel();
      assert.equal(recording.track.stops, 1);
    },
  );

  await t.test(
    "requires samples from an unmuted track and a running context",
    async (t) => {
      const resume = deferred<void>();
      const recording = setup({ muted: true, resume: resume.promise });
      t.after(() => recording.restore());
      await turn();
      recording.node.emit(new Float32Array(512));
      resume.resolve();
      await turn();
      assert.equal(recording.ready, false);
      recording.track.muted = false;
      recording.track.dispatchEvent(new Event("unmute"));
      await turn();
      assert.equal(recording.ready, false);
      recording.node.emit(new Float32Array(512));
      (await recording.opening).cancel();
    },
  );

  await t.test(
    "cancelling before the first frame rejects and releases all resources",
    async (t) => {
      const recording = setup();
      t.after(() => recording.restore());
      const rejected = assert.rejects(recording.opening, {
        name: "AbortError",
      });
      await turn();
      recording.controller.abort();
      await rejected;
      assert.equal(recording.track.stops, 1);
      assert.equal(recording.context.closed, true);
      assert.equal(recording.node.disconnected, true);
    },
  );

  await t.test(
    "a microphone granted after cancellation is immediately stopped",
    async (t) => {
      const permission = deferred<unknown>();
      const recording = setup({ permission: permission.promise });
      t.after(() => recording.restore());
      const rejected = assert.rejects(recording.opening, {
        name: "AbortError",
      });
      await turn();
      recording.controller.abort();
      await rejected;
      permission.resolve(undefined);
      await turn();
      assert.equal(recording.track.stops, 1);
      assert.equal(recording.context.closed, true);
      assert.equal(recording.node, undefined);
    },
  );

  await t.test(
    "a track ending before readiness fails immediately and cleans up",
    async (t) => {
      const recording = setup();
      t.after(() => recording.restore());
      const rejected = assert.rejects(
        recording.opening,
        (error: { key?: string }) => error.key === "speechMicFailed",
      );
      await turn();
      recording.track.readyState = "ended";
      recording.track.dispatchEvent(new Event("ended"));
      await rejected;
      assert.equal(recording.ready, false);
      assert.equal(recording.context.closed, true);
      assert.equal(recording.node.port.closed, true);
    },
  );

  await t.test(
    "cancelling module preparation never opens the microphone",
    async (t) => {
      const module = deferred<void>();
      const recording = setup({ module: module.promise });
      t.after(() => recording.restore());
      const rejected = assert.rejects(recording.opening, {
        name: "AbortError",
      });
      recording.controller.abort();
      await rejected;
      module.resolve();
      await turn();
      assert.deepEqual(recording.order, ["module"]);
      assert.equal(recording.context.closed, true);
    },
  );

  await t.test(
    "missing audio times out and cleans up instead of claiming readiness",
    async (t) => {
      t.mock.timers.enable({ apis: ["setTimeout"] });
      const recording = setup();
      t.after(() => recording.restore());
      const rejected = assert.rejects(
        recording.opening,
        (error: { key?: string }) => error.key === "speechMicTimeout",
      );
      await turn();
      t.mock.timers.tick(30_000);
      await rejected;
      assert.equal(recording.ready, false);
      assert.equal(recording.track.stops, 1);
      assert.equal(recording.context.closed, true);
    },
  );

  await t.test(
    "permission denial closes the prepared processing context",
    async (t) => {
      const permission = deferred<unknown>();
      const recording = setup({
        permission: permission.promise.then(() => {
          throw new DOMException("denied", "NotAllowedError");
        }),
      });
      t.after(() => recording.restore());
      const rejected = assert.rejects(
        recording.opening,
        (error: { key?: string }) => error.key === "speechMicDenied",
      );
      permission.resolve(undefined);
      await rejected;
      assert.equal(recording.context.closed, true);
    },
  );
});
