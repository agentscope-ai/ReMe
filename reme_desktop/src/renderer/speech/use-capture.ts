import { useEffect, useRef, useState } from "react";
import { deserializeError, UiError } from "../../shared/i18n.ts";
import { MAX_RECORD_MS, MIN_RECORD_MS } from "../../shared/speech.ts";
import type { DesktopState } from "../../shared/protocol.ts";
import { action } from "../desktop.ts";
import { openRecorder, type Recorder } from "./recorder.ts";

interface Capture {
  id: string;
  controller: AbortController;
  recorder?: Recorder;
  phase: "starting" | "recording" | "transcribing";
  bound: boolean;
  timer?: ReturnType<typeof setTimeout>;
}

export function useCapture(
  state: DesktopState,
  insert: (text: string) => void,
  fail: (error: unknown) => void,
) {
  const [phase, setPhase] = useState<Capture["phase"] | null>(null);
  const [seconds, setSeconds] = useState(0);
  const [level, setLevel] = useState(0);
  const active = useRef<Capture | null>(null);
  const latest = useRef({ state, insert, fail });
  latest.current = { state, insert, fail };

  const cancel = () => {
    const capture = active.current;
    active.current = null;
    if (capture) {
      clearTimeout(capture.timer);
      capture.controller.abort();
      capture.recorder?.cancel();
      void action({ type: "speech.cancel", id: capture.id }).catch(() => {});
    }
    setPhase(null);
    setLevel(0);
  };
  const finish = async () => {
    const capture = active.current;
    if (!capture || capture.phase === "transcribing") return;
    if (!capture.recorder) {
      cancel();
      return;
    }
    capture.phase = "transcribing";
    clearTimeout(capture.timer);
    setPhase("transcribing");
    try {
      const clip = await capture.recorder.stop();
      if (active.current !== capture) return;
      if (clip.duration < MIN_RECORD_MS) return;
      if (clip.peak < 0.002) throw new UiError("speechSilent");
      const result = await window.desktop.transcribe({
        id: capture.id,
        audio: clip.audio,
      });
      if (active.current !== capture) return;
      if (result.error) throw deserializeError(result.error);
      latest.current.insert(result.text);
    } catch (error) {
      if (
        active.current === capture &&
        !capture.controller.signal.aborted &&
        !(error instanceof UiError && error.key === "speechCancelled")
      )
        latest.current.fail(error);
    } finally {
      if (active.current === capture) cancel();
    }
  };
  const start = async () => {
    if (active.current || !latest.current.state.chatVisible) return;
    latest.current.fail(null);
    const capture: Capture = {
      id: crypto.randomUUID(),
      controller: new AbortController(),
      phase: "starting",
      bound: false,
    };
    active.current = capture;
    setSeconds(0);
    setPhase("starting");
    try {
      await action({ type: "speech.capture", id: capture.id });
      if (active.current !== capture) return;
      const recorder = await openRecorder(
        capture.controller.signal,
        (seconds, level) => {
          if (active.current !== capture) return;
          setSeconds(Math.floor(seconds));
          setLevel(level);
        },
      );
      if (active.current !== capture) {
        recorder.cancel();
        return;
      }
      capture.recorder = recorder;
      await action({ type: "speech.recording", id: capture.id });
      if (active.current !== capture) {
        recorder.cancel();
        return;
      }
      capture.phase = "recording";
      setPhase("recording");
      capture.timer = setTimeout(() => void finish(), MAX_RECORD_MS);
    } catch (error) {
      if (active.current === capture && !capture.controller.signal.aborted) {
        latest.current.fail(error);
        cancel();
      }
    }
  };
  useEffect(() => {
    const capture = active.current;
    if (!capture) return;
    if (state.speech.capture?.id === capture.id) capture.bound = true;
    else if (capture.bound && capture.phase !== "transcribing") cancel();
  }, [state.speech.capture]);
  useEffect(() => {
    cancel();
  }, [state.sessionId, state.settings.endpoint]);
  useEffect(() => {
    if (!state.chatVisible || state.connection !== "connected") cancel();
  }, [state.chatVisible, state.connection]);
  useEffect(
    () => () => {
      cancel();
    },
    [],
  );
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (!latest.current.state.chatVisible || event.repeat) return;
      if (event.key === "Escape" && active.current) {
        event.preventDefault();
        cancel();
      } else if (
        (event.metaKey || event.ctrlKey) &&
        event.shiftKey &&
        event.code === "KeyM"
      ) {
        event.preventDefault();
        if (active.current) void finish();
        else void start();
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);
  return {
    phase,
    seconds,
    level,
    start: () => void start(),
    finish: () => void finish(),
    cancel,
    toggle: () => (active.current ? void finish() : void start()),
  };
}
