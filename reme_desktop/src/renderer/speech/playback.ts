import { useEffect, useRef } from "react";
import { encodeWave } from "../../shared/speech.ts";
import type { DesktopState } from "../../shared/protocol.ts";
import { action } from "../desktop.ts";

/** The chat renderer is the sole audio host, including while its window is hidden. */
export function useSpeechPlayback(state: DesktopState): void {
  const current = useRef<{
    id: number;
    audio: HTMLAudioElement;
    url: string;
  } | null>(null);
  const volume = useRef(state.settings.voiceVolume);
  volume.current = state.settings.voiceVolume;
  useEffect(() => {
    if (current.current) current.current.audio.volume = volume.current;
  }, [state.settings.voiceVolume]);
  useEffect(() => {
    const token = crypto.randomUUID();
    const dispose = () => {
      const active = current.current;
      current.current = null;
      if (!active) return;
      active.audio.onended = null;
      active.audio.onerror = null;
      active.audio.pause();
      active.audio.removeAttribute("src");
      active.audio.load();
      URL.revokeObjectURL(active.url);
    };
    const stop = window.desktop.onCommand((command) => {
      if (command.type === "settings.open") return;
      if (command.type === "speech.stop") {
        if (current.current?.id === command.id) dispose();
        return;
      }
      dispose();
      const report = (status: "started" | "ended" | "failed") => {
        void action({ type: "speech.report", id: command.id, status }).catch(
          () => {},
        );
      };
      try {
        let bytes: Uint8Array<ArrayBuffer> = Uint8Array.from(
          atob(command.audio.audio),
          (char) => char.charCodeAt(0),
        );
        let mime = {
          mp3: "audio/mpeg",
          wav: "audio/wav",
          pcm: "audio/wav",
          opus: "audio/ogg; codecs=opus",
        }[command.audio.format];
        if (command.audio.format === "pcm") {
          if (bytes.length % 2) throw new Error("Invalid PCM length");
          const view = new DataView(bytes.buffer);
          const pcm = new Int16Array(bytes.length / 2);
          for (let i = 0; i < pcm.length; i++)
            pcm[i] = view.getInt16(i * 2, true);
          bytes = new Uint8Array(encodeWave(pcm, command.audio.sample_rate));
          mime = "audio/wav";
        }
        const url = URL.createObjectURL(new Blob([bytes], { type: mime }));
        const audio = new Audio(url);
        audio.volume = volume.current;
        const active = { id: command.id, audio, url };
        current.current = active;
        audio.onended = () => {
          if (current.current !== active) return;
          dispose();
          report("ended");
        };
        audio.onerror = () => {
          if (current.current !== active) return;
          dispose();
          report("failed");
        };
        void audio
          .play()
          .then(() => {
            if (current.current === active) report("started");
          })
          .catch(() => {
            if (current.current === active) {
              dispose();
              report("failed");
            }
          });
      } catch {
        dispose();
        report("failed");
      }
    });
    void action({ type: "speech.ready", ready: true, token }).catch(() => {});
    return () => {
      stop();
      dispose();
      void action({ type: "speech.ready", ready: false, token }).catch(
        () => {},
      );
    };
  }, []);
}
