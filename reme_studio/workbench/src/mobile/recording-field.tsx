import { LoaderCircle, Mic, X } from "lucide-react";
import { translator, type Language } from "../../../client/src/i18n.ts";

export default function RecordingField({
  phase,
  canceling,
  seconds,
  level,
  language,
  onCancel,
}: {
  phase: "starting" | "recording" | "transcribing";
  canceling: boolean;
  seconds: number;
  level: number;
  language: Language;
  onCancel: () => void;
}) {
  const t = translator(language);
  const transcribing = phase === "transcribing";
  return (
    <div
      className="mobile-recording-field"
      data-canceling={canceling}
      role="status"
      aria-live="polite"
    >
      {transcribing ? (
        <LoaderCircle size={22} className="spinning" aria-hidden />
      ) : (
        <Mic size={22} aria-hidden />
      )}
      <div className="mobile-recording-copy" key={`${phase}:${canceling}`}>
        <strong>
          {t(
            canceling
              ? "releaseToCancel"
              : transcribing
                ? "speechSendingRecording"
                : phase === "starting"
                  ? "speechOpeningMic"
                  : "releaseToSend",
          )}
        </strong>
        {!transcribing && (
          <small>
            {t(canceling ? "slideBackToRecord" : "slideUpToCancel")}
          </small>
        )}
      </div>
      {phase === "recording" && (
        <>
          <meter
            min={0}
            max={1}
            value={level}
            aria-label={t("speechListening")}
          />
          <time>
            {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, "0")}
          </time>
        </>
      )}
      {transcribing && (
        <button
          type="button"
          className="icon-button"
          aria-label={t("cancelVoiceSend")}
          onClick={onCancel}
        >
          <X size={18} aria-hidden />
        </button>
      )}
    </div>
  );
}
