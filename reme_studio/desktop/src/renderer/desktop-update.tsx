import { useEffect, useId, useRef, useState } from "react";
import { ArrowDownToLine } from "lucide-react";
import {
  translator,
  type Language,
  type TranslationKey,
} from "../../../client/src/i18n.ts";
import type { UpdateCommand, UpdateState } from "../protocol.ts";
import { useHost } from "../../../workbench/src/host.ts";
import { githubUrl } from "../../../workbench/src/workspace-titlebar.tsx";

const messages = {
  checking: "desktopUpdateChecking",
  current: "desktopUpdateCurrent",
  available: "desktopUpdateAvailable",
  downloading: "desktopUpdateDownloading",
  ready: "desktopUpdateReady",
  installing: "desktopUpdateInstalling",
  manual: "desktopUpdateManual",
  error: "desktopUpdateError",
} as const;

const actions: Partial<
  Record<
    UpdateState["status"],
    { label: TranslationKey; command: UpdateCommand }
  >
> = {
  idle: { label: "checkDesktopUpdate", command: "check" },
  current: { label: "checkDesktopUpdate", command: "check" },
  error: { label: "checkDesktopUpdate", command: "check" },
  available: { label: "desktopUpdateDownload", command: "download" },
  ready: { label: "desktopUpdateInstall", command: "install" },
};

export default function DesktopUpdate({
  language,
  version,
}: {
  language: Language;
  version: string;
}) {
  const t = translator(language);
  const { action } = useHost();
  const modal = useRef<HTMLDialogElement>(null);
  const id = useId();
  const [state, setState] = useState<UpdateState>({ status: "idle" });
  useEffect(() => window.desktop.onUpdate(setState), []);
  const run = async (command: UpdateCommand) => {
    try {
      setState(await window.desktop.update(command));
    } catch {
      setState({ status: "error" });
    }
  };
  const label = t("checkDesktopUpdate");
  const status = state.status;
  const message =
    status === "idle"
      ? null
      : t(messages[status], {
          version: "version" in state ? state.version : "",
          percent: state.status === "downloading" ? state.percent : 0,
        });
  const primary = actions[status];
  return (
    <>
      <button
        type="button"
        className="rail-button"
        aria-label={label}
        aria-haspopup="dialog"
        onClick={() => {
          modal.current?.showModal();
          void run("check");
        }}
      >
        <ArrowDownToLine size={18} aria-hidden />
        <span className="rail-label" aria-hidden>
          {label}
        </span>
      </button>
      <dialog
        ref={modal}
        className="workspace-dialog"
        aria-labelledby={`${id}-title`}
      >
        <h2 id={`${id}-title`}>{label}</h2>
        <p>
          {t("versionLabel")} {version}
        </p>
        {message && (
          <p role={status === "downloading" ? undefined : "status"}>
            {message}
          </p>
        )}
        {state.status === "downloading" && (
          <progress
            max={100}
            value={state.percent}
            aria-label={t("desktopUpdateDownload")}
          />
        )}
        <footer>
          <button
            type="button"
            className="secondary"
            onClick={() => modal.current?.close()}
          >
            {t("desktopUpdateClose")}
          </button>
          {(status === "manual" || status === "error") && (
            <button
              type="button"
              className="primary"
              onClick={() => {
                void action({
                  type: "link.open",
                  url: `${githubUrl}/releases/latest`,
                }).catch(() => setState({ status: "error" }));
              }}
            >
              {t("desktopUpdateDownloads")}
            </button>
          )}
          {primary && (
            <button
              type="button"
              className="primary"
              onClick={() => void run(primary.command)}
            >
              {t(primary.label)}
            </button>
          )}
        </footer>
      </dialog>
    </>
  );
}
