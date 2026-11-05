import { useCallback, useEffect, useState } from "react";
import type { StatusRpc } from "../../client/status-page.js";
import { safeError } from "../../endpoint.js";
import type { PythonDetection, PythonEnv } from "../types.js";
import { fill, type EnvTranslator } from "./locales.js";
import { InterpreterBadges, originLabel } from "./presentation.js";
import { envCall, parseDetection } from "./rpc.js";

const MINICONDA_URL = "https://docs.conda.io/en/latest/miniconda.html";

/** Interpreter rows shown before the list asks to expand. */
const ENV_PAGE = 6;

export interface DetectionState {
  phase: "loading" | "ready" | "error";
  data?: PythonDetection;
  error?: string;
}

export interface PythonSelection {
  detection: DetectionState;
  /** Interpreter the Host recommends for this machine. */
  suggested: string;
  /** Selected interpreter from settings, or the recommendation when unset. */
  selectedPath: string;
  selectedEnv: PythonEnv | undefined;
  /** Interpreter path whose selection is being written. */
  saving: string;
  /** Probe again; the Host caches detection for a minute. */
  refresh(): void;
  select(env: PythonEnv): void;
}

/**
 * The interpreter section's asynchronous state: one detection probe per
 * refresh, plus the selection write that settings owns.
 */
export function usePython(
  rpc: StatusRpc,
  options: {
    pythonPath: string;
    writable: boolean;
    /** False for a remote endpoint, whose local steps are hidden. */
    enabled?: boolean;
    onSelect(exe: string): Promise<unknown>;
  },
): PythonSelection {
  const { pythonPath, writable, enabled = true, onSelect } = options;
  const [detection, setDetection] = useState<DetectionState>({
    phase: "loading",
  });
  const [tick, setTick] = useState(0);
  const [saving, setSaving] = useState("");

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    void (async () => {
      try {
        const value = await envCall(
          rpc,
          "python",
          { refresh: tick > 0 },
          controller.signal,
        );
        if (controller.signal.aborted) return;
        setDetection({ phase: "ready", data: parseDetection(value) });
      } catch (error) {
        if (!controller.signal.aborted)
          setDetection((current) => ({
            phase: "error",
            error: safeError(error),
            ...(current.data ? { data: current.data } : {}),
          }));
      }
    })();
    return () => controller.abort();
  }, [rpc, tick, enabled]);

  // A refresh keeps the last detection data, so the selected interpreter (and
  // the install/progress panels keyed off it) does not flash away while the
  // new probe is in flight.
  const detected = detection.data;
  const suggested = detected?.suggested ?? "";
  const selectedPath = pythonPath || suggested;

  const refresh = useCallback(() => {
    setDetection((current) => ({
      phase: "loading",
      ...(current.data ? { data: current.data } : {}),
    }));
    setTick((current) => current + 1);
  }, []);

  const select = useCallback(
    (env: PythonEnv) => {
      if (!writable || saving) return;
      setSaving(env.exe);
      void onSelect(env.exe)
        .catch(() => undefined)
        .finally(() => setSaving(""));
    },
    [onSelect, saving, writable],
  );

  return {
    detection,
    suggested,
    selectedPath,
    selectedEnv: detected?.envs.find((env) => env.exe === selectedPath),
    saving,
    refresh,
    select,
  };
}

/** Step 1: the interpreter a local service runs on. */
export function PythonSection({
  t,
  selection,
  writable,
}: {
  t: EnvTranslator;
  selection: PythonSelection;
  writable: boolean;
}): JSX.Element {
  const { detection, suggested, selectedPath, saving } = selection;
  // A refresh keeps the last detection data, so the list must keep rendering it
  // instead of flashing "no usable Python" while the new probe is in flight.
  const detected = detection.data;
  const envs = detected?.envs ?? [];
  const [listOpen, setListOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [showAll, setShowAll] = useState(false);

  const selectedEnv = selection.selectedEnv;
  const others = envs.filter((env) => env.exe !== selectedPath);
  // The selected interpreter stays in the radio group, checked and disabled,
  // so assistive technology can hear which one is current. Dropping it from the
  // group left every radio unchecked, which conveys no selection at all.
  const listed = selectedEnv ? [selectedEnv, ...others] : others;
  const filter = query.trim().toLowerCase();
  const matches = filter
    ? listed.filter((env) =>
        `${env.version} ${env.exe} ${originLabel(t, env.origin)} ${env.source}`
          .toLowerCase()
          .includes(filter),
      )
    : listed;
  const visible = showAll ? matches : matches.slice(0, ENV_PAGE);
  const hidden = matches.length - visible.length;
  const withReme = envs.filter((env) => env.reme?.installed === true).length;

  return (
    <section className="reme-env-card">
      <header>
        <span className="reme-env-step" aria-hidden="true">
          1
        </span>
        <div>
          <h4>{t("pythonTitle")}</h4>
          <p>{t("pythonHint")}</p>
        </div>
        {envs.length > 0 ? (
          <aside>
            <span className="reme-env-count">
              {fill(t("envCount"), { count: envs.length })}
              {withReme > 0
                ? ` · ${fill(t("envRemeCount"), { count: withReme })}`
                : ""}
            </span>
          </aside>
        ) : null}
      </header>
      <div className="reme-env-card-body">
        {detection.phase === "loading" ? (
          <p className="reme-settings-muted">{t("detecting")}</p>
        ) : null}
        {detection.phase === "error" ? (
          <p className="reme-settings-error">
            {t("detectFailed")}
            {detection.error ? `: ${detection.error}` : ""}
          </p>
        ) : null}
        {(detection.phase === "error" && !detected) ||
        (detection.phase === "ready" && envs.length === 0) ? (
          <div className="reme-env-hint">
            <strong>{t("emptyTitle")}</strong>
            <span>{t("emptyHint")}</span>
            <a
              className="reme-settings-button"
              href={MINICONDA_URL}
              target="_blank"
              rel="noreferrer"
            >
              {t("miniconda")}
            </a>
          </div>
        ) : null}
        {selectedEnv ? (
          <div className="reme-env-picked">
            <span className="reme-env-picked-mark" aria-hidden="true" />
            <span className="reme-env-picked-main">
              <span className="reme-env-title">
                <strong>{`Python ${selectedEnv.version || "?"}`}</strong>
                <span className="reme-env-tag">
                  {originLabel(t, selectedEnv.origin)}
                </span>
                <InterpreterBadges
                  t={t}
                  env={selectedEnv}
                  suggested={suggested}
                />
              </span>
              <code className="reme-env-path" title={selectedEnv.exe}>
                {selectedEnv.exe}
              </code>
            </span>
            {saving === selectedEnv.exe ? (
              <span className="reme-settings-muted">
                {t("savingSelection")}
              </span>
            ) : null}
          </div>
        ) : selectedPath ? (
          <div className="reme-env-picked">
            <span className="reme-env-picked-mark" aria-hidden="true" />
            <span className="reme-env-picked-main">
              <span className="reme-env-title">
                <strong>{t("currentInterpreter")}</strong>
                <span className="reme-env-tag bad">{t("notInDetection")}</span>
              </span>
              <code className="reme-env-path" title={selectedPath}>
                {selectedPath}
              </code>
            </span>
          </div>
        ) : null}
        {others.length > 0 ? (
          <div className="reme-env-picker">
            <div className="reme-env-picker-bar">
              <button
                type="button"
                className="reme-settings-button"
                onClick={() => setListOpen((open) => !open)}
              >
                {t(listOpen ? "collapseList" : "changeInterpreter")}
              </button>
              {listOpen ? (
                <input
                  className="reme-env-search"
                  value={query}
                  placeholder={t("searchPlaceholder")}
                  onChange={(event) => {
                    setQuery(event.target.value);
                    setShowAll(false);
                  }}
                />
              ) : (
                <span className="reme-settings-muted">
                  {fill(t("envOthers"), { count: others.length })}
                </span>
              )}
            </div>
            {listOpen ? (
              <>
                {visible.length > 0 ? (
                  <div
                    className="reme-env-options"
                    role="radiogroup"
                    aria-label={t("pythonTitle")}
                  >
                    {visible.map((env) => (
                      <label key={env.exe} className="reme-env-option">
                        <input
                          type="radio"
                          name="reme-python-interpreter"
                          checked={env.exe === selectedPath}
                          disabled={
                            env.exe === selectedPath ||
                            !env.ok ||
                            saving !== "" ||
                            !writable
                          }
                          onChange={() => {
                            selection.select(env);
                            setListOpen(false);
                          }}
                        />
                        <strong>{`Python ${env.version || "?"}`}</strong>
                        <em>
                          {env.exe === selectedPath
                            ? t("currentInterpreter")
                            : originLabel(t, env.origin)}
                        </em>
                        <InterpreterBadges
                          t={t}
                          env={env}
                          suggested={suggested}
                        />
                        <code className="reme-env-option-path" title={env.exe}>
                          {env.exe}
                        </code>
                      </label>
                    ))}
                  </div>
                ) : (
                  <p className="reme-settings-muted">{t("noMatch")}</p>
                )}
                {hidden > 0 ? (
                  <button
                    type="button"
                    className="reme-settings-reset reme-env-more"
                    onClick={() => setShowAll(true)}
                  >
                    {fill(t("showAllEnvs"), { count: matches.length })}
                  </button>
                ) : matches.length > ENV_PAGE ? (
                  <button
                    type="button"
                    className="reme-settings-reset reme-env-more"
                    onClick={() => setShowAll(false)}
                  >
                    {t("showFewer")}
                  </button>
                ) : null}
              </>
            ) : null}
          </div>
        ) : null}
        {detected && detected.errors.length > 0 ? (
          <details className="reme-settings-details">
            <summary>{t("errorsTitle")}</summary>
            <pre>{detected.errors.join("\n")}</pre>
          </details>
        ) : null}
      </div>
    </section>
  );
}
