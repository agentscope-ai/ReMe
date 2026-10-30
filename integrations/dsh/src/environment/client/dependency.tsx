import { useCallback, useEffect, useRef, useState } from "react";
import type { StatusRpc } from "../../client/status-page.js";
import { safeError } from "../../endpoint.js";
import { interpreterKey } from "../interpreter-key.js";
import type { InstallStatus, RemeCheck } from "../types.js";
import { fill, type EnvTranslator } from "./locales.js";
import { StatusBadge } from "./presentation.js";
import { envCall, parseInstallStatus, parseRemeCheck } from "./rpc.js";

export interface CheckState {
  phase: "idle" | "ready" | "error";
  data?: RemeCheck;
  error?: string;
  /** The interpreter this result belongs to. */
  forPath: string;
}

export interface InstallState {
  phase: "idle" | "running" | "done";
  /** Interpreter this state belongs to; another one cannot replace it. */
  pythonPath: string;
  /** Job this section polls; another installation cannot replace it. */
  jobId?: string;
  job?: InstallStatus;
  error?: string;
  /**
   * Request generation. An answer from an older request never paints this
   * card, so a late response cannot resurrect a job the user moved on from.
   */
  generation: number;
}

/** The state of an interpreter that has nothing installed or running. */
const IDLE_INSTALL: InstallState = {
  phase: "idle",
  pythonPath: "",
  generation: 0,
};

export interface DependencyState {
  /** Check result, only when it belongs to the interpreter on screen. */
  checkData: RemeCheck | undefined;
  checkError: string | undefined;
  checking: boolean;
  install: InstallState;
  /** Wall clock tick while an installation runs, for the elapsed label. */
  now: number;
  recheck(): void;
  start(
    upgrade: boolean,
    indexUrl?: string,
    forceReinstall?: boolean,
  ): Promise<void>;
  cancel(): Promise<void>;
  dismiss(): void;
}

/**
 * The dependency section's asynchronous state: one reme-ai check per
 * interpreter, and one installation job bound to its own interpreter, job id
 * and request generation.
 */
export function useDependency(
  rpc: StatusRpc,
  options: {
    pythonPath: string;
    /** False for a remote endpoint, whose local steps are hidden. */
    enabled?: boolean;
    onInstalled(): void;
  },
): DependencyState {
  const { pythonPath, enabled = true, onInstalled } = options;
  const [check, setCheck] = useState<CheckState>({
    phase: "idle",
    forPath: "",
  });
  const [install, setInstall] = useState<InstallState>(IDLE_INSTALL);
  const [tick, setTick] = useState(0);
  const [now, setNow] = useState(0);
  const installed = useRef(onInstalled);
  installed.current = onInstalled;
  /** Monotonic request counter: only the newest answer may paint the card. */
  const generation = useRef(0);
  /** Interpreter the state on screen belongs to. */
  const currentPath = useRef(pythonPath);
  /**
   * The installation on screen belongs to the interpreter on screen: a job for
   * another interpreter is neither shown nor cancellable from here. The card
   * and the actions read this one state, so they cannot disagree.
   */
  const visibleInstall =
    install.pythonPath === pythonPath
      ? install
      : { ...IDLE_INSTALL, pythonPath };

  useEffect(() => {
    if (!enabled || !pythonPath) return;
    const controller = new AbortController();
    void (async () => {
      try {
        const value = await envCall(
          rpc,
          "checkReme",
          { pythonPath },
          controller.signal,
        );
        if (controller.signal.aborted) return;
        setCheck({
          phase: "ready",
          data: parseRemeCheck(value),
          error: undefined,
          forPath: pythonPath,
        });
      } catch (error) {
        if (!controller.signal.aborted)
          setCheck({
            phase: "error",
            error: safeError(error),
            forPath: pythonPath,
          });
      }
    })();
    return () => controller.abort();
  }, [rpc, pythonPath, tick, enabled]);

  const recheck = useCallback(() => {
    setCheck({ phase: "idle", forPath: "" });
    setTick((current) => current + 1);
  }, []);

  /**
   * The job finished. A successful install changed what the interpreter has,
   * so the dependency check has to run again — otherwise the card keeps
   * reporting "not installed" and Start stays disabled.
   */
  const finishInstall = useCallback(
    (status: InstallStatus, path: string) => {
      if (status.phase !== "succeeded") return;
      // The success belongs to the interpreter on screen; anything else is a
      // job the user already moved on from.
      if (currentPath.current !== path) return;
      installed.current();
      recheck();
    },
    [recheck],
  );

  // Progress belongs to an interpreter: switching interpreters must not keep
  // showing (or cancelling) a job that was started for the previous one. The
  // new generation retires every answer the previous interpreter is owed.
  useEffect(() => {
    currentPath.current = pythonPath;
    const request = (generation.current += 1);
    setInstall({ phase: "idle", pythonPath, generation: request });
  }, [pythonPath]);

  // Restoring a running job is per interpreter: a job started for another
  // interpreter belongs to that interpreter's section, not this one. The Host
  // looks the job up by interpreter, because two interpreters may install at
  // the same time and the newest job overall is not necessarily this one.
  useEffect(() => {
    if (!enabled || !pythonPath) return;
    const controller = new AbortController();
    const request = generation.current;
    void (async () => {
      try {
        const status = parseInstallStatus(
          await envCall(
            rpc,
            "installStatus",
            { pythonPath },
            controller.signal,
          ),
        );
        if (controller.signal.aborted) return;
        // An installation the user started meanwhile owns the card now.
        if (request !== generation.current) return;
        if (!status || status.phase !== "running") return;
        // The Host looks the job up by interpreter, with the path semantics of
        // the path's own platform; the tab compares the same way, so a job the
        // Host found is not dropped again here.
        if (interpreterKey(status.pythonPath) !== interpreterKey(pythonPath))
          return;
        setInstall({
          phase: "running",
          pythonPath,
          jobId: status.id,
          job: status,
          generation: request,
        });
      } catch {
        // Restoring is best-effort; a later action reports real failures.
      }
    })();
    return () => controller.abort();
  }, [rpc, pythonPath, enabled]);

  const jobId = install.jobId;
  const installPath = install.pythonPath;
  // The generation is not read in the effect: it is listed so that a replaced
  // state restarts the poll instead of leaving the old one running.
  const installGeneration = install.generation;
  useEffect(() => {
    // A running job always has an id; without one the Host would answer with
    // the newest job overall, which may belong to another interpreter.
    if (install.phase !== "running" || !jobId) return;
    const controller = new AbortController();
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async (): Promise<void> => {
      try {
        const status = parseInstallStatus(
          await envCall(rpc, "installStatus", { jobId }, controller.signal),
        );
        if (cancelled) return;
        if (!status) {
          setInstall((current) => ({ ...current, phase: "idle" }));
          return;
        }
        if (status.phase === "running") {
          setInstall((current) => ({ ...current, job: status }));
          timer = setTimeout(() => void poll(), 2000);
          return;
        }
        setInstall((current) => ({ ...current, phase: "done", job: status }));
        finishInstall(status, installPath);
      } catch (error) {
        if (cancelled) return;
        setInstall((current) => ({
          ...current,
          phase: "done",
          error: safeError(error),
        }));
      }
    };
    void poll();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [
    install.phase,
    jobId,
    installPath,
    installGeneration,
    rpc,
    finishInstall,
  ]);

  useEffect(() => {
    if (install.phase !== "running") return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [install.phase]);

  const start = useCallback(
    async (upgrade: boolean, indexUrl?: string, forceReinstall?: boolean) => {
      // The card shows the derived state, so the guard reads it too: a job for
      // another interpreter must not block this one.
      if (!pythonPath || visibleInstall.phase === "running") return;
      const target = pythonPath;
      // Every request gets its own generation, so a late answer cannot
      // replace a newer installation or survive an interpreter switch.
      const request = (generation.current += 1);
      setInstall({ phase: "idle", pythonPath: target, generation: request });
      try {
        const status = parseInstallStatus(
          await envCall(rpc, "install", {
            pythonPath: target,
            upgrade,
            ...(indexUrl ? { indexUrl } : {}),
            ...(forceReinstall ? { forceReinstall } : {}),
          }),
        );
        // The answer belongs to the request that asked for it.
        if (request !== generation.current || target !== currentPath.current)
          return;
        if (!status) {
          setInstall({
            phase: "idle",
            pythonPath: target,
            generation: request,
          });
          return;
        }
        setInstall({
          phase: status.phase === "running" ? "running" : "done",
          pythonPath: target,
          jobId: status.id,
          job: status,
          generation: request,
        });
        if (status.phase !== "running") finishInstall(status, target);
      } catch (error) {
        if (request !== generation.current || target !== currentPath.current)
          return;
        setInstall({
          phase: "done",
          pythonPath: target,
          generation: request,
          error: safeError(error),
        });
      }
    },
    [finishInstall, pythonPath, rpc, visibleInstall.phase],
  );

  const cancel = useCallback(async () => {
    if (!jobId) return;
    try {
      await envCall(rpc, "installCancel", { jobId });
    } catch {
      // Polling reflects the final phase either way.
    }
  }, [jobId, rpc]);

  const dismiss = useCallback(() => {
    const request = (generation.current += 1);
    setInstall({
      phase: "idle",
      pythonPath: currentPath.current,
      generation: request,
    });
  }, []);

  const current = check.forPath === pythonPath ? check : undefined;
  return {
    checkData: current?.data,
    checkError: current?.error,
    checking: Boolean(pythonPath) && check.forPath !== pythonPath,
    // The installation on screen belongs to the interpreter on screen.
    install: visibleInstall,
    now,
    recheck,
    start,
    cancel,
    dismiss,
  };
}

function elapsedText(job: InstallStatus, now: number): string {
  const started = Date.parse(job.startedAt);
  if (!Number.isFinite(started) || now < started) return "";
  const seconds = Math.floor((now - started) / 1000);
  const minutes = Math.floor(seconds / 60);
  return minutes > 0 ? `${minutes}m ${seconds % 60}s` : `${seconds}s`;
}

/** Step 2: reme-ai in the selected interpreter, and its one-click install. */
export function DependencySection({
  t,
  pythonPath,
  dependency,
}: {
  t: EnvTranslator;
  pythonPath: string;
  dependency: DependencyState;
}): JSX.Element {
  const { checkData, checkError, checking, install, now } = dependency;
  const [confirming, setConfirming] = useState(false);
  const [mirror, setMirror] = useState("");
  // The confirm copy follows the CTA: a Reinstall downloads again with
  // `--force-reinstall`, so the first-install wording would be misleading.
  const reinstall = checkData?.installed === true && checkData.versionOk;

  const badge = !checkData
    ? undefined
    : !checkData.installed
    ? { tone: "progress" as const, text: t("depMissing") }
    : checkData.versionOk && checkData.hasCore
    ? { tone: "ok" as const, text: t("depReady") }
    : checkData.versionOk
    ? { tone: "progress" as const, text: t("coreMissingBadge") }
    : { tone: "progress" as const, text: t("depUpgrade") };

  return (
    <section className="reme-env-card">
      <header>
        <span className="reme-env-step" aria-hidden="true">
          2
        </span>
        <div>
          <h4>{t("dependencyTitle")}</h4>
          <p>{t("dependencyHint")}</p>
        </div>
        {badge ? (
          <aside>
            <StatusBadge tone={badge.tone}>{badge.text}</StatusBadge>
          </aside>
        ) : null}
      </header>
      <div className="reme-env-card-body">
        <div className="reme-env-factrow">
          <span className="reme-env-field-label">{t("interpreterLabel")}</span>
          <code className="reme-env-path" title={pythonPath}>
            {pythonPath || t("noInterpreter")}
          </code>
          <button
            type="button"
            className="reme-settings-button"
            disabled={!pythonPath}
            onClick={dependency.recheck}
          >
            {t("recheck")}
          </button>
        </div>
        {!pythonPath ? (
          <p className="reme-settings-muted">{t("selectInterpreterFirst")}</p>
        ) : checking ? (
          <p className="reme-settings-muted">{t("checkingReme")}</p>
        ) : null}
        {checkError ? (
          <p className="reme-settings-error">
            {t("checkRemeFailed")}
            {checkError ? `: ${checkError}` : ""}
          </p>
        ) : null}
        {checkData?.error ? (
          <>
            <p className="reme-settings-error">{t("checkRemeFailed")}</p>
            <details className="reme-settings-details">
              <summary>{t("errorsTitle")}</summary>
              <pre>{checkData.error}</pre>
            </details>
          </>
        ) : null}
        {checkData ? (
          <div className="reme-env-metrics">
            <div className="reme-env-metric">
              <span>{t("remeLabel")}</span>
              <div>
                {checkData.installed ? (
                  <>
                    <StatusBadge
                      tone={checkData.versionOk ? "ok" : "progress"}
                    >{`reme-ai ${checkData.version}`}</StatusBadge>
                    {checkData.versionOk ? null : (
                      <small>{t("upgradeRecommended")}</small>
                    )}
                  </>
                ) : (
                  <StatusBadge tone="bad">{t("notInstalled")}</StatusBadge>
                )}
              </div>
            </div>
            <div className="reme-env-metric">
              <span>{t("coreLabel")}</span>
              <div>
                {checkData.installed ? (
                  <StatusBadge tone={checkData.hasCore ? "ok" : "progress"}>
                    {checkData.hasCore ? t("coreOk") : t("coreMissing")}
                  </StatusBadge>
                ) : (
                  <span className="reme-settings-muted">—</span>
                )}
              </div>
            </div>
            <div className="reme-env-metric">
              <span>{t("cliLabel")}</span>
              <div>
                {checkData.installed ? (
                  checkData.cliPath ? (
                    <code
                      className="reme-env-metric-path"
                      title={checkData.cliPath}
                    >
                      {checkData.cliPath}
                    </code>
                  ) : (
                    <StatusBadge tone="progress">
                      {t("cliFallback")}
                    </StatusBadge>
                  )
                ) : (
                  <span className="reme-settings-muted">—</span>
                )}
              </div>
            </div>
          </div>
        ) : null}
        {checkData && install.phase !== "running" ? (
          <div className="reme-env-cta">
            <label className="reme-env-mirror">
              <span>{t("mirrorLabel")}</span>
              <select
                value={mirror}
                onChange={(event) => setMirror(event.target.value)}
              >
                <option value="">{t("mirrorDefault")}</option>
                <option value="https://pypi.org/simple">
                  {t("mirrorPypi")}
                </option>
                <option value="https://pypi.tuna.tsinghua.edu.cn/simple">
                  {t("mirrorTsinghua")}
                </option>
                <option value="https://mirrors.aliyun.com/pypi/simple">
                  {t("mirrorAliyun")}
                </option>
              </select>
            </label>
            {confirming ? (
              <div className="reme-env-confirm">
                <p>
                  {fill(t(reinstall ? "reinstallConfirm" : "installConfirm"), {
                    path: pythonPath,
                  })}
                </p>
                <button
                  type="button"
                  className="reme-settings-save"
                  onClick={() => {
                    setConfirming(false);
                    // Fresh install: no upgrade. Outdated: upgrade. Current
                    // but suspect (Reinstall): force pip to reinstall.
                    // The mirror choice reaches pip as `--index-url`; without
                    // it the selector would be decorative.
                    void dependency.start(
                      checkData.installed && !checkData.versionOk,
                      mirror || undefined,
                      reinstall,
                    );
                  }}
                >
                  {t("confirmInstall")}
                </button>
                <button
                  type="button"
                  className="reme-settings-discard"
                  onClick={() => setConfirming(false)}
                >
                  {t("cancel")}
                </button>
              </div>
            ) : checkData.installed && checkData.versionOk ? (
              <button
                type="button"
                className="reme-settings-button"
                onClick={() => setConfirming(true)}
              >
                {t("reinstall")}
              </button>
            ) : checkData.installed ? (
              <button
                type="button"
                className="reme-settings-save"
                onClick={() => setConfirming(true)}
              >
                {t("upgradeButton")}
              </button>
            ) : (
              <button
                type="button"
                className="reme-settings-save"
                onClick={() => setConfirming(true)}
              >
                {t("installButton")}
              </button>
            )}
            <p className="reme-settings-muted reme-env-hint-inline">
              {t("condaHint")}
            </p>
          </div>
        ) : null}
        {install.phase === "running" && install.job ? (
          <div className="reme-env-progress">
            <div className="reme-env-row">
              <StatusBadge tone="progress">
                {t("installing")}
                {elapsedText(install.job, now)
                  ? ` · ${elapsedText(install.job, now)}`
                  : ""}
              </StatusBadge>
              <button
                type="button"
                className="reme-settings-discard"
                disabled={install.job.cancelling === true}
                onClick={() => void dependency.cancel()}
              >
                {t(install.job.cancelling ? "cancelling" : "cancelInstall")}
              </button>
            </div>
            {install.job.cancelling ? (
              <p className="reme-settings-muted">{t("cancellingHint")}</p>
            ) : null}
            <pre className="reme-env-log">
              {install.job.logTail || t("waitingForOutput")}
            </pre>
          </div>
        ) : null}
        {install.phase === "done" ? (
          <div className="reme-env-result">
            {install.job?.phase === "succeeded" ? (
              <p className="reme-settings-success">{t("installSucceeded")}</p>
            ) : null}
            {install.job?.phase === "cancelled" ? (
              <p className="reme-settings-muted">{t("installCancelled")}</p>
            ) : null}
            {install.job?.phase === "failed" ? (
              <>
                <p className="reme-settings-error">{t("installFailed")}</p>
                {install.job.logTail ? (
                  <pre className="reme-env-log">{install.job.logTail}</pre>
                ) : null}
                <p className="reme-settings-muted">{install.job.logPath}</p>
              </>
            ) : null}
            {install.error ? (
              <p className="reme-settings-error">{install.error}</p>
            ) : null}
            <button
              type="button"
              className="reme-settings-button"
              onClick={() => {
                setConfirming(false);
                dependency.dismiss();
              }}
            >
              {t("dismiss")}
            </button>
          </div>
        ) : null}
      </div>
    </section>
  );
}
