import { useCallback, useEffect, useRef, useState } from "react";
import type { StatusRpc } from "../../client/status-page.js";
import { safeError } from "../../endpoint.js";
import { serviceEndpointKey } from "../endpoint-key.js";
import { canHostService } from "../dependency.js";
import type {
  HealthErrorCode,
  RemeCheck,
  ServiceRecordRef,
  ServiceStartResult,
  ServiceState,
} from "../types.js";
import { fill, type EnvKey, type EnvTranslator } from "./locales.js";
import { StatusBadge } from "./presentation.js";
import {
  envCall,
  parseHealth,
  parseServiceForget,
  parseServiceLog,
  parseServiceStart,
  parseServiceState,
  parseServiceStop,
} from "./rpc.js";

/**
 * How long a detached start may take to answer `/health_check`. Kept at the
 * Host's start grace period, so the client never gives up while the Host still
 * considers the launch in flight.
 */
const START_DEADLINE_MS = 120_000;

/** Host start errors carry a stable code; older/unknown codes fall back. */
const START_ERROR_KEYS: Record<string, EnvKey> = {
  "other-endpoint-starting": "startErrorOtherEndpoint",
  conflict: "startErrorConflict",
  "other-endpoint-running": "startErrorOtherEndpointRunning",
  "previous-not-stopped": "startErrorPreviousNotStopped",
  "unverified-record": "startErrorUnverifiedRecord",
  "no-python": "startErrorNoPython",
  "not-installed": "startErrorNotInstalled",
  "version-too-old": "startErrorVersionTooOld",
  "start-failed": "startErrorFailed",
};

function startErrorText(t: EnvTranslator, result: ServiceStartResult): string {
  // The unverified record carries the pid it refused to touch, so the message
  // can tell the user what to look for instead of a generic sentence.
  if (result.code === "unverified-record" && result.pid !== undefined)
    return fill(t("startErrorUnverifiedRecordPid"), { pid: result.pid });
  const key = result.code ? START_ERROR_KEYS[result.code] : undefined;
  return key ? t(key) : result.error || t("serviceError");
}

const STOP_REASON_KEYS: Record<string, EnvKey> = {
  "no-record": "stopNoRecord",
  foreign: "stopForeign",
  unverified: "stopUnknownIdentity",
  "not-exited": "stopNotExited",
  "record-changed": "stopRecordChanged",
};

function stopReasonText(
  t: EnvTranslator,
  result: { code?: string; reason?: string },
): string {
  const key = result.code ? STOP_REASON_KEYS[result.code] : undefined;
  return key ? t(key) : result.reason || t("serviceError");
}

/**
 * Codes whose refusal leaves the record as the only way out: Start and Stop
 * both refuse to act on such a record, so it can only be forgotten.
 */
const RECORD_ONLY_CODES = new Set([
  "unverified-record",
  "previous-not-stopped",
  "unverified",
  "not-exited",
]);

/**
 * The recovery a refused action allows.
 *
 * The Host names the record it refused on: that record may belong to another
 * endpoint than the one on screen, and forgetting the endpoint on screen
 * would be refused as foreign. A refusal that names no record (a conflict, a
 * missing interpreter) has no record-only way out. An older Host names no
 * record either, so the endpoint on screen stays the fallback.
 */
function recoveryFor(
  result: { code?: string; record?: ServiceRecordRef },
  endpoint: string,
): ServiceRecordRef | undefined {
  if (!result.code || !RECORD_ONLY_CODES.has(result.code)) return undefined;
  return result.record ?? { endpoint };
}

export interface ServiceViewState {
  phase: "checking" | "running" | "starting" | "stopped" | "conflict" | "error";
  data?: ServiceState;
  error?: string;
}

export interface ServiceLogView {
  open: boolean;
  lines: string;
  logPath: string;
}

export interface ServiceControl {
  state: ServiceViewState;
  stopping: boolean;
  /** Why the last Stop request left the service alone, if it did. */
  stopError: string;
  /**
   * The record the Host refused to act on. It can only be forgotten, and
   * forgetting targets this record, not the endpoint on screen.
   */
  forgetTarget?: ServiceRecordRef;
  forgetting: boolean;
  start(): Promise<void>;
  stop(): Promise<void>;
  forget(): Promise<void>;
  refresh(): void;
  log: ServiceLogView;
  viewLog(): Promise<void>;
  dismissLog(): void;
}

/**
 * The service section's asynchronous state, keyed by endpoint: one status read
 * per endpoint, plus the start and stop operations for the managed instance.
 */
export function useService(
  rpc: StatusRpc,
  options: {
    t: EnvTranslator;
    endpoint: string;
    /** Only a loopback endpoint can be managed, so remote ones are read-only. */
    local: boolean;
    pythonPath: string;
    workspaceDir: string;
    authToken: string;
  },
): ServiceControl {
  const { t, endpoint, local, pythonPath, workspaceDir, authToken } = options;
  const [state, setState] = useState<ServiceViewState>({ phase: "checking" });
  const [stopping, setStopping] = useState(false);
  const [stopError, setStopError] = useState("");
  const [forgetTarget, setForgetTarget] = useState<
    ServiceRecordRef | undefined
  >(undefined);
  const [forgetting, setForgetting] = useState(false);
  const [tick, setTick] = useState(0);
  const [log, setLog] = useState<ServiceLogView>({
    open: false,
    lines: "",
    logPath: "",
  });
  const startController = useRef<AbortController | undefined>(undefined);
  /** True only while a start this tab requested is being polled. */
  const startInFlight = useRef(false);
  /** The endpoint the callbacks below belong to; stale answers are dropped. */
  const endpointRef = useRef(endpoint);
  /** Sequence of status answers; a slow one must not overwrite a newer phase. */
  const statusSeq = useRef(0);

  const applyState = useCallback((next: ServiceViewState) => {
    // An answer that settles a start is newer than any status read that was
    // already in flight, so that read is discarded when it lands.
    statusSeq.current += 1;
    setState(next);
  }, []);

  // A new endpoint has no known phase yet; keeping the previous endpoint's
  // "running + old pid/version" on screen would mislabel the new one. A start
  // poll for the old endpoint must stop before it can paint that old data.
  useEffect(() => {
    endpointRef.current = endpoint;
    startInFlight.current = false;
    startController.current?.abort();
    // Answers requested for the old endpoint are no longer relevant.
    statusSeq.current += 1;
    // Resetting the view when the endpoint prop changes is intentional; the
    // hook has no key, so this is the reset path.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setState({ phase: "checking" });
    setStopError("");
    setStopping(false);
    setForgetTarget(undefined);
    setForgetting(false);
  }, [endpoint]);

  useEffect(() => () => startController.current?.abort(), []);

  useEffect(() => {
    if (!local || !endpoint) return;
    const controller = new AbortController();
    const request = statusSeq.current + 1;
    statusSeq.current = request;
    void (async () => {
      try {
        const next = parseServiceState(
          await envCall(
            rpc,
            "serviceStatus",
            { endpoint, authToken },
            controller.signal,
          ),
        );
        if (controller.signal.aborted || request !== statusSeq.current) return;
        // The record is gone (dead pid cleared by the Host, or just forgotten),
        // so there is nothing left to forget. This status read only covers the
        // endpoint on screen, so a record for another endpoint stays.
        if (next.phase === "stopped" && next.pid === undefined)
          setForgetTarget((current) =>
            current &&
            serviceEndpointKey(current.endpoint) ===
              serviceEndpointKey(endpoint)
              ? undefined
              : current,
          );
        // A start this tab requested maps "stopped with a live pid" (the Host's
        // "launch still booting" signal) to `starting`; a plain status read
        // keeps the Host's phase, so a leftover process never locks the card.
        setState({
          phase:
            next.phase === "stopped" &&
            next.pid !== undefined &&
            startInFlight.current
              ? "starting"
              : next.phase,
          data: next,
        });
      } catch (error) {
        if (controller.signal.aborted || request !== statusSeq.current) return;
        setState({ phase: "error", error: safeError(error) });
      }
    })();
    return () => controller.abort();
  }, [rpc, endpoint, local, authToken, tick]);

  const refresh = useCallback(() => setTick((current) => current + 1), []);

  const start = useCallback(async () => {
    if (!pythonPath || state.phase === "starting") return;
    startController.current?.abort();
    const controller = new AbortController();
    startController.current = controller;
    startInFlight.current = true;
    // A status read requested before this click is older than the start.
    statusSeq.current += 1;
    setState({ phase: "starting" });
    setStopError("");
    setLog({ open: false, lines: "", logPath: "" });
    try {
      const started = parseServiceStart(
        await envCall(
          rpc,
          "serviceStart",
          { endpoint, pythonPath, workspaceDir, authToken },
          controller.signal,
        ),
      );
      if (controller.signal.aborted) return;
      if (started.error) {
        if (started.code === "conflict") {
          // The Host refused because the port answered. Showing that as a
          // conflict card (with Stop when it is ours) matches what the poll
          // below would report, instead of an error whose Start just repeats.
          try {
            const status = parseServiceState(
              await envCall(
                rpc,
                "serviceStatus",
                { endpoint, authToken },
                controller.signal,
              ),
            );
            if (controller.signal.aborted) return;
            if (status.phase === "conflict") {
              applyState({ phase: "conflict", data: status });
              return;
            }
          } catch {
            // A switch of endpoint aborts this call; its error belongs to the
            // old endpoint and must not be painted (or bump the sequence) on
            // the new card.
            if (controller.signal.aborted) return;
            // Otherwise fall back to the reported start error.
          }
        }
        // The Host refused because the record describes a live process it
        // cannot stop or verify. Neither Start nor Stop can act on it, so offer
        // the record-only escape hatch, aimed at the record the Host named.
        // A refusal that names no record leaves an existing offer alone: the
        // record it describes is still there.
        const recovery = recoveryFor(started, endpoint);
        if (recovery) setForgetTarget(recovery);
        applyState({ phase: "error", error: startErrorText(t, started) });
        return;
      }
      const deadline = Date.now() + START_DEADLINE_MS;
      let ownedConflict: ServiceState | undefined;
      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        if (controller.signal.aborted) return;
        if (Date.now() > deadline) {
          // A service that holds the port but never turns healthy cannot be
          // recovered by waiting; an *owned* one is still stoppable, so keep
          // that way out instead of a dead-end timeout error.
          applyState(
            ownedConflict
              ? { phase: "conflict", data: ownedConflict }
              : { phase: "error", error: t("startTimeout") },
          );
          return;
        }
        try {
          const status = parseServiceState(
            await envCall(
              rpc,
              "serviceStatus",
              { endpoint, authToken },
              controller.signal,
            ),
          );
          if (controller.signal.aborted) return;
          if (status.phase === "running") {
            setForgetTarget(undefined);
            applyState({ phase: "running", data: status });
            return;
          }
          if (status.phase === "conflict") {
            if (status.managed) {
              // Our own launch can hold the port before `/health_check`
              // passes. Keep polling instead of offering Stop too early; if it
              // never becomes healthy the deadline shows the conflict card.
              ownedConflict = status;
              continue;
            }
            applyState({ phase: "conflict", data: status });
            return;
          }
          if (status.phase === "stopped" && status.pid === undefined) {
            // The Host already dropped the start marker: the launch died before
            // it ever listened, so waiting out the 120 s deadline is wrong.
            applyState({ phase: "error", error: t("startFailedStopped") });
            return;
          }
          // stopped with a live pid: the process is still booting. Carry the
          // pid so a launch that never becomes healthy can still be stopped.
          if (status.pid !== undefined)
            setState((current) =>
              current.phase === "starting"
                ? { phase: "starting", data: status }
                : current,
            );
        } catch {
          // Keep polling: a starting service answers once it is listening.
        }
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      applyState({ phase: "error", error: safeError(error) });
    } finally {
      if (startController.current === controller) {
        startController.current = undefined;
        startInFlight.current = false;
      }
    }
  }, [
    applyState,
    authToken,
    endpoint,
    pythonPath,
    rpc,
    state.phase,
    t,
    workspaceDir,
  ]);

  const stop = useCallback(async () => {
    if (stopping) return;
    const target = endpoint;
    // A Stop during a start must cancel that poll, or its next tick would
    // repaint the launch the user just cancelled.
    startController.current?.abort();
    startInFlight.current = false;
    setStopping(true);
    try {
      const result = parseServiceStop(
        await envCall(rpc, "serviceStop", { endpoint: target }),
      );
      if (endpointRef.current !== target) return;
      setStopError(result.stopped ? "" : stopReasonText(t, result));
      // A record the Host refuses to stop (unverifiable, or a process that
      // survived the force-kill) cannot be cleared by either button: offer the
      // record-only way out instead of leaving two actions that both fail.
      const recovery = recoveryFor(result, target);
      if (recovery) setForgetTarget(recovery);
      // A stopped service has no record left; a refusal that names none (a
      // foreign record, a missing one) leaves an existing offer alone.
      else if (result.stopped || result.code === "no-record")
        setForgetTarget(undefined);
      refresh();
    } catch (error) {
      if (endpointRef.current !== target) return;
      applyState({ phase: "error", error: safeError(error) });
    } finally {
      // A result for the previous endpoint must not disable the new card; that
      // endpoint's effect already reset the flag.
      if (endpointRef.current === target) setStopping(false);
    }
  }, [applyState, endpoint, refresh, rpc, stopping, t]);

  const forget = useCallback(async () => {
    if (forgetting || !forgetTarget) return;
    const card = endpoint;
    // The record is named by the Host, not by the endpoint on screen, so the
    // request carries the record's own endpoint and the pid it described.
    const target = forgetTarget;
    setForgetting(true);
    try {
      const result = parseServiceForget(
        await envCall(rpc, "serviceForget", {
          endpoint: target.endpoint,
          ...(target.pid === undefined ? {} : { expectPid: target.pid }),
        }),
      );
      if (endpointRef.current !== card) return;
      if (result.forgotten) {
        setForgetTarget(undefined);
        setStopError("");
        // Drop the stale pid immediately; the refresh confirms it.
        applyState({ phase: "checking" });
        refresh();
      } else {
        setStopError(stopReasonText(t, result));
        if (result.code === "no-record") {
          // The record is already gone: there is nothing left to forget, so the
          // button must not offer an action that can never succeed.
          setForgetTarget(undefined);
        } else if (result.code === "record-changed") {
          // The record the user confirmed is not the one that is there now.
          // The offer is dropped instead of being re-aimed at a record the
          // user never saw: the next refusal names the current one.
          setForgetTarget(undefined);
        }
      }
    } catch (error) {
      if (endpointRef.current !== card) return;
      applyState({ phase: "error", error: safeError(error) });
    } finally {
      if (endpointRef.current === card) setForgetting(false);
    }
  }, [applyState, endpoint, forgetTarget, forgetting, refresh, rpc, t]);

  const viewLog = useCallback(async () => {
    try {
      const value = parseServiceLog(
        await envCall(rpc, "serviceLog", { workspaceDir }),
      );
      setLog({ open: true, lines: value.lines, logPath: value.logPath });
    } catch {
      // Log viewing is diagnostic only.
    }
  }, [rpc, workspaceDir]);

  const dismissLog = useCallback(
    () => setLog({ open: false, lines: "", logPath: "" }),
    [],
  );

  return {
    state,
    stopping,
    stopError,
    forgetTarget,
    forgetting,
    start,
    stop,
    forget,
    refresh,
    log,
    viewLog,
    dismissLog,
  };
}

export interface ConnectionState {
  phase: "loading" | "ok" | "fail";
  endpoint: string;
  version?: string;
  error?: string;
  /** Stable Host reason, localized by the hero instead of showing `error`. */
  code?: HealthErrorCode;
}

/**
 * Health of a ReMe service that this plugin cannot manage locally (a remote
 * URL), so the hero still reports whether the configured endpoint answers.
 */
export function useHealth(options: {
  rpc: StatusRpc;
  endpoint: string;
  enabled: boolean;
  tick: number;
  authToken: string;
}): ConnectionState {
  const { rpc, endpoint, enabled, tick, authToken } = options;
  const [health, setHealth] = useState<ConnectionState>({
    phase: "loading",
    endpoint: "",
  });

  useEffect(() => {
    if (!enabled || !endpoint) return;
    const controller = new AbortController();
    void (async () => {
      try {
        const result = parseHealth(
          await envCall(
            rpc,
            "health",
            { endpoint, authToken },
            controller.signal,
          ),
        );
        if (controller.signal.aborted) return;
        setHealth(
          result.ok
            ? { phase: "ok", endpoint, version: result.version }
            : {
                phase: "fail",
                endpoint,
                error: result.error,
                ...(result.code === undefined ? {} : { code: result.code }),
              },
        );
      } catch (error) {
        if (!controller.signal.aborted)
          setHealth({ phase: "fail", endpoint, error: safeError(error) });
      }
    })();
    return () => controller.abort();
  }, [authToken, endpoint, enabled, rpc, tick]);

  return health;
}

/** Step 4: the local ReMe service that DSH talks to. */
export function ServiceSection({
  t,
  endpoint,
  pythonPath,
  control,
  remeCheck,
  autoStart,
  writable,
  current,
  onAutoStart,
}: {
  t: EnvTranslator;
  endpoint: string;
  pythonPath: string;
  control: ServiceControl;
  /** What the selected interpreter has; the shared rule decides a start. */
  remeCheck: RemeCheck | undefined;
  autoStart: boolean;
  writable: boolean;
  /** False while the state on screen belongs to another endpoint. */
  current: boolean;
  onAutoStart(checked: boolean): void;
}): JSX.Element {
  const { state, stopping, stopError, forgetTarget, forgetting, log } = control;
  const [confirming, setConfirming] = useState(false);
  const [forgetConfirm, setForgetConfirm] = useState(false);
  // The confirmation belongs to one phase and endpoint: keeping it open across
  // a transition would re-show "Stop?" on top of a live service later.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setConfirming(false);
    setForgetConfirm(false);
  }, [endpoint, state.phase, state.data?.pid, forgetTarget]);
  // A conflict can still be our own unhealthy service (booting, wrong token);
  // offer the same Stop affordance instead of leaving the tab without a way
  // out other than editing the record by hand. `stopped`/`starting` with a
  // live pid is a recorded launch that never became healthy: Stop is the only
  // way to clear it, so it must be offered there too.
  const canStop =
    current &&
    (state.phase === "running" ||
      (state.phase === "conflict" && state.data?.managed === true) ||
      ((state.phase === "stopped" || state.phase === "starting") &&
        state.data?.pid !== undefined));
  // Only a pid this plugin recorded can be stopped; a foreign running service
  // has no managed flag and keeps the button disabled.
  const canSignal =
    state.data?.managed === true ||
    ((state.phase === "stopped" || state.phase === "starting") &&
      state.data?.pid !== undefined);
  // A stale record may be a recycled pid that now belongs to a foreign
  // process. Stop then refuses, so Start must stay available: the Host stops a
  // provably owned previous launch, discards a foreign record, and reports an
  // unverifiable one. Hiding Start would leave that card without a way out.
  const canRestart = state.phase === "stopped" && state.data?.pid !== undefined;
  // The record can only be forgotten when the Host named one: it is the
  // record-only way out of a process the Host refuses to signal.
  const canForget = current && forgetTarget !== undefined;
  // The confirmation names the record, so the user sees what they are about
  // to delete and not only where it lives.
  const forgetText = forgetTarget
    ? `${fill(t("forgetConfirmRecord"), { endpoint: forgetTarget.endpoint })}${
        forgetTarget.pid === undefined
          ? ""
          : ` ${fill(t("forgetConfirmPid"), { pid: forgetTarget.pid })}`
      }`
    : "";

  return (
    <section className="reme-env-card">
      <header>
        <span className="reme-env-step" aria-hidden="true">
          4
        </span>
        <div>
          <h4>{t("controlTitle")}</h4>
          <p>{t("serviceHint")}</p>
        </div>
        <aside>
          {!current ||
          state.phase === "checking" ||
          state.phase === "starting" ? (
            <StatusBadge tone="progress">
              {t(
                !current
                  ? "checking"
                  : state.phase === "starting"
                  ? "serviceBadgeStarting"
                  : "checking",
              )}
            </StatusBadge>
          ) : state.phase === "running" ? (
            <StatusBadge tone="ok">
              {state.data?.version
                ? `ReMe ${state.data.version}`
                : t("serviceRunning")}
            </StatusBadge>
          ) : state.phase === "conflict" ? (
            <StatusBadge tone="progress">
              {t(
                state.data?.managed
                  ? "serviceBadgeUnhealthy"
                  : "serviceBadgeConflict",
              )}
            </StatusBadge>
          ) : state.phase === "stopped" && state.data?.pid !== undefined ? (
            <StatusBadge tone="progress">{t("serviceBadgeStale")}</StatusBadge>
          ) : (
            <StatusBadge tone="bad">{t("serviceBadgeStopped")}</StatusBadge>
          )}
        </aside>
      </header>
      <div className="reme-env-card-body">
        <div className="reme-env-factrow">
          <span className="reme-env-field-label">{t("endpointLabel")}</span>
          <code className="reme-env-path" title={endpoint}>
            {endpoint}
          </code>
          <button
            type="button"
            className="reme-settings-button"
            onClick={control.refresh}
          >
            {t("recheck")}
          </button>
        </div>
        {current &&
        (state.phase === "running" || state.phase === "conflict") ? (
          <p className="reme-settings-muted">
            {state.data?.managed
              ? fill(t("managedNote"), { pid: state.data?.pid ?? "" })
              : state.phase === "running"
              ? t("foreignNote")
              : null}
          </p>
        ) : null}
        {current && state.phase === "conflict" ? (
          <p className="reme-settings-muted">
            {state.data?.managed ? t("conflictManagedNote") : t("conflictNote")}
          </p>
        ) : null}
        {current &&
        state.phase === "stopped" &&
        state.data?.pid !== undefined ? (
          <p className="reme-settings-muted">{t("staleNote")}</p>
        ) : null}
        {current && state.phase === "error" && state.error ? (
          <p className="reme-settings-error">{state.error}</p>
        ) : null}
        {stopError ? (
          <p className="reme-settings-error" role="alert">
            {stopError}
          </p>
        ) : null}
        {current && canForget ? (
          <p className="reme-settings-muted">{t("forgetHint")}</p>
        ) : null}
        <div className="reme-env-cta">
          {canStop ? (
            confirming ? (
              <div className="reme-env-confirm">
                <p>{t("stopConfirm")}</p>
                <button
                  type="button"
                  className="reme-settings-save"
                  disabled={stopping}
                  onClick={() => {
                    setConfirming(false);
                    void control.stop();
                  }}
                >
                  {t("stopService")}
                </button>
                <button
                  type="button"
                  className="reme-settings-discard"
                  onClick={() => setConfirming(false)}
                >
                  {t("cancel")}
                </button>
              </div>
            ) : (
              <button
                type="button"
                className="reme-settings-button"
                disabled={stopping || !canSignal}
                onClick={() => setConfirming(true)}
              >
                {t("stopService")}
              </button>
            )
          ) : null}
          {!canStop || canRestart ? (
            <button
              type="button"
              className="reme-settings-save"
              disabled={
                !current ||
                state.phase === "starting" ||
                state.phase === "conflict" ||
                !canHostService(remeCheck) ||
                !pythonPath
              }
              onClick={() => void control.start()}
            >
              {t("startService")}
            </button>
          ) : null}
          {current && canForget ? (
            forgetConfirm ? (
              <div className="reme-env-confirm">
                <p>{forgetText}</p>
                <button
                  type="button"
                  className="reme-settings-discard"
                  disabled={forgetting}
                  onClick={() => {
                    setForgetConfirm(false);
                    void control.forget();
                  }}
                >
                  {t("forgetRecord")}
                </button>
                <button
                  type="button"
                  className="reme-settings-button"
                  onClick={() => setForgetConfirm(false)}
                >
                  {t("cancel")}
                </button>
              </div>
            ) : (
              <button
                type="button"
                className="reme-settings-discard"
                disabled={forgetting}
                onClick={() => setForgetConfirm(true)}
              >
                {t("forgetRecord")}
              </button>
            )
          ) : null}
          <button
            type="button"
            className="reme-settings-button"
            onClick={() => void control.viewLog()}
          >
            {t("viewLogs")}
          </button>
          <label className="reme-settings-check reme-env-autostart">
            <input
              type="checkbox"
              checked={autoStart}
              disabled={!writable}
              onChange={(event) => {
                if (!writable) return;
                onAutoStart(event.target.checked);
              }}
            />
            <span>{t("autoStartLabel")}</span>
          </label>
        </div>
        {log.open ? (
          <div className="reme-env-result">
            <p className="reme-settings-muted">
              {`${t("envPath")}: ${log.logPath}`}
            </p>
            <pre className="reme-env-log">{log.lines || t("logEmpty")}</pre>
            <button
              type="button"
              className="reme-settings-button"
              onClick={control.dismissLog}
            >
              {t("dismiss")}
            </button>
          </div>
        ) : null}
      </div>
    </section>
  );
}
