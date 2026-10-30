import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { InstallJobs } from "../dist/environment/host/install.js";

/**
 * A fake `InstallSpawner`: no pip is really spawned and every termination the
 * job requests is recorded.
 *
 * The fake child models a stubborn process: it survives the graceful signal
 * and dies only when the force-kill arrives. It never reports an exit on its
 * own, because whether the Host ever observes one is exactly what these
 * tests are about.
 */
function fakeSpawner() {
  const children = [];
  const calls = { starts: 0 };
  const spawner = {
    start(options) {
      calls.starts += 1;
      // The real spawner opens the job log before spawning, so pruning must
      // see the new file too.
      if (options?.logPath) writeFileSync(options.logPath, "");
      const child = {
        pid: 4000 + children.length,
        /** Whether the process still exists. */
        alive: true,
        /** Every termination the job asked for, in order. */
        kills: [],
        onExit(callback) {
          child.exit = callback;
        },
        kill(force = false) {
          child.kills.push(force ? "force" : "graceful");
          if (force) child.alive = false;
        },
        /** The Host observed the process exiting. */
        exit(code) {
          child.alive = false;
          child.exit?.(code);
        },
      };
      children.push(child);
      return child;
    },
  };
  return { spawner, children, calls };
}

/**
 * A controllable stand-in for `setTimeout`.
 *
 * The install job uses one timer only - the escalation from the graceful
 * signal to the force-kill - so driving it by hand keeps the tests
 * deterministic and free of real waiting.
 */
function fakeTimers(t) {
  const originalSetTimeout = globalThis.setTimeout;
  const originalClearTimeout = globalThis.clearTimeout;
  const pending = new Map();
  let sequence = 0;
  globalThis.setTimeout = (callback, delay) => {
    sequence += 1;
    const timer = {
      id: sequence,
      delay,
      /** Whether the job unref()ed the timer. */
      unrefed: false,
      unref() {
        timer.unrefed = true;
        return timer;
      },
      ref() {
        return timer;
      },
    };
    pending.set(timer.id, { timer, callback });
    return timer;
  };
  globalThis.clearTimeout = (timer) => {
    if (timer && typeof timer.id === "number") pending.delete(timer.id);
  };
  t.after(() => {
    globalThis.setTimeout = originalSetTimeout;
    globalThis.clearTimeout = originalClearTimeout;
  });
  return {
    /** The timers that are still pending. */
    pending: () => [...pending.values()].map((entry) => entry.timer),
    /** Fire every pending timer, oldest first. */
    fire() {
      const entries = [...pending.values()];
      pending.clear();
      for (const { callback } of entries) callback();
      return entries.length;
    },
    /**
     * Fire every pending timer, including the ones they schedule themselves.
     *
     * This is what "after every escalation timer has fired" means: a chain of
     * timers must be drained, not only the first one.
     */
    fireAll() {
      let fired = 0;
      for (let round = 0; round < 10; round += 1) {
        const entries = [...pending.values()];
        if (entries.length === 0) break;
        pending.clear();
        for (const { callback } of entries) callback();
        fired += entries.length;
      }
      return fired;
    },
  };
}

function makeJobs(t, options = {}) {
  const directory = mkdtempSync(join(tmpdir(), "reme-install-jobs-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const { spawner, children, calls } = fakeSpawner();
  const jobs = new InstallJobs({ spawner, logDir: directory, ...options });
  return { jobs, children, calls, directory };
}

test("a cancelled job stays running and cancelling until the exit is observed", (t) => {
  const timers = fakeTimers(t);
  const { jobs, children } = makeJobs(t, { killGraceMs: 5_000 });
  const started = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  assert.equal(jobs.cancel(started.id), true);
  // The graceful signal goes out first.
  assert.deepEqual(children[0].kills, ["graceful"]);

  // Every escalation timer has fired and the child still never exited.
  timers.fireAll();
  const status = jobs.status(started.id);
  assert.equal(status?.phase, "running");
  assert.equal(status?.cancelling, true);
  // A failed termination must not be reported as a completed cancellation.
  assert.notEqual(status?.phase, "cancelled");
  assert.notEqual(status?.phase, "failed");
  assert.equal(status?.exitCode, undefined);
  assert.equal(status?.finishedAt, undefined);
  // The force-kill was sent, so the job is waiting for a real exit.
  assert.deepEqual(children[0].kills, ["graceful", "force"]);
});

test("an unconfirmed cancellation keeps the interpreter locked", (t) => {
  const timers = fakeTimers(t);
  const { jobs, children, calls } = makeJobs(t, { killGraceMs: 5_000 });
  const started = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  jobs.cancel(started.id);
  timers.fireAll(); // The escalation fired; the child never exited.

  // A second installation for the same interpreter must not spawn a second
  // child while the first one is still unconfirmed.
  const again = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  assert.equal(again.id, started.id);
  assert.equal(calls.starts, 1);
  assert.equal(children.length, 1);
  // The lock is visible through the interpreter lookup as well.
  const locked = jobs.status("", "C:\\env\\python.exe");
  assert.equal(locked?.id, started.id);
  assert.equal(locked?.cancelling, true);
});

test("the interpreter lock folds case on Windows paths", (t) => {
  const { jobs, calls } = makeJobs(t);
  const started = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  // The Host finds the same interpreter in another spelling.
  const again = jobs.start({ pythonPath: "c:/env/python.exe" });
  assert.equal(again.id, started.id);
  assert.equal(calls.starts, 1);
  assert.equal(jobs.status("", "C:\\ENV\\PYTHON.EXE")?.id, started.id);
  // A POSIX path stays case-sensitive.
  const posix = jobs.start({ pythonPath: "/opt/venv/bin/python" });
  assert.notEqual(posix.id, started.id);
  assert.equal(jobs.status("", "/opt/Venv/bin/python"), null);
});

test("cancel escalates to the force-kill after the grace period", (t) => {
  const timers = fakeTimers(t);
  const { jobs, children } = makeJobs(t, { killGraceMs: 1_234 });
  const started = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  jobs.cancel(started.id);

  // The force-kill is not sent before the grace period is over.
  assert.deepEqual(children[0].kills, ["graceful"]);
  const pending = timers.pending();
  assert.equal(pending.length, 1);
  assert.equal(pending[0].delay, 1_234);
  // A pending escalation must never keep the Host process alive.
  assert.equal(pending[0].unrefed, true);

  timers.fire();
  assert.deepEqual(children[0].kills, ["graceful", "force"]);
});

test("the default cancellation grace gives pip five seconds to clean up", (t) => {
  const timers = fakeTimers(t);
  const { jobs } = makeJobs(t);
  const started = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  jobs.cancel(started.id);
  const pending = timers.pending();
  assert.equal(pending.length, 1);
  assert.equal(pending[0].delay, 5_000);
});

test("dispose force-kills a child that ignores the graceful signal", (t) => {
  const timers = fakeTimers(t);
  const { jobs, children } = makeJobs(t); // The default dispose grace is 0.
  jobs.start({ pythonPath: "C:\\env\\python.exe" });
  jobs.dispose();

  // The force-kill follows the graceful signal immediately, so a child that
  // ignores it cannot outlive the Host.
  assert.deepEqual(children[0].kills, ["graceful", "force"]);
  assert.equal(children[0].alive, false);
  // Nothing is left pending: the plugin is unloading.
  assert.equal(timers.pending().length, 0);
});

test("dispose escalates after the configured grace period", (t) => {
  const timers = fakeTimers(t);
  const { jobs, children } = makeJobs(t, { disposeGraceMs: 250 });
  jobs.start({ pythonPath: "C:\\env\\python.exe" });
  jobs.dispose();

  assert.deepEqual(children[0].kills, ["graceful"]);
  assert.equal(children[0].alive, true);
  const pending = timers.pending();
  assert.equal(pending.length, 1);
  assert.equal(pending[0].delay, 250);
  assert.equal(pending[0].unrefed, true);

  timers.fire();
  assert.deepEqual(children[0].kills, ["graceful", "force"]);
  assert.equal(children[0].alive, false);
});

test("an observed exit settles the job once, as cancelled", (t) => {
  const timers = fakeTimers(t);
  const { jobs, children } = makeJobs(t, { killGraceMs: 5_000 });
  const started = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  jobs.cancel(started.id);

  children[0].exit(1);
  const settled = jobs.status(started.id);
  assert.equal(settled?.phase, "cancelled");
  assert.equal(settled?.cancelling, undefined);
  assert.equal(settled?.exitCode, 1);
  // The pending escalation is dropped, so it cannot fire after the exit.
  assert.equal(timers.pending().length, 0);

  // A second, late exit must not settle the job a second time.
  children[0].exit(1);
  const again = jobs.status(started.id);
  assert.equal(again?.phase, "cancelled");
  assert.equal(again?.finishedAt, settled?.finishedAt);
});

test("dispose does not settle the job; the observed exit does", (t) => {
  const timers = fakeTimers(t);
  const { jobs, children } = makeJobs(t);
  const started = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  jobs.dispose();

  const status = jobs.status(started.id);
  assert.equal(status?.phase, "running");
  assert.equal(status?.cancelling, true);
  assert.equal(status?.finishedAt, undefined);
  // The default dispose grace is zero, so the force-kill already went out and
  // nothing is left pending.
  assert.equal(timers.pending().length, 0);
  // The interpreter is still locked: the child may still be dying.
  assert.equal(
    jobs.start({ pythonPath: "C:\\env\\python.exe" }).id,
    started.id,
  );

  children[0].exit(null);
  assert.equal(jobs.status(started.id)?.phase, "cancelled");
});

test("the status of an unconfirmed cancellation stays JSON-safe", (t) => {
  const timers = fakeTimers(t);
  const { jobs } = makeJobs(t, { killGraceMs: 5_000 });
  const started = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  jobs.cancel(started.id);
  timers.fireAll();

  const status = jobs.status(started.id);
  assert.equal(status?.phase, "running");
  // Optional keys are omitted, never written as undefined.
  assert.equal("exitCode" in status, false);
  assert.equal("finishedAt" in status, false);
  assert.deepEqual(JSON.parse(JSON.stringify(status)), status);
});
