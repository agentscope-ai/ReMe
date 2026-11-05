import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

import {
  checkReme,
  compareVersions,
  parseRemeVersion,
  remeCliPath,
  remeVersionAtLeast,
} from "../dist/environment/host/reme.js";
import {
  InstallJobs,
  pipArgs,
  tailLog,
} from "../dist/environment/host/install.js";

const SNIPPETS = {
  version: "importlib.metadata",
  core: "import agentscope",
};

function remeIO(handlers, platform = "win32") {
  return {
    run: async (exe, args) => {
      const handler = handlers[exe];
      if (!handler) return { code: 1, stdout: "", stderr: "no handler" };
      return handler(args);
    },
    platform,
    exists: (path) =>
      path.endsWith("Scripts\\reme.exe") || path.endsWith("bin\\reme"),
  };
}

test("compares dotted versions across ReMe's four-segment scheme", () => {
  assert.equal(compareVersions("0.4.0.9", "0.4.0"), 1);
  assert.equal(compareVersions("0.4.0", "0.4.0"), 0);
  assert.equal(compareVersions("0.3.9.9", "0.4.0"), -1);
  assert.equal(compareVersions("1.0", "0.4.0.9"), 1);
  assert.equal(remeVersionAtLeast("0.4.0.9"), true);
  assert.equal(remeVersionAtLeast("0.4.0"), true);
  assert.equal(remeVersionAtLeast("0.4.1.11"), true);
  assert.equal(remeVersionAtLeast("0.3.2"), false);
  assert.equal(remeVersionAtLeast(""), false);
  assert.equal(remeVersionAtLeast("v0.4.0"), false);
});

test("parses a version line out of noisy metadata output", () => {
  assert.equal(parseRemeVersion("0.4.0.9\n"), "0.4.0.9");
  assert.equal(parseRemeVersion("warning\n0.4.0.9\ntrailer"), "0.4.0.9");
  assert.equal(parseRemeVersion("PackageNotFoundError"), "");
  assert.equal(parseRemeVersion(""), "");
});

test("checks an installed interpreter including [core] and the CLI", async () => {
  const io = remeIO({
    "C:\\env\\python.exe": (args) =>
      String(args[1]).includes(SNIPPETS.version)
        ? { code: 0, stdout: "0.4.0.9\n", stderr: "" }
        : { code: 0, stdout: "", stderr: "" },
  });
  const check = await checkReme(io, "C:\\env\\python.exe");
  assert.equal(check.installed, true);
  assert.equal(check.version, "0.4.0.9");
  assert.equal(check.versionOk, true);
  assert.equal(check.hasCore, true);
  assert.equal(check.cliPath, "C:\\env\\Scripts\\reme.exe");
  assert.equal(check.error, "");
});

test("finds the CLI for venv interpreters that live inside Scripts", async () => {
  const io = remeIO({
    "C:\\venv\\Scripts\\python.exe": (args) =>
      String(args[1]).includes(SNIPPETS.version)
        ? { code: 0, stdout: "0.4.1.1\n", stderr: "" }
        : { code: 0, stdout: "", stderr: "" },
  });
  io.exists = (path) => path.toLowerCase() === "c:\\venv\\scripts\\reme.exe";
  const check = await checkReme(io, "C:\\venv\\Scripts\\python.exe");
  assert.equal(check.cliPath, "C:\\venv\\Scripts\\reme.exe");
});

test("flags old versions and missing [core] extras", async () => {
  const io = remeIO({
    "C:\\env\\python.exe": (args) =>
      String(args[1]).includes(SNIPPETS.version)
        ? { code: 0, stdout: "0.3.2\n", stderr: "" }
        : { code: 1, stdout: "", stderr: "ModuleNotFoundError" },
  });
  const check = await checkReme(io, "C:\\env\\python.exe");
  assert.equal(check.installed, true);
  assert.equal(check.versionOk, false);
  assert.equal(check.hasCore, false);
});

test("reports a missing installation without core or CLI", async () => {
  const io = remeIO({
    "C:\\env\\python.exe": () => ({ code: 1, stdout: "", stderr: "not found" }),
  });
  const check = await checkReme(io, "C:\\env\\python.exe");
  assert.equal(check.installed, false);
  assert.equal(check.versionOk, false);
  assert.equal(check.hasCore, false);
  assert.equal(check.cliPath, "");
});

test("rejects relative interpreter paths", async () => {
  const check = await checkReme(remeIO({}), "python.exe");
  assert.equal(check.installed, false);
  assert.match(check.error, /absolute/);
});

test("composes pip arguments for install, upgrade and mirrors", () => {
  assert.deepEqual(pipArgs({ pythonPath: "C:\\env\\python.exe" }), [
    "-m",
    "pip",
    "install",
    "--disable-pip-version-check",
    "--no-input",
    "reme-ai[core]",
  ]);
  assert.deepEqual(
    pipArgs({ pythonPath: "C:\\env\\python.exe", upgrade: true }),
    [
      "-m",
      "pip",
      "install",
      "--disable-pip-version-check",
      "--no-input",
      "-U",
      "reme-ai[core]",
    ],
  );
  assert.deepEqual(
    pipArgs({
      pythonPath: "C:\\env\\python.exe",
      indexUrl: "https://pypi.tuna.tsinghua.edu.cn/simple",
    }).slice(-3),
    [
      "--index-url",
      "https://pypi.tuna.tsinghua.edu.cn/simple",
      "reme-ai[core]",
    ],
  );
  assert.ok(
    pipArgs({
      pythonPath: "C:\\env\\python.exe",
      forceReinstall: true,
    }).includes("--force-reinstall"),
  );
  assert.throws(() =>
    pipArgs({ pythonPath: "C:\\env\\python.exe", indexUrl: "ftp://x" }),
  );
  // Credentials would be echoed in the job command, the log and the browser.
  assert.throws(
    () =>
      pipArgs({
        pythonPath: "C:\\env\\python.exe",
        indexUrl: "https://user:token@mirror.example/simple",
      }),
    /must not embed credentials/,
  );
});

test("tails a log with CRLF and progress carriage returns", (t) => {
  const directory = mkdtempSync(join(tmpdir(), "reme-install-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const path = join(directory, "install.log");
  writeFileSync(path, "line1\r\nline2\r\nprogress 40%\rprogress 80%\rline3\n");
  const tail = tailLog(path);
  assert.ok(tail.includes("line1"));
  assert.ok(tail.includes("line3"));
  assert.ok(tail.includes("progress 80%"));
  assert.equal(tailLog(join(directory, "missing.log")), "");
});

function fakeSpawner() {
  const children = [];
  const spawner = {
    start(options) {
      // The real spawner opens the job log before spawning, so pruning must see
      // the new file too.
      if (options?.logPath) writeFileSync(options.logPath, "");
      const child = {
        pid: 4000 + children.length,
        killed: false,
        onExit(callback) {
          child.exit = callback;
        },
        kill(force = false) {
          child.killed = true;
          child.force = force;
        },
      };
      children.push(child);
      return child;
    },
  };
  return { spawner, children };
}

function makeJobs(t) {
  const directory = mkdtempSync(join(tmpdir(), "reme-jobs-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const { spawner, children } = fakeSpawner();
  const jobs = new InstallJobs({ spawner, logDir: directory });
  return { jobs, children, directory };
}

test("finds the reme CLI beside a Windows interpreter on any host", () => {
  const seen = [];
  const cli = remeCliPath(
    {
      run: async () => ({ code: 0, stdout: "", stderr: "" }),
      platform: "win32",
      exists: (path) => {
        seen.push(path);
        return path.endsWith("reme.exe");
      },
    },
    "C:\\env\\python.exe",
  );
  assert.equal(cli, "C:\\env\\Scripts\\reme.exe");
  assert.deepEqual(seen, ["C:\\env\\Scripts\\reme.exe"]);
});

test("finds the reme CLI in a POSIX bin directory", () => {
  const cli = remeCliPath(
    {
      run: async () => ({ code: 0, stdout: "", stderr: "" }),
      platform: "linux",
      exists: () => true,
    },
    "/home/u/venv/bin/python",
  );
  assert.equal(cli, "/home/u/venv/bin/reme");
});

test("keeps exactly the documented number of install logs", (t) => {
  const { jobs, directory } = makeJobs(t);
  for (let index = 0; index < 24; index += 1)
    writeFileSync(join(directory, `install-old-${index}.log`), "x");
  const started = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  const remaining = readdirSync(directory).filter((name) =>
    name.endsWith(".log"),
  );
  // The steady state is the documented count, not one more than it.
  assert.equal(remaining.length, 20);
  assert.ok(remaining.includes(basename(started.logPath)));
});

test("tracks a successful pip job through exit", (t) => {
  const { jobs, children, directory } = makeJobs(t);
  const started = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  assert.equal(started.phase, "running");
  assert.ok(started.logPath.startsWith(directory));
  assert.match(started.command, /reme-ai\[core\]/);
  assert.equal(jobs.status()?.id, started.id);

  children[0].exit(0);
  const done = jobs.status(started.id);
  assert.equal(done?.phase, "succeeded");
  assert.equal(done?.exitCode, 0);
  assert.ok(done?.finishedAt);
});

test("tracks a failed pip job and keeps the log tail", (t) => {
  const { jobs, children } = makeJobs(t);
  const started = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  writeFileSync(started.logPath, "ERROR: Could not find a version\n");
  children[0].exit(1);
  const done = jobs.status(started.id);
  assert.equal(done?.phase, "failed");
  assert.equal(done?.exitCode, 1);
  assert.match(done?.logTail ?? "", /Could not find a version/);
});

test("keeps the interpreter locked until a cancelled job exits", (t) => {
  const { jobs, children } = makeJobs(t);
  const started = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  assert.equal(jobs.cancel(started.id), true);
  assert.equal(children[0].killed, true);
  // The process still owns the interpreter, so the job stays running until the
  // exit is observed and no second installation may start.
  const cancelling = jobs.status(started.id);
  assert.equal(cancelling?.phase, "running");
  assert.equal(cancelling?.cancelling, true);
  assert.equal(
    jobs.start({ pythonPath: "C:\\env\\python.exe" }).id,
    started.id,
  );
  assert.equal(jobs.cancel(started.id), false);
  children[0].exit(1); // A late exit settles the cancelled job, not a failure.
  assert.equal(jobs.status(started.id)?.phase, "cancelled");
  assert.equal(jobs.status(started.id)?.cancelling, undefined);
});

test("dispose stops the installations the manager still owns", (t) => {
  const { jobs, children } = makeJobs(t);
  const started = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  jobs.dispose();
  assert.equal(children[0].killed, true);
  children[0].exit(null);
  assert.equal(jobs.status(started.id)?.phase, "cancelled");
});

test("reuses the running job for the same interpreter and validates paths", (t) => {
  const { jobs } = makeJobs(t);
  const first = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  const second = jobs.start({ pythonPath: "C:\\env\\python.exe" });
  assert.equal(first.id, second.id);
  assert.throws(() => jobs.start({ pythonPath: "relative\\python.exe" }));
  // Root-relative Windows paths pass win32.isAbsolute but are not absolute in
  // the sense the install needs (they resolve against the current drive).
  assert.throws(() => jobs.start({ pythonPath: "\\python.exe" }));
  assert.equal(jobs.status("unknown-job"), null);
});

test("restores the running job of one interpreter, not the newest job", (t) => {
  // Two interpreters may install at the same time, so "the newest job" is not
  // necessarily the one a tab is looking for.
  const { jobs } = makeJobs(t);
  const first = jobs.start({ pythonPath: "C:\\env\\a\\python.exe" });
  const second = jobs.start({ pythonPath: "C:\\env\\b\\python.exe" });
  assert.notEqual(first.id, second.id);
  assert.equal(jobs.status("", "C:\\env\\a\\python.exe")?.id, first.id);
  assert.equal(jobs.status("", "C:\\env\\b\\python.exe")?.id, second.id);
  // An interpreter without a running job has nothing to restore.
  assert.equal(jobs.status("", "C:\\env\\c\\python.exe"), null);
  // Without an interpreter the newest job overall is still reported.
  assert.equal(jobs.status("")?.id, second.id);
  assert.equal(jobs.status(first.id)?.id, first.id);
  // A named job wins over the interpreter lookup: the poller asks for its own
  // job by id.
  assert.equal(jobs.status(first.id, "C:\\env\\b\\python.exe")?.id, first.id);
});

test("an interpreter is matched by its own path semantics", (t) => {
  const { jobs } = makeJobs(t);
  const started = jobs.start({ pythonPath: "C:\\env\\a\\python.exe" });
  // Windows paths fold case and separators, so the same interpreter is found.
  assert.equal(jobs.status("", "c:/env/a/python.exe")?.id, started.id);
  // A POSIX path stays case-sensitive.
  const posix = jobs.start({ pythonPath: "/opt/venv/bin/python" });
  assert.equal(jobs.status("", "/opt/venv/bin/python")?.id, posix.id);
  assert.equal(jobs.status("", "/opt/Venv/bin/python"), null);
});
