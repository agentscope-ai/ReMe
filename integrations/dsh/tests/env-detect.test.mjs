import assert from "node:assert/strict";
import test from "node:test";

import {
  condaRootFromPath,
  detectPythonEnvs,
  isWindowsAppsAlias,
  makeCondaRunner,
  parseCondaEnvs,
  parseCondaRoot,
  parsePathList,
  parsePyListPaths,
  parseRegQuery,
  parseVerifyOutput,
  suggestInterpreter,
  versionAtLeast,
} from "../dist/environment/host/detect.js";

function fakeRun(rules) {
  return async (exe, args) => {
    for (const rule of rules) {
      if (rule.match(exe, args))
        return typeof rule.result === "function"
          ? rule.result(args)
          : rule.result;
    }
    throw new Error(`unexpected command: ${exe} ${args.join(" ")}`);
  };
}

const VERIFY_ARG0 = "-c";

function verifyRule(exe, version, base) {
  return {
    match: (actual, args) =>
      actual.toLowerCase() === exe.toLowerCase() &&
      args[0] === VERIFY_ARG0 &&
      String(args[1]).includes("import sys"),
    result: {
      code: 0,
      stdout: `note\n{"exe":"${exe.replaceAll(
        "\\",
        "\\\\",
      )}","ver":"${version}","base":"${base.replaceAll("\\", "\\\\")}"}\n`,
      stderr: "",
    },
  };
}

/** reme-ai metadata probe; version null means "not installed". */
function remeRule(exe, version) {
  return {
    match: (actual, args) =>
      actual.toLowerCase() === exe.toLowerCase() &&
      args[0] === VERIFY_ARG0 &&
      String(args[1]).includes("importlib.metadata"),
    result: version
      ? { code: 0, stdout: `${version}\n`, stderr: "" }
      : { code: 1, stdout: "", stderr: "PackageNotFoundError" },
  };
}

function windowsFixture() {
  const miniconda = "D:\\ProgramFils\\Miniconda";
  const remeServe = `${miniconda}\\envs\\reme-serve`;
  const old38 = `${miniconda}\\envs\\old38`;
  const storePackage =
    "C:\\Users\\u\\AppData\\Local\\Microsoft\\WindowsApps\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0";
  const rules = [
    {
      match: (exe, args) =>
        exe === "reg.exe" &&
        args[0] === "query" &&
        args[1] === "HKCU\\Software\\Python",
      result: {
        code: 0,
        stdout: [
          "HKEY_CURRENT_USER\\Software\\Python\\PythonCore",
          "",
          "HKEY_CURRENT_USER\\Software\\Python\\PythonCore\\3.12",
          "    (默认)    REG_SZ    3.12",
          "",
          `HKEY_CURRENT_USER\\Software\\Python\\PythonCore\\3.12\\InstallPath`,
          `    (默认)    REG_SZ    ${miniconda}\\`,
          "",
          "HKEY_CURRENT_USER\\Software\\Python\\PythonOrg",
          "",
          "HKEY_CURRENT_USER\\Software\\Python\\PythonOrg\\3.12",
          "",
          `HKEY_CURRENT_USER\\Software\\Python\\PythonOrg\\3.12\\InstallPath`,
          "    (Default)    REG_SZ    C:\\Python312\\",
          "",
          `HKEY_CURRENT_USER\\Software\\Python\\PythonLegacy\\3.8\\InstallPath`,
          "    (Default)    REG_SZ    C:\\Python38",
          "",
          "HKEY_CURRENT_USER\\Software\\Python\\PythonSoftwareFoundation",
          "",
          "HKEY_CURRENT_USER\\Software\\Python\\PythonSoftwareFoundation\\Python.3.11",
          "",
          `HKEY_CURRENT_USER\\Software\\Python\\PythonSoftwareFoundation\\Python.3.11\\InstallPath`,
          `    (Default)    REG_SZ    ${storePackage}\\`,
          "",
        ].join("\r\n"),
        stderr: "",
      },
    },
    {
      match: (exe, args) => exe === "reg.exe" && args[0] === "query",
      result: { code: 1, stdout: "", stderr: "key not found" },
    },
    {
      match: (exe) => exe === "py.exe",
      result: {
        code: null,
        stdout: "",
        stderr: "",
        error: "spawn py.exe ENOENT",
      },
    },
    {
      match: (exe, args) => exe === "where.exe" && args[0] === "python",
      result: {
        code: 0,
        stdout: [
          miniconda + "\\python.exe",
          "C:\\Users\\u\\AppData\\Local\\Microsoft\\WindowsApps\\python.exe",
          "",
        ].join("\r\n"),
        stderr: "",
      },
    },
    {
      match: (exe, args) => exe === "where.exe" && args[0] === "conda",
      result: {
        code: 0,
        stdout: `${miniconda}\\Library\\bin\\conda.bat\r\n`,
        stderr: "",
      },
    },
    {
      match: (exe, args) =>
        exe === `${miniconda}\\Scripts\\conda.exe` &&
        args[0] === "info" &&
        args[1] === "--json",
      result: {
        code: 0,
        stdout: JSON.stringify({ root_prefix: miniconda }),
        stderr: "",
      },
    },
    {
      match: (exe, args) =>
        exe === `${miniconda}\\Scripts\\conda.exe` &&
        args[0] === "env" &&
        args[1] === "list",
      result: {
        code: 0,
        stdout: JSON.stringify({
          envs: [miniconda, remeServe, old38],
        }),
        stderr: "",
      },
    },
    verifyRule(`${miniconda}\\python.exe`, "3.12.3", miniconda),
    verifyRule(`${remeServe}\\python.exe`, "3.12.13", remeServe),
    verifyRule(`${old38}\\python.exe`, "3.8.10", old38),
    verifyRule("C:\\Python312\\python.exe", "3.12.2", "C:\\Python312"),
    verifyRule("C:\\Python38\\python.exe", "3.8.10", "C:\\Python38"),
    verifyRule(`${storePackage}\\python.exe`, "3.11.9", storePackage),
    remeRule(`${miniconda}\\python.exe`, "0.4.0.9"),
    remeRule(`${remeServe}\\python.exe`, "0.4.0.9"),
    remeRule(`${old38}\\python.exe`, null),
    remeRule("C:\\Python312\\python.exe", "0.4.1.11"),
    remeRule("C:\\Python38\\python.exe", null),
    remeRule(`${storePackage}\\python.exe`, null),
  ];
  return {
    rules,
    miniconda,
    remeServe,
    old38,
    storePackage,
    io: {
      run: fakeRun(rules),
      platform: "win32",
      env: { USERPROFILE: "C:\\Users\\u", ProgramData: "C:\\ProgramData" },
      homedir: "C:\\Users\\u",
      exists: (path) => path === `${miniconda}\\Scripts\\conda.exe`,
    },
  };
}

test("parses PEP 514 registry output across locales, mojibake and empty values", () => {
  const found = parseRegQuery(
    [
      "HKEY_CURRENT_USER\\Software\\Python\\PythonCore\\3.12\\InstallPath",
      "    (\uFFFD\uFFFD)    REG_SZ    D:\\ProgramFils\\Miniconda\\",
      "",
      "HKEY_CURRENT_USER\\Software\\Python\\PythonOrg\\3.11\\InstallPath",
      "    (Default)    REG_SZ    C:\\Python311",
      "",
      "HKEY_CURRENT_USER\\Software\\Python\\Localized\\3.10\\InstallPath",
      "    (默认)    REG_SZ    C:\\Python310",
      "",
      "HKEY_CURRENT_USER\\Software\\Python\\Empty\\InstallPath",
      "    (\uFFFD\uFFFD)    REG_SZ    (\uFFFD\uFFFD\uFFFD\uFFFD)",
      "",
    ].join("\r\n"),
  );
  assert.deepEqual(
    found.map((entry) => entry.exe),
    [
      "D:\\ProgramFils\\Miniconda\\python.exe",
      "C:\\Python311\\python.exe",
      "C:\\Python310\\python.exe",
    ],
  );
  assert.equal(
    found[0].source,
    "registry HKEY_CURRENT_USER\\Software\\Python\\PythonCore\\3.12\\InstallPath",
  );
});

test("parses py launcher and path listings", () => {
  assert.deepEqual(
    parsePyListPaths(
      [
        " -V:3.12 *        D:\\ProgramFils\\Miniconda\\python.exe",
        " -V:3.11          C:\\Python311\\python.exe",
        " -V:2.7           C:\\Python27\\python.exe",
      ].join("\n"),
    ),
    [
      "D:\\ProgramFils\\Miniconda\\python.exe",
      "C:\\Python311\\python.exe",
      "C:\\Python27\\python.exe",
    ],
  );
  assert.deepEqual(parsePathList("a\\python.exe\r\n\r\nb\\python.exe\n"), [
    "a\\python.exe",
    "b\\python.exe",
  ]);
});

test("drops Store alias stubs but keeps real Store interpreters", () => {
  assert.equal(
    isWindowsAppsAlias(
      "C:\\Users\\u\\AppData\\Local\\Microsoft\\WindowsApps\\python3.exe",
    ),
    true,
  );
  assert.equal(
    isWindowsAppsAlias(
      "C:\\Users\\u\\AppData\\Local\\Microsoft\\WindowsApps\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\python.exe",
    ),
    false,
  );
  assert.equal(isWindowsAppsAlias("C:\\Python312\\python.exe"), false);
});

test("compares interpreter versions numerically", () => {
  assert.equal(versionAtLeast("3.12.3"), true);
  assert.equal(versionAtLeast("3.11.0"), true);
  assert.equal(versionAtLeast("3.12"), true);
  assert.equal(versionAtLeast("3.10.8"), false);
  assert.equal(versionAtLeast("3.8"), false);
  assert.equal(versionAtLeast(""), false);
  assert.equal(versionAtLeast("abc"), false);
});

test("parses interpreter verify output embedded in noisy stdout", () => {
  const info = parseVerifyOutput(
    'Warning: something\n{"exe":"D:\\\\x\\\\python.exe","ver":"3.12.3","base":"D:\\\\x"}\n',
  );
  assert.deepEqual(info, {
    exe: "D:\\x\\python.exe",
    version: "3.12.3",
    basePrefix: "D:\\x",
  });
  assert.equal(parseVerifyOutput("no json here"), undefined);
});

test("parses conda json output defensively", () => {
  assert.deepEqual(parseCondaEnvs('{"envs":["D:\\\\a","D:\\\\b"]}'), [
    "D:\\a",
    "D:\\b",
  ]);
  assert.deepEqual(parseCondaEnvs("not json"), []);
  assert.deepEqual(parseCondaEnvs('{"envs":42}'), []);
  assert.equal(parseCondaRoot('{"root_prefix":"D:\\\\a"}'), "D:\\a");
  assert.equal(parseCondaRoot("garbage"), "");
});

test("resolves every fixture with its own platform's path semantics", () => {
  // The flavor comes from the path, not from the host running the test, so
  // these assertions hold on Windows, macOS and Linux alike.
  assert.equal(
    condaRootFromPath("D:\\ProgramFils\\Miniconda\\Scripts\\conda.exe"),
    "D:\\ProgramFils\\Miniconda",
  );
  assert.equal(
    condaRootFromPath("D:\\ProgramFils\\Miniconda\\condabin\\conda.bat"),
    "D:\\ProgramFils\\Miniconda",
  );
  assert.equal(
    condaRootFromPath("/opt/homebrew/Caskroom/miniconda/base/bin/conda"),
    "/opt/homebrew/Caskroom/miniconda/base",
  );
  assert.equal(condaRootFromPath("D:\\mc\\conda.bat"), "");
});

test("derives conda roots and prefers the real conda.exe over the .bat", async () => {
  assert.equal(condaRootFromPath("D:\\mc\\Library\\bin\\conda.bat"), "D:\\mc");
  assert.equal(condaRootFromPath("D:\\mc\\Scripts\\conda.exe"), "D:\\mc");
  assert.equal(
    condaRootFromPath("/home/u/miniconda3/bin/conda"),
    "/home/u/miniconda3",
  );
  assert.equal(condaRootFromPath("D:\\mc\\conda.bat"), "");

  const calls = [];
  const run = async (exe, args) => {
    calls.push([exe, args]);
    return { code: 0, stdout: "{}", stderr: "" };
  };
  const runner = makeCondaRunner(
    run,
    "D:\\mc\\Library\\bin\\conda.bat",
    true,
    (path) => path === "D:\\mc\\Scripts\\conda.exe",
  );
  await runner(["info", "--json"]);
  assert.deepEqual(calls[0][0], "D:\\mc\\Scripts\\conda.exe");

  const fallback = [];
  const runner2 = makeCondaRunner(
    async (exe, args) => {
      fallback.push([exe, args]);
      return { code: null, stdout: "", stderr: "", error: "spawn ENOENT" };
    },
    "D:\\mc\\Library\\bin\\conda.bat",
    true,
    () => false,
  );
  await runner2(["env", "list", "--json"]);
  assert.equal(fallback.at(-1)[0], "cmd.exe");
  assert.ok(fallback.at(-1)[1].includes("/s"));
});

test("detects, classifies and suggests Windows interpreters end to end", async () => {
  const { io, miniconda, remeServe, old38, storePackage } = windowsFixture();
  const detection = await detectPythonEnvs(io);

  assert.equal(detection.platform, "win32");
  assert.equal(detection.condaRoot, miniconda);
  assert.deepEqual(detection.errors, []);
  assert.equal(detection.suggested, `${remeServe}\\python.exe`);

  const exes = detection.envs.map((env) => env.exe.toLowerCase());
  assert.ok(!exes.some((exe) => exe.includes("windowsapps\\\\python.exe")));
  assert.ok(
    !exes.some(
      (exe) =>
        exe ===
        "c:\\users\\u\\appdata\\local\\microsoft\\windowsapps\\python.exe",
    ),
  );
  assert.equal(detection.envs.length, 6);

  const base = detection.envs.find(
    (env) => env.origin.kind === "conda" && env.origin.envName === "base",
  );
  assert.ok(base);
  assert.equal(base.exe, `${miniconda}\\python.exe`);
  assert.equal(base.version, "3.12.3");
  assert.equal(base.ok, true);
  assert.equal(base.origin.root, miniconda);
  assert.ok(base.source.includes("PATH"));
  assert.ok(base.source.includes("conda env list"));
  assert.ok(base.source.includes("registry"));

  const named = detection.envs.find(
    (env) => env.exe === `${remeServe}\\python.exe`,
  );
  assert.ok(named);
  assert.equal(named.origin.kind, "conda");
  assert.equal(named.origin.envName, "reme-serve");

  const legacy = detection.envs.find(
    (env) => env.exe === `${old38}\\python.exe`,
  );
  assert.ok(legacy);
  assert.equal(legacy.ok, false);
  assert.ok(detection.envs.indexOf(legacy) > detection.envs.indexOf(base));

  const pythonOrg = detection.envs.find(
    (env) => env.exe === "C:\\Python312\\python.exe",
  );
  assert.ok(pythonOrg);
  assert.equal(pythonOrg.origin.kind, "python.org");

  const legacy38 = detection.envs.find(
    (env) => env.exe === "C:\\Python38\\python.exe",
  );
  assert.ok(legacy38);
  assert.equal(legacy38.ok, false);

  const store = detection.envs.find(
    (env) => env.exe === `${storePackage}\\python.exe`,
  );
  assert.ok(store);
  assert.equal(store.origin.kind, "windows-store");
  assert.equal(store.ok, true);

  assert.deepEqual(base.reme, {
    installed: true,
    version: "0.4.0.9",
    versionOk: true,
  });
  assert.deepEqual(named.reme, {
    installed: true,
    version: "0.4.0.9",
    versionOk: true,
  });
  assert.deepEqual(pythonOrg.reme, {
    installed: true,
    version: "0.4.1.11",
    versionOk: true,
  });
  assert.deepEqual(store.reme, {
    installed: false,
    version: "",
    versionOk: false,
  });
});

test("suggests nothing usable by preferring reme installs, then named conda envs", () => {
  assert.equal(
    suggestInterpreter([
      {
        exe: "a",
        version: "3.12.0",
        ok: true,
        origin: { kind: "conda", envName: "base", root: "r" },
        source: "",
      },
      {
        exe: "b",
        version: "3.12.0",
        ok: true,
        origin: { kind: "conda", envName: "reme", root: "r" },
        source: "",
      },
    ]),
    "b",
  );
  assert.equal(
    suggestInterpreter([
      {
        exe: "a",
        version: "3.8.0",
        ok: false,
        origin: { kind: "conda", envName: "base", root: "r" },
        source: "",
      },
      {
        exe: "c",
        version: "3.11.9",
        ok: true,
        origin: { kind: "python.org" },
        source: "",
      },
    ]),
    "c",
  );
  // A reme-enabled interpreter outranks a bare named conda env.
  assert.equal(
    suggestInterpreter([
      {
        exe: "a",
        version: "3.12.0",
        ok: true,
        origin: { kind: "conda", envName: "tools", root: "r" },
        source: "",
        reme: { installed: false, version: "" },
      },
      {
        exe: "b",
        version: "3.12.0",
        ok: true,
        origin: { kind: "python.org" },
        source: "",
        reme: { installed: true, version: "0.4.0.9" },
      },
    ]),
    "b",
  );
  // An outdated reme does not make an interpreter ready.
  assert.equal(
    suggestInterpreter([
      {
        exe: "a",
        version: "3.12.0",
        ok: true,
        origin: { kind: "conda", envName: "tools", root: "r" },
        source: "",
        reme: { installed: true, version: "0.3.1.8" },
      },
      {
        exe: "b",
        version: "3.12.0",
        ok: true,
        origin: { kind: "conda", envName: "other", root: "r" },
        source: "",
        reme: { installed: false, version: "" },
      },
    ]),
    "a",
  );
  assert.equal(suggestInterpreter([]), "");
});

test("detects Unix interpreters and merges paths that resolve together", async () => {
  const rules = [
    {
      match: (exe, args) =>
        exe === "which" && args[0] === "-a" && args[1] === "python3",
      result: { code: 0, stdout: "/usr/bin/python3\n", stderr: "" },
    },
    {
      match: (exe, args) =>
        exe === "which" && args[0] === "-a" && args[1] === "python",
      result: { code: 0, stdout: "/usr/bin/python\n", stderr: "" },
    },
    {
      match: (exe, args) =>
        exe === "which" && args[0] === "-a" && args[1] === "conda",
      result: { code: 0, stdout: "/home/u/miniconda3/bin/conda\n", stderr: "" },
    },
    {
      match: (exe, args) =>
        exe === "/home/u/miniconda3/bin/conda" && args[0] === "info",
      result: {
        code: 0,
        stdout: JSON.stringify({ root_prefix: "/home/u/miniconda3" }),
        stderr: "",
      },
    },
    {
      match: (exe, args) =>
        exe === "/home/u/miniconda3/bin/conda" && args[0] === "env",
      result: {
        code: 0,
        stdout: JSON.stringify({
          envs: ["/home/u/miniconda3", "/home/u/miniconda3/envs/reme"],
        }),
        stderr: "",
      },
    },
    verifyRule("/usr/bin/python3", "3.10.12", "/usr"),
    // /usr/bin/python is a shim whose sys.executable reports python3.
    {
      match: (actual, args) =>
        actual === "/usr/bin/python" &&
        args[0] === VERIFY_ARG0 &&
        String(args[1]).includes("import sys"),
      result: {
        code: 0,
        stdout: '{"exe":"/usr/bin/python3","ver":"3.10.12","base":"/usr"}\n',
        stderr: "",
      },
    },
    verifyRule(
      "/home/u/miniconda3/bin/python3",
      "3.12.1",
      "/home/u/miniconda3",
    ),
    verifyRule(
      "/home/u/miniconda3/envs/reme/bin/python3",
      "3.11.9",
      "/home/u/miniconda3/envs/reme",
    ),
    remeRule("/usr/bin/python3", null),
    remeRule("/usr/bin/python", null),
    remeRule("/home/u/miniconda3/bin/python3", "0.4.1.11"),
    remeRule("/home/u/miniconda3/envs/reme/bin/python3", "0.4.1.11"),
  ];
  const detection = await detectPythonEnvs({
    run: fakeRun(rules),
    platform: "linux",
    env: {},
    homedir: "/home/u",
    exists: () => false,
  });
  assert.equal(detection.platform, "linux");
  assert.equal(detection.condaRoot, "/home/u/miniconda3");
  assert.equal(detection.suggested, "/home/u/miniconda3/envs/reme/bin/python3");

  const system = detection.envs.filter((env) => env.origin.kind === "system");
  assert.equal(system.length, 1);
  assert.equal(system[0].exe, "/usr/bin/python3");
  assert.equal(system[0].ok, false);

  const named = detection.envs.find((env) => env.origin.envName === "reme");
  assert.ok(named);
  assert.equal(named.ok, true);
  assert.equal(named.version, "3.11.9");
});

test("probes well-known Unix locations beyond the Host PATH", async () => {
  const homebrew = "/opt/homebrew/bin/python3";
  const pyenv = "/Users/u/.pyenv/shims/python3";
  const rules = [
    {
      match: (exe) => exe === "which",
      result: { code: 1, stdout: "", stderr: "" },
    },
    verifyRule(homebrew, "3.12.4", "/opt/homebrew"),
    // pyenv shims report the real interpreter they dispatch to.
    {
      match: (actual, args) =>
        actual === pyenv &&
        args[0] === VERIFY_ARG0 &&
        String(args[1]).includes("import sys"),
      result: {
        code: 0,
        stdout:
          '{"exe":"/Users/u/.pyenv/versions/3.12.4/bin/python3","ver":"3.12.4","base":"/Users/u/.pyenv/versions/3.12.4"}\n',
        stderr: "",
      },
    },
    remeRule(homebrew, "0.4.1.11"),
    remeRule(pyenv, null),
    remeRule("/Users/u/.pyenv/versions/3.12.4/bin/python3", null),
  ];
  const detection = await detectPythonEnvs({
    run: fakeRun(rules),
    platform: "darwin",
    env: {},
    homedir: "/Users/u",
    exists: (path) => path === homebrew || path === pyenv,
  });
  assert.deepEqual(
    detection.envs.map((env) => env.exe).sort(),
    [homebrew, "/Users/u/.pyenv/versions/3.12.4/bin/python3"].sort(),
  );
  assert.ok(
    detection.envs.every((env) => env.origin.kind === "system"),
    "well-known paths are classified as system installations",
  );
});

test("collects conda failures without rejecting the whole detection", async () => {
  const { io, miniconda } = windowsFixture();
  io.run = async (exe, args) => {
    if (exe === `${miniconda}\\Scripts\\conda.exe`)
      return { code: null, stdout: "", stderr: "", error: "spawn failed" };
    const original = windowsFixture();
    const rule = original.rules.find((entry) => entry.match(exe, args));
    if (!rule) throw new Error(`unexpected: ${exe} ${args.join(" ")}`);
    return typeof rule.result === "function" ? rule.result(args) : rule.result;
  };
  const detection = await detectPythonEnvs(io);
  assert.equal(detection.condaRoot, "");
  assert.ok(detection.errors.some((error) => error.startsWith("conda:")));
  assert.ok(
    detection.envs.some((env) => env.exe === `${miniconda}\\python.exe`),
  );
});
