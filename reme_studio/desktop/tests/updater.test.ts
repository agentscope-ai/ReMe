import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";
import type { UpdateState } from "../src/protocol.ts";
import { DesktopUpdater } from "../src/main/updater.ts";

test("desktop updates only download and install in response to their commands", async () => {
  let checks = 0;
  let downloads = 0;
  let installs = 0;
  const version = "0.4.4";
  const info = { version, files: [], path: "", sha512: "", releaseDate: "" };
  const published: UpdateState[] = [];
  let finish = () => {};
  const nativeUpdater = Object.assign(new EventEmitter(), {
    autoDownload: true,
    autoInstallOnAppQuit: true,
    allowPrerelease: true,
    disableWebInstaller: false,
    async checkForUpdates() {
      checks++;
      if (checks === 1) throw new Error("network unavailable");
      return { isUpdateAvailable: true, updateInfo: info, versionInfo: info };
    },
    downloadUpdate() {
      downloads++;
      return new Promise<string[]>((resolve) => {
        finish = () => resolve([]);
      });
    },
    quitAndInstall(silent?: boolean, restart?: boolean) {
      assert.equal(silent, true);
      assert.equal(restart, true);
      installs++;
    },
  });
  const updater = new DesktopUpdater(nativeUpdater, (state) =>
    published.push(state),
  );
  assert.equal(nativeUpdater.autoDownload, false);
  assert.equal(nativeUpdater.autoInstallOnAppQuit, false);
  assert.deepEqual(await updater.run("check"), { status: "error" });
  assert.deepEqual(await updater.run("check"), {
    status: "available",
    version,
  });
  assert.equal(downloads, 0);
  await updater.run("install");
  assert.equal(installs, 0);

  const downloading = updater.run("download");
  nativeUpdater.emit("download-progress", { percent: 37.4 });
  assert.deepEqual(await updater.run("check"), {
    status: "downloading",
    version,
    percent: 37,
  });
  await updater.run("download");
  assert.equal(checks, 2);
  assert.equal(downloads, 1);
  finish();
  assert.deepEqual(await downloading, { status: "ready", version });
  assert.deepEqual(await updater.run("check"), { status: "ready", version });
  assert.equal(checks, 2);
  assert.equal(installs, 0);

  assert.deepEqual(await updater.run("install"), {
    status: "installing",
    version,
  });
  await updater.run("install");
  assert.equal(installs, 1);
  assert.deepEqual(
    published.map(({ status }) => status),
    [
      "checking",
      "error",
      "checking",
      "available",
      "downloading",
      "downloading",
      "ready",
      "installing",
    ],
  );
});

function nativeUpdater() {
  const updateInfo = {
    version: "1.2.3",
    files: [],
    path: "",
    sha512: "",
    releaseDate: "",
  };
  return Object.assign(new EventEmitter(), {
    autoDownload: true,
    autoInstallOnAppQuit: true,
    allowPrerelease: true,
    disableWebInstaller: false,
    async checkForUpdates() {
      return { isUpdateAvailable: true, updateInfo, versionInfo: updateInfo };
    },
    async downloadUpdate(): Promise<string[]> {
      return [];
    },
    quitAndInstall(_silent?: boolean, _restart?: boolean) {},
  });
}

test("builds without an update channel keep the manual update route", async () => {
  const updater = new DesktopUpdater(null, () =>
    assert.fail("no update event"),
  );
  for (const command of ["check", "download", "install"] as const) {
    assert.deepEqual(await updater.run(command), { status: "manual" });
  }
});

for (const settle of ["resolve", "reject"] as const) {
  test(`an old update check cannot overwrite a retry when it ${settle}s`, async () => {
    const native = nativeUpdater();
    const result = await native.checkForUpdates();
    let finish = () => {};
    const pending = new Promise<typeof result>((resolve, reject) => {
      finish = () =>
        settle === "resolve"
          ? resolve({ ...result, isUpdateAvailable: false })
          : reject(new Error("old request failed"));
    });
    let checks = 0;
    native.checkForUpdates = () =>
      ++checks === 1 ? pending : Promise.resolve(result);
    const updater = new DesktopUpdater(native, () => {});
    const checking = updater.run("check");
    native.emit("error", new Error("connection interrupted"));
    assert.deepEqual(await updater.run("check"), {
      status: "available",
      version: result.updateInfo.version,
    });
    finish();
    assert.deepEqual(await checking, {
      status: "available",
      version: result.updateInfo.version,
    });
  });
}

test("a download error cannot become ready when the old download finishes", async () => {
  const native = nativeUpdater();
  let finish = () => {};
  native.downloadUpdate = () =>
    new Promise<string[]>((resolve) => {
      finish = () => resolve([]);
    });
  const updater = new DesktopUpdater(native, () => {});
  await updater.run("check");
  const downloading = updater.run("download");
  native.emit("error", new Error("download interrupted"));
  native.emit("download-progress", { percent: 100 });
  finish();
  assert.deepEqual(await downloading, { status: "error" });
  assert.deepEqual(await updater.run("install"), { status: "error" });
  await updater.run("check");
  native.downloadUpdate = async () => [];
  assert.deepEqual(await updater.run("download"), {
    status: "ready",
    version: "1.2.3",
  });
});

test("native installation errors are visible and allow checking again", async () => {
  const native = nativeUpdater();
  native.quitAndInstall = () => {
    native.emit("error", new Error("installer failed"));
  };
  const updater = new DesktopUpdater(native, () => {});
  await updater.run("check");
  await updater.run("download");
  assert.deepEqual(await updater.run("install"), { status: "error" });
  assert.deepEqual(await updater.run("check"), {
    status: "available",
    version: "1.2.3",
  });
});
