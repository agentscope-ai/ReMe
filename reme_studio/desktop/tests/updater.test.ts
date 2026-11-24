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
    quitAndInstall() {
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
