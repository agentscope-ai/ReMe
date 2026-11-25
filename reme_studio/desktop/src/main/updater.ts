import type { AppUpdater } from "electron-updater";
import type { UpdateCommand, UpdateState } from "../protocol.ts";

type Updater = Pick<
  AppUpdater,
  | "autoDownload"
  | "autoInstallOnAppQuit"
  | "allowPrerelease"
  | "disableWebInstaller"
  | "checkForUpdates"
  | "downloadUpdate"
  | "quitAndInstall"
  | "on"
>;

export class DesktopUpdater {
  private state: UpdateState;
  private operation = 0;

  constructor(
    private readonly updater: Updater | null,
    private readonly publish: (state: UpdateState) => void,
  ) {
    this.state = { status: updater ? "idle" : "manual" };
    if (!updater) return;
    updater.autoDownload = false;
    updater.autoInstallOnAppQuit = false;
    updater.allowPrerelease = false;
    updater.disableWebInstaller = true;
    updater.on("download-progress", ({ percent }) => {
      if (this.state.status === "downloading" && Number.isFinite(percent)) {
        const rounded = Math.max(0, Math.min(100, Math.round(percent)));
        if (rounded !== this.state.percent)
          this.set({ ...this.state, percent: rounded });
      }
    });
    updater.on("error", () => {
      this.operation++;
      this.set({ status: "error" });
    });
  }

  private set(state: UpdateState): void {
    this.state = state;
    this.publish(state);
  }

  async run(command: UpdateCommand): Promise<UpdateState> {
    const { state, updater } = this;
    if (!updater) return state;
    let operation = this.operation;
    try {
      switch (command) {
        case "check": {
          if (
            state.status !== "idle" &&
            state.status !== "current" &&
            state.status !== "error"
          )
            break;
          operation = ++this.operation;
          this.set({ status: "checking" });
          const result = await updater.checkForUpdates();
          if (operation !== this.operation) break;
          this.set(
            result?.isUpdateAvailable
              ? { status: "available", version: result.updateInfo.version }
              : { status: result ? "current" : "manual" },
          );
          break;
        }
        case "download":
          if (state.status !== "available") break;
          operation = ++this.operation;
          this.set({
            status: "downloading",
            version: state.version,
            percent: 0,
          });
          await updater.downloadUpdate();
          if (operation !== this.operation) break;
          this.set({ status: "ready", version: state.version });
          break;
        case "install":
          if (state.status !== "ready") break;
          this.set({ status: "installing", version: state.version });
          updater.quitAndInstall(true, true);
          break;
      }
    } catch {
      if (operation === this.operation) this.set({ status: "error" });
    }
    return this.state;
  }
}
