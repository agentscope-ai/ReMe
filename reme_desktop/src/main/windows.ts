import { app, BrowserWindow, screen, nativeTheme } from "electron";
import { companionPlacement, workspacePlacement } from "./placement.ts";
import { StateUpdates } from "./view-state.ts";
import { channels } from "../shared/protocol.ts";
import type {
  DesktopCommand,
  DesktopState,
  WindowLayout,
} from "../shared/protocol.ts";
import type { Storage } from "./storage.ts";
import { surfaceKey, type SurfaceSize } from "../shared/surface.ts";

export class DesktopWindows {
  readonly companion: BrowserWindow;
  readonly chat: BrowserWindow;
  layout: WindowLayout = {
    visible: true,
    body: { left: 12, top: 12, size: 180 },
    bubble: {
      left: 0,
      top: 0,
      width: 320,
      maxWidth: 320,
      height: 180,
    },
  };
  private anchor: { x: number; y: number };
  private visible = true;
  private dragging: {
    cursor: { x: number; y: number };
    anchor: { x: number; y: number };
    moved: boolean;
  } | null = null;
  private state: DesktopState | null = null;
  private disposed = false;
  private readonly companionUpdates = new StateUpdates(true);
  private readonly chatUpdates = new StateUpdates(false);
  private companionBounds = "";
  private surfaceSize: SurfaceSize | undefined;

  constructor(
    readonly storage: Storage,
    private readonly paths: { preload: string; renderer: string; icon: string },
    private readonly onVisibility: (visible: boolean) => void,
    onFocus: (focused: boolean) => void,
    private readonly onCompanionVisibility: (visible: boolean) => void,
  ) {
    const area = screen.getPrimaryDisplay().workArea;
    this.anchor = storage.settings.anchor ?? {
      x: area.x + area.width - 110,
      y: area.y + area.height - 24,
    };
    const webPreferences = {
      preload: paths.preload,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      autoplayPolicy: "no-user-gesture-required" as const,
    };
    this.companion = new BrowserWindow({
      width: 204,
      height: 204,
      show: false,
      frame: false,
      transparent: true,
      hasShadow: false,
      resizable: false,
      maximizable: false,
      fullscreenable: false,
      skipTaskbar: true,
      focusable: false,
      acceptFirstMouse: true,
      alwaysOnTop: true,
      type: process.platform === "darwin" ? "panel" : undefined,
      webPreferences,
    });
    const display = screen.getDisplayNearestPoint(this.anchor);
    this.chat = new BrowserWindow({
      ...workspacePlacement(
        display.workArea,
        display.bounds,
        storage.settings.windowBounds,
      ),
      minWidth: Math.min(900, display.workArea.width),
      minHeight: Math.min(560, display.workArea.height),
      show: false,
      frame: process.platform === "darwin",
      titleBarStyle: process.platform === "darwin" ? "hiddenInset" : undefined,
      trafficLightPosition:
        process.platform === "darwin" ? { x: 20, y: 25 } : undefined,
      transparent: process.platform === "darwin",
      backgroundColor: process.platform === "darwin" ? "#00000000" : "#f4f5f7",
      visualEffectState: "followWindow",
      title: "Remy · ReMe",
      icon: paths.icon,
      // This renderer also plays audio while the workspace is hidden.
      webPreferences: { ...webPreferences, backgroundThrottling: false },
    });
    if (process.platform === "darwin") {
      // The optional native material owns no tasks; its views are owned by NSWindow.
      this.chat.webContents.once("did-finish-load", async () => {
        if (this.disposed || nativeTheme.prefersReducedTransparency) return;
        try {
          const { default: glass } = await import("electron-liquid-glass");
          if (
            !this.disposed &&
            glass.isGlassSupported() &&
            glass.addView(this.chat.getNativeWindowHandle(), {
              cornerRadius: 18,
            }) >= 0
          ) {
            return;
          }
        } catch {
          /* Built-in vibrancy remains available without the native addon. */
        }
        if (!this.disposed) this.chat.setVibrancy("sidebar");
      });
    }
    const rememberBounds = () => {
      if (
        this.disposed ||
        this.chat.isMinimized() ||
        this.chat.isMaximized() ||
        this.chat.isFullScreen()
      )
        return;
      storage.settings.windowBounds = this.chat.getBounds();
      storage.changed();
    };
    this.chat.on("move", rememberBounds);
    this.chat.on("resize", rememberBounds);
    for (const window of [this.companion, this.chat]) {
      if (process.platform !== "darwin") window.setMenu(null);
      window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
      window.webContents.on("will-navigate", (event) => event.preventDefault());
      window.on("close", (event) => {
        if (!this.disposed) {
          event.preventDefault();
          if (window === this.companion) this.hideCompanion();
          else this.closeChat();
        }
      });
      const view = window === this.chat ? "chat" : "companion";
      if (process.env.ELECTRON_RENDERER_URL)
        void window.loadURL(
          `${process.env.ELECTRON_RENDERER_URL}?view=${view}`,
        );
      else void window.loadFile(this.paths.renderer, { query: { view } });
    }
    this.companion.setIgnoreMouseEvents(true, { forward: true });
    if (process.platform === "darwin")
      this.companion.setVisibleOnAllWorkspaces(true, {
        visibleOnFullScreen: true,
      });
    this.companion.once("ready-to-show", this.reposition);
    const updateVisibility = () => {
      const visible = this.chat.isVisible() && !this.chat.isMinimized();
      if (this.state?.chatVisible !== visible) this.onVisibility(visible);
    };
    this.chat.on("show", updateVisibility);
    this.chat.on("hide", updateVisibility);
    this.chat.on("minimize", updateVisibility);
    this.chat.on("restore", updateVisibility);
    this.chat.on("focus", () => onFocus(true));
    this.chat.on("blur", () => {
      this.chat.webContents.setIgnoreMenuShortcuts(false);
      onFocus(false);
    });
    this.chat.webContents.on("before-input-event", (_event, input) => {
      const modifier =
        input.control || (process.platform === "darwin" && input.meta);
      // Suppress Reload while leaving keydown/keyup available to the audio host.
      this.chat.webContents.setIgnoreMenuShortcuts(
        input.type === "keyDown" &&
          input.code === "KeyR" &&
          modifier &&
          !input.alt &&
          !input.shift &&
          this.state?.activePage === "remy",
      );
    });
    screen.on("display-removed", this.reposition);
    screen.on("display-added", this.reposition);
    screen.on("display-metrics-changed", this.reposition);
  }

  isSender(id: number): boolean {
    return (
      !this.disposed &&
      [this.companion, this.chat].some((window) => window.webContents.id === id)
    );
  }

  get companionVisible(): boolean {
    return this.visible;
  }

  isCompanionSender(id: number): boolean {
    return this.visible && id === this.companion.webContents.id;
  }

  update(state: DesktopState): void {
    this.state = state;
    this.reposition();
    for (const [window, updates] of [
      [this.companion, this.companionUpdates],
      [this.chat, this.chatUpdates],
    ] as const) {
      const patch = updates.update(state);
      if (patch && !window.isDestroyed())
        window.webContents.send(channels.state, patch);
    }
  }

  private reposition = (): void => {
    if (this.disposed) return;
    const area = screen.getDisplayNearestPoint(this.anchor).workArea;
    const placement = companionPlacement(
      this.anchor,
      this.storage.settings.scale,
      this.state?.surface?.kind ?? null,
      area,
      this.surfaceSize?.key === surfaceKey(this.state?.surface ?? null)
        ? this.surfaceSize
        : undefined,
    );
    this.anchor = placement.anchor;
    const encoded = JSON.stringify(placement.bounds);
    if (encoded !== this.companionBounds) {
      this.companionBounds = encoded;
      this.companion.setBounds(placement.bounds, false);
    }
    const layout = { ...placement.layout, visible: this.visible };
    if (JSON.stringify(this.layout) !== JSON.stringify(layout)) {
      this.layout = layout;
      this.companion.webContents.send(channels.layout, this.layout);
    }
    if (this.visible && !this.companion.isVisible())
      this.companion.showInactive();
    else if (!this.visible && this.companion.isVisible()) this.companion.hide();
  };

  measureSurface(size: SurfaceSize): void {
    if (
      this.disposed ||
      !this.state?.surface ||
      !size ||
      size.key !== surfaceKey(this.state.surface) ||
      !Number.isFinite(size.width) ||
      !Number.isFinite(size.height) ||
      size.width < 40 ||
      size.width > 640 ||
      size.height < 24 ||
      size.height > 480
    )
      return;
    const next = {
      key: size.key,
      width: Math.ceil(size.width),
      height: Math.ceil(size.height),
    };
    if (
      this.surfaceSize?.key === next.key &&
      this.surfaceSize.width === next.width &&
      this.surfaceSize.height === next.height
    )
      return;
    this.surfaceSize = next;
    this.reposition();
  }

  openChat(): void {
    if (this.chat.isMinimized()) this.chat.restore();
    if (!this.chat.isMaximized() && !this.chat.isFullScreen()) {
      const bounds = this.chat.getBounds();
      const display = screen.getDisplayMatching(bounds);
      this.chat.setBounds(
        workspacePlacement(display.workArea, display.bounds, bounds),
      );
    }
    this.chat.show();
    if (process.platform === "darwin") app.focus({ steal: true });
    this.chat.focus();
  }

  closeChat(): void {
    this.chat.hide();
  }
  showCompanion(): void {
    if (this.visible) return;
    this.visible = true;
    this.reposition();
    this.onCompanionVisibility(true);
  }
  hideCompanion(): void {
    if (!this.visible) return;
    this.visible = false;
    this.reposition();
    this.onCompanionVisibility(false);
  }

  command(command: DesktopCommand): void {
    if (!this.chat.isDestroyed())
      this.chat.webContents.send(channels.command, command);
  }

  resetPosition(): void {
    const area = screen.getDisplayNearestPoint(this.anchor).workArea;
    this.anchor = {
      x: area.x + area.width - 110,
      y: area.y + area.height - 24,
    };
    this.reposition();
    this.storage.settings.anchor = { ...this.anchor };
    this.storage.changed();
    this.showCompanion();
  }

  pointer(interactive: boolean): void {
    if (!this.disposed)
      this.companion.setIgnoreMouseEvents(!interactive && !this.dragging, {
        forward: true,
      });
  }

  drag(phase: "start" | "move" | "end", sender: number): boolean {
    if (
      sender !== this.companion.webContents.id ||
      (phase === "start" ? !this.visible : !this.dragging)
    )
      return false;
    if (phase === "start") {
      this.dragging = {
        cursor: screen.getCursorScreenPoint(),
        anchor: { ...this.anchor },
        moved: false,
      };
      this.companion.setIgnoreMouseEvents(false);
    } else if (this.dragging) {
      if (phase === "move") this.dragging.moved = true;
      if (this.dragging.moved) {
        const cursor = screen.getCursorScreenPoint();
        this.anchor = {
          x: this.dragging.anchor.x + cursor.x - this.dragging.cursor.x,
          y: this.dragging.anchor.y + cursor.y - this.dragging.cursor.y,
        };
        this.reposition();
      }
      if (phase === "end") {
        this.dragging = null;
        this.storage.settings.anchor = { ...this.anchor };
        this.storage.changed();
      }
    }
    return true;
  }

  destroy(): void {
    this.disposed = true;
    screen.removeListener("display-removed", this.reposition);
    screen.removeListener("display-added", this.reposition);
    screen.removeListener("display-metrics-changed", this.reposition);
    this.companion.destroy();
    this.chat.destroy();
  }
}
