import { BrowserWindow, screen, nativeTheme } from "electron";
import {
  companionPlacement,
  workspacePlacement,
  type Rect,
} from "./placement.ts";
import { CompanionHabitat, intersection } from "./habitat.ts";
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
  private readonly habitat: CompanionHabitat;
  private appLayout: WindowLayout;
  private slot: Rect | null = null;
  private reducedMotion = false;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private timerAt = Infinity;
  private visible = true;
  private presented: boolean | undefined;
  private dragging: {
    cursor: { x: number; y: number };
    anchor: { x: number; y: number };
    moved: boolean;
    sender: number;
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
    this.habitat = new CompanionHabitat(this.anchor);
    this.appLayout = {
      ...this.layout,
      presence: { host: "desktop", visible: true, opacity: 1 },
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
    this.chat.on("move", this.reposition);
    this.chat.on("resize", this.reposition);
    this.chat.on("show", this.reposition);
    this.chat.on("hide", this.reposition);
    this.chat.on("minimize", this.reposition);
    this.chat.on("restore", this.reposition);
    this.chat.on("focus", this.reposition);
    this.chat.on("blur", this.reposition);
    this.chat.on("hide", () => this.habitat.unpin());
    this.chat.webContents.on("did-start-loading", () => {
      this.slot = null;
      this.reposition();
    });
    this.chat.webContents.on("render-process-gone", () => {
      this.slot = null;
      this.reposition();
    });
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
    this.chat.on("minimize", () => this.habitat.unpin());
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

  layoutFor(id: number): WindowLayout {
    return id === this.chat.webContents.id ? this.appLayout : this.layout;
  }

  isCompanionSender(id: number): boolean {
    return (
      this.visible &&
      id ===
        (this.layout.presence?.host === "app" ? this.chat : this.companion)
          .webContents.id
    );
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
    const now = Date.now();
    const bounds = this.chat.getContentBounds();
    const zoom = this.chat.webContents.getZoomFactor();
    const available =
      this.chat.isVisible() &&
      !this.chat.isMinimized() &&
      this.state?.activePage === "remy";
    const reportedSlot =
      available && this.slot
        ? {
            x: bounds.x + this.slot.x * zoom,
            y: bounds.y + this.slot.y * zoom,
            width: this.slot.width * zoom,
            height: this.slot.height * zoom,
          }
        : null;
    const slot = reportedSlot ? intersection(reportedSlot, bounds) : null;
    const frame = this.habitat.update(
      {
        displays: screen.getAllDisplays(),
        slot: this.dragging?.moved || !this.visible ? null : slot,
        window: available ? bounds : null,
        scale: this.storage.settings.scale,
        reducedMotion: this.reducedMotion,
        surfaceHeight: this.state?.surface
          ? this.surfaceSize?.key === surfaceKey(this.state.surface)
            ? this.surfaceSize.height
            : this.state.surface.kind === "message"
              ? 180
              : 36
          : 0,
      },
      now,
    );
    this.anchor = frame.anchor;
    const area = screen.getDisplayNearestPoint(this.anchor).workArea;
    const placement = companionPlacement(
      this.anchor,
      this.storage.settings.scale,
      frame.motion ? null : (this.state?.surface?.kind ?? null),
      area,
      this.surfaceSize?.key === surfaceKey(this.state?.surface ?? null)
        ? this.surfaceSize
        : undefined,
    );
    const encoded = JSON.stringify(placement.bounds);
    if (encoded !== this.companionBounds) {
      this.companionBounds = encoded;
      this.companion.setBounds(placement.bounds, false);
    }
    const presence = {
      host: frame.host,
      visible: this.visible,
      opacity: frame.opacity,
      ...(frame.motion ? { motion: frame.motion } : {}),
    };
    const nextLayout: WindowLayout = { ...placement.layout, presence };
    if (JSON.stringify(this.layout) !== JSON.stringify(nextLayout)) {
      this.layout = nextLayout;
      this.companion.webContents.send(channels.layout, this.layout);
    }
    const appArea = slot ? intersection(slot, area) : area;
    const embedded = companionPlacement(
      this.anchor,
      this.storage.settings.scale,
      this.state?.surface?.kind ?? null,
      appArea,
      this.surfaceSize?.key === surfaceKey(this.state?.surface ?? null)
        ? this.surfaceSize
        : undefined,
    );
    const local = (x: number, origin: number) => (x - origin) / zoom;
    const appLayout: WindowLayout =
      frame.host === "app"
        ? {
            body: {
              left: local(
                embedded.bounds.x + embedded.layout.body.left,
                bounds.x,
              ),
              top: local(
                embedded.bounds.y + embedded.layout.body.top,
                bounds.y,
              ),
              size: embedded.layout.body.size / zoom,
            },
            bubble: {
              left: local(
                embedded.bounds.x + embedded.layout.bubble.left,
                bounds.x,
              ),
              top: local(
                embedded.bounds.y + embedded.layout.bubble.top,
                bounds.y,
              ),
              width: embedded.layout.bubble.width / zoom,
              maxWidth: embedded.layout.bubble.maxWidth / zoom,
              height: embedded.layout.bubble.height / zoom,
            },
            presence,
          }
        : {
            ...this.appLayout,
            presence: { host: "desktop", visible: this.visible, opacity: 1 },
          };
    if (JSON.stringify(this.appLayout) !== JSON.stringify(appLayout)) {
      this.appLayout = appLayout;
      this.chat.webContents.send(channels.layout, this.appLayout);
    }
    const show = this.visible && frame.host === "desktop";
    if (show && !this.companion.isVisible()) this.companion.showInactive();
    else if (!show && this.companion.isVisible()) this.companion.hide();
    const presented =
      this.visible && (frame.host === "desktop" || this.chat.isFocused());
    if (presented !== this.presented) {
      this.presented = presented;
      this.onCompanionVisibility(presented);
    }
    if (this.habitat.pending && !this.dragging) {
      const delay = this.habitat.travelling ? 16 : 300;
      if (now + delay < this.timerAt) {
        clearTimeout(this.timer);
        this.timerAt = now + delay;
        this.timer = setTimeout(() => {
          this.timerAt = Infinity;
          this.reposition();
        }, delay);
      }
    }
  };

  measureHabitat(
    value: Parameters<
      import("../shared/protocol.ts").DesktopApi["measureHabitat"]
    >[0],
  ): void {
    if (!value || typeof value.reducedMotion !== "boolean") return;
    const slot = value.slot;
    if (
      slot &&
      (![slot.x, slot.y, slot.width, slot.height].every(Number.isFinite) ||
        slot.x < 0 ||
        slot.y < 0 ||
        slot.width < 0 ||
        slot.height < 0 ||
        [slot.x, slot.y, slot.width, slot.height].some((n) => n > 32768))
    )
      return;
    this.slot = slot;
    this.reducedMotion = value.reducedMotion;
    this.reposition();
  }

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
    this.habitat.unpin();
    if (this.chat.isMinimized()) this.chat.restore();
    this.reposition();
    if (!this.chat.isMaximized() && !this.chat.isFullScreen()) {
      const bounds = this.chat.getBounds();
      const display = screen.getDisplayMatching(bounds);
      this.chat.setBounds(
        workspacePlacement(display.workArea, display.bounds, bounds),
      );
    }
    this.chat.show();
    this.chat.focus();
  }

  closeChat(): void {
    this.habitat.unpin();
    this.chat.hide();
  }
  showCompanion(): void {
    this.visible = true;
    this.reposition();
  }
  hideCompanion(): void {
    this.visible = false;
    this.reposition();
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
    this.habitat.place(this.anchor);
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
      phase === "start"
        ? !this.isCompanionSender(sender)
        : this.dragging?.sender !== sender
    )
      return false;
    if (phase === "start") {
      this.dragging = {
        cursor: screen.getCursorScreenPoint(),
        anchor: { ...this.anchor },
        moved: false,
        sender,
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
        this.habitat.place(this.anchor);
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
    clearTimeout(this.timer);
    screen.removeListener("display-removed", this.reposition);
    screen.removeListener("display-added", this.reposition);
    screen.removeListener("display-metrics-changed", this.reposition);
    this.companion.destroy();
    this.chat.destroy();
  }
}
