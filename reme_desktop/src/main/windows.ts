import { BrowserWindow, screen } from "electron";
import { companionPlacement, clamp } from "./placement.ts";
import { StateUpdates } from "./view-state.ts";
import { channels } from "../shared/protocol.ts";
import type {
  DesktopCommand,
  DesktopState,
  WindowLayout,
} from "../shared/protocol.ts";
import type { Storage } from "./storage.ts";

export class DesktopWindows {
  readonly companion: BrowserWindow;
  readonly chat: BrowserWindow;
  layout: WindowLayout = {
    body: { left: 12, top: 12, size: 180 },
    bubble: { left: 0, top: 0, width: 320, below: false, tail: 270 },
  };
  private anchor: { x: number; y: number };
  private dragging: {
    cursor: { x: number; y: number };
    anchor: { x: number; y: number };
    moved: boolean;
  } | null = null;
  private state: DesktopState | null = null;
  private disposed = false;
  private positionChatOnOpen = true;
  private readonly companionUpdates = new StateUpdates(true);
  private readonly chatUpdates = new StateUpdates(false);
  private geometry = "";
  private companionBounds = "";

  constructor(
    readonly storage: Storage,
    private readonly paths: { preload: string; renderer: string },
    private readonly onVisibility: (visible: boolean) => void,
    onFocus: (focused: boolean) => void,
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
    this.chat = new BrowserWindow({
      width: 440,
      height: 640,
      minWidth: 340,
      minHeight: 430,
      show: false,
      frame: false,
      backgroundColor: "#f5f7f4",
      title: "Remy · ReMe",
      webPreferences,
    });
    for (const window of [this.companion, this.chat]) {
      window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
      window.webContents.on("will-navigate", (event) => event.preventDefault());
      window.on("close", (event) => {
        if (!this.disposed) {
          event.preventDefault();
          window.hide();
          if (window === this.chat) this.onVisibility(false);
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
    this.companion.once("ready-to-show", () => this.companion.showInactive());
    this.chat.on("show", () => this.onVisibility(true));
    this.chat.on("hide", () => this.onVisibility(false));
    this.chat.on("focus", () => onFocus(true));
    this.chat.on("blur", () => onFocus(false));
    screen.on("display-removed", this.reposition);
    screen.on("display-metrics-changed", this.reposition);
  }

  isSender(id: number): boolean {
    return (
      !this.disposed &&
      [this.companion, this.chat].some((window) => window.webContents.id === id)
    );
  }

  update(state: DesktopState): void {
    this.state = state;
    const geometry = `${state.settings.scale}:${state.surface?.kind ?? "none"}`;
    if (geometry !== this.geometry) {
      this.geometry = geometry;
      this.reposition();
    }
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
    );
    this.anchor = placement.anchor;
    const encoded = JSON.stringify(placement.bounds);
    if (encoded !== this.companionBounds) {
      this.companionBounds = encoded;
      this.companion.setBounds(placement.bounds, false);
    }
    if (JSON.stringify(this.layout) !== JSON.stringify(placement.layout)) {
      this.layout = placement.layout;
      this.companion.webContents.send(channels.layout, this.layout);
    }
    const bounds = this.chat.getBounds();
    const width = Math.min(bounds.width, area.width);
    const height = Math.min(bounds.height, area.height);
    if (
      this.positionChatOnOpen ||
      bounds.x + bounds.width < area.x ||
      bounds.x > area.x + area.width ||
      bounds.y + bounds.height < area.y ||
      bounds.y > area.y + area.height
    ) {
      this.chat.setBounds(
        {
          width,
          height,
          x: Math.round(
            clamp(
              this.anchor.x - width - 80,
              area.x,
              area.x + area.width - width,
            ),
          ),
          y: Math.round(
            clamp(
              this.anchor.y - height,
              area.y,
              area.y + area.height - height,
            ),
          ),
        },
        false,
      );
    }
  };

  openChat(): void {
    this.positionChatOnOpen = !this.chat.isVisible();
    this.reposition();
    this.positionChatOnOpen = false;
    this.chat.show();
    this.chat.focus();
  }

  closeChat(): void {
    this.chat.hide();
  }
  showCompanion(): void {
    this.companion.showInactive();
  }

  command(command: DesktopCommand): void {
    const window =
      command.type === "settings.open" ? this.chat : this.companion;
    window.webContents.send(channels.command, command);
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

  drag(phase: "start" | "move" | "end"): void {
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
  }

  destroy(): void {
    this.disposed = true;
    screen.removeListener("display-removed", this.reposition);
    screen.removeListener("display-metrics-changed", this.reposition);
    this.companion.destroy();
    this.chat.destroy();
  }
}
