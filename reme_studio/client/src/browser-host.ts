import { ClientController } from "./controller.ts";
import { serializeError, UiError } from "./i18n.ts";
import { resolveWorkspaceLink } from "./links.ts";
import type {
  StudioAction,
  StudioApi,
  StudioCommand,
  StudioState,
  Presentation,
  Settings,
} from "./protocol.ts";
import { defaultSettings, type ClientStorage } from "./settings.ts";

function read<T>(key: string, fallback: T): T {
  try {
    return JSON.parse(localStorage.getItem(key) ?? "null") ?? fallback;
  } catch {
    return fallback;
  }
}

class BrowserStorage implements ClientStorage {
  settings: Settings;
  private readonly presentations = new Map(
    Object.entries(
      read<Record<string, Presentation>>("reme:presentations", {}),
    ),
  );
  private dirty = false;

  constructor(endpoint: string) {
    this.settings = {
      ...defaultSettings,
      ...read<Partial<Settings>>("reme:settings", {}),
      endpoint,
      transparentBackground: false,
    };
  }

  scope(endpoint: string, workspace: string): string {
    return JSON.stringify([endpoint, workspace]);
  }

  presentation(key: string): Presentation {
    const current = this.presentations.get(key) ?? {
      initialized: false,
      receipts: {},
    };
    this.presentations.set(key, current);
    return current;
  }

  changed(): void {
    this.dirty = true;
  }

  async flush(): Promise<void> {
    if (!this.dirty) return;
    const { endpoint: _, ...preferences } = this.settings;
    localStorage.setItem("reme:settings", JSON.stringify(preferences));
    localStorage.setItem(
      "reme:presentations",
      JSON.stringify(Object.fromEntries(this.presentations)),
    );
    this.dirty = false;
  }
}

export class BrowserHost implements StudioApi {
  private readonly controller: ClientController;
  private readonly listeners = new Set<() => void>();
  private readonly commands = new Set<(command: StudioCommand) => void>();
  private state: StudioState;

  constructor(
    endpoint: string,
    private readonly openExternal?: (url: string) => Promise<void>,
  ) {
    this.controller = new ClientController(
      new BrowserStorage(endpoint),
      (state) => {
        this.state = { ...state };
        for (const listener of this.listeners) listener();
      },
      undefined,
      (notice, shown) => {
        if (
          !("Notification" in globalThis) ||
          Notification.permission !== "granted"
        )
          return;
        const notification = new Notification(notice.title, {
          body: notice.body,
        });
        notification.onshow = shown;
        notification.onclick = () => {
          window.focus();
          this.controller.setChatVisible(
            true,
            notice.messageId
              ? { kind: "message", id: notice.messageId }
              : undefined,
          );
          notification.close();
        };
      },
      (command) => {
        for (const callback of this.commands) callback(command);
      },
    );
    this.state = { ...this.controller.state };
  }

  snapshot = (): StudioState => this.state;

  subscribe = (callback: () => void): (() => void) => {
    this.listeners.add(callback);
    return () => {
      this.listeners.delete(callback);
    };
  };

  private updateVisibility = (): void => {
    this.controller.setChatVisible(!document.hidden);
    this.controller.setChatFocused(!document.hidden && document.hasFocus());
  };

  start(): () => void {
    this.updateVisibility();
    this.controller.start();
    document.addEventListener("visibilitychange", this.updateVisibility);
    window.addEventListener("focus", this.updateVisibility);
    window.addEventListener("blur", this.updateVisibility);
    return () => {
      document.removeEventListener("visibilitychange", this.updateVisibility);
      window.removeEventListener("focus", this.updateVisibility);
      window.removeEventListener("blur", this.updateVisibility);
      void this.controller.close().catch(console.error);
    };
  }

  onCommand: StudioApi["onCommand"] = (callback) => {
    this.commands.add(callback);
    return () => {
      this.commands.delete(callback);
    };
  };

  workspace: StudioApi["workspace"] = async (request) => {
    try {
      return await this.controller.workspace(request);
    } catch (error) {
      return { error: serializeError(error) };
    }
  };

  transcribe: StudioApi["transcribe"] = async ({ id, audio }) => {
    try {
      return { text: await this.controller.speech.transcribe(id, audio) };
    } catch (error) {
      return { error: serializeError(error) };
    }
  };

  action: StudioApi["action"] = async (action) => {
    try {
      await this.dispatch(action);
      return null;
    } catch (error) {
      return serializeError(error);
    }
  };

  private async dispatch(action: StudioAction): Promise<void> {
    const { controller } = this;
    const { speech } = controller;
    switch (action.type) {
      case "chat.send":
        return controller.send(action.query);
      case "chat.stop":
        return controller.stopReply();
      case "page.select":
        return controller.selectPage(action.page);
      case "agent.select":
        return controller.selectAgent(action.agentName);
      case "session.new":
        return controller.newSession();
      case "messages.read":
        return controller.markRead(action.sessionId, action.ids);
      case "items.read":
        return controller.markAllRead();
      case "interaction":
        return controller.interact();
      case "speech.ready":
        return speech.rendererReady(action.ready, action.token);
      case "speech.capture":
        return speech.beginCapture(action.id);
      case "speech.recording":
        return speech.recording(action.id);
      case "speech.cancel":
        return speech.cancelCapture(action.id);
      case "speech.report":
        return speech.report(action.id, action.status);
      case "speech.clearError":
        return speech.clearError();
      case "speech.retry":
        return speech.retry();
      case "speech.stop":
        return speech.cancel();
      case "speech.speak":
        if (
          action.sessionId === controller.model.sessionId &&
          controller.state.selectedAgentName !== "Remy"
        )
          controller.selectAgent("Remy");
        if (action.sessionId !== controller.state.sessionId)
          throw new UiError("sessionUnavailable");
        return controller.speakMessage(action.messageId);
      case "settings.update": {
        const { type: _, endpoint: _endpoint, ...update } = action;
        if (update.notifications && "Notification" in globalThis)
          update.notifications =
            (await Notification.requestPermission()) === "granted";
        return controller.settings(update);
      }
      case "document.open":
        return this.openLink(action.path);
      case "link.open":
        return this.openLink(action.url, action.sourcePath);
      case "agent.reveal": {
        const agent = controller.state.agents?.find(
          (item) => item.agent_name === action.agentName,
        );
        if (!agent?.task_path) throw new UiError("invalidWorkspaceRequest");
        return this.openLink(agent.task_path);
      }
      case "item.action":
        switch (action.action) {
          case "open":
            return controller.setChatVisible(true, action.target);
          case "explain":
            return controller.explain(action.target);
          case "copy":
            return navigator.clipboard.writeText(
              (await controller.item(action.target)).content,
            );
          case "remind": {
            const message = await controller.item(action.target);
            if (
              "Notification" in globalThis &&
              Notification.permission === "granted"
            )
              new Notification("Remy", { body: message.content });
            return controller.setChatVisible(true, action.target);
          }
        }
        return;
      default:
        throw new UiError("invalidAction");
    }
  }

  private async openLink(href: string, sourcePath = ""): Promise<void> {
    if (/^https?:\/\//i.test(href)) {
      const url = new URL(href);
      if (url.username || url.password) throw new UiError("invalidAction");
      if (this.openExternal) await this.openExternal(url.href);
      else window.open(url, "_blank", "noopener,noreferrer");
      return;
    }
    const path = resolveWorkspaceLink(
      href,
      sourcePath,
      this.state.clientInfo?.workspace_dir,
    );
    if (!path) throw new UiError("invalidWorkspaceRequest");
    this.controller.openDocument(path);
  }
}
