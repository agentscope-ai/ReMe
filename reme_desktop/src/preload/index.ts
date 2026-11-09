import { contextBridge, ipcRenderer } from "electron";
import { channels } from "../shared/protocol.ts";
import type { DesktopApi } from "../shared/protocol.ts";

const subscribe = <T>(channel: string, callback: (value: T) => void) => {
  const listener = (_event: Electron.IpcRendererEvent, value: T) =>
    callback(value);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
};

const api: DesktopApi = {
  bootstrap: () => ipcRenderer.invoke(channels.bootstrap),
  action: (action) => ipcRenderer.invoke(channels.action, action),
  workspace: (request) => ipcRenderer.invoke(channels.workspace, request),
  transcribe: (request) => ipcRenderer.invoke(channels.transcribe, request),
  onState: (callback) => subscribe(channels.state, callback),
  onLayout: (callback) => subscribe(channels.layout, callback),
  onCommand: (callback) => subscribe(channels.command, callback),
  pointer: (interactive) => ipcRenderer.send(channels.pointer, interactive),
  measureSurface: (size) => ipcRenderer.send(channels.surfaceSize, size),
  drag: (phase) => ipcRenderer.send(channels.drag, phase),
};

contextBridge.exposeInMainWorld("desktop", api);
