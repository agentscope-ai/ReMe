import { createContext, useContext } from "react";
import { deserializeError } from "../../client/src/i18n.ts";
import type { StudioApi, StudioAction } from "../../client/src/protocol.ts";
import type { openRecorder } from "./speech/recorder.ts";

export type WorkbenchHost = Omit<StudioApi, "action"> & {
  action(action: StudioAction): Promise<void>;
  openRecorder?: typeof openRecorder;
  desktop?: {
    companionAvatar: string;
    minimize(): Promise<void>;
    toggleMaximize(): Promise<void>;
    close(): Promise<void>;
    setCompanionVisible(visible: boolean): Promise<void>;
    resetCompanion(): Promise<void>;
  };
};

export const HostContext = createContext<WorkbenchHost | null>(null);

export function useHost(): WorkbenchHost {
  const host = useContext(HostContext);
  if (!host) throw new Error("The workbench requires a host.");
  return host;
}

export function workbenchHost(api: StudioApi): WorkbenchHost {
  return {
    workspace: (request) => api.workspace(request),
    onCommand: (callback) => api.onCommand(callback),
    transcribe: (request) => api.transcribe(request),
    async action(action) {
      const error = await api.action(action);
      if (error) throw deserializeError(error);
    },
  };
}
