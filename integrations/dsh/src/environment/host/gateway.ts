import type { Context } from "@deepseek-ai/cordis";
import { Remote, TypertRemoteService } from "@deepseek-ai/dsh-typert-protocol";

import type {
  ConnectionHealth,
  EnvSummary,
  InstallStatus,
  LlmTestResult,
  PythonDetection,
  RemeCheck,
  ServiceForgetResult,
  ServiceLog,
  ServiceStartResult,
  ServiceState,
  ServiceStopResult,
} from "../types.js";
import type { ReMeEnvManager } from "./manager.js";

declare module "@deepseek-ai/cordis" {
  interface Context {
    /** Local environment manager mounted by the DSH adapter. */
    remeEnv: ReMeEnvManager;
  }
}

/**
 * Host projection for the environment management tab.
 *
 * DSH exposes these methods in SRC mode (no generated typert artifact), which
 * parses the method source: every method takes exactly one plain identifier
 * parameter named `input`, with no destructuring, defaults, or rest, and the
 * client sends `{ args: { input: ... } }`.
 */
export class ReMeEnvGateway extends TypertRemoteService {
  static inject = ["remeEnv"];

  constructor(ctx: Context) {
    // The service key must differ from the manager registered as `remeEnv`:
    // TypertRemoteService registers itself under this key, and two providers
    // for one key fail plugin activation. The wire namespace stays `remeEnv`,
    // which is the endpoint the client calls.
    super(ctx, "remeEnvGateway", { namespace: "remeEnv" });
  }

  /** Detect local Python interpreters; cached briefly unless refreshed. */
  @Remote("python")
  async python(
    input: { refresh?: boolean } | undefined,
  ): Promise<PythonDetection> {
    return this.ctx.remeEnv.detectPython(input ?? {});
  }

  /** Inspect reme-ai, the [core] extras and the CLI for one interpreter. */
  @Remote("checkReme")
  async checkReme(
    input: { pythonPath?: string } | undefined,
  ): Promise<RemeCheck> {
    return this.ctx.remeEnv.checkReme(input?.pythonPath ?? "");
  }

  /** Start the one-click pip install and return the running job. */
  @Remote("install")
  async install(
    input:
      | {
          pythonPath?: string;
          upgrade?: boolean;
          indexUrl?: string;
          forceReinstall?: boolean;
        }
      | undefined,
  ): Promise<InstallStatus> {
    return this.ctx.remeEnv.install({
      pythonPath: input?.pythonPath ?? "",
      upgrade: input?.upgrade === true,
      indexUrl: typeof input?.indexUrl === "string" ? input.indexUrl : "",
      forceReinstall: input?.forceReinstall === true,
    });
  }

  /** Latest (or named) install job progress, including the log tail. */
  @Remote("installStatus")
  async installStatus(
    input: { jobId?: string; pythonPath?: string } | undefined,
  ): Promise<InstallStatus | null> {
    return this.ctx.remeEnv.installStatus(
      input?.jobId ?? "",
      input?.pythonPath ?? "",
    );
  }

  /** Ask a running install to stop; polling observes the cancelled phase. */
  @Remote("installCancel")
  async installCancel(input: { jobId?: string } | undefined): Promise<boolean> {
    return this.ctx.remeEnv.installCancel(input?.jobId ?? "");
  }

  /** Masked .env values for the workspace; secrets are never returned. */
  @Remote("readEnv")
  async readEnv(
    input: { workspaceDir?: string } | undefined,
  ): Promise<EnvSummary> {
    return this.ctx.remeEnv.readEnv(input?.workspaceDir ?? "");
  }

  /** Merge edited values into the workspace .env and return fresh state. */
  @Remote("writeEnv")
  async writeEnv(
    input:
      | { workspaceDir?: string; values?: Record<string, string> }
      | undefined,
  ): Promise<EnvSummary> {
    return this.ctx.remeEnv.writeEnv(
      input?.workspaceDir ?? "",
      input?.values ?? {},
    );
  }

  /** Probe the configured LLM endpoint with the selected interpreter. */
  @Remote("testLlm")
  async testLlm(
    input: { workspaceDir?: string; pythonPath?: string } | undefined,
  ): Promise<LlmTestResult> {
    return this.ctx.remeEnv.testLlm(
      input?.workspaceDir ?? "",
      input?.pythonPath ?? "",
    );
  }

  /**
   * Probe the configured endpoint from this Host. The Environment hero and the
   * service card both read Host results, so a browser on another machine cannot
   * disagree with the Host about "local".
   */
  @Remote("health")
  async health(
    input: { endpoint?: string; authToken?: string } | undefined,
  ): Promise<ConnectionHealth> {
    return this.ctx.remeEnv.health(
      input?.endpoint ?? "",
      input?.authToken ?? "",
    );
  }

  /** Phase of the local ReMe service: running, conflict or stopped. */
  @Remote("serviceStatus")
  async serviceStatus(
    input: { endpoint?: string; authToken?: string } | undefined,
  ): Promise<ServiceState> {
    return this.ctx.remeEnv.serviceStatus(
      input?.endpoint ?? "",
      input?.authToken ?? "",
    );
  }

  /** Spawn a detached `reme start`; poll serviceStatus until healthy. */
  @Remote("serviceStart")
  async serviceStart(
    input:
      | {
          endpoint?: string;
          pythonPath?: string;
          workspaceDir?: string;
          authToken?: string;
        }
      | undefined,
  ): Promise<ServiceStartResult> {
    return this.ctx.remeEnv.serviceStart(
      input?.endpoint ?? "",
      input?.pythonPath ?? "",
      input?.workspaceDir ?? "",
      input?.authToken ?? "",
    );
  }

  /** Stop the plugin-managed service; foreign instances are untouched. */
  @Remote("serviceStop")
  async serviceStop(
    input: { endpoint?: string } | undefined,
  ): Promise<ServiceStopResult> {
    return this.ctx.remeEnv.serviceStop(input?.endpoint ?? "");
  }

  /**
   * Drop the service record without signalling the process. This is the way
   * out of a live record whose identity cannot be verified, where Start and
   * Stop both refuse to act.
   *
   * `endpoint` is the record's own endpoint, which is not necessarily the one
   * on screen, and `expectPid` pins the record so a replaced one is not
   * dropped silently.
   */
  @Remote("serviceForget")
  async serviceForget(
    input: { endpoint?: string; expectPid?: number } | undefined,
  ): Promise<ServiceForgetResult> {
    return this.ctx.remeEnv.serviceForget(
      input?.endpoint ?? "",
      typeof input?.expectPid === "number" ? input.expectPid : undefined,
    );
  }

  /** Tail of the workspace service log for start-failure diagnosis. */
  @Remote("serviceLog")
  async serviceLog(
    input: { workspaceDir?: string } | undefined,
  ): Promise<ServiceLog> {
    return this.ctx.remeEnv.serviceLog(input?.workspaceDir ?? "");
  }
}

export default ReMeEnvGateway;
