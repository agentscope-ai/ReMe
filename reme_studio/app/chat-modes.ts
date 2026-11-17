import type { AppConfig, ChatMode } from "./types";

export function isJobAvailable(
  config: AppConfig | undefined,
  name: string,
  backend: string,
): boolean {
  const job = config?.jobs[name];
  return Boolean(
    job &&
      job.backend === backend &&
      job.enabled &&
      job.enable_serve &&
      (!config?.service.jobs || config.service.jobs.includes(name)),
  );
}

export function isChatAvailable(
  config: AppConfig | undefined,
  mode: ChatMode,
): boolean {
  const serves = (name: string, backend: string) =>
    isJobAvailable(config, name, backend);
  return mode === "chat"
    ? serves("chat", "stream")
    : serves("pa_chat", "stream") &&
        serves("pa_session", "base") &&
        serves("pa_new_session", "base");
}
