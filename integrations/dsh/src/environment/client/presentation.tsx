import type { ReactNode } from "react";

import type { PythonEnv, PythonOrigin } from "../types.js";
import type { EnvTranslator } from "./locales.js";

/** Tone of a status chip: neutral, healthy, broken, or in progress. */
export type BadgeTone = "" | "ok" | "bad" | "progress";

export interface Badge {
  tone: BadgeTone;
  text: string;
}

/** One dot-plus-text chip, shared by the hero, the cards and the metrics. */
export function StatusBadge({
  tone,
  children,
}: {
  tone: BadgeTone;
  children: ReactNode;
}): JSX.Element {
  return (
    <span className={`reme-status-badge ${tone}`}>
      <i />
      {children}
    </span>
  );
}

/** One pill used for origins, versions and capability markers. */
export function Tag({
  tone = "",
  title,
  children,
}: {
  tone?: "" | "ok" | "bad" | "warn" | "brand";
  title?: string;
  children: ReactNode;
}): JSX.Element {
  return (
    <span className={`reme-env-tag ${tone}`} title={title}>
      {children}
    </span>
  );
}

export function originLabel(t: EnvTranslator, origin: PythonOrigin): string {
  switch (origin.kind) {
    case "conda":
      return t("originConda").replace("{name}", origin.envName);
    case "python.org":
      return t("originPythonOrg");
    case "windows-store":
      return t("originStore");
    case "venv":
      return t("originVenv");
    case "system":
      return t("originSystem");
    default:
      return t("originUnknown");
  }
}

/** Human-readable reason behind the "recommended" marker. */
export function recommendReason(t: EnvTranslator, env: PythonEnv): string {
  const named = env.origin.kind === "conda" && env.origin.envName !== "base";
  if (env.reme?.installed)
    return t(named ? "recommendWhyReme" : "recommendWhyRemeAny");
  return t(named ? "recommendWhyNamed" : "recommendWhyFallback");
}

/** The reme-ai, recommendation and version pills one interpreter carries. */
export function InterpreterBadges({
  t,
  env,
  suggested,
}: {
  t: EnvTranslator;
  env: PythonEnv;
  suggested: string;
}): JSX.Element {
  const outdated = env.reme?.versionOk === false;
  return (
    <>
      {env.reme?.installed ? (
        <Tag
          tone={outdated ? "warn" : "ok"}
          title={outdated ? t("upgradeRecommended") : undefined}
        >{`ReMe ${env.reme.version}`}</Tag>
      ) : null}
      {env.exe === suggested ? (
        <Tag tone="brand" title={recommendReason(t, env)}>
          {t("recommended")}
        </Tag>
      ) : null}
      {env.ok ? null : <Tag tone="bad">{t("tooOld")}</Tag>}
    </>
  );
}

/** Label, control and hint of one settings field. */
export function EnvField({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}): JSX.Element {
  return (
    <div className="reme-settings-field">
      <label>{label}</label>
      {children}
      {hint ? <small>{hint}</small> : null}
    </div>
  );
}
