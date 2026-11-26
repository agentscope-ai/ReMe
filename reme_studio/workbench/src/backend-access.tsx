import { createContext, useContext, useState } from "react";
import type {
  BackendRequest,
  BackendStatus,
} from "../../client/src/backend.ts";
import {
  errorText,
  translator,
  UiError,
  type Language,
} from "../../client/src/i18n.ts";
import { useResource } from "./resources.ts";

export const BackendAccess = createContext({
  endpoint: "",
  token: "",
  setToken: (_token: string) => {},
});

export function needsBackendToken(error: unknown) {
  return error instanceof UiError && error.key === "backendAuthRequired";
}

export function useBackendRequest() {
  const { endpoint, token } = useContext(BackendAccess);
  const request = useResource();
  return {
    token,
    request: <T,>(value: BackendRequest) =>
      request<T>({
        ...value,
        credentials: token ? { endpoint, token } : undefined,
      }),
  };
}

export function BackendUnlock({ language }: { language: Language }) {
  const { endpoint, setToken } = useContext(BackendAccess);
  const request = useResource();
  const t = translator(language);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [pending, setPending] = useState(false);
  return (
    <form
      className="backend-connection"
      onSubmit={async (event) => {
        event.preventDefault();
        const token = draft.trim();
        if (!token || pending) return;
        setPending(true);
        setError(null);
        try {
          await request<BackendStatus>({
            type: "backend",
            action: "status",
            credentials: { endpoint, token },
          });
          setToken(token);
          setDraft("");
        } catch (reason) {
          setError(reason);
        } finally {
          setPending(false);
        }
      }}
    >
      <label className="backend-field">
        <span>{t("backendAdminToken")}</span>
        <input
          type="password"
          autoComplete="off"
          required
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
        />
      </label>
      <button type="submit" className="secondary" disabled={pending}>
        {t("connect")}
      </button>
      {error != null && <p role="alert">{errorText(error, language)}</p>}
    </form>
  );
}
