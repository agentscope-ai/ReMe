"use client";

import { MessageSquarePlus, Sparkles } from "lucide-react";
import { isChatAvailable } from "./chat-modes";
import { useI18n } from "./i18n";
import { useWorkspaceStore } from "./store";
import type { AppConfig, ChatMode } from "./types";

export default function ChatEntries({ config }: { config?: AppConfig }) {
  const { t } = useI18n();
  const openAgent = useWorkspaceStore((state) => state.openAgent);
  return (
    <div className="chat-entries">
      {(["chat", "pa"] as ChatMode[]).map((mode) => {
        const available = isChatAvailable(config, mode);
        const label = t(mode === "pa" ? "paChat" : "chatTab");
        const hint = !config
          ? t("loadingSession")
          : t(mode === "pa" ? "paChatRequiresConfig" : "chatNotEnabled");
        return (
          <button
            key={mode}
            disabled={!available}
            onClick={() => openAgent(mode)}
            aria-label={t("openChat", { mode: label })}
            title={
              available
                ? label
                : mode === "pa" && config
                ? t("paChatSetup")
                : hint
            }
          >
            <span>
              {mode === "pa" ? (
                <Sparkles size={14} />
              ) : (
                <MessageSquarePlus size={14} />
              )}
              {label}
            </span>
            {!available && <small>{hint}</small>}
          </button>
        );
      })}
    </div>
  );
}
