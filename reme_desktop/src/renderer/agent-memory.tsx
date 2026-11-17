import { useEffect, useState } from "react";
import { ArrowUpRight } from "lucide-react";
import type { DesktopState } from "../shared/protocol.ts";
import { errorText, translator } from "../shared/i18n.ts";
import { action } from "./desktop.ts";
import { requestResource } from "./resources.ts";
import DocumentPreview from "./knowledge/DocumentPreview.tsx";

export default function AgentMemory({
  state,
  path,
}: {
  state: DesktopState;
  path: string;
}) {
  const [content, setContent] = useState<string>();
  const [error, setError] = useState<unknown>(null);
  const t = translator(state.settings.language);
  const completed = state.agents?.find(
    (agent) => agent.agent_name === state.selectedAgentName,
  )?.last_completed_at;
  useEffect(() => {
    let live = true;
    setContent(undefined);
    setError(null);
    void requestResource<string>({ type: "load", path })
      .then((value) => {
        if (live) setContent(value);
      })
      .catch((reason) => {
        if (live) setError(reason);
      });
    return () => {
      live = false;
    };
  }, [path, state.workspace, state.settings.endpoint, completed]);
  return (
    <div className="agent-memory document-pane">
      <div className="resource-memory-toolbar">
        <span>{path}</span>
        <button
          className="subtle"
          onClick={() =>
            void action({ type: "document.open", path }).catch(setError)
          }
        >
          {t("openInKnowledge")}
          <ArrowUpRight size={14} />
        </button>
      </div>
      {error != null ? (
        <p className="action-error" role="alert">
          {errorText(error, state.settings.language)}
        </p>
      ) : content === undefined ? (
        <p className="document-loading">{t("documentLoading")}</p>
      ) : (
        <DocumentPreview
          path={path}
          content={content}
          tagKey="memory_tags"
          supportsTags={false}
          onTag={() => {}}
          language={state.settings.language}
          onLink={(url) =>
            void action({ type: "link.open", url, sourcePath: path }).catch(
              setError,
            )
          }
        />
      )}
    </div>
  );
}
