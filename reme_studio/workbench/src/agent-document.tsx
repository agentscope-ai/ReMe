import { useEffect, useState } from "react";
import { ArrowUpRight } from "lucide-react";
import type { StudioState } from "../../client/src/protocol.ts";
import { errorText, translator } from "../../client/src/i18n.ts";
import { useHost } from "./host.ts";
import { useResource } from "./resources.ts";
import DocumentPreview from "./knowledge/DocumentPreview.tsx";

export default function AgentDocument({
  state,
  path,
  onEdit,
}: {
  state: StudioState;
  path: string;
  onEdit?: () => void;
}) {
  const requestResource = useResource();
  const { action } = useHost();
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
  }, [path, completed]);
  return (
    <div className="agent-document document-pane">
      <div className="agent-document-toolbar">
        <span>{path}</span>
        {onEdit && (
          <button className="subtle" onClick={onEdit}>
            {t("taskEdit")}
          </button>
        )}
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
