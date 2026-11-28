import { useEffect, useState } from "react";
import type { AgentView, StudioState } from "../../../client/src/protocol.ts";
import type { TaskDocument as Document } from "../../../client/src/tasks.ts";
import { errorText, translator } from "../../../client/src/i18n.ts";
import { useHost } from "../host.ts";
import DocumentEditor from "../knowledge/DocumentEditor.tsx";
import DocumentPreview from "../knowledge/DocumentPreview.tsx";
import { useResource } from "../resources.ts";

export default function TaskDocument({
  state,
  agent,
  editing,
  navigationBlocked,
  onEditingChange,
}: {
  state: StudioState;
  agent: AgentView;
  editing: boolean;
  navigationBlocked: boolean;
  onEditingChange: (editing: boolean) => void;
}) {
  const request = useResource();
  const { action } = useHost();
  const t = translator(state.settings.language);
  const [document, setDocument] = useState<Document>();
  const [error, setError] = useState<unknown>(null);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (editing) return;
    let live = true;
    setDocument(undefined);
    setError(null);
    void request<Document>({
      type: "task",
      action: "read",
      agent_name: agent.agent_name,
    })
      .then((value) => {
        if (live) setDocument(value);
      })
      .catch((reason) => {
        if (live) setError(reason);
      });
    return () => {
      live = false;
    };
  }, [agent.agent_name, agent.last_completed_at, editing, retry]);
  return (
    <div className="agent-document document-pane">
      {error != null ? (
        <p className="action-error" role="alert">
          {errorText(error, state.settings.language)}{" "}
          <button
            type="button"
            className="subtle"
            onClick={() => setRetry((value) => value + 1)}
          >
            {t("retry")}
          </button>
        </p>
      ) : !document ? (
        <p className="document-loading">{t("documentLoading")}</p>
      ) : editing ? (
        <DocumentEditor
          path={document.path}
          content={document.content}
          mtime={document.mtime}
          language={state.settings.language}
          navigationBlocked={navigationBlocked}
          save={async (content) => {
            const saved = await request<Pick<Document, "path" | "mtime">>({
              type: "task",
              action: "save",
              agent_name: agent.agent_name,
              content,
              expected_mtime: document.mtime,
            });
            setDocument({ ...saved, content });
          }}
          onClose={() => onEditingChange(false)}
        />
      ) : (
        <DocumentPreview
          path={document.path}
          content={document.content}
          mtime={document.mtime}
          language={state.settings.language}
          onCopy={(text) => action({ type: "clipboard.copy", text })}
          onEdit={
            state.connection === "connected"
              ? () => onEditingChange(true)
              : undefined
          }
          onLink={(url) =>
            void action({
              type: "link.open",
              url,
              sourcePath: document.path,
            }).catch(setError)
          }
        />
      )}
    </div>
  );
}
