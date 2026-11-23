import { useEffect, useState } from "react";
import { ArrowUpRight, Archive, Pencil, RotateCcw } from "lucide-react";
import {
  errorText,
  translator,
  type Language,
} from "../../../client/src/i18n.ts";
import type { ResourcePage } from "../../../client/src/protocol.ts";
import {
  isTodo,
  projectLabel,
  type Idea,
  type NotebookItem,
  type NotebookKind,
  type NotebookRequest,
  type Todo,
  type TodoStatus,
} from "../../../client/src/notebook.ts";
import { formatDate, useResource } from "../resources.ts";

const statuses = {
  open: "notebookOpen",
  partial: "notebookPartial",
  done: "notebookDone",
  cancelled: "notebookCancelled",
} as const;

export default function RecordDetail({
  item,
  language,
  busy,
  canCreateTodo,
  active,
  revision,
  draft,
  onDraft,
  onEdit,
  onWrite,
  onOpen,
}: {
  item: NotebookItem;
  language: Language;
  busy: boolean;
  canCreateTodo: boolean;
  active: boolean;
  revision: number;
  draft: string;
  onDraft: (value: string) => void;
  onEdit: () => void;
  onWrite: (request: NotebookRequest) => Promise<boolean>;
  onOpen: (kind: NotebookKind, item: Idea) => void;
}) {
  const requestResource = useResource();
  const t = translator(language);
  const todo = isTodo(item);
  const [source, setSource] = useState<Idea>();
  const [linked, setLinked] = useState<ResourcePage<Todo>>();
  const [linkOffset, setLinkOffset] = useState(0);
  const [linkError, setLinkError] = useState<unknown>(null);
  const [linkRetry, setLinkRetry] = useState(0);
  const sourceId = todo ? item.source_idea : null;
  useEffect(() => {
    if (!active) return;
    let live = true;
    setLinkError(null);
    if (todo) {
      setSource(undefined);
      if (sourceId)
        void requestResource<Idea>({
          type: "idea",
          action: "get",
          id: sourceId,
        })
          .then((value) => {
            if (live) setSource(value);
          })
          .catch((error) => {
            if (live) setLinkError(error);
          });
    } else {
      void requestResource<ResourcePage<Todo>>({
        type: "todo",
        action: "list",
        source_idea: item.id,
        scope: "all",
        offset: linkOffset,
        limit: 20,
      })
        .then((value) => {
          if (live) setLinked(value);
        })
        .catch((error) => {
          if (live) setLinkError(error);
        });
    }
    return () => {
      live = false;
    };
  }, [item.id, todo, sourceId, active, revision, linkOffset, linkRetry]);
  return (
    <article className="notebook-record">
      <div className="notebook-record-heading">
        <span className="notebook-project">
          {projectLabel(item.project, language)}
        </span>
        <button className="subtle" onClick={onEdit} disabled={busy}>
          <Pencil size={14} />
          {t("notebookEdit")}
        </button>
      </div>
      <h1>{item.text}</h1>
      <div className="notebook-record-meta">
        {todo && (
          <label>
            <span className="sr-only">{t("notebookStatus")}</span>
            <select
              aria-label={t("notebookStatus")}
              value={item.status}
              disabled={busy}
              onChange={(event) =>
                void onWrite({
                  type: "todo",
                  action: "update",
                  id: item.id,
                  status: event.target.value as TodoStatus,
                })
              }
            >
              {Object.entries(statuses).map(([value, key]) => (
                <option key={value} value={value}>
                  {t(key)}
                </option>
              ))}
            </select>
          </label>
        )}
        {todo && item.due_date && (
          <span>{t("notebookDue", { date: item.due_date })}</span>
        )}
        {item.closed_at && (
          <span>
            {t("notebookClosed", {
              date: formatDate(item.closed_at, language),
            })}
          </span>
        )}
      </div>
      {source && (
        <section className="notebook-source">
          <h2>{t("notebookFromIdea")}</h2>
          <button
            className="text-action"
            onClick={() => onOpen("idea", source)}
          >
            {source.text}
            <ArrowUpRight size={14} />
          </button>
        </section>
      )}
      {!todo && (
        <div className="notebook-actions">
          <button
            className="primary"
            disabled={busy || !canCreateTodo}
            onClick={() =>
              void onWrite({
                type: "todo",
                action: "create",
                source_idea: item.id,
              })
            }
          >
            {t("notebookCreateTodo")}
          </button>
          <button
            className="secondary"
            disabled={busy}
            onClick={() =>
              void onWrite({
                type: "idea",
                action: item.closed_at ? "restore" : "archive",
                id: item.id,
              })
            }
          >
            {item.closed_at ? <RotateCcw size={14} /> : <Archive size={14} />}
            {t(item.closed_at ? "notebookReopen" : "notebookArchiveIdea")}
          </button>
        </div>
      )}
      {todo ? (
        <section className="notebook-progress">
          <h2>{t("notebookProgress")}</h2>
          {item.progress.length ? (
            <ol>
              {item.progress.map((note, index) => (
                <li key={`${index}:${note.time}`}>
                  <time dateTime={note.time}>
                    {formatDate(note.time, language)}
                  </time>
                  <p>{note.text}</p>
                </li>
              ))}
            </ol>
          ) : (
            <p className="notebook-muted">{t("notebookNoProgress")}</p>
          )}
          {!item.closed_at && (
            <form
              onSubmit={(event) => {
                event.preventDefault();
                if (draft.trim())
                  void onWrite({
                    type: "todo",
                    action: "update",
                    id: item.id,
                    progress_text: draft.trim(),
                  }).then((success) => {
                    if (success) onDraft("");
                  });
              }}
            >
              <label htmlFor={`progress-${item.id}`}>
                {t("notebookAddProgress")}
              </label>
              <textarea
                id={`progress-${item.id}`}
                rows={2}
                maxLength={100000}
                value={draft}
                disabled={busy}
                placeholder={t("notebookProgressPlaceholder")}
                onChange={(event) => onDraft(event.target.value)}
              />
              <button className="secondary" disabled={busy || !draft.trim()}>
                {t("notebookAddProgress")}
              </button>
            </form>
          )}
        </section>
      ) : (
        <section className="notebook-links">
          <h2>{t("notebookRelatedTodos")}</h2>
          {linked?.items.map((task) => (
            <button
              key={task.id}
              className="notebook-linked"
              onClick={() => onOpen("todo", task)}
            >
              <span>{task.text}</span>
              <small>{t(statuses[task.status])}</small>
              <ArrowUpRight size={14} />
            </button>
          ))}
          {linked && !linked.total && (
            <p className="notebook-muted">{t("notebookNoRelated")}</p>
          )}
          {linked && linked.total > linked.limit && (
            <div className="pagination">
              <button
                className="text-action"
                disabled={!linkOffset}
                onClick={() =>
                  setLinkOffset(Math.max(0, linkOffset - linked.limit))
                }
              >
                {t("previousPage")}
              </button>
              <span>
                {t("pageLabel", {
                  page: Math.floor(linkOffset / linked.limit) + 1,
                  total: Math.ceil(linked.total / linked.limit),
                })}
              </span>
              <button
                className="text-action"
                disabled={linkOffset + linked.limit >= linked.total}
                onClick={() => setLinkOffset(linkOffset + linked.limit)}
              >
                {t("nextPage")}
              </button>
            </div>
          )}
        </section>
      )}
      {linkError != null && (
        <p className="action-error" role="alert">
          {errorText(linkError, language)}{" "}
          <button
            className="text-action"
            onClick={() => setLinkRetry((value) => value + 1)}
          >
            {t("retry")}
          </button>
        </p>
      )}
    </article>
  );
}

export { statuses };
