import { useEffect, useRef, useState } from "react";
import {
  Archive,
  ArrowRight,
  CheckSquare,
  ChevronLeft,
  ChevronRight,
  Leaf,
  Lightbulb,
  Plus,
  RefreshCw,
  Search,
} from "lucide-react";
import { errorText, translator, UiError } from "../../shared/i18n.ts";
import type { DesktopState } from "../../shared/protocol.ts";
import {
  isTodo,
  DEFAULT_PROJECT,
  normalizeProject,
  projectLabel,
  projectValue,
  type NotebookItem,
  type NotebookKind,
  type NotebookRequest,
} from "../../shared/notebook.ts";
import { requestResource, useResourcePage } from "../resources.ts";
import SplitPane from "../split-pane.tsx";
import RecordDetail, { statuses } from "./detail.tsx";
import RecordEditor, { type RecordDraft } from "./editor.tsx";
import "./notebook.css";

type Editor = { draft: RecordDraft; original?: NotebookItem };

export default function Notebook({
  state,
  collapsed,
}: {
  state: DesktopState;
  collapsed?: boolean;
}) {
  const language = state.settings.language;
  const t = translator(language);
  const active = state.activePage === "notebook";
  const connected = state.connection === "connected";
  const [kind, setKind] = useState<NotebookKind>("todo");
  const [scope, setScope] = useState<"active" | "archive">("active");
  const [query, setQuery] = useState("");
  const [project, setProject] = useState("");
  const [queryDraft, setQueryDraft] = useState("");
  const [projectDraft, setProjectDraft] = useState("");
  const [offset, setOffset] = useState(0);
  const [selection, setSelection] = useState<
    Partial<Record<NotebookKind, string>>
  >({});
  const [details, setDetails] = useState<
    Partial<Record<NotebookKind, NotebookItem>>
  >({});
  const [editors, setEditors] = useState<Partial<Record<NotebookKind, Editor>>>(
    {},
  );
  const [progressDrafts, setProgressDrafts] = useState<Record<string, string>>(
    {},
  );
  const missing = ["todo", "idea"].find(
    (name) => !state.clientInfo?.jobs.includes(name),
  );
  const available = Boolean(state.clientInfo && !missing);
  const capabilityError =
    state.clientInfo && missing
      ? new UiError("missingJob", { job: missing })
      : null;
  const [detailError, setDetailError] = useState<unknown>(null);
  const [actionError, setActionError] = useState<unknown>(null);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState("");
  const [revision, setRevision] = useState(0);
  const [detailLoading, setDetailLoading] = useState(false);
  const mounted = useRef(false);
  const writing = useRef(false);
  const epoch = useRef(0);
  const selected = selection[kind] ?? "";
  const editor = editors[kind];
  const item = details[kind]?.id === selected ? details[kind] : undefined;
  const latest = useRef({ kind, selected, selection });
  latest.current = { kind, selected, selection };
  const ready = active && connected && available;
  const disabled = saving || !connected || !available;
  const page = useResourcePage<NotebookItem>(
    {
      type: kind,
      action: "list",
      scope,
      offset,
      limit: 20,
      ...(query ? { query } : {}),
      ...(project ? { project } : {}),
    },
    ready,
    revision,
  );
  const summary = JSON.stringify(
    page.page?.items.find((value) => value.id === selected),
  );
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    if (page.page && offset > 0 && offset >= page.page.total)
      setOffset(Math.max(0, Math.floor((page.page.total - 1) / 20) * 20));
  }, [page.page?.total, offset]);
  useEffect(() => {
    setDetailError(null);
    if (!ready || !selected || editor) {
      setDetailLoading(false);
      return;
    }
    let live = true;
    const version = epoch.current;
    setDetailLoading(true);
    setDetailError(null);
    void requestResource<NotebookItem>({
      type: kind,
      action: "get",
      id: selected,
    })
      .then((value) => {
        if (live && version === epoch.current)
          setDetails((current) => ({ ...current, [kind]: value }));
      })
      .catch((error) => {
        if (live && version === epoch.current) setDetailError(error);
      })
      .finally(() => {
        if (live && version === epoch.current) setDetailLoading(false);
      });
    return () => {
      live = false;
    };
  }, [ready, kind, selected, editor, revision, summary]);

  async function write(request: NotebookRequest): Promise<boolean> {
    if (disabled || writing.current) return false;
    writing.current = true;
    const origin = latest.current;
    epoch.current++;
    setSaving(true);
    setActionError(null);
    setNotice("");
    try {
      const result = await requestResource<NotebookItem>(request);
      if (!mounted.current) return false;
      if (request.action === "create") {
        setSelection((current) => ({ ...current, [request.type]: result.id }));
        setDetails((current) => ({ ...current, [request.type]: result }));
        if (
          latest.current.kind === origin.kind &&
          latest.current.selected === origin.selected
        ) {
          setKind(request.type);
          setScope("active");
          setOffset(0);
          setQuery("");
          setQueryDraft("");
          setProject("");
          setProjectDraft("");
        }
      } else if (latest.current.selection[request.type] === result.id) {
        setDetails((current) => ({ ...current, [request.type]: result }));
      }
      setNotice(t("notebookSaved"));
      return true;
    } catch (error) {
      if (mounted.current) setActionError(error);
      return false;
    } finally {
      writing.current = false;
      if (mounted.current) {
        setSaving(false);
        setRevision((value) => value + 1);
      }
    }
  }

  function openEditor(original?: NotebookItem) {
    setEditors((current) => ({
      ...current,
      [kind]: {
        original,
        draft: {
          ...(original ? { id: original.id } : {}),
          project: normalizeProject(original?.project ?? project),
          text: original?.text ?? "",
          due_date:
            original && isTodo(original) ? (original.due_date ?? "") : "",
        },
      },
    }));
    setActionError(null);
  }

  async function saveEditor() {
    if (!editor || !editor.draft.text.trim()) return;
    const { draft, original } = editor;
    const values: {
      text?: string;
      project?: string;
      due_date?: string | null;
    } = {};
    if (!original || draft.text.trim() !== original.text)
      values.text = draft.text.trim();
    if (!original || normalizeProject(draft.project) !== original.project)
      values.project = normalizeProject(draft.project);
    if (
      kind === "todo" &&
      (!original ||
        draft.due_date !== (isTodo(original) ? (original.due_date ?? "") : ""))
    )
      values.due_date = draft.due_date || null;
    if (original && !Object.keys(values).length) {
      setEditors((current) => ({ ...current, [kind]: undefined }));
      return;
    }
    const request: NotebookRequest =
      kind === "todo"
        ? original
          ? { type: "todo", action: "update", id: original.id, ...values }
          : { type: "todo", action: "create", ...values }
        : original
          ? { type: "idea", action: "update", id: original.id, ...values }
          : { type: "idea", action: "create", ...values };
    if ((await write(request)) && mounted.current)
      setEditors((current) => ({ ...current, [kind]: undefined }));
  }

  const error = actionError ?? detailError ?? capabilityError;
  return (
    <SplitPane
      name="notebook"
      collapsed={collapsed}
      language={language}
      className="notebook-browser"
      sidebar={
        <>
          <div className="sidebar-heading">
            <span className="eyebrow">A LITTLE ROOM TO THINK</span>
            <h1>
              {t("notebook")}
              <button
                className="icon-button"
                aria-label={t("notebookRefresh")}
                disabled={!connected || saving}
                onClick={() => {
                  setRevision((value) => value + 1);
                  setActionError(null);
                }}
              >
                <RefreshCw
                  size={15}
                  className={page.loading ? "spinning" : ""}
                />
              </button>
            </h1>
            <p>{t("notebookCaption")}</p>
          </div>
          <div className="source-tabs" role="group" aria-label={t("notebook")}>
            {(["todo", "idea"] as const).map((value) => (
              <button
                key={value}
                aria-pressed={kind === value}
                className={kind === value ? "active" : ""}
                onClick={() => {
                  setKind(value);
                  setOffset(0);
                  setActionError(null);
                }}
              >
                {value === "todo" ? (
                  <CheckSquare size={14} />
                ) : (
                  <Lightbulb size={14} />
                )}
                {t(value === "todo" ? "notebookTodos" : "notebookIdeas")}
              </button>
            ))}
          </div>
          <form
            className="notebook-filters"
            onSubmit={(event) => {
              event.preventDefault();
              setQuery(projectValue(queryDraft.trim(), language));
              setProject(projectDraft.trim());
              setOffset(0);
              setSelection({});
            }}
          >
            <label className="knowledge-search">
              <Search size={15} />
              <input
                aria-label={t("notebookSearch")}
                placeholder={t("notebookSearch")}
                value={queryDraft}
                disabled={!!editor}
                onChange={(event) => setQueryDraft(event.target.value)}
              />
              <button aria-label={t("searchAction")} disabled={!!editor}>
                <ArrowRight size={14} />
              </button>
            </label>
            <label className="notebook-project-filter">
              <span className="sr-only">{t("notebookProjectFilter")}</span>
              <input
                aria-label={t("notebookProjectFilter")}
                placeholder={t("notebookAllProjects")}
                value={
                  projectDraft === DEFAULT_PROJECT
                    ? projectLabel(projectDraft, language)
                    : projectDraft
                }
                disabled={!!editor}
                onChange={(event) =>
                  setProjectDraft(projectValue(event.target.value, language))
                }
                onBlur={() => {
                  if (project !== projectDraft.trim()) {
                    setProject(projectDraft.trim());
                    setOffset(0);
                    setSelection({});
                  }
                }}
              />
            </label>
          </form>
          <div className="notebook-navigator" aria-busy={page.loading}>
            <div className="list-section-label">
              <span>
                {t(scope === "archive" ? "notebookArchive" : "notebookCurrent")}
              </span>
              <span>{page.page?.total ?? ""}</span>
            </div>
            {page.error != null && (
              <p className="navigator-error" role="alert">
                {errorText(page.error, language)}{" "}
                <button className="text-action" onClick={page.refresh}>
                  {t("retry")}
                </button>
              </p>
            )}
            {!connected && (
              <p className="navigator-empty">{t("waitingService")}</p>
            )}
            {connected && page.loading && !page.page && (
              <div className="tree-skeleton">
                <i />
                <i />
                <i />
              </div>
            )}
            {page.page && !page.page.total && (
              <p className="navigator-empty">
                {t(query || project ? "notebookNoMatches" : "notebookEmpty")}
              </p>
            )}
            <ul className="notebook-list">
              {page.page?.items.map((record) => (
                <li
                  key={record.id}
                  className={`notebook-row ${selected === record.id ? "selected" : ""}`}
                >
                  {isTodo(record) ? (
                    <input
                      type="checkbox"
                      aria-label={
                        record.closed_at
                          ? `${t("notebookReopen")}：${record.text}`
                          : t("notebookComplete", { text: record.text })
                      }
                      checked={!!record.closed_at}
                      disabled={disabled || !!editor}
                      onChange={() =>
                        void write({
                          type: "todo",
                          action: "update",
                          id: record.id,
                          status: record.closed_at ? "open" : "done",
                        })
                      }
                    />
                  ) : (
                    <Lightbulb size={15} />
                  )}
                  <button
                    className="notebook-row-content"
                    aria-pressed={selected === record.id}
                    disabled={!!editor}
                    onClick={() => {
                      setSelection((current) => ({
                        ...current,
                        [kind]: record.id,
                      }));
                      setActionError(null);
                    }}
                  >
                    <span className="notebook-project">
                      {projectLabel(record.project, language)}
                    </span>
                    <span
                      className="notebook-row-title"
                      data-cancelled={
                        isTodo(record) && record.status === "cancelled"
                      }
                    >
                      {record.text}
                    </span>
                    {isTodo(record) && record.progress.length > 0 && (
                      <span className="notebook-row-preview">
                        {record.progress.at(-1)!.text}
                      </span>
                    )}
                    {isTodo(record) && record.status !== "open" && (
                      <span className="notebook-row-status">
                        {t(statuses[record.status])}
                      </span>
                    )}
                    {isTodo(record) && record.due_date && (
                      <span className="notebook-row-preview">
                        {t("notebookDue", { date: record.due_date })}
                      </span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          </div>
          {page.page && page.page.total > page.page.limit && (
            <nav className="resource-pagination" aria-label={t("navigation")}>
              <button
                className="icon-button"
                aria-label={t("previousPage")}
                disabled={!offset || !!editor}
                onClick={() => setOffset(Math.max(0, offset - 20))}
              >
                <ChevronLeft size={16} />
              </button>
              <span>
                {t("pageLabel", {
                  page: Math.floor(offset / 20) + 1,
                  total: Math.ceil(page.page.total / 20),
                })}
              </span>
              <button
                className="icon-button"
                aria-label={t("nextPage")}
                disabled={offset + 20 >= page.page.total || !!editor}
                onClick={() => setOffset(offset + 20)}
              >
                <ChevronRight size={16} />
              </button>
            </nav>
          )}
          <div className="sidebar-footer notebook-footer">
            <button
              className="text-action"
              aria-pressed={scope === "archive"}
              disabled={!!editor}
              onClick={() => {
                setScope(scope === "active" ? "archive" : "active");
                setOffset(0);
                setSelection({});
                setActionError(null);
              }}
            >
              <Archive size={14} />
              {t(
                scope === "active" ? "notebookArchive" : "notebookShowCurrent",
              )}
            </button>
          </div>
        </>
      }
    >
      <section className="document-pane notebook-main">
        <header className="document-toolbar">
          <Leaf size={16} />
          <span>{t("notebook")}</span>
          <button
            className="primary"
            disabled={disabled || !!editor}
            onClick={() => openEditor()}
          >
            <Plus size={14} />
            {t("notebookCapture")}
          </button>
        </header>
        {error != null && (
          <div className="action-error" role="alert">
            {errorText(error, language)}{" "}
            <button
              className="text-action"
              onClick={() => {
                setRevision((value) => value + 1);
                setActionError(null);
              }}
            >
              {t("retry")}
            </button>
          </div>
        )}
        <div className="notebook-scroll">
          {editor ? (
            <RecordEditor
              kind={kind}
              draft={editor.draft}
              busy={disabled}
              language={language}
              onChange={(draft) =>
                setEditors((current) => ({
                  ...current,
                  [kind]: { ...editor, draft },
                }))
              }
              onSave={() => void saveEditor()}
              onCancel={() =>
                setEditors((current) => ({ ...current, [kind]: undefined }))
              }
            />
          ) : item ? (
            <RecordDetail
              key={`${kind}:${item.id}`}
              item={item}
              language={language}
              busy={disabled}
              canCreateTodo={!editors.todo}
              active={ready}
              revision={revision}
              draft={progressDrafts[item.id] ?? ""}
              onDraft={(value) =>
                setProgressDrafts((current) => ({
                  ...current,
                  [item.id]: value,
                }))
              }
              onEdit={() => openEditor(item)}
              onWrite={write}
              onOpen={(targetKind, target) => {
                setKind(targetKind);
                setScope(target.closed_at ? "archive" : "active");
                setOffset(0);
                setQuery("");
                setQueryDraft("");
                setProject("");
                setProjectDraft("");
                setSelection((current) => ({
                  ...current,
                  [targetKind]: target.id,
                }));
                setActionError(null);
              }}
            />
          ) : detailLoading && selected ? (
            <p className="document-loading">{t("documentLoading")}</p>
          ) : (
            <div className="knowledge-empty notebook-empty">
              <Leaf size={40} strokeWidth={1.2} />
              <h2>{t("notebookSelect")}</h2>
              <p>{t("notebookSelectHint")}</p>
            </div>
          )}
        </div>
        <div className="notebook-notice" role="status" aria-live="polite">
          {saving ? t("notebookSaving") : notice}
        </div>
      </section>
    </SplitPane>
  );
}
