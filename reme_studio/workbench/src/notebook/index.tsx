import { useEffect, useId, useRef, useState } from "react";
import {
  Archive,
  ArrowLeft,
  ArrowRight,
  CalendarDays,
  CheckSquare,
  ChevronLeft,
  ChevronRight,
  Folder,
  Folders,
  Lightbulb,
  Plus,
  RefreshCw,
  Search,
} from "lucide-react";
import { errorText, translator, UiError } from "../../../client/src/i18n.ts";
import type {
  ResourcePage,
  StudioState,
} from "../../../client/src/protocol.ts";
import {
  DEFAULT_PROJECT,
  isTodo,
  normalizeProject,
  projectLabel,
  splitNotebookText,
  type NotebookItem,
  type NotebookKind,
  type NotebookProjects,
  type NotebookRequest,
  type Todo,
} from "../../../client/src/notebook.ts";
import { usePollingResource, useResource } from "../resources.ts";
import SplitPane, { SidebarHeading } from "../split-pane.tsx";
import RecordDetail from "./detail.tsx";
import RecordEditor, { type RecordDraft } from "./editor.tsx";
import NotebookList from "./list.tsx";
import "./notebook.css";

type Editor = { draft: RecordDraft; original?: NotebookItem };
const PAGE_SIZE = 20;

export default function Notebook({
  state,
  collapsed,
  onNavigate,
}: {
  state: StudioState;
  collapsed?: boolean;
  onNavigate?: () => void;
}) {
  const requestResource = useResource();
  const language = state.settings.language;
  const t = translator(language);
  const active = state.activePage === "notebook";
  const connected = state.connection === "connected";
  const [kind, setKind] = useState<NotebookKind>("todo");
  const [scope, setScope] = useState<"active" | "archive">("active");
  const [query, setQuery] = useState("");
  const [project, setProject] = useState("");
  const [queryDraft, setQueryDraft] = useState("");
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
  const [closingTodo, setClosingTodo] = useState<Todo>();
  const [closeNote, setCloseNote] = useState("");
  const closeDialog = useRef<HTMLDialogElement>(null);
  const closeTitleId = useId();
  const scroll = useRef<HTMLDivElement>(null);
  const listScroll = useRef(0);
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
  const editing = Boolean(editor);
  const item = details[kind]?.id === selected ? details[kind] : undefined;
  const latest = useRef({ kind, selected, selection });
  latest.current = { kind, selected, selection };
  const ready = active && connected && available;
  const disabled = saving || !connected || !available;
  const {
    data: page,
    error: pageError,
    loading: pageLoading,
    refresh: refreshPage,
  } = usePollingResource<ResourcePage<NotebookItem>>(
    {
      type: kind,
      action: "list",
      scope,
      offset,
      limit: PAGE_SIZE,
      ...(query ? { query } : {}),
      ...(project ? { project } : {}),
    },
    ready,
    revision,
  );
  const projects = usePollingResource<NotebookProjects>(
    { type: "todo", action: "list_projects", scope: "all" },
    ready,
    revision,
  );
  const projectNames = [
    "",
    DEFAULT_PROJECT,
    ...(projects.data?.items
      .map(({ project }) => project)
      .filter((name) => name !== DEFAULT_PROJECT) ?? []),
  ];
  const summary = JSON.stringify(
    page?.items.find((value) => value.id === selected),
  );
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    if (page && offset > 0 && offset >= page.total)
      setOffset(
        Math.max(0, Math.floor((page.total - 1) / PAGE_SIZE) * PAGE_SIZE),
      );
  }, [page?.total, offset]);
  useEffect(() => {
    if (scroll.current)
      scroll.current.scrollTop = selected || editor ? 0 : listScroll.current;
  }, [kind, scope, selected, editing, offset, project, query]);
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
          setProject((current) => (current === result.project ? current : ""));
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

  function openCloseDialog(todo: Todo) {
    setClosingTodo(todo);
    setActionError(null);
    closeDialog.current?.showModal();
  }

  async function closeTodo() {
    if (!closingTodo) return;
    const note = closeNote.trim();
    if (
      await write({
        type: "todo",
        action: "update",
        id: closingTodo.id,
        status: "closed",
        ...(note ? { close_note: note } : {}),
      })
    )
      closeDialog.current?.close();
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
  const openList = () => {
    setSelection((current) => ({ ...current, [kind]: undefined }));
    setActionError(null);
  };
  const todo = kind === "todo";
  const archived = scope === "archive";
  const kindLabel = t(todo ? "notebookTodos" : "notebookIdeas");
  const newLabel = t(todo ? "notebookNewTodo" : "notebookNewIdea");
  const archiveLabel = t(todo ? "notebookClosed" : "notebookArchive");
  const currentLabel = t(
    todo ? "notebookCurrentTodos" : "notebookCurrentIdeas",
  );
  const listHeading = archived ? archiveLabel : kindLabel;
  const KindIcon = todo ? CheckSquare : Lightbulb;
  const orderLabel = todo
    ? t(archived ? "notebookClosedOrder" : "notebookDueOrder")
    : t(archived ? "notebookArchiveOrder" : "notebookCaptureOrder");
  const emptyLabel = todo
    ? t(archived ? "notebookEmptyClosed" : "notebookEmptyTodos")
    : t(archived ? "notebookEmptyArchive" : "notebookEmptyIdeas");
  const kindTabs = (className = "") => (
    <div
      className={`source-tabs notebook-kind-tabs ${className}`}
      data-kind={kind}
      role="group"
      aria-label={t("notebook")}
    >
      {(["todo", "idea"] as const).map((value) => (
        <button
          key={value}
          type="button"
          aria-pressed={kind === value}
          className={kind === value ? "active" : ""}
          onClick={() => {
            if (kind !== value) {
              listScroll.current = 0;
              setKind(value);
              setScope("active");
              setOffset(0);
              setSelection({});
              setActionError(null);
            } else if (!editor) openList();
            onNavigate?.();
          }}
        >
          {value === "todo" ? (
            <CheckSquare size={15} />
          ) : (
            <Lightbulb size={15} />
          )}
          {t(value === "todo" ? "notebookTodos" : "notebookIdeas")}
        </button>
      ))}
    </div>
  );
  return (
    <SplitPane
      name="notebook"
      path={[
        {
          label: t("notebook"),
          onSelect: editor ? undefined : openList,
          revealSidebar: !editor,
        },
        {
          label: kindLabel,
          onSelect: editor ? undefined : openList,
          revealSidebar: !editor,
        },
        ...(item && !editor
          ? [{ label: splitNotebookText(item.text).title }]
          : []),
      ]}
      collapsed={collapsed}
      language={language}
      className={`notebook-browser notebook-${kind}`}
      sidebar={
        <>
          <SidebarHeading eyebrow="NOTES & TO-DOS" title={t("notebook")}>
            <button
              type="button"
              className="icon-button"
              aria-label={t("notebookRefresh")}
              title={t("notebookRefresh")}
              disabled={!connected || saving}
              onClick={() => {
                setRevision((value) => value + 1);
                setActionError(null);
              }}
            >
              <RefreshCw
                size={15}
                aria-hidden
                className={pageLoading ? "spinning" : ""}
              />
            </button>
          </SidebarHeading>
          <div className="sidebar-controls">
            {kindTabs()}
            <form
              className="sidebar-search"
              onSubmit={(event) => {
                event.preventDefault();
                setQuery(queryDraft.trim());
                setOffset(0);
                setSelection({});
                listScroll.current = 0;
                if (scroll.current) scroll.current.scrollTop = 0;
                onNavigate?.();
              }}
            >
              <Search size={15} aria-hidden />
              <input
                aria-label={t("notebookSearch")}
                placeholder={t("notebookSearch")}
                value={queryDraft}
                disabled={!!editor}
                onChange={(event) => setQueryDraft(event.target.value)}
              />
              <button
                type="submit"
                aria-label={t("searchAction")}
                disabled={!!editor}
              >
                <ArrowRight size={14} aria-hidden />
              </button>
            </form>
          </div>
          <section className="notebook-projects">
            <h3 className="list-section-label">{t("notebookProjects")}</h3>
            <ul
              aria-label={t("notebookProjectFilter")}
              aria-busy={ready && projects.loading}
            >
              {projectNames.map((name) => (
                <li key={name}>
                  <button
                    type="button"
                    className={`file-tree-row ${project === name ? "selected" : ""}`}
                    aria-pressed={project === name}
                    disabled={editing}
                    onClick={() => {
                      if (project !== name) {
                        setProject(name);
                        setOffset(0);
                        listScroll.current = 0;
                      }
                      openList();
                      onNavigate?.();
                    }}
                  >
                    {name ? (
                      <Folder size={15} aria-hidden />
                    ) : (
                      <Folders size={15} aria-hidden />
                    )}
                    <span
                      className="notebook-project-name"
                      title={name ? projectLabel(name, language) : undefined}
                    >
                      {name
                        ? projectLabel(name, language)
                        : t("notebookAllProjects")}
                    </span>
                    {name === DEFAULT_PROJECT && (
                      <span className="notebook-project-default">
                        {t("notebookDefaultProjectBadge")}
                      </span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
            {projects.error != null && (
              <p className="navigator-error" role="alert">
                {errorText(projects.error, language)}{" "}
                <button className="text-action" onClick={projects.refresh}>
                  {t("retry")}
                </button>
              </p>
            )}
          </section>
        </>
      }
    >
      <section className="document-pane notebook-main">
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
        <div className="notebook-scroll" ref={scroll}>
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
          ) : selected ? (
            <>
              <div className="notebook-back">
                <button className="text-action" onClick={openList}>
                  <ArrowLeft size={15} aria-hidden />
                  {t("notebookBackToList")}
                </button>
              </div>
              {item ? (
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
                  onCloseTodo={openCloseDialog}
                  onWrite={write}
                  onOpen={(targetKind, target) => {
                    listScroll.current = 0;
                    setKind(targetKind);
                    setScope(target.closed_at ? "archive" : "active");
                    setOffset(0);
                    setQuery("");
                    setQueryDraft("");
                    setProject((current) =>
                      current === target.project ? current : "",
                    );
                    setSelection((current) => ({
                      ...current,
                      [targetKind]: target.id,
                    }));
                    setActionError(null);
                  }}
                />
              ) : detailLoading ? (
                <p className="document-loading">{t("documentLoading")}</p>
              ) : null}
            </>
          ) : (
            <div className="notebook-overview">
              {kindTabs("notebook-compact-tabs")}
              <header className="notebook-masthead">
                <div className="notebook-masthead-icon" aria-hidden>
                  {archived ? <Archive size={21} /> : <KindIcon size={22} />}
                </div>
                <h1>{listHeading}</h1>
                <button
                  className="primary notebook-new"
                  disabled={disabled}
                  onClick={() => openEditor()}
                >
                  <Plus size={16} aria-hidden />
                  {newLabel}
                </button>
              </header>
              <div className="notebook-list-toolbar">
                <div
                  className="notebook-scope-tabs"
                  role="group"
                  aria-label={t("navigation")}
                >
                  {(["active", "archive"] as const).map((value) => (
                    <button
                      type="button"
                      key={value}
                      aria-pressed={scope === value}
                      onClick={() => {
                        if (scope === value) return;
                        listScroll.current = 0;
                        setScope(value);
                        setOffset(0);
                        setActionError(null);
                      }}
                    >
                      {value === "archive" && <Archive size={14} aria-hidden />}
                      {value === "archive" ? archiveLabel : currentLabel}
                    </button>
                  ))}
                </div>
                {page && (
                  <span className="notebook-list-summary">
                    {project && (
                      <span
                        className="notebook-project-context"
                        title={projectLabel(project, language)}
                      >
                        {projectLabel(project, language)}
                      </span>
                    )}
                    <CalendarDays size={14} aria-hidden />
                    {orderLabel} ·{" "}
                    {t("notebookResultCount", { count: page.total })}
                  </span>
                )}
              </div>
              <div aria-busy={ready && pageLoading}>
                {pageError != null && (
                  <p className="navigator-error" role="alert">
                    {errorText(pageError, language)}{" "}
                    <button className="text-action" onClick={refreshPage}>
                      {t("retry")}
                    </button>
                  </p>
                )}
                {!connected && (
                  <p className="notebook-list-empty">{t("waitingService")}</p>
                )}
                {ready && pageLoading && (
                  <div className="tree-skeleton">
                    <i />
                    <i />
                    <i />
                  </div>
                )}
                {page && !page.total && (
                  <div className="notebook-list-empty">
                    <KindIcon size={24} aria-hidden />
                    <p>
                      {query || project ? t("notebookNoMatches") : emptyLabel}
                    </p>
                  </div>
                )}
                {page && page.items.length > 0 && (
                  <NotebookList
                    items={page.items}
                    kind={kind}
                    archived={archived}
                    language={language}
                    disabled={disabled}
                    onOpen={(record) => {
                      listScroll.current = scroll.current?.scrollTop ?? 0;
                      setSelection((current) => ({
                        ...current,
                        [kind]: record.id,
                      }));
                      setActionError(null);
                    }}
                    onToggleTodo={(record) => {
                      if (record.status !== "closed") {
                        openCloseDialog(record);
                        return;
                      }
                      void write({
                        type: "todo",
                        action: "update",
                        id: record.id,
                        status: "pending",
                      });
                    }}
                  />
                )}
              </div>
              {page && page.total > page.limit && (
                <nav
                  className="resource-pagination"
                  aria-label={t("navigation")}
                >
                  <button
                    className="icon-button"
                    aria-label={t("previousPage")}
                    disabled={!offset}
                    onClick={() => {
                      listScroll.current = 0;
                      setOffset(Math.max(0, offset - PAGE_SIZE));
                    }}
                  >
                    <ChevronLeft size={16} />
                  </button>
                  <span>
                    {t("pageLabel", {
                      page: Math.floor(offset / PAGE_SIZE) + 1,
                      total: Math.ceil(page.total / PAGE_SIZE),
                    })}
                  </span>
                  <button
                    className="icon-button"
                    aria-label={t("nextPage")}
                    disabled={offset + PAGE_SIZE >= page.total}
                    onClick={() => {
                      listScroll.current = 0;
                      setOffset(offset + PAGE_SIZE);
                    }}
                  >
                    <ChevronRight size={16} />
                  </button>
                </nav>
              )}
            </div>
          )}
        </div>
        <div className="notebook-notice" role="status" aria-live="polite">
          {saving ? t("notebookSaving") : notice}
        </div>
        <dialog
          ref={closeDialog}
          className="workspace-dialog notebook-close-dialog"
          aria-labelledby={closeTitleId}
          onClose={() => {
            setClosingTodo(undefined);
            setCloseNote("");
          }}
          onCancel={(event) => {
            if (saving) event.preventDefault();
          }}
        >
          <h2 id={closeTitleId}>{t("notebookCloseTodo")}</h2>
          <p>{closingTodo?.text}</p>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void closeTodo();
            }}
          >
            <label>
              <span>{t("notebookCloseNoteOptional")}</span>
              <textarea
                autoFocus
                rows={3}
                maxLength={100000}
                value={closeNote}
                disabled={saving}
                onChange={(event) => setCloseNote(event.target.value)}
              />
            </label>
            {actionError != null && (
              <p className="action-error" role="alert">
                {errorText(actionError, language)}
              </p>
            )}
            <footer>
              <button
                type="button"
                className="secondary"
                disabled={saving}
                onClick={() => closeDialog.current?.close()}
              >
                {t("notebookCancel")}
              </button>
              <button className="primary" disabled={disabled || !closingTodo}>
                {t(saving ? "notebookSaving" : "notebookCloseTodo")}
              </button>
            </footer>
          </form>
        </dialog>
      </section>
    </SplitPane>
  );
}
