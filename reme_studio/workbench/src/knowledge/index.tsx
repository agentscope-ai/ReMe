import { useContext, useEffect, useMemo, useRef, useState } from "react";
import {
  BookOpen,
  CalendarDays,
  FolderTree,
  Search,
  RefreshCw,
  Tags,
  X,
  ChevronRight,
  ChevronDown,
  FileText,
  Folder,
  ArrowLeft,
  ArrowRight,
} from "lucide-react";
import type { StudioState, TagFiles } from "../../../client/src/protocol.ts";
import { errorText, translator, UiError } from "../../../client/src/i18n.ts";
import { useHost } from "../host.ts";
import { useWorkspace } from "../resources.ts";
import SplitPane, { SidebarHeading } from "../split-pane.tsx";
import { buildTree, type FileTree } from "./model.ts";
import DocumentPreview from "./DocumentPreview.tsx";
import DocumentEditor from "./DocumentEditor.tsx";
import { useNavigationBlocker, useNavigationState } from "../navigation.tsx";
import { WorkspaceLayout } from "../workspace-titlebar.tsx";
import MobileSections from "../mobile/sections.tsx";
import {
  TimelineHeading,
  TimelineList,
  TimelinePane,
  timelineDay,
} from "../timeline.tsx";

const sources = [
  ["workspace", FolderTree, "workspaceTab"],
  ["daily", CalendarDays, "dailyTab"],
  ["digest", BookOpen, "digestTab"],
  ["tags", Tags, "memoryTagsTab"],
] as const;

type KnowledgeSource = (typeof sources)[number][0];
const DAILY_PAGE_SIZE = 40;

function Tree({
  nodes,
  selected,
  onOpen,
  depth = 0,
}: {
  nodes: FileTree[];
  selected: string;
  onOpen: (path: string) => void;
  depth?: number;
}) {
  return (
    <>
      {nodes.map((node) => (
        <TreeNode
          key={node.path}
          node={node}
          depth={depth}
          selected={selected}
          onOpen={onOpen}
        />
      ))}
    </>
  );
}
function TreeNode({
  node,
  depth,
  selected,
  onOpen,
}: {
  node: FileTree;
  depth: number;
  selected: string;
  onOpen: (path: string) => void;
}) {
  const [expanded, setExpanded] = useState(depth === 0);
  useEffect(() => {
    if (selected.startsWith(node.path + "/")) setExpanded(true);
  }, [selected, node.path]);
  return (
    <>
      <button
        className={`file-tree-row ${selected === node.path ? "selected" : ""}`}
        style={{
          paddingInlineStart: `calc(var(--sidebar-row-inset) + ${depth * 15}px)`,
        }}
        title={node.path}
        aria-expanded={node.children ? expanded : undefined}
        onClick={() =>
          node.children ? setExpanded(!expanded) : onOpen(node.path)
        }
      >
        {node.children ? (
          <>
            <ChevronRight
              className="tree-chevron"
              size={12}
              style={{ rotate: expanded ? "90deg" : undefined }}
            />
            <Folder size={15} />
          </>
        ) : (
          <>
            <span className="tree-spacer" />
            <FileText size={15} />
          </>
        )}
        <span>{node.name}</span>
      </button>
      {node.children && expanded && (
        <Tree
          nodes={node.children}
          selected={selected}
          onOpen={onOpen}
          depth={depth + 1}
        />
      )}
    </>
  );
}

export default function Knowledge({
  state,
  collapsed,
  onNavigate,
}: {
  state: StudioState;
  collapsed?: boolean;
  onNavigate?: () => void;
}) {
  const mobile = useContext(WorkspaceLayout)?.mobile;
  const request = useWorkspace();
  const { action } = useHost();
  const t = translator(state.settings.language);
  const active = state.activePage === "knowledge";
  const config = state.clientInfo;
  const available = Boolean(
    config?.jobs.includes("list_workspace") &&
    config.jobs.includes("load_file"),
  );
  const [source, setSource] = useNavigationState<KnowledgeSource>(
    "knowledge",
    "source",
    "workspace",
  );
  const [paths, setPaths] = useState<string[]>([]);
  const [limited, setLimited] = useState(false);
  const [selected, setSelected] = useNavigationState(
    "knowledge",
    "selected",
    "",
  );
  const [document, setDocument] = useState<{
    content: string;
    image?: string;
    mtime?: string;
  }>();
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useNavigationState("knowledge", "query", "");
  const [tags, setTags] = useState<[string, number][]>([]);
  const [tagsPage, setTagsPage] = useState(1);
  const [moreTags, setMoreTags] = useState(false);
  const [tagsLoading, setTagsLoading] = useState(true);
  const [selectedTags, setSelectedTags] = useNavigationState<string[]>(
    "knowledge",
    "tags",
    [],
  );
  const [tagError, setTagError] = useState<unknown>(null);
  const [page, setPage] = useNavigationState("knowledge", "page", 1);
  const [pages, setPages] = useState(1);
  const [dailyPage, setDailyPage] = useNavigationState(
    "knowledge",
    "dailyPage",
    1,
  );
  const [refresh, setRefresh] = useState(0);
  const [loading, setLoading] = useState(false);
  const [fileLoading, setFileLoading] = useState(false);
  const [requestError, setError] = useState<unknown>(null);
  const error =
    config && !available ? new UiError("workspaceUnavailable") : requestError;
  const [fileError, setFileError] = useState<unknown>(null);
  const [editorOpen, setEditorOpen] = useState(false);
  useNavigationBlocker("knowledge", active && editorOpen);
  useEffect(() => setDraft(query), [query]);
  const [navigationBlocked, setNavigationBlocked] = useState(false);
  const editButton = useRef<HTMLButtonElement>(null);
  const restoreEditFocus = useRef(false);
  const openedTarget = useRef<number>(undefined);
  const loadedPath = useRef("");
  const documentScroll = useRef<HTMLDivElement>(null);
  const root =
    (source === "daily"
      ? config?.daily_dir
      : source === "digest"
        ? config?.digest_dir
        : ""
    )
      ?.replace(/\\/g, "/")
      .replace(/^\/+|\/+$/g, "") ?? "";
  const supportsTags = Boolean(
    config?.jobs.includes("list_tags") &&
    config.jobs.includes("list_tag_files"),
  );
  const browsingTags = source === "tags";
  const activeTags = browsingTags ? selectedTags : undefined;
  const showFiles = !browsingTags || Boolean(query || activeTags?.length);
  const daily = source === "daily";
  const showTimeline = daily || (browsingTags && showFiles);
  const timelineItems = useMemo(() => {
    if (!showTimeline) return [];
    const items = paths.map((path) => {
      const relative =
        root && path.startsWith(root + "/")
          ? path.slice(root.length + 1)
          : path;
      const title = relative.split("/").at(-1) || relative;
      return {
        id: path,
        title,
        date: daily
          ? timelineDay(/^(\d{4}-\d{2}-\d{2})(?:\.md$|\/)/i.exec(relative)?.[1])
          : undefined,
        preview: relative === title ? undefined : relative,
        leading: (
          <span className="timeline-card-leading">
            <FileText size={15} aria-hidden />
          </span>
        ),
      };
    });
    if (daily) items.sort((a, b) => (b.date ?? "").localeCompare(a.date ?? ""));
    return items;
  }, [showTimeline, paths, root, daily]);
  const dailyPages = Math.max(
    1,
    Math.ceil(timelineItems.length / DAILY_PAGE_SIZE),
  );
  const timelinePage = daily ? Math.min(dailyPage, dailyPages) : page;
  useEffect(() => {
    const target = state.documentTarget;
    if (!target) {
      openedTarget.current = undefined;
      return;
    }
    if (editorOpen || openedTarget.current === target.request) return;
    openedTarget.current = target.request;
    setSource("workspace");
    setQuery("");
    setDraft("");
    setSelectedTags([]);
    setPage(1);
    setSelected(target.path);
    setRefresh((value) => value + 1);
  }, [state.documentTarget, editorOpen]);
  useEffect(() => {
    if (!editorOpen && restoreEditFocus.current) {
      restoreEditFocus.current = false;
      editButton.current?.focus({ preventScroll: true });
    }
  }, [editorOpen]);
  useEffect(() => {
    if (!active || !browsingTags || !supportsTags) return;
    let live = true;
    setTagsLoading(true);
    void Promise.all(
      Array.from({ length: tagsPage }, (_, index) =>
        request<{ items: [string, number][]; total_pages: number }>({
          type: "tags",
          page: index + 1,
        }),
      ),
    )
      .then((results) => {
        if (!live) return;
        setTags([
          ...new Map(
            results
              .flatMap(({ answer }) => answer.items)
              .map((item) => [item[0], item]),
          ).values(),
        ]);
        setMoreTags(tagsPage < results[0].answer.total_pages);
        setTagError(null);
      })
      .catch((error) => {
        if (live) setTagError(error);
      })
      .finally(() => {
        if (live) setTagsLoading(false);
      });
    return () => {
      live = false;
    };
  }, [active, browsingTags, config, supportsTags, tagsPage, refresh]);
  useEffect(() => {
    if (!active || !showFiles || !available || state.connection !== "connected")
      return;
    let live = true;
    setLoading(true);
    setError(null);
    const load = async () => {
      let next: string[];
      let filter: { applied?: boolean; tags?: string[] } | undefined;
      if (query) {
        const result = await request<string>({
          type: "search",
          query,
          tags: activeTags ?? [],
          path: root,
        });
        filter = result.metadata.tag_filter as typeof filter;
        const results = result.metadata.results as
          { path: string }[] | undefined;
        next = [...new Set((results ?? []).map((item) => item.path))];
        if (live) {
          setLimited(false);
          setPages(1);
        }
      } else if (activeTags?.length) {
        const result = await request<TagFiles>({
          type: "tagFiles",
          tags: activeTags,
          path: root,
          page,
        });
        filter = result.metadata.tag_filter as typeof filter;
        next = result.answer.items.map((item) => item.path);
        if (live) {
          setPages(Math.max(1, result.answer.total_pages));
          setLimited(false);
        }
      } else {
        const result = await request<string>({ type: "list", path: root });
        next = (result.metadata.items as string[] | undefined) ?? [];
        if (live) {
          setLimited(next.length >= 5000);
          setPages(1);
        }
      }
      if (activeTags?.length && filter?.applied !== true)
        throw new UiError("tagFilterUnavailable");
      if (live) {
        if (
          activeTags &&
          filter?.tags &&
          JSON.stringify(filter.tags) !== JSON.stringify(activeTags)
        )
          setSelectedTags(filter.tags);
        setPaths(next);
      }
    };
    void load()
      .catch((error) => {
        if (live) {
          setError(error);
          setPaths([]);
        }
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, [
    active,
    source,
    showFiles,
    available,
    state.connection,
    root,
    activeTags,
    query,
    page,
    refresh,
  ]);
  useEffect(() => {
    if (!active || !config) return;
    const timer = window.setInterval(() => {
      if (!window.document.hidden) setRefresh((n) => n + 1);
    }, 30_000);
    return () => clearInterval(timer);
  }, [active, config]);
  useEffect(() => {
    if (!selected || editorOpen) return;
    let live = true;
    if (loadedPath.current !== selected) {
      setFileLoading(true);
      if (documentScroll.current) documentScroll.current.scrollTop = 0;
    }
    setFileError(null);
    const image = /\.(png|jpe?g|webp|gif|bmp)$/i.test(selected);
    void request<string>({ type: image ? "image" : "load", path: selected })
      .then((result) => {
        if (
          image &&
          (!/^image\/(png|jpeg|webp|gif|bmp)$/.test(
            String(result.metadata.mime),
          ) ||
            result.metadata.oversized)
        )
          throw new UiError("invalidWorkspaceRequest");
        if (live) {
          loadedPath.current = selected;
          setDocument({
            content: image ? "" : result.answer,
            image: image
              ? `data:${result.metadata.mime};base64,${result.answer}`
              : undefined,
            mtime:
              typeof result.metadata.mtime === "string"
                ? result.metadata.mtime
                : undefined,
          });
        }
      })
      .catch((error) => {
        if (live) {
          setFileError(error);
          setDocument(undefined);
        }
      })
      .finally(() => {
        if (live) setFileLoading(false);
      });
    return () => {
      live = false;
    };
  }, [selected, refresh, editorOpen]);
  const mtime = document?.mtime;
  const canEdit = Boolean(
    mtime &&
    config?.jobs.includes("save_file") &&
    /\.(mdx?|txt|csv|jsonl?|ya?ml)$/i.test(selected),
  );
  const tree = useMemo(() => buildTree(paths, root), [paths, root]);
  const selectSource = (next: KnowledgeSource) => {
    if (next === source) return;
    if (editorOpen) {
      setNavigationBlocked(true);
      return;
    }
    if (next === "daily" || source === "daily" || next === "tags")
      setSelected("");
    setDailyPage(1);
    setSource(next);
    setPage(1);
    setPages(1);
    setLimited(false);
    setPaths([]);
  };
  const chooseTag = (tag: string) => {
    if (editorOpen) {
      setNavigationBlocked(true);
      return;
    }
    setSelected("");
    if (!browsingTags) {
      selectSource("tags");
      setSelectedTags([tag]);
      setDraft("");
      setQuery("");
      return;
    }
    setPage(1);
    setSelectedTags((previous) =>
      previous.includes(tag)
        ? previous.filter((value) => value !== tag)
        : [...previous, tag],
    );
  };
  const openLink = (url: string) =>
    void action({ type: "link.open", url, sourcePath: selected }).catch(
      setFileError,
    );
  const selectDocument = (path: string) => {
    if (editorOpen && path !== selected) {
      setNavigationBlocked(true);
      return;
    }
    setSelected(path);
  };
  const preview = selected && (
    <div className="document-pane">
      <div className="document-scroll" ref={documentScroll}>
        {fileError !== null ? (
          <div className="document-error" role="alert">
            {errorText(fileError, state.settings.language)}
            <button
              className="secondary"
              onClick={() => setRefresh((n) => n + 1)}
            >
              {t("retry")}
            </button>
          </div>
        ) : fileLoading || loadedPath.current !== selected ? (
          <div className="document-loading">{t("documentLoading")}</div>
        ) : (
          document &&
          (document.image ? (
            <img
              className="knowledge-image"
              src={document.image}
              alt={selected}
            />
          ) : editorOpen && mtime ? (
            <DocumentEditor
              path={selected}
              content={document.content}
              mtime={mtime}
              language={state.settings.language}
              navigationBlocked={navigationBlocked}
              save={async (content) => {
                const { metadata } = await request({
                  type: "save",
                  path: selected,
                  content,
                  expectedMtime: mtime,
                });
                setDocument({
                  content,
                  mtime:
                    typeof metadata.mtime === "string"
                      ? metadata.mtime
                      : undefined,
                });
                setRefresh((n) => n + 1);
              }}
              onClose={() => {
                restoreEditFocus.current = true;
                setEditorOpen(false);
                setNavigationBlocked(false);
              }}
            />
          ) : (
            <DocumentPreview
              key={selected}
              path={selected}
              content={document.content}
              mtime={document.mtime}
              tagKey={config?.tag_key ?? "memory_tags"}
              onTag={supportsTags ? chooseTag : undefined}
              onLink={openLink}
              onCopy={(text) => action({ type: "clipboard.copy", text })}
              onEdit={canEdit ? () => setEditorOpen(true) : undefined}
              editRef={editButton}
              language={state.settings.language}
            />
          ))
        )}
      </div>
    </div>
  );

  return (
    <SplitPane
      name="knowledge"
      path={
        selected
          ? [
              {
                label: t("memory"),
                onSelect: () => selectDocument(""),
                revealSidebar: !editorOpen,
              },
              {
                label: selected.split(/[\\/]/).at(-1) || selected,
                title: selected,
              },
            ]
          : [{ label: t("memory") }]
      }
      collapsed={collapsed}
      mobileView={
        mobile && !selected && !showTimeline ? "navigation" : "content"
      }
      contentKey={JSON.stringify([source, selected, editorOpen])}
      mobileNavigation={
        !selected && (
          <MobileSections
            label={t("memory")}
            value={source}
            items={sources.map(([value, icon, key]) => ({
              value,
              icon,
              label: t(key),
            }))}
            disabled={editorOpen}
            onSelect={(value) => {
              selectSource(value);
              onNavigate?.();
            }}
          />
        )
      }
      language={state.settings.language}
      actions={
        mobile && (
          <button
            type="button"
            className="titlebar-button"
            aria-label={t("refreshMemory")}
            disabled={editorOpen}
            onClick={() => setRefresh((value) => value + 1)}
          >
            <RefreshCw size={20} aria-hidden />
          </button>
        )
      }
      sidebar={
        <>
          <SidebarHeading eyebrow="KNOWLEDGE & HISTORY" title={t("memory")}>
            <button
              type="button"
              className="icon-button"
              aria-label={t("refreshMemory")}
              title={t("refreshMemory")}
              onClick={() => setRefresh((n) => n + 1)}
            >
              <RefreshCw
                size={15}
                aria-hidden
                className={
                  (showFiles && loading) ||
                  (browsingTags && supportsTags && tagsLoading)
                    ? "spinning"
                    : ""
                }
              />
            </button>
          </SidebarHeading>
          <div className="sidebar-controls">
            {!mobile && (
              <div
                className="source-tabs knowledge-source-tabs"
                data-source={source}
                role="group"
                aria-label={t("memory")}
              >
                {sources.map(([id, Icon, key]) => (
                  <button
                    key={id}
                    type="button"
                    aria-pressed={source === id}
                    className={source === id ? "active" : ""}
                    onClick={() => {
                      selectSource(id);
                      onNavigate?.();
                    }}
                  >
                    <Icon size={14} aria-hidden />
                    <span>{t(key)}</span>
                  </button>
                ))}
              </div>
            )}
            <form
              className="sidebar-search"
              onSubmit={(event) => {
                event.preventDefault();
                setQuery(draft.trim());
                setPage(1);
                setDailyPage(1);
              }}
            >
              <Search size={15} aria-hidden />
              <input
                aria-label={t("searchKnowledge")}
                placeholder={t("searchKnowledge")}
                value={draft}
                onChange={(event) => {
                  const value = event.target.value;
                  setDraft(value);
                  if (query && !value.trim()) {
                    setQuery("");
                    setPage(1);
                    setDailyPage(1);
                  }
                }}
              />
              <button type="submit" aria-label={t("searchAction")}>
                <ArrowRight size={14} aria-hidden />
              </button>
            </form>
          </div>
          {browsingTags && (
            <section
              className="knowledge-tags"
              data-show-files={showFiles}
              aria-label={t("memoryTags")}
            >
              <div className="tags-heading">
                <p>{t("tagsGlobalHint")}</p>
                {showFiles && (
                  <button
                    aria-label={t("clearFilters")}
                    onClick={() => {
                      setSelectedTags([]);
                      setDraft("");
                      setQuery("");
                      setPage(1);
                      setPaths([]);
                    }}
                  >
                    <X size={13} />
                  </button>
                )}
              </div>
              <div
                className="knowledge-tags-content"
                aria-busy={tagsLoading && supportsTags}
              >
                {(!supportsTags && config) || tagError !== null ? (
                  <p className="navigator-note" role="status">
                    {t("tagsUnavailable")}
                  </p>
                ) : tagsLoading && tags.length === 0 ? (
                  <div className="tree-skeleton">
                    <i />
                    <i />
                    <i />
                  </div>
                ) : tags.length === 0 ? (
                  <p className="navigator-note">{t("noMemoryTags")}</p>
                ) : (
                  <div className="tag-cloud">
                    {tags.map(([tag, count]) => (
                      <button
                        key={tag}
                        aria-pressed={selectedTags.includes(tag)}
                        className={selectedTags.includes(tag) ? "selected" : ""}
                        onClick={() => chooseTag(tag)}
                      >
                        <span>{tag}</span>
                        <small className="count-badge">{count}</small>
                      </button>
                    ))}
                  </div>
                )}
                {moreTags && supportsTags && tagError === null && (
                  <button
                    className="text-action more-tags"
                    disabled={tagsLoading}
                    onClick={() => setTagsPage(tagsPage + 1)}
                  >
                    {t("moreTags")}
                    <ChevronDown size={13} />
                  </button>
                )}
              </div>
            </section>
          )}
          {showFiles && (
            <div className="knowledge-navigator">
              <div className="list-section-label">
                <span>{t("files")}</span>
                <span className="count-badge">{paths.length}</span>
              </div>
              {error !== null && (
                <div className="navigator-error" role="alert">
                  {errorText(error, state.settings.language)}
                  <button
                    className="text-action"
                    onClick={() => setRefresh((n) => n + 1)}
                  >
                    {t("retry")}
                  </button>
                </div>
              )}
              {!error && !loading && tree.length === 0 && (
                <p className="navigator-empty">
                  {t(activeTags?.length || query ? "noMatches" : "noFiles")}
                </p>
              )}
              {loading && (
                <div className="tree-skeleton">
                  <i />
                  <i />
                  <i />
                </div>
              )}
              <Tree
                nodes={tree}
                selected={selected}
                onOpen={(path) => {
                  selectDocument(path);
                  onNavigate?.();
                }}
              />
              {limited && <p className="navigator-note">{t("limitedFiles")}</p>}
              {pages > 1 && (
                <div className="pagination">
                  <button
                    aria-label={t("previousPage")}
                    disabled={page <= 1}
                    onClick={() => setPage(page - 1)}
                  >
                    <ArrowLeft size={14} />
                  </button>
                  <span>{t("pageLabel", { page, total: pages })}</span>
                  <button
                    aria-label={t("nextPage")}
                    disabled={page >= pages}
                    onClick={() => setPage(page + 1)}
                  >
                    <ArrowRight size={14} />
                  </button>
                </div>
              )}
            </div>
          )}
        </>
      }
    >
      <div className="knowledge-content">
        {showTimeline ? (
          <TimelinePane
            name={daily ? "daily-timeline" : "tag-timeline"}
            language={state.settings.language}
            listKey={JSON.stringify([source, query, timelinePage, activeTags])}
            detailKey={selected}
            onClose={editorOpen ? undefined : () => selectDocument("")}
            heading={
              <TimelineHeading
                title={t(daily ? "dailyTab" : "memoryTagsTab")}
                icon={daily ? <CalendarDays size={22} /> : <Tags size={22} />}
                controls={
                  daily && (
                    <span className="timeline-order">
                      {t("timelineNewest")}
                    </span>
                  )
                }
              />
            }
            list={
              error !== null ? (
                <p className="timeline-empty navigator-error" role="alert">
                  {errorText(error, state.settings.language)}{" "}
                  <button
                    type="button"
                    className="text-action"
                    onClick={() => setRefresh((n) => n + 1)}
                  >
                    {t("retry")}
                  </button>
                </p>
              ) : timelineItems.length ? (
                <>
                  <TimelineList
                    key={JSON.stringify([
                      source,
                      query,
                      timelinePage,
                      activeTags,
                    ])}
                    language={state.settings.language}
                    selected={selected}
                    disabled={editorOpen}
                    holdNew={daily && timelinePage === 1}
                    undatedTogether={!daily}
                    onSelect={selectDocument}
                    items={
                      daily
                        ? timelineItems.slice(
                            (timelinePage - 1) * DAILY_PAGE_SIZE,
                            timelinePage * DAILY_PAGE_SIZE,
                          )
                        : timelineItems
                    }
                  />
                  {daily && dailyPages > 1 && (
                    <nav
                      className="timeline-pagination"
                      aria-label={t("navigation")}
                    >
                      <button
                        className="icon-button"
                        aria-label={t("previousPage")}
                        disabled={editorOpen || timelinePage <= 1}
                        onClick={() => {
                          selectDocument("");
                          setDailyPage(timelinePage - 1);
                        }}
                      >
                        <ArrowLeft size={14} aria-hidden />
                      </button>
                      <span>
                        {t("pageLabel", {
                          page: timelinePage,
                          total: dailyPages,
                        })}
                      </span>
                      <button
                        className="icon-button"
                        aria-label={t("nextPage")}
                        disabled={editorOpen || timelinePage >= dailyPages}
                        onClick={() => {
                          selectDocument("");
                          setDailyPage(timelinePage + 1);
                        }}
                      >
                        <ArrowRight size={14} aria-hidden />
                      </button>
                    </nav>
                  )}
                </>
              ) : (
                <p className="timeline-empty">
                  {t(
                    loading
                      ? "documentLoading"
                      : query
                        ? "noMatches"
                        : "noFiles",
                  )}
                </p>
              )
            }
          >
            {preview}
          </TimelinePane>
        ) : selected ? (
          preview
        ) : (
          <div className="knowledge-empty">
            <div className="knowledge-illustration">
              <BookOpen size={44} strokeWidth={1} />
              <span />
              <span />
            </div>
            <span className="eyebrow">A LITTLE MORE CONNECTED</span>
            <h2>{t("selectDocument")}</h2>
            <p>{t("selectDocumentHint")}</p>
          </div>
        )}
      </div>
    </SplitPane>
  );
}
