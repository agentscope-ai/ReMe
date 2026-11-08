import { useEffect, useMemo, useRef, useState } from "react";
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
import type {
  DesktopState,
  JobResponse,
  WorkspaceConfig,
  WorkspaceRequest,
  TagFiles,
} from "../../shared/protocol.ts";
import {
  deserializeError,
  errorText,
  translator,
  UiError,
} from "../../shared/i18n.ts";
import { action } from "../desktop.ts";
import SplitPane from "../split-pane.tsx";
import { buildTree, resolveDocumentLink, type FileTree } from "./model.ts";
import { workspaceLinkTarget } from "./wikilinks.ts";
import DocumentPreview from "./DocumentPreview.tsx";

async function request<T>(value: WorkspaceRequest): Promise<JobResponse<T>> {
  const result = await window.desktop.workspace(value);
  if ("error" in result) throw deserializeError(result.error);
  return result as JobResponse<T>;
}

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
  return (
    <>
      <button
        className={`file-tree-row ${selected === node.path ? "selected" : ""}`}
        style={{ paddingInlineStart: 12 + depth * 15 }}
        title={node.path}
        aria-expanded={node.children ? expanded : undefined}
        onClick={() =>
          node.children ? setExpanded(!expanded) : onOpen(node.path)
        }
      >
        {node.children ? (
          <>
            {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
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
}: {
  state: DesktopState;
  collapsed?: boolean;
}) {
  const t = translator(state.settings.language);
  const active = state.activePage === "knowledge";
  const scope = state.settings.endpoint + "\n" + state.workspace;
  const [config, setConfig] = useState<WorkspaceConfig>();
  const [source, setSource] = useState("workspace");
  const [paths, setPaths] = useState<string[]>([]);
  const [allPaths, setAllPaths] = useState<string[]>([]);
  const [limited, setLimited] = useState(false);
  const [selected, setSelected] = useState("");
  const [document, setDocument] = useState<{
    content: string;
    mtime?: string;
  }>();
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useState("");
  const [tags, setTags] = useState<[string, number][]>([]);
  const [tagsPage, setTagsPage] = useState(1);
  const [moreTags, setMoreTags] = useState(false);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [tagError, setTagError] = useState<unknown>(null);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [refresh, setRefresh] = useState(0);
  const [loading, setLoading] = useState(false);
  const [fileLoading, setFileLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [fileError, setFileError] = useState<unknown>(null);
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
  useEffect(() => {
    setConfig(undefined);
    setPaths([]);
    setAllPaths([]);
    setSelected("");
    setDocument(undefined);
    loadedPath.current = "";
    setTags([]);
    setTagsPage(1);
    setSelectedTags([]);
    setPage(1);
    setQuery("");
    setDraft("");
    setError(null);
    setTagError(null);
  }, [scope]);
  useEffect(() => {
    if (!active || state.connection !== "connected" || config) return;
    let live = true;
    void request<WorkspaceConfig>({ type: "config" })
      .then((result) => {
        if (!live) return;
        if (
          !result.answer.jobs.includes("list") ||
          !result.answer.jobs.includes("load")
        )
          throw new UiError("workspaceUnavailable");
        setConfig(result.answer);
        setError(null);
      })
      .catch((error) => {
        if (live) setError(error);
      });
    return () => {
      live = false;
    };
  }, [active, state.connection, config, scope, refresh]);
  useEffect(() => {
    if (!active || !config || !supportsTags) return;
    let live = true;
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
      });
    return () => {
      live = false;
    };
  }, [active, config, supportsTags, tagsPage, refresh]);
  useEffect(() => {
    if (!active || !config || state.connection !== "connected") return;
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
          tags: selectedTags,
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
      } else if (selectedTags.length) {
        const result = await request<TagFiles>({
          type: "tagFiles",
          tags: selectedTags,
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
      if (selectedTags.length && filter?.applied !== true)
        throw new UiError("tagFilterUnavailable");
      if (live) {
        if (
          filter?.tags &&
          JSON.stringify(filter.tags) !== JSON.stringify(selectedTags)
        )
          setSelectedTags(filter.tags);
        setPaths(next);
        setAllPaths((previous) => [...new Set([...previous, ...next])]);
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
    config,
    state.connection,
    root,
    selectedTags,
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
    if (!selected) return;
    let live = true;
    if (loadedPath.current !== selected) {
      setFileLoading(true);
      if (documentScroll.current) documentScroll.current.scrollTop = 0;
    }
    setFileError(null);
    void request<string>({ type: "load", path: selected })
      .then((result) => {
        if (live) {
          loadedPath.current = selected;
          setDocument({
            content: result.answer,
            mtime: result.metadata.mtime as string | undefined,
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
  }, [selected, scope, refresh]);
  const tree = useMemo(() => buildTree(paths, root), [paths, root]);
  const chooseTag = (tag: string) => {
    setPage(1);
    setSelectedTags((previous) =>
      previous.includes(tag)
        ? previous.filter((value) => value !== tag)
        : [...previous, tag],
    );
  };
  const openLink = (href: string) => {
    if (/^https?:\/\//i.test(href)) {
      void action({ type: "link.open", url: href });
      return;
    }
    const wiki = workspaceLinkTarget(href);
    try {
      const target = wiki ?? decodeURIComponent(href.split("#")[0]);
      if (!target) return;
      const path = resolveDocumentLink(
        target,
        wiki === undefined ? selected : "",
        allPaths,
      );
      if (path) {
        setSelected(path);
        return;
      }
    } catch {
      /* Malformed URL escapes are invalid workspace links. */
    }
    setFileError(new UiError("invalidWorkspaceRequest"));
  };
  return (
    <SplitPane
      name="knowledge"
      collapsed={collapsed}
      language={state.settings.language}
      sidebar={
        <>
          <div className="sidebar-heading">
            <span className="eyebrow">YOUR SECOND MEMORY</span>
            <h1>
              {t("knowledge")}
              <button
                className="icon-button"
                aria-label={t("refreshFiles")}
                onClick={() => setRefresh((n) => n + 1)}
              >
                <RefreshCw size={15} className={loading ? "spinning" : ""} />
              </button>
            </h1>
            <p>{t("knowledgeCaption")}</p>
          </div>
          <form
            className="knowledge-search"
            onSubmit={(event) => {
              event.preventDefault();
              setQuery(draft.trim());
              setPage(1);
            }}
          >
            <Search size={15} />
            <input
              aria-label={t("searchKnowledge")}
              placeholder={t("searchKnowledge")}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
            />
            <button aria-label={t("searchAction")}>
              <ArrowRight size={14} />
            </button>
          </form>
          <div className="source-tabs">
            {(
              [
                ["workspace", FolderTree, "workspaceTab"],
                ["daily", CalendarDays, "dailyTab"],
                ["digest", BookOpen, "digestTab"],
              ] as const
            ).map(([id, Icon, key]) => (
              <button
                key={id}
                aria-pressed={source === id}
                className={source === id ? "active" : ""}
                onClick={() => {
                  setSource(id);
                  setPage(1);
                }}
              >
                <Icon size={14} />
                {t(key)}
              </button>
            ))}
          </div>
          <div className="knowledge-navigator">
            <div className="list-section-label">
              <span>{t("files")}</span>
              <span>{paths.length}</span>
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
                {t(selectedTags.length || query ? "noMatches" : "noFiles")}
              </p>
            )}
            {loading && (
              <div className="tree-skeleton">
                <i />
                <i />
                <i />
              </div>
            )}
            <Tree nodes={tree} selected={selected} onOpen={setSelected} />
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
            <div className="tags-heading">
              <Tags size={13} />
              <span>{t("memoryTags")}</span>
              {(selectedTags.length > 0 || query) && (
                <button
                  aria-label={t("clearFilters")}
                  onClick={() => {
                    setSelectedTags([]);
                    setDraft("");
                    setQuery("");
                    setPage(1);
                  }}
                >
                  <X size={13} />
                </button>
              )}
            </div>
            <p className="navigator-note">{t("tagsGlobalHint")}</p>
            {((!supportsTags && config) || tagError !== null) && (
              <p className="navigator-note" role="status">
                {t("tagsUnavailable")}
              </p>
            )}
            <div className="tag-cloud">
              {tags.map(([tag, count]) => (
                <button
                  key={tag}
                  aria-pressed={selectedTags.includes(tag)}
                  className={selectedTags.includes(tag) ? "selected" : ""}
                  onClick={() => chooseTag(tag)}
                >
                  <span>{tag}</span>
                  <small>{count}</small>
                </button>
              ))}
            </div>
            {moreTags && (
              <button
                className="text-action more-tags"
                onClick={() => setTagsPage(tagsPage + 1)}
              >
                {t("moreTags")}
                <ChevronDown size={13} />
              </button>
            )}
          </div>
          <div className="sidebar-footer">
            <BookOpen size={13} />
            {t("privateMemory")}
          </div>
        </>
      }
    >
      {selected ? (
        <div className="document-pane">
          <header className="document-toolbar">
            <FileText size={16} />
            <span title={selected}>{selected}</span>
            <span className="reading-label">{t("reading")}</span>
          </header>
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
            ) : fileLoading ? (
              <div className="document-loading">{t("documentLoading")}</div>
            ) : (
              document && (
                <DocumentPreview
                  key={selected}
                  path={selected}
                  content={document.content}
                  mtime={document.mtime}
                  tagKey={config?.tag_key ?? "memory_tags"}
                  supportsTags={supportsTags}
                  onTag={chooseTag}
                  onLink={openLink}
                  language={state.settings.language}
                />
              )
            )}
          </div>
        </div>
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
    </SplitPane>
  );
}
