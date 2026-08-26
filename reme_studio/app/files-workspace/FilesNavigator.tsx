"use client";

/* eslint-disable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex -- ARIA separators are adjustable controls when they expose a value and keyboard handlers. */

import { useEffect, useMemo, useState } from "react";
import {
  BookOpen,
  CalendarDays,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  File,
  FileText,
  Folder,
  FolderOpen,
  FolderTree,
  LoaderCircle,
  MessageSquarePlus,
  Network,
} from "lucide-react";
import {
  getAppConfig,
  listWorkspaceFiles,
  readWorkspaceFile,
  REME_API_ENDPOINT,
} from "../api";
import { useI18n } from "../i18n";
import { useWorkspaceStore } from "../store";
import type {
  AppConfig,
  MemoryGraphRoot,
  TreeNode,
  WorkspaceSource,
} from "../types";
import {
  absoluteWorkspacePath,
  WORKSPACE_FILE_DRAG_TYPE,
} from "../workspace-drag";
import {
  buildTree,
  filterPathsBySource,
  parseWorkspaceExtensions,
  sourceDirectory,
  WORKSPACE_FILE_LIMIT,
} from "../workspace-files";

const extensions = parseWorkspaceExtensions(
  process.env.NEXT_PUBLIC_REME_WORKSPACE_EXTENSIONS,
);
const AUTO_REFRESH_MS = 10_000;
const isEditable = (name: string) =>
  extensions.has(name.split(".").pop()?.toLowerCase() || "");
const loadWorkspace = () =>
  Promise.all([getAppConfig(), listWorkspaceFiles([...extensions])]);
const remeEndpoint = (() => {
  try {
    return new URL(REME_API_ENDPOINT).host;
  } catch {
    return REME_API_ENDPOINT;
  }
})();

function DirectoryNode({
  node,
  workspaceDir,
  source,
  depth = 0,
}: {
  node: TreeNode;
  workspaceDir: string;
  source: WorkspaceSource;
  depth?: number;
}) {
  const { t } = useI18n();
  const [expanded, setExpanded] = useState(depth === 0);
  const {
    tabs,
    activeTabId,
    openMarkdown,
    openGraph,
    hydrateMarkdown,
    failMarkdown,
  } = useWorkspaceStore();
  const active = tabs.find((tab) => tab.id === activeTabId);
  const selected = active?.type === "markdown" && active.path === node.path;
  if (node.type === "directory") {
    const graphRoot =
      source === "digest" &&
      depth === 0 &&
      ["wiki", "personal", "procedure"].includes(node.name)
        ? (node.name as MemoryGraphRoot)
        : undefined;
    return (
      <>
        <div
          className={`tree-directory-row ${
            active?.type === "graph" && active.root === graphRoot
              ? "graph-active"
              : ""
          }`}
        >
          <button
            className="tree-row"
            style={{ paddingInlineStart: 10 + depth * 15 }}
            onClick={() => setExpanded(!expanded)}
            aria-expanded={expanded}
          >
            {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            {expanded ? <FolderOpen size={15} /> : <Folder size={15} />}
            <span>{node.name}</span>
          </button>
          {graphRoot && (
            <button
              className="tree-graph-button"
              onClick={() => openGraph(graphRoot)}
              aria-label={`${t("memoryGraph")} · ${node.name}`}
              title={`${t("memoryGraph")} · ${node.name}`}
            >
              <Network size={13} />
              <span>{t("memoryGraphShort")}</span>
            </button>
          )}
        </div>
        {expanded &&
          node.children.map((child) => (
            <DirectoryNode
              key={child.path}
              node={child}
              workspaceDir={workspaceDir}
              source={source}
              depth={depth + 1}
            />
          ))}
      </>
    );
  }
  const open = async () => {
    if (!isEditable(node.name)) return;
    const existing = tabs.some(
      (tab) => tab.type === "markdown" && tab.path === node.path,
    );
    const id = openMarkdown(node.path);
    if (existing) return;
    try {
      const file = await readWorkspaceFile(node.path);
      hydrateMarkdown(id, file.content, file.stat.mtime);
    } catch (error) {
      failMarkdown(
        id,
        error instanceof Error ? error.message : t("fileReadFailed"),
      );
    }
  };
  const absolutePath = workspaceDir
    ? absoluteWorkspacePath(workspaceDir, node.path)
    : "";
  return (
    <button
      className={`tree-row file-row ${selected ? "selected" : ""}`}
      disabled={!isEditable(node.name)}
      draggable={Boolean(absolutePath) && isEditable(node.name)}
      onDragStart={(event) => {
        if (!absolutePath) return;
        event.dataTransfer.effectAllowed = "copy";
        event.dataTransfer.setData(WORKSPACE_FILE_DRAG_TYPE, absolutePath);
        event.dataTransfer.setData("text/plain", absolutePath);
      }}
      style={{ paddingInlineStart: 28 + depth * 15 }}
      onClick={() => void open()}
      title={node.path}
    >
      {isEditable(node.name) ? <FileText size={15} /> : <File size={15} />}
      <span>{node.name}</span>
    </button>
  );
}

/** ReMe adapter of QwenPaw FilesNavigator. Profile/archive, project switching and Git are intentionally omitted. */
interface FilesNavigatorProps {
  open: boolean;
  width: number;
  resizing: boolean;
  onResizeStart: (event: React.PointerEvent<HTMLDivElement>) => void;
  onResizeKeyDown: (event: React.KeyboardEvent<HTMLDivElement>) => void;
}

export default function FilesNavigator({
  open,
  width,
  resizing,
  onResizeStart,
  onResizeKeyDown,
}: FilesNavigatorProps) {
  const { t } = useI18n();
  const [config, setConfig] = useState<AppConfig>();
  const [paths, setPaths] = useState<string[]>([]);
  const [limited, setLimited] = useState(false);
  const [status, setStatus] = useState<"loading" | "ready" | "error">(
    "loading",
  );
  const [source, setSource] = useState<WorkspaceSource>("workspace");
  const openAgent = useWorkspaceStore((state) => state.openAgent);
  useEffect(() => {
    let mounted = true;
    let loading = false;
    const refresh = async () => {
      if (loading || !mounted) return;
      loading = true;
      try {
        const [nextConfig, listing] = await loadWorkspace();
        if (!mounted) return;
        setConfig(nextConfig);
        setPaths(listing.paths);
        setLimited(listing.limited);
        setStatus("ready");
      } catch {
        if (mounted) setStatus("error");
      } finally {
        loading = false;
      }
    };
    void refresh();
    const timer = window.setInterval(() => {
      if (!document.hidden) void refresh();
    }, AUTO_REFRESH_MS);
    const onVisibilityChange = () => {
      if (!document.hidden) void refresh();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      mounted = false;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, []);
  const visiblePaths = useMemo(
    () => (config ? filterPathsBySource(paths, source, config) : paths),
    [config, paths, source],
  );
  const tree = useMemo(
    () =>
      buildTree(
        visiblePaths,
        extensions,
        config ? sourceDirectory(source, config) : "",
      ),
    [config, source, visiblePaths],
  );
  return (
    <aside
      id="workspace-navigator"
      className={`navigator ${open ? "" : "navigator-closed"} ${
        resizing ? "navigator-resizing" : ""
      }`}
      style={{ width: open ? width : 0 }}
      aria-hidden={!open}
      inert={!open}
    >
      <div
        className="navigator-resize-handle"
        role="separator"
        aria-orientation="vertical"
        aria-label={t("resizeNavigator")}
        aria-valuemin={220}
        aria-valuenow={Math.round(width)}
        tabIndex={0}
        onPointerDown={onResizeStart}
        onKeyDown={onResizeKeyDown}
      />
      <div className="navigator-content">
        <div className="source-tabs" role="tablist">
          <button
            role="tab"
            aria-selected={source === "workspace"}
            className={source === "workspace" ? "active" : ""}
            onClick={() => setSource("workspace")}
          >
            <FolderTree size={13} />
            {t("workspaceTab")}
          </button>
          <button
            role="tab"
            aria-selected={source === "daily"}
            className={source === "daily" ? "active" : ""}
            onClick={() => setSource("daily")}
          >
            <CalendarDays size={13} />
            {t("dailyTab")}
          </button>
          <button
            role="tab"
            aria-selected={source === "digest"}
            className={source === "digest" ? "active" : ""}
            onClick={() => setSource("digest")}
          >
            <BookOpen size={13} />
            {t("knowledgeTab")}
          </button>
          <button
            className="chat-tab"
            role="tab"
            aria-selected={false}
            aria-label={t("newAgentChat")}
            onClick={openAgent}
          >
            <MessageSquarePlus size={13} />
            {t("chatTab")}
          </button>
        </div>
        <div className="tree" role="tree" aria-busy={status === "loading"}>
          {status === "loading" && !paths.length && (
            <div className="side-state">
              <LoaderCircle className="spin" size={15} />
              {t("loadingWorkspace")}
            </div>
          )}
          {status === "error" && (
            <div className="side-state error">
              <CircleAlert size={15} />
              {t("connectionFailed")}
            </div>
          )}
          {status === "ready" && !tree.length && (
            <div className="side-state">{t("emptyWorkspace")}</div>
          )}
          {status === "ready" && limited && (
            <div className="side-state warning" role="status">
              <CircleAlert size={15} />
              {t("workspaceFileLimit", {
                limit: WORKSPACE_FILE_LIMIT.toLocaleString(),
              })}
            </div>
          )}
          {tree.map((node) => (
            <DirectoryNode
              key={node.path}
              node={node}
              workspaceDir={config?.workspace_dir || ""}
              source={source}
            />
          ))}
        </div>
        <div className="workspace-path" title={REME_API_ENDPOINT}>
          <span
            className={`status-dot ${status === "error" ? "offline" : ""}`}
          />
          <span>
            {status === "error"
              ? `${t("connectionFailed")} · ${remeEndpoint}`
              : remeEndpoint}
          </span>
        </div>
      </div>
    </aside>
  );
}
