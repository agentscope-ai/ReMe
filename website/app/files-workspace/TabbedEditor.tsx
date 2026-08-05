"use client";

import { useCallback, useEffect, useState } from "react";
import Editor from "@monaco-editor/react";
import { Check, Code2, Download, Eye, LoaderCircle, Save } from "lucide-react";
import { saveWorkspaceFile } from "../api";
import { useI18n } from "../i18n";
import "../monaco-setup";
import { useWorkspaceStore } from "../store";
import { useThemeStore } from "../theme";
import type { WorkspaceTab } from "../types";
import FilePreview from "./FilePreview";
import { getLanguage } from "./get-language";
import styles from "./files-workspace.module.css";

type FileTab = Extract<WorkspaceTab, { type: "markdown" }>;

/** Minimal ReMe port of QwenPaw's TabbedEditor: preview-first, Monaco edit, download and Cmd/Ctrl+S. */
export default function TabbedEditor({ tab }: { tab: FileTab }) {
  const { t } = useI18n();
  const [preview, setPreview] = useState(true);
  const [saving, setSaving] = useState(false);
  const update = useWorkspaceStore((state) => state.updateMarkdown);
  const markSaved = useWorkspaceStore((state) => state.markSaved);
  const dirty = tab.content !== tab.savedContent;
  const theme = useThemeStore((state) => state.resolved);

  const save = useCallback(async () => {
    if (!dirty || saving) return;
    setSaving(true);
    try {
      const stat = await saveWorkspaceFile(tab.path, tab.content, tab.mtime);
      markSaved(tab.id, stat.mtime);
    } finally {
      setSaving(false);
    }
  }, [dirty, markSaved, saving, tab]);

  useEffect(() => {
    const shortcut = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        void save();
      }
    };
    window.addEventListener("keydown", shortcut);
    return () => window.removeEventListener("keydown", shortcut);
  }, [save]);

  const download = () => {
    const url = URL.createObjectURL(
      new Blob([tab.content], { type: "text/plain;charset=utf-8" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = tab.title;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  if (tab.loading)
    return (
      <div className="center">
        <LoaderCircle className="spin" />
        {t("openingFile", { path: tab.path })}
      </div>
    );
  if (tab.error) return <div className="center error">{tab.error}</div>;

  return (
    <div className={styles.wrap}>
      <div className={styles.toolbar}>
        <span className={styles.fileName}>{tab.path}</span>
        <div className={styles.documentActions}>
          <div className={styles.modeSwitch}>
            <button
              className={preview ? styles.modeActive : ""}
              onClick={() => setPreview(true)}
            >
              <Eye size={12} />
              {t("preview")}
            </button>
            <button
              className={!preview ? styles.modeActive : ""}
              onClick={() => setPreview(false)}
            >
              <Code2 size={12} />
              {t("edit")}
            </button>
          </div>
          <button
            className={styles.iconBtn}
            onClick={download}
            aria-label={t("download")}
            title={t("download")}
          >
            <Download size={13} />
          </button>
          {!preview && (
            <button
              className={styles.iconBtn}
              onClick={() => void save()}
              disabled={!dirty || saving}
              aria-label={t("save")}
              title={t("save")}
            >
              {saving ? (
                <LoaderCircle className="spin" size={13} />
              ) : dirty ? (
                <Save size={13} />
              ) : (
                <Check size={13} />
              )}
            </button>
          )}
        </div>
      </div>
      <div className={styles.editor}>
        {preview ? (
          <FilePreview filePath={tab.path} content={tab.content} />
        ) : (
          <Editor
            path={tab.path}
            language={getLanguage(tab.path)}
            value={tab.content}
            theme={theme === "dark" ? "vs-dark" : "vs"}
            onChange={(value) => update(tab.id, value ?? "")}
            options={{
              minimap: { enabled: false },
              fontSize: 13,
              wordWrap: "on",
              scrollBeyondLastLine: false,
              automaticLayout: true,
            }}
          />
        )}
      </div>
    </div>
  );
}
