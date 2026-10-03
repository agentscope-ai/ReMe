import { useEffect, useRef, useState } from "react";
import { BookOpen, Network, RotateCcw, Search, X } from "lucide-react";
import { useI18n } from "../app/i18n";
import { openWorkspaceFile } from "../app/open-file";
import { useWorkspaceStore } from "../app/store";
import { WORKSPACE_CHANGED } from "../app/studio-mode";
import { demoWorkspace } from "./api";
import { DEMO_SEARCH } from "./DemoTags";
import { WELCOME_PATH } from "./workspace";
import styles from "./demo.module.css";

export default function DemoBar() {
  const { language, t } = useI18n();
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [, setRevision] = useState(0);
  const [error, setError] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const workspace = demoWorkspace();
  const results = workspace.search(query, selected);
  const tags = workspace.tags();
  const refresh = () => setRevision((value) => value + 1);
  useEffect(() => {
    const search = (event: Event) => {
      setQuery("");
      setSelected([(event as CustomEvent<string>).detail]);
      dialog.current?.showModal();
    };
    window.addEventListener(WORKSPACE_CHANGED, refresh);
    window.addEventListener("storage", refresh);
    window.addEventListener(DEMO_SEARCH, search);
    return () => {
      window.removeEventListener(WORKSPACE_CHANGED, refresh);
      window.removeEventListener("storage", refresh);
      window.removeEventListener(DEMO_SEARCH, search);
    };
  }, []);
  const reset = () => {
    if (!window.confirm(t("demoResetConfirm"))) return;
    try {
      workspace.reset();
      useWorkspaceStore.persist.clearStorage();
      window.location.reload();
    } catch (error) {
      setError(error instanceof Error ? error.message : t("unknownError"));
    }
  };
  const showFile = (path: string) => {
    dialog.current?.close();
    void openWorkspaceFile(path);
  };
  return (
    <>
      <div className={styles.bar}>
        <div className={styles.description}>
          <strong>{t("demoTitle")}</strong>
          <span>{t("demoDescription")}</span>
        </div>
        <div className={styles.actions}>
          <button onClick={() => void openWorkspaceFile(WELCOME_PATH)}>
            <BookOpen size={14} />
            {t("demoGuide")}
          </button>
          <button
            onClick={() => useWorkspaceStore.getState().openGraph("wiki")}
          >
            <Network size={14} />
            {t("memoryGraphShort")}
          </button>
          <button onClick={() => dialog.current?.showModal()}>
            <Search size={14} />
            {t("demoSearch")}
          </button>
          <button onClick={reset}>
            <RotateCcw size={14} />
            {t("demoReset")}
          </button>
        </div>
        {error && <span role="alert">{error}</span>}
      </div>
      <dialog
        ref={dialog}
        className={styles.dialog}
        aria-labelledby="demo-search-title"
      >
        <header>
          <div>
            <h2 id="demo-search-title">{t("demoSearch")}</h2>
            <p>{t("demoSearchDescription")}</p>
          </div>
          <button
            onClick={() => dialog.current?.close()}
            aria-label={t("demoCloseSearch")}
          >
            <X size={18} />
          </button>
        </header>
        <label className={styles.search}>
          <Search size={16} />
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={t("demoSearchPlaceholder")}
            aria-label={t("demoSearchPlaceholder")}
          />
        </label>
        <div className={styles.tagHeading}>
          <strong>memory_tags</strong>
          <span>{t("demoTagHint")}</span>
          {selected.length > 0 && (
            <button onClick={() => setSelected([])}>
              {t("demoClearTags")}
            </button>
          )}
        </div>
        <div className={styles.tags}>
          {tags.map(({ tag, count }) => (
            <button
              key={tag}
              className={styles.tag}
              aria-pressed={selected.includes(tag)}
              onClick={() =>
                setSelected((current) =>
                  current.includes(tag)
                    ? current.filter((value) => value !== tag)
                    : [...current, tag],
                )
              }
            >
              {tag}
              <span>{count}</span>
            </button>
          ))}
        </div>
        <p className={styles.count} role="status">
          {t("demoMatches", { count: String(results.length) })}
        </p>
        <div className={styles.results} lang={language}>
          {results.map((file) => (
            <button
              className={styles.result}
              key={file.path}
              onClick={() => showFile(file.path)}
            >
              <strong>{file.title}</strong>
              <code>{file.path}</code>
              <p>{file.snippet}</p>
              <span>{file.tags.join(" · ")}</span>
            </button>
          ))}
          {!results.length && <p>{t("demoNoResults")}</p>}
        </div>
      </dialog>
    </>
  );
}
