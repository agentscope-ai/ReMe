import { useEffect, useState } from "react";
import {
  errorText,
  translator,
  type Language,
} from "../../../client/src/i18n.ts";
import { DocumentHeading } from "./DocumentPreview.tsx";

export default function DocumentEditor({
  path,
  content,
  mtime,
  language,
  navigationBlocked,
  save,
  onClose,
}: {
  path: string;
  content: string;
  mtime: string;
  language: Language;
  navigationBlocked: boolean;
  save: (content: string) => Promise<void>;
  onClose: () => void;
}) {
  const t = translator(language);
  const [initialText] = useState(() => content.replace(/\r\n?/g, "\n"));
  const [draft, setDraft] = useState(initialText);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const dirty = draft !== initialText;

  useEffect(() => {
    if (!dirty) return;
    const preventUnload = (event: BeforeUnloadEvent) => event.preventDefault();
    window.addEventListener("beforeunload", preventUnload);
    return () => window.removeEventListener("beforeunload", preventUnload);
  }, [dirty]);

  const submit = async () => {
    if (saving || !dirty) return;
    setSaving(true);
    setError(null);
    try {
      // A textarea normalizes newlines to LF. Keep the file's newline convention.
      const newline = content.match(/\r\n|\n|\r/)?.[0] ?? "\n";
      await save(newline === "\n" ? draft : draft.replaceAll("\n", newline));
      onClose();
    } catch (reason) {
      setError(reason);
      setSaving(false);
    }
  };

  return (
    <form
      className="document-content document-editor"
      onSubmit={(event) => {
        event.preventDefault();
        void submit();
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape" && !dirty && !saving) onClose();
        if (
          (event.metaKey || event.ctrlKey) &&
          !event.shiftKey &&
          !event.altKey &&
          event.key.toLowerCase() === "s"
        ) {
          event.preventDefault();
          void submit();
        }
      }}
    >
      <DocumentHeading path={path} mtime={mtime} language={language}>
        <div className="document-actions">
          <button
            type="button"
            className="subtle"
            disabled={saving}
            onClick={onClose}
          >
            {t(dirty ? "documentDiscard" : "cancel")}
          </button>
          <button type="submit" className="primary" disabled={!dirty || saving}>
            {t("documentSave")}
          </button>
        </div>
      </DocumentHeading>
      {error != null && (
        <p className="document-action-error" role="alert">
          {errorText(error, language)}
        </p>
      )}
      <textarea
        aria-label={t("documentSource")}
        autoFocus
        autoCapitalize="off"
        autoCorrect="off"
        spellCheck={false}
        value={draft}
        readOnly={saving}
        onChange={(event) => setDraft(event.target.value)}
      />
      <div className="document-edit-status" role="status">
        {saving
          ? t("documentSaving")
          : navigationBlocked
            ? t("documentFinishEditing")
            : dirty && t("documentUnsaved")}
      </div>
    </form>
  );
}
