import { translator, type Language } from "../../../client/src/i18n.ts";
import {
  DEFAULT_PROJECT,
  projectLabel,
  projectValue,
  type NotebookKind,
} from "../../../client/src/notebook.ts";

export interface RecordDraft {
  id?: string;
  project: string;
  text: string;
  due_date: string;
}

export default function RecordEditor({
  kind,
  draft,
  onChange,
  onSave,
  onCancel,
  busy,
  language,
}: {
  kind: NotebookKind;
  draft: RecordDraft;
  onChange: (draft: RecordDraft) => void;
  onSave: () => void;
  onCancel: () => void;
  busy: boolean;
  language: Language;
}) {
  const t = translator(language);
  return (
    <form
      className="notebook-editor"
      onSubmit={(event) => {
        event.preventDefault();
        onSave();
      }}
    >
      <h1>{t(draft.id ? "notebookEdit" : "notebookCapture")}</h1>
      <label>
        <span>{t("notebookText")}</span>
        <textarea
          autoFocus
          required
          aria-label={t("notebookText")}
          rows={3}
          maxLength={100000}
          value={draft.text}
          placeholder={t("notebookTextPlaceholder")}
          disabled={busy}
          onChange={(event) => onChange({ ...draft, text: event.target.value })}
        />
      </label>
      <label>
        <span>{t("notebookProject")}</span>
        <input
          aria-label={t("notebookProject")}
          value={
            draft.project === DEFAULT_PROJECT
              ? projectLabel(draft.project, language)
              : draft.project
          }
          placeholder={t("notebookDefaultProject")}
          disabled={busy}
          maxLength={100000}
          onChange={(event) =>
            onChange({
              ...draft,
              project: projectValue(event.target.value, language),
            })
          }
        />
      </label>
      {kind === "todo" && (
        <label>
          <span>{t("notebookDeadline")}</span>
          <input
            type="date"
            aria-label={t("notebookDeadline")}
            value={draft.due_date}
            disabled={busy}
            onChange={(event) =>
              onChange({ ...draft, due_date: event.target.value })
            }
          />
        </label>
      )}
      <div className="notebook-actions">
        <button className="primary" disabled={busy || !draft.text.trim()}>
          {t(busy ? "notebookSaving" : "notebookSave")}
        </button>
        <button
          type="button"
          className="secondary"
          disabled={busy}
          onClick={onCancel}
        >
          {t("notebookCancel")}
        </button>
      </div>
    </form>
  );
}
