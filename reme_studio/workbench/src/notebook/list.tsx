import { ChevronRight, Lightbulb } from "lucide-react";
import { translator, type Language } from "../../../client/src/i18n.ts";
import {
  isTodo,
  projectLabel,
  splitNotebookText,
  type NotebookItem,
  type NotebookKind,
  type Todo,
} from "../../../client/src/notebook.ts";
import { statuses } from "./detail.tsx";

function localDay(value: string) {
  const date = new Date(value);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export default function NotebookList({
  items,
  kind,
  archived,
  language,
  disabled,
  onOpen,
  onToggleTodo,
}: {
  items: NotebookItem[];
  kind: NotebookKind;
  archived: boolean;
  language: Language;
  disabled: boolean;
  onOpen: (item: NotebookItem) => void;
  onToggleTodo: (item: Todo) => void;
}) {
  const t = translator(language);
  const ideas = kind === "idea";
  const dateFormat = new Intl.DateTimeFormat(
    language === "zh" ? "zh-CN" : "en",
    {
      month: "short",
      day: "numeric",
      weekday: "short",
    },
  );
  const dated = !ideas || archived;
  const groups = new Map<string, NotebookItem[]>();
  for (const item of items) {
    const day =
      archived && item.closed_at
        ? localDay(item.closed_at)
        : isTodo(item)
          ? (item.due_date ?? "")
          : "";
    const group = groups.get(day);
    if (group) group.push(item);
    else groups.set(day, [item]);
  }
  const days = [...groups.keys()].sort((a, b) => {
    if (!a || !b) return Number(!a) - Number(!b);
    return archived ? b.localeCompare(a) : a.localeCompare(b);
  });

  return (
    <div className="notebook-groups">
      {days.map((day) => (
        <section key={day} className="notebook-list-group" data-dated={dated}>
          {dated && (
            <h2 className="notebook-group-date">
              {day ? (
                <time dateTime={day}>
                  {dateFormat.format(new Date(`${day}T12:00:00`))}
                </time>
              ) : (
                t("notebookUnscheduled")
              )}
            </h2>
          )}
          <ul className="notebook-list" data-grid={ideas && !archived}>
            {groups.get(day)?.map((record) => {
              const todo = isTodo(record);
              const { title, body } = splitNotebookText(record.text);
              const preview = todo
                ? (record.close_note ?? record.progress.at(-1)?.text ?? body)
                : body;
              return (
                <li key={record.id} className="notebook-row">
                  {todo ? (
                    <input
                      type="checkbox"
                      aria-label={t(
                        record.status === "closed"
                          ? "notebookReopenItem"
                          : "notebookCloseItem",
                        { text: title },
                      )}
                      checked={record.status === "closed"}
                      disabled={disabled}
                      onChange={() => onToggleTodo(record)}
                    />
                  ) : (
                    <span className="notebook-idea-icon" aria-hidden>
                      <Lightbulb size={16} />
                    </span>
                  )}
                  <button
                    type="button"
                    className="notebook-row-content"
                    onClick={() => onOpen(record)}
                  >
                    <span className="notebook-row-meta">
                      <span className="notebook-project">
                        {projectLabel(record.project, language)}
                      </span>
                      {todo && (
                        <span
                          className="notebook-row-status"
                          data-status={record.status}
                        >
                          {t(statuses[record.status])}
                        </span>
                      )}
                    </span>
                    <span className="notebook-row-title">{title}</span>
                    {preview && (
                      <span className="notebook-row-preview">{preview}</span>
                    )}
                    <ChevronRight
                      className="notebook-row-chevron"
                      size={15}
                      aria-hidden
                    />
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </div>
  );
}
