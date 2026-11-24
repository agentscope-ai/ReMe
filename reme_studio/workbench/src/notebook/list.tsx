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

const ideaGroups = {
  today: "notebookIdeasToday",
  recent: "notebookIdeasRecent",
  earlier: "notebookIdeasEarlier",
  undated: "notebookIdeasUndated",
} as const;

function localDay(date: Date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function ideaGroup(
  value: string | null | undefined,
  today: string,
  recentStart: string,
) {
  if (!value) return "undated";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "undated";
  const day = localDay(date);
  if (day === today) return "today";
  return day >= recentStart ? "recent" : "earlier";
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
  const ideaGrid = kind === "idea" && !archived;
  const locale = language === "zh" ? "zh-CN" : "en";
  const now = new Date();
  const today = localDay(now);
  const recentStart = localDay(
    new Date(now.getFullYear(), now.getMonth(), now.getDate() - 7),
  );
  const dateFormat = new Intl.DateTimeFormat(locale, {
    month: "short",
    day: "numeric",
    weekday: "short",
  });
  const groups = new Map<string, NotebookItem[]>();
  for (const item of items) {
    const key = archived
      ? item.closed_at
        ? localDay(new Date(item.closed_at))
        : ""
      : isTodo(item)
        ? (item.due_date ?? "")
        : ideaGrid
          ? ideaGroup(item.created_at, today, recentStart)
          : "";
    const group = groups.get(key);
    if (group) group.push(item);
    else groups.set(key, [item]);
  }
  const days = [...groups.keys()].sort((a, b) => {
    if (!a || !b) return Number(!a) - Number(!b);
    return archived ? b.localeCompare(a) : a.localeCompare(b);
  });
  const sections = ideaGrid
    ? Object.entries(ideaGroups).map(([key, label]) => ({
        key,
        label: t(label),
      }))
    : days.map((key) => ({
        key,
        label: key ? (
          <time dateTime={key}>
            {dateFormat.format(new Date(`${key}T12:00:00`))}
          </time>
        ) : (
          t("notebookUnscheduled")
        ),
      }));

  return (
    <div className="notebook-groups">
      {sections
        .filter(({ key }) => groups.has(key))
        .map(({ key, label }) => (
          <section
            key={key}
            className="notebook-list-group"
            data-date-rail={!ideaGrid}
          >
            <h2 className="notebook-group-date">{label}</h2>
            <ul className="notebook-list" data-grid={ideaGrid}>
              {groups.get(key)?.map((record) => {
                const todo = isTodo(record);
                const { title, body } = splitNotebookText(record.text);
                const capturedAt = ideaGrid && !todo ? record.created_at : null;
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
                        {capturedAt &&
                          (key === "recent" || key === "earlier") && (
                            <time
                              className="notebook-idea-date"
                              dateTime={capturedAt}
                            >
                              {new Date(capturedAt).toLocaleDateString(locale, {
                                month: "short",
                                day: "numeric",
                                year: key === "earlier" ? "numeric" : undefined,
                              })}
                            </time>
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
