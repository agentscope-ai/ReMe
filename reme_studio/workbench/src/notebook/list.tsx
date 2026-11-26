import { Lightbulb } from "lucide-react";
import { translator, type Language } from "../../../client/src/i18n.ts";
import {
  isTodo,
  projectLabel,
  splitNotebookText,
  type NotebookItem,
  type Todo,
} from "../../../client/src/notebook.ts";
import { TimelineBadge, TimelineList } from "../timeline.tsx";
import { statuses } from "./detail.tsx";

export default function NotebookList({
  items,
  archived,
  order,
  language,
  disabled,
  editing,
  selected,
  onOpen,
  onToggleTodo,
}: {
  items: NotebookItem[];
  archived: boolean;
  order: "due" | "created";
  language: Language;
  disabled: boolean;
  editing: boolean;
  selected: string;
  onOpen: (id: string) => void;
  onToggleTodo: (item: Todo) => void;
}) {
  const t = translator(language);
  const due = !archived && order === "due";
  return (
    <TimelineList
      language={language}
      selected={selected}
      onSelect={onOpen}
      disabled={editing}
      ascending={due}
      relative={!due}
      unknownDate={due ? t("notebookUnscheduled") : undefined}
      items={items.map((record) => {
        const todo = isTodo(record);
        const { title, body } = splitNotebookText(record.text);
        return {
          id: record.id,
          title,
          date: archived
            ? record.closed_at
            : due && todo
              ? record.due_date
              : record.created_at,
          preview: todo
            ? (record.close_note ?? record.progress.at(-1)?.text ?? body)
            : body,
          leading: todo ? (
            <input
              type="checkbox"
              aria-label={t(
                record.status === "closed"
                  ? "notebookReopenItem"
                  : "notebookCloseItem",
                { text: title },
              )}
              checked={record.status === "closed"}
              disabled={disabled || editing}
              onChange={() => onToggleTodo(record)}
            />
          ) : (
            <span className="timeline-card-leading">
              <Lightbulb size={16} aria-hidden />
            </span>
          ),
          metadata: (
            <>
              <span>{projectLabel(record.project, language)}</span>
              {todo && (
                <TimelineBadge
                  tone={record.status === "in_progress" ? "accent" : undefined}
                >
                  {t(statuses[record.status])}
                </TimelineBadge>
              )}
            </>
          ),
        };
      })}
    />
  );
}
