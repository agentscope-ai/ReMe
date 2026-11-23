import { memo } from "react";
import { LoaderCircle, Paperclip } from "lucide-react";
import type { Language } from "../../client/src/i18n.ts";
import { translator } from "../../client/src/i18n.ts";
import type {
  DisplayMessage,
  SummaryItem,
  TranscriptItem,
} from "../../client/src/protocol.ts";
import { previewContent, replyText } from "../../client/src/surface.ts";
import { Activity } from "./activity.tsx";
import type { ProcessItem } from "./activity.tsx";
import MarkdownText from "./markdown.tsx";

type ActivityEntry = { type: "activity"; id: string; items: ProcessItem[] };
type Entry = Exclude<TranscriptItem, ProcessItem> | SummaryItem | ActivityEntry;

/** Only visible prose, plans and attachments separate consecutive activity. */
export function timelineItems(items: TranscriptItem[]): Entry[] {
  const entries: Entry[] = [];
  for (const item of items) {
    if ((item.type === "text" || item.type === "summary") && !item.text.trim())
      continue;
    const previous = entries.at(-1);
    if (
      item.type === "tool" ||
      (item.type === "summary" && item.kind === "reasoning")
    ) {
      if (previous?.type === "activity") previous.items.push(item);
      else entries.push({ type: "activity", id: item.id, items: [item] });
    } else entries.push(item);
  }
  return entries;
}

/** Main history follows assistant prose; reasoning and raw output stay in disclosure. */
export default memo(function MessageBody({
  message,
  language,
  compact = false,
  onLink,
}: {
  message: DisplayMessage;
  language: Language;
  compact?: boolean;
  onLink: (url: string) => void;
}) {
  const items = message.items ?? [];
  const content = compact
    ? previewContent(replyText(message))
    : message.content;
  const entries = compact ? [] : timelineItems(items);
  const complete =
    !compact &&
    message.role === "assistant" &&
    message.source === "chat" &&
    !message.streaming &&
    (message.completed || message.status?.tone === "success") &&
    !message.error &&
    (!message.status || message.status.tone === "success");
  const hasContent = content || entries.length;
  return (
    <div className="message-body">
      {entries.length ? (
        entries.map((item) => {
          if (item.type === "activity")
            return (
              <Activity
                key={item.id}
                items={item.items}
                language={language}
                onLink={onLink}
              />
            );
          if (item.type === "artifact")
            return /^data:image\/(png|jpeg|gif|webp);base64,/.test(item.url) ? (
              <figure key={item.id} className="attachment-image">
                <img src={item.url} alt={item.label} />
              </figure>
            ) : (
              <button
                className="attachment-link"
                key={item.id}
                onClick={() => onLink(item.url)}
              >
                <Paperclip size={14} />
                {item.label}
              </button>
            );
          return item.text ? (
            <div
              className={`message-content${item.type === "summary" ? " message-plan" : ""}`}
              key={item.id}
              data-phase={
                item.type === "text" ? (item.phase ?? undefined) : undefined
              }
            >
              <MarkdownText
                text={item.text}
                completed={item.status !== "running"}
                onLink={onLink}
              />
            </div>
          ) : null;
        })
      ) : content ? (
        <div className="message-content">
          <MarkdownText
            text={content}
            completed={!message.streaming}
            onLink={onLink}
          />
        </div>
      ) : null}
      {complete && (
        <div className="turn-completion">
          {translator(language)("completed")}
        </div>
      )}
      {!complete &&
        message.status &&
        (!hasContent || ["error", "waiting"].includes(message.status.tone)) && (
          <span
            className={`message-status ${message.status.tone}`}
            role="status"
            aria-live="polite"
          >
            {message.streaming && (
              <LoaderCircle size={12} className="activity-spinner" />
            )}
            {message.status.text}
          </span>
        )}
      {message.error && message.error !== message.status?.text && (
        <p className="message-error">{message.error}</p>
      )}
    </div>
  );
});
