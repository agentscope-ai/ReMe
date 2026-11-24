import { Children, isValidElement, memo, type ComponentProps } from "react";
import { LoaderCircle, Paperclip } from "lucide-react";
import { fromMarkdown } from "mdast-util-from-markdown";
import type { Language } from "../../client/src/i18n.ts";
import { translator } from "../../client/src/i18n.ts";
import { resolveWorkspaceLink } from "../../client/src/links.ts";
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

const reminderMarkdown = {
  p({ children }: ComponentProps<"p">) {
    const first = Children.toArray(children)[0];
    const lead =
      isValidElement<{ children?: unknown }>(first) && first.type === "strong"
        ? first.props.children
        : first;
    const source =
      typeof lead === "string" && /^\s*(?:来源|Sources?)\s*[:：]/i.test(lead);
    return <p className={source ? "message-source" : undefined}>{children}</p>;
  },
};

/** Lift a standalone source from the end of a delivered reminder for display. */
export function splitReminderSource(
  message: DisplayMessage,
  workspace?: string,
) {
  if (
    message.source !== "proactive" ||
    message.role !== "assistant" ||
    !message.completed ||
    message.streaming
  )
    return;

  const items = message.items ?? [];
  const lastItem = items.at(-1);
  if (lastItem && lastItem.type !== "text") return;
  const text = (lastItem?.text ?? message.content).trimEnd();
  const start = text.lastIndexOf("\n") + 1;
  const citation =
    /^(?:\*\*)?(?:来源|Sources?)[ \t]*[:：](?:\*\*)?[ \t]*(?:[\p{L}\p{N}_-]+[ \t]*·[ \t]*)?(.+)$/iu.exec(
      text.slice(start),
    )?.[1];
  if (!citation) return;

  const target =
    /^\[[^\]]+\]\(([^()\s]+)\)$/.exec(citation)?.[1] ??
    /^`([^`]+)`$/.exec(citation)?.[1] ??
    citation;
  if (!/^[\p{L}\p{N}._%/-]+\.md$/iu.test(target) || !target.includes("/"))
    return;
  const path = resolveWorkspaceLink(target, "", workspace);
  if (!path) return;

  const paragraph = fromMarkdown(text).children.at(-1);
  if (
    paragraph?.type !== "paragraph" ||
    paragraph.position?.start.offset !== start
  )
    return;

  const body = text.slice(0, start).trimEnd();
  const display = {
    ...message,
    content: body,
    items: lastItem
      ? [...items.slice(0, -1), { ...lastItem, text: body }]
      : message.items,
  };
  return { path, display };
}

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

export function isCompletedTurn(message: DisplayMessage): boolean {
  return (
    message.role === "assistant" &&
    message.source === "chat" &&
    !message.streaming &&
    (message.completed || message.status?.tone === "success") &&
    !message.error &&
    (!message.status || message.status.tone === "success") &&
    Boolean(message.content.trim() || timelineItems(message.items ?? []).length)
  );
}

/** Main history follows assistant prose; reasoning and raw output stay in disclosure. */
export default memo(function MessageBody({
  message,
  language,
  compact = false,
  showCompletion = true,
  onLink,
}: {
  message: DisplayMessage;
  language: Language;
  compact?: boolean;
  showCompletion?: boolean;
  onLink: (url: string) => void;
}) {
  const items = message.items ?? [];
  const content = compact
    ? previewContent(replyText(message))
    : message.content;
  const entries = compact ? [] : timelineItems(items);
  const components =
    !compact && message.source === "proactive" && message.role === "assistant"
      ? reminderMarkdown
      : undefined;
  const complete = !compact && isCompletedTurn(message);
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
                components={components}
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
            components={components}
            onLink={onLink}
          />
        </div>
      ) : null}
      {complete && showCompletion && (
        <div className="turn-completion">
          {translator(language)("completed")}
        </div>
      )}
      {!complete &&
        message.status &&
        message.status.tone !== "success" &&
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
