import { memo, useState } from "react";
import {
  ChevronRight,
  LoaderCircle,
  Terminal,
  FileText,
  Search,
  Wrench,
  CircleAlert,
  Paperclip,
} from "lucide-react";
import { translator } from "../shared/i18n.ts";
import type { Language } from "../shared/i18n.ts";
import type {
  DisplayMessage,
  TranscriptItem,
  ToolItem,
} from "../shared/protocol.ts";
import { previewContent, replyText } from "../shared/surface.ts";
import MarkdownText from "./markdown.tsx";

function toolKind(name: string) {
  const words = name
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/[_./:]+/g, " ")
    .toLowerCase();
  if (/\b(write|edit|patch)\b|file change/.test(words))
    return {
      icon: FileText,
      label: "activityWriteFiles" as const,
      running: "activityWritingFiles" as const,
    };
  if (/web extract|fetch|browse|open page/.test(words))
    return {
      icon: Search,
      label: "activityPages" as const,
      running: "activityReadingPages" as const,
    };
  if (/web search|search web/.test(words))
    return {
      icon: Search,
      label: "activityWebSearch" as const,
      running: "activitySearchingWeb" as const,
    };
  if (/\b(search|find|grep)\b/.test(words))
    return {
      icon: Search,
      label: "activitySearch" as const,
      running: "activitySearching" as const,
    };
  if (
    /\b(read|load|list)(?:$| (?:files?|directories|directory|dir|paths?)\b)/.test(
      words,
    )
  )
    return {
      icon: FileText,
      label: "activityFiles" as const,
      running: "activityReadingFiles" as const,
    };
  if (/\b(command|exec|bash|shell|terminal)\b/.test(words))
    return {
      icon: Terminal,
      label: "activityCommands" as const,
      running: "activityRunningCommands" as const,
    };
  return {
    icon: Wrench,
    label: "activityTools" as const,
    running: "activityCallingTools" as const,
  };
}

type Entry =
  | Exclude<TranscriptItem, ToolItem>
  | { type: "activity"; id: string; tools: ToolItem[] };
export function timelineItems(items: TranscriptItem[]): Entry[] {
  const entries: Entry[] = [];
  for (const item of items) {
    const previous = entries.at(-1);
    if (item.type === "tool") {
      if (previous?.type === "activity") previous.tools.push(item);
      else entries.push({ type: "activity", id: item.id, tools: [item] });
    } else entries.push(item);
  }
  return entries;
}

const ToolOutput = memo(function ToolOutput({
  text,
  language,
}: {
  text: string;
  language: Language;
}) {
  const [full, setFull] = useState(false);
  const truncated = !full && text.length > 24000;
  return (
    <>
      <pre>
        {truncated ? `${text.slice(0, 8000)}\n…\n${text.slice(-16000)}` : text}
      </pre>
      {truncated && (
        <button className="tool-show-output" onClick={() => setFull(true)}>
          {translator(language)("showFullOutput")}
        </button>
      )}
    </>
  );
});

const Activity = memo(function Activity({
  tools,
  language,
}: {
  tools: ToolItem[];
  language: Language;
}) {
  const t = translator(language);
  const [expanded, setExpanded] = useState(false);
  const running = tools.some((tool) => tool.status === "running");
  const failed = tools.some(
    (tool) => tool.status === "failed" || tool.status === "interrupted",
  );
  const status = running
    ? "running"
    : tools.some((tool) => tool.status === "failed")
      ? "failed"
      : tools.some((tool) => tool.status === "interrupted")
        ? "interrupted"
        : tools.some((tool) => tool.status === "waiting")
          ? "waiting"
          : tools.some((tool) => tool.status === "unknown")
            ? "unknown"
            : "completed";
  const groups = new Map<
    string,
    { kind: ReturnType<typeof toolKind>; running: boolean }
  >();
  for (const tool of tools) {
    const kind = toolKind(tool.name);
    groups.set(kind.label, {
      kind,
      running: groups.get(kind.label)?.running || tool.status === "running",
    });
  }
  const summary =
    status === "completed" || status === "running"
      ? [...groups.values()]
          .map(({ kind, running }) => t(running ? kind.running : kind.label))
          .join(t("activitySeparator"))
      : t(status);
  const Icon = running
    ? LoaderCircle
    : failed
      ? CircleAlert
      : toolKind(tools[0].name).icon;
  return (
    <details
      className={`activity-group ${failed ? "activity-failed" : ""}`}
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary>
        <Icon size={14} className={running ? "activity-spinner" : ""} />
        <span>{summary}</span>
        <ChevronRight size={13} className="activity-chevron" />
      </summary>
      {expanded && (
        <div className="activity-details">
          {tools.map((tool) => (
            <section
              key={tool.id}
              className="tool-detail"
              data-status={tool.status}
            >
              <header>
                <code>{tool.name}</code>
                <span>{t(tool.status)}</span>
              </header>
              {tool.input && (
                <ToolOutput text={tool.input} language={language} />
              )}
              {tool.output && (
                <ToolOutput text={tool.output} language={language} />
              )}
            </section>
          ))}
        </div>
      )}
    </details>
  );
});

/** Both surfaces share the transcript; the companion receives a text-only preview. */
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
  const t = translator(language);
  const items = message.items ?? [];
  const content = compact
    ? previewContent(replyText(message))
    : message.content;
  const entries = compact ? [] : timelineItems(items);
  const hasContent = Boolean(content || entries.length);
  return (
    <div className="message-body">
      {entries.length ? (
        entries.map((item) => {
          if (item.type === "activity")
            return (
              <Activity key={item.id} tools={item.tools} language={language} />
            );
          if (item.type === "summary")
            return item.text ? (
              <details className="reasoning-summary" key={item.id}>
                <summary>{t("activityReasoning")}</summary>
                <MarkdownText
                  text={item.text}
                  completed={item.status !== "running"}
                  onLink={onLink}
                />
              </details>
            ) : null;
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
              className="message-content"
              key={item.id}
              data-phase={item.phase}
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
      {message.status && (!hasContent || message.status.tone !== "success") && (
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
