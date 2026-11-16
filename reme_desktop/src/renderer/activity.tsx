import { memo, useState } from "react";
import {
  BookOpen,
  ChevronRight,
  Folder,
  Image,
  LoaderCircle,
  Pencil,
  Search,
  Terminal,
  Wrench,
} from "lucide-react";
import { translator } from "../shared/i18n.ts";
import type { Language, TranslationKey } from "../shared/i18n.ts";
import type { SummaryItem, ToolAction, ToolItem } from "../shared/protocol.ts";
import MarkdownText from "./markdown.tsx";

export type ProcessItem = ToolItem | SummaryItem;

const kinds = {
  read: {
    icon: BookOpen,
    past: "activityFiles",
    running: "activityReadingFiles",
    detail: "actionRead",
    active: "actionReadRunning",
  },
  list: {
    icon: Folder,
    past: "activityListFiles",
    running: "activityListingFiles",
    detail: "actionList",
    active: "actionListRunning",
  },
  search: {
    icon: Search,
    past: "activitySearch",
    running: "activitySearching",
    detail: "actionSearch",
    active: "actionSearchRunning",
  },
  command: {
    icon: Terminal,
    past: "activityCommands",
    running: "activityRunningCommands",
    detail: "actionCommand",
    active: "actionCommandRunning",
  },
  edit: {
    icon: Pencil,
    past: "activityWriteFiles",
    running: "activityWritingFiles",
    detail: "actionEdit",
    active: "actionEditRunning",
  },
  webSearch: {
    icon: Search,
    past: "activityWebSearch",
    running: "activitySearchingWeb",
    detail: "actionWebSearch",
    active: "actionWebSearchRunning",
  },
  browse: {
    icon: BookOpen,
    past: "activityPages",
    running: "activityReadingPages",
    detail: "actionBrowse",
    active: "actionBrowseRunning",
  },
  image: {
    icon: Image,
    past: "activityImages",
    running: "activityViewingImages",
    detail: "actionImage",
    active: "actionImageRunning",
  },
  tool: {
    icon: Wrench,
    past: "activityTools",
    running: "activityCallingTools",
    detail: "actionTool",
    active: "actionToolRunning",
  },
} as const satisfies Record<
  ToolAction["kind"],
  {
    icon: unknown;
    past: TranslationKey;
    running: TranslationKey;
    detail: TranslationKey;
    active: TranslationKey;
  }
>;

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

function ActionLabel({
  action,
  language,
  status,
  onLink,
}: {
  action: ToolAction;
  language: Language;
  status: ToolItem["status"];
  onLink: (url: string) => void;
}) {
  const t = translator(language);
  const { kind, target, query } = action;
  const style = kinds[kind];
  const displayTarget =
    target && ["read", "edit", "image"].includes(kind)
      ? target.split(/[\\/]/).at(-1) || target
      : target;
  const link = target && ["read", "edit", "image", "browse"].includes(kind);
  if (kind === "search" && target && query) {
    const [before, after] = t(
      status === "running" ? "actionSearchInRunning" : "actionSearchIn",
      { query },
    ).split("{target}");
    return (
      <span className="action-label">
        {status === "completed" && t("actionCompletedPrefix")}
        {before}
        <span title={target}>{displayTarget}</span>
        {after}
      </span>
    );
  }
  return (
    <span className="action-label">
      {status === "completed" ? t("actionCompletedPrefix") : ""}
      {t(status === "running" ? style.active : style.detail)}
      {target && (
        <>
          {" "}
          {link ? (
            <button
              className="action-target"
              title={target}
              onClick={() => onLink(target)}
            >
              {displayTarget}
            </button>
          ) : (
            <span className="action-target" title={target}>
              {displayTarget}
            </span>
          )}
        </>
      )}
      {kind === "list" && target && t("actionListSuffix")}
      {query && <> {t("actionQuery", { query })}</>}
    </span>
  );
}

const ExecutionDetails = memo(function ExecutionDetails({
  items,
  language,
  onLink,
}: {
  items: ProcessItem[];
  language: Language;
  onLink: (url: string) => void;
}) {
  const t = translator(language);
  return (
    <div className="activity-details">
      {items
        .filter((item): item is ToolItem => item.type === "tool")
        .map((tool) => (
          <ToolDetail
            key={tool.id}
            tool={tool}
            language={language}
            onLink={onLink}
          />
        ))}
      {items.some((item) => item.type === "summary") && (
        <details className="reasoning-details">
          <summary>
            {t("activityReasoning")}
            <ChevronRight size={13} className="activity-chevron" />
          </summary>
          {items
            .filter((item): item is SummaryItem => item.type === "summary")
            .map((item) => (
              <MarkdownText
                key={item.id}
                text={item.text}
                completed={item.status !== "running"}
                onLink={onLink}
              />
            ))}
        </details>
      )}
    </div>
  );
});

const ToolDetail = memo(function ToolDetail({
  tool,
  language,
  onLink,
}: {
  tool: ToolItem;
  language: Language;
  onLink: (url: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const t = translator(language);
  return (
    <details
      className="tool-detail"
      data-status={tool.status}
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary>
        {tool.actions.map((action, index) => {
          const Icon = kinds[action.kind].icon;
          return (
            <span className="tool-action" key={index}>
              <Icon size={15} />
              <ActionLabel
                action={action}
                language={language}
                status={tool.status}
                onLink={onLink}
              />
            </span>
          );
        })}
      </summary>
      {expanded && (
        <div>
          <header>
            <code>{tool.name}</code>
            <span>{t(tool.status)}</span>
          </header>
          {tool.input && <ToolOutput text={tool.input} language={language} />}
          {tool.output && <ToolOutput text={tool.output} language={language} />}
        </div>
      )}
      {["failed", "interrupted", "waiting", "unknown"].includes(
        tool.status,
      ) && <span className="tool-outcome">{t(tool.status)}</span>}
    </details>
  );
});

export const Activity = memo(function Activity({
  items,
  language,
  onLink,
  streaming,
}: {
  items: ProcessItem[];
  language: Language;
  onLink: (url: string) => void;
  streaming: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const t = translator(language);
  const tools = items.filter((item): item is ToolItem => item.type === "tool");
  const running =
    tools.some((tool) => tool.status === "running") ||
    (streaming && items.at(-1)?.type === "summary");
  const groups = new Map<ToolAction["kind"], boolean>();
  for (const tool of tools)
    for (const action of tool.actions) {
      const kind =
        action.kind === "list" || action.kind === "search"
          ? action.target?.startsWith("http")
            ? "browse"
            : "read"
          : action.kind;
      groups.set(kind, Boolean(groups.get(kind)) || tool.status === "running");
    }
  if (!tools.length && !running) return null;
  const Icon = running
    ? LoaderCircle
    : kinds[groups.keys().next().value ?? "tool"].icon;
  const issue = (["failed", "interrupted", "waiting", "unknown"] as const).find(
    (status) => tools.some((tool) => tool.status === status),
  );
  const failed = issue === "failed" || issue === "interrupted";
  const label = groups.size
    ? [...groups]
        .map(([kind, running]) => t(kinds[kind][running ? "running" : "past"]))
        .join(t("activitySeparator"))
    : t("thinking");
  return (
    <details
      className={`activity-group ${failed ? "activity-failed" : ""}`}
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary>
        <Icon size={15} className={running ? "activity-spinner" : ""} />
        <span>{label}</span>
        {issue && <span className="activity-outcome">{t(issue)}</span>}
        <ChevronRight size={13} className="activity-chevron" />
      </summary>
      {expanded && (
        <ExecutionDetails items={items} language={language} onLink={onLink} />
      )}
    </details>
  );
});

export function Completion({
  items,
  language,
  onLink,
}: {
  items: ProcessItem[];
  language: Language;
  onLink: (url: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const t = translator(language);
  return items.length ? (
    <details
      className="turn-completion"
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary>
        {t("completed")}
        <ChevronRight size={13} className="activity-chevron" />
      </summary>
      {expanded && (
        <ExecutionDetails items={items} language={language} onLink={onLink} />
      )}
    </details>
  ) : (
    <div className="turn-completion">{t("completed")}</div>
  );
}
