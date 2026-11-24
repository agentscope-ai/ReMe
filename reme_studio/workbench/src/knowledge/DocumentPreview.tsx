import { useMemo } from "react";
import MarkdownText from "../markdown.tsx";
import { ChevronRight, FileText, ListFilter } from "lucide-react";
import { stringify } from "yaml";
import { translator, type Language } from "../../../client/src/i18n.ts";
import { parseMarkdown } from "./model.ts";

export default function DocumentPreview({
  path,
  content,
  mtime,
  tagKey,
  onTag,
  onLink,
  language,
}: {
  path: string;
  content: string;
  mtime?: string;
  tagKey?: string;
  onTag?: (tag: string) => void;
  onLink: (href: string) => void;
  language: Language;
}) {
  const t = translator(language);
  const markdown = /\.mdx?$/i.test(path);
  const filename = path.split("/").at(-1) || path;
  const extension = /.+\.([^.]+)$/.exec(filename)?.[1];
  const title =
    markdown && extension ? filename.slice(0, -extension.length - 1) : filename;
  const modified = !markdown && mtime ? new Date(mtime) : null;
  const parsed = useMemo(
    () => parseMarkdown(markdown ? content : ""),
    [content, markdown],
  );
  const entries = Object.entries(parsed.metadata);
  return (
    <article className="document-content">
      <header className="document-heading">
        <h1 className="document-title" data-long={title.length > 80}>
          {title}
        </h1>
        {modified && !Number.isNaN(modified.getTime()) && (
          <time
            className="document-modified"
            dateTime={mtime}
            title={t("modifiedTime")}
          >
            {modified.toLocaleString(language === "zh" ? "zh-CN" : "en", {
              year: "numeric",
              month: "short",
              day: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            })}
          </time>
        )}
      </header>
      {(entries.length > 0 || parsed.frontmatterError) && (
        <details className="document-properties" open>
          <summary>
            <ChevronRight size={13} />
            <ListFilter size={13} />
            <span>{t("documentProperties")}</span>
            <small className={entries.length ? "count-badge" : undefined}>
              {entries.length || "YAML"}
            </small>
          </summary>
          {parsed.frontmatterError ? (
            <div className="frontmatter-error" role="status">
              <p>{t("invalidFrontmatter")}</p>
              <pre>{parsed.rawFrontmatter}</pre>
            </div>
          ) : (
            <dl>
              {entries.map(([key, value]) => {
                const linkedTags =
                  onTag &&
                  key === tagKey &&
                  Array.isArray(value) &&
                  value.length > 0 &&
                  value.every(
                    (tag) => typeof tag === "string" || typeof tag === "number",
                  )
                    ? value
                    : null;
                return (
                  <div className="frontmatter-row" key={key}>
                    <dt>{key}</dt>
                    <dd>
                      {onTag && linkedTags ? (
                        <div className="frontmatter-tags">
                          {linkedTags.map((tag, index) => (
                            <button
                              type="button"
                              key={index}
                              className="document-tag"
                              onClick={() => onTag(String(tag))}
                            >
                              {String(tag)}
                            </button>
                          ))}
                        </div>
                      ) : value !== null && typeof value === "object" ? (
                        <pre>
                          {stringify(value, { lineWidth: 0 }).trimEnd()}
                        </pre>
                      ) : (
                        <span
                          className={
                            key === "status" ? "property-status" : undefined
                          }
                        >
                          {String(value)}
                        </span>
                      )}
                    </dd>
                  </div>
                );
              })}
            </dl>
          )}
        </details>
      )}
      {markdown ? (
        <div className="document-markdown">
          <MarkdownText
            text={parsed.body}
            onLink={onLink}
            components={{
              img: ({ alt }) => (
                <span className="image-placeholder">{alt || t("image")}</span>
              ),
            }}
          />
        </div>
      ) : (
        <div className="document-source">
          <div className="document-source-format">
            <FileText size={13} aria-hidden />
            <span>{extension?.toUpperCase() ?? t("plainTextFile")}</span>
          </div>
          <pre className="document-text">
            <code>{content}</code>
          </pre>
        </div>
      )}
    </article>
  );
}
