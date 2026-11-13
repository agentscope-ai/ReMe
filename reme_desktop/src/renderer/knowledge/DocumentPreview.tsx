import { useMemo } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChevronRight, ListFilter } from "lucide-react";
import { stringify } from "yaml";
import { translator, type Language } from "../../shared/i18n.ts";
import { documentPresentation, parseMarkdown } from "./model.ts";
import { remarkWikilinks } from "./wikilinks.ts";

export default function DocumentPreview({
  path,
  content,
  mtime,
  tagKey,
  supportsTags,
  onTag,
  onLink,
  language,
}: {
  path: string;
  content: string;
  mtime?: string;
  tagKey: string;
  supportsTags: boolean;
  onTag: (tag: string) => void;
  onLink: (href: string) => void;
  language: Language;
}) {
  const t = translator(language);
  const markdown = /\.mdx?$/i.test(path);
  const parsed = useMemo(
    () => parseMarkdown(markdown ? content : "", tagKey),
    [content, tagKey, markdown],
  );
  const presentation = useMemo(
    () => documentPresentation(path, parsed),
    [path, parsed],
  );
  const entries = Object.entries(parsed.metadata);
  const created =
    typeof parsed.metadata.created_at === "string"
      ? parsed.metadata.created_at
      : undefined;
  const timestamp = created ?? mtime;
  const date = timestamp ? new Date(timestamp) : null;
  const tag = (value: string) => (
    <button
      key={value}
      className="document-tag"
      disabled={!supportsTags}
      onClick={() => onTag(value)}
    >
      # {value}
    </button>
  );
  return (
    <article className="document-content">
      <h1 className="document-title" data-long={presentation.title.length > 80}>
        {presentation.title}
      </h1>
      <div className="document-meta">
        {date && !Number.isNaN(date.getTime()) && (
          <time
            dateTime={timestamp}
            title={t(created ? "createdTime" : "modifiedTime")}
          >
            {date.toLocaleString(language === "zh" ? "zh-CN" : "en", {
              year: "numeric",
              month: "short",
              day: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            })}
          </time>
        )}
        {parsed.tags.map(tag)}
      </div>
      {typeof parsed.metadata.description === "string" && (
        <p className="document-description">{parsed.metadata.description}</p>
      )}
      {(entries.length > 0 || parsed.frontmatterError) && (
        <details className="document-properties" open>
          <summary>
            <ChevronRight size={13} />
            <ListFilter size={13} />
            <span>{t("documentProperties")}</span>
            <small>{entries.length || "YAML"}</small>
          </summary>
          {parsed.frontmatterError ? (
            <div className="frontmatter-error" role="status">
              <p>{t("invalidFrontmatter")}</p>
              <pre>{parsed.rawFrontmatter}</pre>
            </div>
          ) : (
            <dl>
              {entries.map(([key, value]) => (
                <div className="frontmatter-row" key={key}>
                  <dt>{key}</dt>
                  <dd>
                    {value !== null && typeof value === "object" ? (
                      <pre>{stringify(value, { lineWidth: 0 }).trimEnd()}</pre>
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
              ))}
            </dl>
          )}
        </details>
      )}
      {markdown ? (
        <div className="document-markdown">
          <Markdown
            remarkPlugins={[remarkGfm, remarkWikilinks]}
            components={{
              a: ({ href, children }) => (
                <a
                  href={href}
                  onClick={(event) => {
                    event.preventDefault();
                    if (href) onLink(href);
                  }}
                >
                  {children}
                </a>
              ),
              img: ({ alt }) => (
                <span className="image-placeholder">{alt || t("image")}</span>
              ),
            }}
          >
            {presentation.body}
          </Markdown>
        </div>
      ) : (
        <pre className="document-text">{content}</pre>
      )}
    </article>
  );
}
