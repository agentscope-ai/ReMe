import { useEffect, useMemo, useState, type ReactNode, type Ref } from "react";
import MarkdownText from "../markdown.tsx";
import {
  Check,
  ChevronRight,
  Copy,
  FileText,
  ListFilter,
  Pencil,
} from "lucide-react";
import { stringify } from "yaml";
import { translator, type Language } from "../../../client/src/i18n.ts";
import { parseMarkdown } from "./model.ts";
import { DetailHeading } from "../timeline.tsx";

export function DocumentHeading({
  path,
  mtime,
  language,
  children,
}: {
  path: string;
  mtime?: string;
  language: Language;
  children?: ReactNode;
}) {
  const t = translator(language);
  const filename = path.split("/").at(-1) || path;
  const title = filename.replace(/\.mdx?$/i, "") || filename;
  const modified = !/\.mdx?$/i.test(path) && mtime ? new Date(mtime) : null;
  return (
    <DetailHeading
      title={title}
      long={title.length > 80}
      actions={children}
      metadata={
        modified &&
        !Number.isNaN(modified.getTime()) && (
          <time dateTime={mtime} title={t("modifiedTime")}>
            {modified.toLocaleString(language === "zh" ? "zh-CN" : "en", {
              year: "numeric",
              month: "short",
              day: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            })}
          </time>
        )
      }
    />
  );
}

export default function DocumentPreview({
  path,
  content,
  mtime,
  tagKey,
  onTag,
  onLink,
  onCopy,
  onEdit,
  editRef,
  language,
}: {
  path: string;
  content: string;
  mtime?: string;
  tagKey?: string;
  onTag?: (tag: string) => void;
  onLink: (href: string) => void;
  onCopy?: (content: string) => Promise<void>;
  onEdit?: () => void;
  editRef?: Ref<HTMLButtonElement>;
  language: Language;
}) {
  const t = translator(language);
  const [copyStatus, setCopyStatus] = useState<"copied" | "failed">();
  useEffect(() => {
    if (!copyStatus) return;
    const timer = window.setTimeout(() => setCopyStatus(undefined), 2400);
    return () => window.clearTimeout(timer);
  }, [copyStatus]);
  const markdown = /\.mdx?$/i.test(path);
  const filename = path.split("/").at(-1) || path;
  const extension = /.+\.([^.]+)$/.exec(filename)?.[1];
  const parsed = useMemo(
    () => parseMarkdown(markdown ? content : ""),
    [content, markdown],
  );
  const entries = Object.entries(parsed.metadata);
  return (
    <article className="document-content">
      <DocumentHeading path={path} mtime={mtime} language={language}>
        {(onCopy || onEdit) && (
          <>
            {onCopy && (
              <button
                type="button"
                className="subtle"
                title={t("copyContent")}
                onClick={() =>
                  void onCopy(content).then(
                    () => setCopyStatus("copied"),
                    () => setCopyStatus("failed"),
                  )
                }
              >
                {copyStatus === "copied" ? (
                  <Check size={14} aria-hidden />
                ) : (
                  <Copy size={14} aria-hidden />
                )}
                <span aria-live="polite">
                  {t(
                    copyStatus === "copied" ? "documentCopied" : "documentCopy",
                  )}
                </span>
              </button>
            )}
            {onEdit && (
              <button
                ref={editRef}
                type="button"
                className="secondary"
                onClick={onEdit}
              >
                <Pencil size={14} aria-hidden />
                {t("documentEdit")}
              </button>
            )}
          </>
        )}
      </DocumentHeading>
      {copyStatus === "failed" && (
        <p className="document-action-error" role="alert">
          {t("documentCopyFailed")}
        </p>
      )}
      {(entries.length > 0 || parsed.frontmatterError) && (
        <details
          className="document-properties"
          open={parsed.frontmatterError || undefined}
        >
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
