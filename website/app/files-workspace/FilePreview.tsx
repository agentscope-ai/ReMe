import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { parseMarkdownFrontmatter } from "./markdown";
import styles from "./files-workspace.module.css";

export type PreviewType = "markdown" | "csv" | "text";

export function getPreviewType(filePath: string): PreviewType {
  const ext = filePath.split(".").pop()?.toLowerCase() ?? "";
  if (ext === "md" || ext === "mdx") return "markdown";
  if (ext === "csv") return "csv";
  return "text";
}

function parseCsv(raw: string): string[][] {
  return raw
    .trimEnd()
    .split(/\r?\n/)
    .map((line) => {
      const cells: string[] = [];
      let current = "";
      let quoted = false;
      for (let index = 0; index < line.length; index += 1) {
        const char = line[index];
        if (char === '"') {
          if (quoted && line[index + 1] === '"') {
            current += '"';
            index += 1;
          } else quoted = !quoted;
        } else if (char === "," && !quoted) {
          cells.push(current);
          current = "";
        } else current += char;
      }
      cells.push(current);
      return cells;
    });
}

function MarkdownPreview({ content }: { content: string }) {
  const { body, entries } = useMemo(
    () => parseMarkdownFrontmatter(content),
    [content],
  );
  return (
    <article className={styles.markdownWrap}>
      {entries.length > 0 && (
        <dl className={styles.frontmatter} aria-label="Front matter">
          {entries.map(({ key, value }, index) => (
            <div className={styles.frontmatterRow} key={`${key}:${index}`}>
              <dt>{key}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      )}
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{body}</ReactMarkdown>
    </article>
  );
}

function CsvPreview({ content }: { content: string }) {
  const rows = useMemo(() => parseCsv(content), [content]);
  const header = rows[0] ?? [];
  return (
    <div className={styles.csvScroll}>
      <table className={styles.csvTable}>
        <thead>
          <tr>
            {header.slice(0, 50).map((cell, index) => (
              <th key={`${index}:${cell}`}>{cell}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(1, 501).map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.slice(0, 50).map((cell, cellIndex) => (
                <td key={cellIndex}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function FilePreview({
  filePath,
  content,
}: {
  filePath: string;
  content: string;
}) {
  const type = getPreviewType(filePath);
  if (type === "markdown") return <MarkdownPreview content={content} />;
  if (type === "csv") return <CsvPreview content={content} />;
  return <pre className={styles.textPreview}>{content}</pre>;
}
