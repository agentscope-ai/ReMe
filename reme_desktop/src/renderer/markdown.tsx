import { memo, useMemo, useRef } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { fromMarkdown } from "mdast-util-from-markdown";
import { remarkWikilinks } from "./knowledge/wikilinks.ts";

interface Segment {
  offset: number;
  text: string;
}

/** Freeze complete top-level blocks; reference definitions invalidate the whole document. */
export class MarkdownBuffer {
  private source = "";
  private stable: Segment[] = [];
  private boundary = 0;
  private references = false;
  private finalized = false;

  update(source: string, completed: boolean): Segment[] {
    if (
      !source.startsWith(this.source) ||
      (this.finalized && source !== this.source)
    ) {
      this.stable = [];
      this.boundary = 0;
      this.references = false;
    }
    this.source = source;
    this.finalized = completed;
    const tail = source.slice(this.boundary);
    const nodes = fromMarkdown(tail).children.filter(
      (node) => "position" in node && node.position,
    );
    this.references ||= nodes.some((node) => node.type === "definition");
    if (this.references) return [{ offset: 0, text: source }];
    const count = completed ? nodes.length : Math.max(0, nodes.length - 1);
    const start = this.boundary;
    for (const node of nodes.slice(0, count)) {
      const end = node.position!.end.offset!;
      this.stable.push({
        offset: start + node.position!.start.offset!,
        text: tail.slice(node.position!.start.offset!, end),
      });
      this.boundary = start + end;
    }
    const last = nodes[count];
    return last
      ? [
          ...this.stable,
          {
            offset: start + last.position!.start.offset!,
            text: tail.slice(last.position!.start.offset!),
          },
        ]
      : this.stable;
  }
}

const plugins = [remarkGfm, remarkWikilinks];
const MarkdownBlock = memo(function MarkdownBlock({
  text,
  onLink,
}: {
  text: string;
  onLink: (url: string) => void;
}) {
  const components = useMemo(
    () => ({
      a: ({ href, children }: React.ComponentProps<"a">) => (
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
      img: ({ alt, src }: React.ComponentProps<"img">) => (
        <button
          type="button"
          className="attachment-link"
          onClick={() => {
            if (typeof src === "string") onLink(src);
          }}
        >
          {alt || "Image"}
        </button>
      ),
    }),
    [onLink],
  );
  return (
    <Markdown remarkPlugins={plugins} components={components}>
      {text}
    </Markdown>
  );
});

export default memo(function MarkdownText({
  text,
  completed = true,
  onLink,
}: {
  text: string;
  completed?: boolean;
  onLink: (url: string) => void;
}) {
  const buffer = useRef<MarkdownBuffer | null>(null);
  buffer.current ??= new MarkdownBuffer();
  const segments = buffer.current.update(text, completed);
  return (
    <>
      {segments.map((segment) => (
        <MarkdownBlock
          key={segment.offset}
          text={segment.text}
          onLink={onLink}
        />
      ))}
    </>
  );
});
