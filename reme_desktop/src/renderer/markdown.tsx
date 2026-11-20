import { memo, useMemo } from "react";
import { Streamdown, type Components } from "streamdown";
import remarkGfm from "remark-gfm";
import { remarkWikilinks } from "./knowledge/wikilinks.ts";

const plugins = [remarkGfm, remarkWikilinks];
const rehypePlugins: [] = [];
const linkSafety = { enabled: false };
const remend = { linkMode: "text-only" as const };

/** Streamdown owns block boundaries and incomplete syntax; workspace links keep their existing routing. */
export default memo(function MarkdownText({
  text,
  completed = true,
  onLink,
  components: overrides,
}: {
  text: string;
  completed?: boolean;
  onLink: (url: string) => void;
  components?: Components;
}) {
  const components = useMemo<Components>(
    () => ({
      p: "p",
      strong: "strong",
      em: "em",
      h1: "h1",
      h2: "h2",
      h3: "h3",
      h4: "h4",
      h5: "h5",
      h6: "h6",
      ul: "ul",
      ol: "ol",
      li: "li",
      blockquote: "blockquote",
      hr: "hr",
      del: "del",
      thead: "thead",
      tbody: "tbody",
      tr: "tr",
      th: "th",
      td: "td",
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
      pre: "pre",
      code: "code",
      table: "table",
      ...overrides,
    }),
    [onLink, overrides],
  );
  return (
    <Streamdown
      className="streaming-markdown"
      components={components}
      remarkPlugins={plugins}
      rehypePlugins={rehypePlugins}
      linkSafety={linkSafety}
      remend={remend}
      controls={false}
      parseIncompleteMarkdown={!completed}
      isAnimating={!completed}
    >
      {text}
    </Streamdown>
  );
});
