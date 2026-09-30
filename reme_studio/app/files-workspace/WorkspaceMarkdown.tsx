import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { openWorkspaceFile } from "../open-file";
import { remarkWikilinks, workspaceLinkTarget } from "./wikilinks";

export default function WorkspaceMarkdown({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkWikilinks]}
      components={{
        a: ({ href, title, children }) => (
          <a
            href={href}
            title={title}
            onClick={(event) => {
              const path = workspaceLinkTarget(href);
              if (!path) return;
              event.preventDefault();
              void openWorkspaceFile(path);
            }}
          >
            {children}
          </a>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
