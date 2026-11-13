import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { translator } from "../shared/i18n.ts";
import type { Language } from "../shared/i18n.ts";
import type { DisplayMessage } from "../shared/protocol.ts";
import { previewContent } from "../shared/surface.ts";

/** Both surfaces render the same message; only the companion's preview is shortened. */
export default function MessageBody({
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
  const content = compact ? previewContent(message.content) : message.content;
  return (
    <div className="message-body">
      {message.role === "user" ? (
        <p>{content}</p>
      ) : content ? (
        <div className="message-content">
          <Markdown
            remarkPlugins={[remarkGfm]}
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
            {content}
          </Markdown>
        </div>
      ) : null}
      {message.status && (!content || message.status.tone !== "success") && (
        <span
          className={`message-status ${message.status.tone}`}
          role="status"
          aria-live="polite"
        >
          {message.status.text}
        </span>
      )}
      {message.error && message.error !== message.status?.text && (
        <p className="message-error">{message.error}</p>
      )}
    </div>
  );
}
