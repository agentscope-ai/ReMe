import type { CompileContext } from "mdast-util-from-markdown";
import type {
  Effects,
  State,
  Token,
  TokenizeContext,
} from "micromark-util-types";
import type { Processor } from "unified";
import type {} from "remark-parse";

declare module "mdast" {
  interface PhrasingContentMap {
    remeWikilink: { type: "remeWikilink"; value: string };
  }
  interface RootContentMap {
    remeWikilink: PhrasingContentMap["remeWikilink"];
  }
}

declare module "micromark-util-types" {
  interface TokenTypeMap {
    remeWikilink: "remeWikilink";
  }
}

const WIKILINK_PATTERN =
  /^\[\[([^[\]|#\n]+?)(?:#[^[\]|\n]+)?(?:\|([^[\]\n]+))?\]\]$/;

/** Keep Markdown punctuation inside a wikilink literal during inline parsing. */
function tokenizeWikilink(
  this: TokenizeContext,
  effects: Effects,
  ok: State,
  nok: State,
): State {
  const serialize = this.sliceSerialize.bind(this);
  let token: Token;
  return start;

  function start(code: number | null) {
    if (code !== 91) return nok(code);
    token = effects.enter("remeWikilink");
    effects.consume(code);
    return secondBracket;
  }

  function secondBracket(code: number | null) {
    if (code !== 91) return nok(code);
    effects.consume(code);
    return inside;
  }

  function inside(code: number | null): State | undefined {
    // Micromark uses negative codes for line endings, tabs, and virtual spaces.
    if (code === null || code <= -3 || code === 91) return nok(code);
    effects.consume(code);
    return code === 93 ? closingBracket : inside;
  }

  function closingBracket(code: number | null) {
    if (code !== 93) return nok(code);
    effects.consume(code);
    effects.exit("remeWikilink");
    const match = WIKILINK_PATTERN.exec(serialize(token));
    return match?.[1].trim() ? ok : nok;
  }
}

function enterWikilink(this: CompileContext, token: Token) {
  // A dedicated node also keeps GFM's autolink transform out of literal aliases.
  this.enter(
    { type: "remeWikilink", value: this.sliceSerialize(token) },
    token,
  );
}

function exitWikilink(this: CompileContext, token: Token) {
  this.exit(token);
}

interface MarkdownNode {
  type: string;
  value?: string;
  url?: string;
  children?: MarkdownNode[];
}

/** Ignore malformed fragments authored directly in Markdown. */
export function workspaceLinkTarget(href?: string): string | undefined {
  const prefix = "#reme-file=";
  if (!href?.startsWith(prefix)) return;
  try {
    return decodeURIComponent(href.slice(prefix.length)) || undefined;
  } catch {
    return;
  }
}

/** Resolve literal workspace wikilinks without rewriting code or existing links. */
export function remarkWikilinks(this: Processor) {
  const data = this.data();
  (data.micromarkExtensions ||= []).push({
    text: { 91: { name: "remeWikilink", tokenize: tokenizeWikilink } },
  });
  (data.fromMarkdownExtensions ||= []).push({
    enter: { remeWikilink: enterWikilink },
    exit: { remeWikilink: exitWikilink },
  });
  return (tree: MarkdownNode) => {
    const literalize = (node: MarkdownNode) => {
      if (node.type === "remeWikilink") node.type = "text";
      node.children?.forEach(literalize);
    };
    const visit = (node: MarkdownNode) => {
      if (
        !node.children ||
        ["link", "image", "code", "inlineCode"].includes(node.type)
      ) {
        literalize(node);
        return;
      }
      node.children = node.children.flatMap((child) => {
        if (
          (child.type !== "text" && child.type !== "remeWikilink") ||
          !child.value
        ) {
          visit(child);
          return [child];
        }
        const parts: MarkdownNode[] = [];
        let end = 0;
        for (const match of child.value.matchAll(
          /\[\[([^[\]|#\n]+?)(?:#[^[\]|\n]+)?(?:\|([^[\]\n]+))?\]\]/g,
        )) {
          const target = match[1].trim();
          if (!target) continue;
          parts.push({
            type: "text",
            value: child.value.slice(end, match.index),
          });
          parts.push({
            type: "link",
            url: `#reme-file=${encodeURIComponent(target)}`,
            children: [{ type: "text", value: match[2]?.trim() || target }],
          });
          end = match.index + match[0].length;
        }
        if (!end) return [child];
        parts.push({ type: "text", value: child.value.slice(end) });
        return parts;
      });
    };
    visit(tree);
  };
}
