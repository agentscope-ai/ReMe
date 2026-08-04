"""Local-link handler for Wikilinks and inline Markdown links.

One class, :class:`WikilinkHandler`, owning every local-link concern:

* **Pure text** — regex, extraction, rewrite, validation:
  :meth:`~WikilinkHandler.extract_links` (used by
  :mod:`reme.components.file_chunker.markdown_file_chunker`),
  :meth:`~WikilinkHandler.scan_and_rewrite`,
  :meth:`~WikilinkHandler.validate_src_dst` /
  :meth:`~WikilinkHandler.validate_scope` /
  :meth:`~WikilinkHandler.within_scope`.
* **Async, file_graph-aware** —
  :meth:`~WikilinkHandler.find_inbound` (called by ``file_delete``
  to surface references the caller might want to clean up) and
  :meth:`~WikilinkHandler.retarget_links` (called by ``file_move``
  post-rename to point inbound links at the new path). Source
  candidates come from the file_graph's reverse index — no fs scan.

Wikilink targets are taken **literally** — ``[[X]]`` →
``target="X"``, no implicit ``.md``, no short-form basename search,
no folder-note expansion. Anchor and alias survive a rewrite
verbatim. Text outside ``[[...]]`` is ignored. Recommended form: full
path relative to the workspace with extension (``[[topics/x.md]]``).
Markdown destinations use document-relative paths and are normalized to
workspace-relative graph targets. External links, images, and code examples
are ignored.

Stale graph entries are harmless (``scan_and_rewrite`` returns
count=0 and the file is skipped), but a graph missing recent writes
will miss those sources — keep the watcher in sync.
"""

import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from ..enumeration import LinkScopeEnum
from ..schema import FileLink


@dataclass(frozen=True)
class WikilinkMatch:
    """The graph-relevant parts and source spans of one local link."""

    target: str
    anchor: str | None
    start: int
    end: int
    target_start: int
    target_end: int
    syntax: str


class WikilinkHandler:
    """Parse, extract, rewrite, and validate local Markdown links."""

    # Captures: optional image marker (``!``), the bare target, an
    # optional ``#anchor`` slice (with ``#``), and an optional ``|alias``
    # slice (with ``|``). The anchor / alias inner classes exclude ``[``
    # defensively so a runaway match on malformed input can't swallow
    # following links.
    WIKILINK_RE = re.compile(
        r"""
        (?P<bang>!?)
        \[\[
            (?P<target>[^\[\]\|\#\n]+?)
            (?P<anchor>\#[^\[\]\|\n]+)?
            (?P<alias>\|[^\[\]\n]+)?
        \]\]
        """,
        re.VERBOSE,
    )

    FORBIDDEN_IN_NEW = ("[", "]", "#", "|", "\n", "\r")

    # -- Low-level scan ------------------------------------------------

    @classmethod
    def _iter_wikilinks(cls, text: str):
        for m in cls.WIKILINK_RE.finditer(text):
            target = m.group("target").strip()
            if not target:
                continue
            anchor_raw = m.group("anchor")
            yield WikilinkMatch(
                target=target,
                anchor=anchor_raw[1:].strip() if anchor_raw else None,
                start=m.start(),
                end=m.end(),
                target_start=m.start("target"),
                target_end=m.end("target"),
                syntax="wikilink",
            )

    @staticmethod
    def _code_spans(text: str) -> list[tuple[int, int]]:
        """Return fenced and inline-code spans so Markdown examples are ignored."""
        spans: list[tuple[int, int]] = []
        fence: tuple[str, int, int] | None = None
        offset = 0
        for line in text.splitlines(keepends=True):
            marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
            if fence is None and marker:
                token = marker.group(1)
                fence = (token[0], len(token), offset)
            elif fence is not None:
                fence_char, fence_length, fence_start = fence
                if re.match(rf"^ {{0,3}}{re.escape(fence_char)}{{{fence_length},}}[ \t]*(?:\n)?$", line):
                    spans.append((fence_start, offset + len(line)))
                    fence = None
            offset += len(line)
        if fence is not None:
            spans.append((fence[2], len(text)))

        fenced = list(spans)
        i = 0
        fence_index = 0
        while i < len(text):
            while fence_index < len(fenced) and i >= fenced[fence_index][1]:
                fence_index += 1
            if fence_index < len(fenced) and fenced[fence_index][0] <= i < fenced[fence_index][1]:
                i = fenced[fence_index][1]
                continue
            if text[i] != "`":
                i += 1
                continue
            run = len(text[i:]) - len(text[i:].lstrip("`"))
            delimiter = "`" * run
            close = text.find(delimiter, i + run)
            if close < 0:
                i += run
                continue
            spans.append((i, close + run))
            i = close + run
        return sorted(spans)

    @staticmethod
    def _matching_paren(text: str, opening: int) -> int | None:
        """Find the closing parenthesis of an inline Markdown link."""
        depth = 1
        quote = ""
        escaped = False
        for i in range(opening + 1, len(text)):
            char = text[i]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if quote:
                if char == quote:
                    quote = ""
                continue
            if char in "\"'" and i > opening + 1 and text[i - 1].isspace():
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return i
        return None

    @staticmethod
    def _markdown_destination(text: str, opening: int, closing: int) -> tuple[int, int] | None:
        """Return the raw destination span, excluding angle brackets and title."""
        i = opening + 1
        while i < closing and text[i].isspace():
            i += 1
        if i >= closing:
            return None
        if text[i] == "<":
            end = i + 1
            while end < closing and text[end] != ">" and text[end] != "\n":
                end += 2 if text[end] == "\\" and end + 1 < closing else 1
            return (i + 1, end) if end < closing and text[end] == ">" else None

        start = i
        depth = 0
        escaped = False
        while i < closing:
            char = text[i]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "(":
                depth += 1
            elif char == ")" and depth:
                depth -= 1
            elif char.isspace() and depth == 0:
                break
            i += 1
        return (start, i) if i > start else None

    @staticmethod
    def _normalize_markdown_destination(destination: str, source_path: str) -> tuple[str, str | None] | None:
        """Resolve one local Markdown destination to a workspace-relative target."""
        destination = re.sub(r"\\([\\()`])", r"\1", destination.strip())
        parsed = urlsplit(destination)
        if parsed.scheme or parsed.netloc or parsed.query or not parsed.path:
            return None
        path = unquote(parsed.path)
        if path.startswith(("/", "~")):
            return None
        base = "" if Path(source_path).is_absolute() else posixpath.dirname(source_path)
        target = posixpath.normpath(posixpath.join(base, path))
        if target in ("", ".", "..") or target.startswith("../"):
            return None
        return target, unquote(parsed.fragment).strip() or None

    @classmethod
    def _iter_markdown_links(cls, text: str, source_path: str, blocked: list[tuple[int, int]] | None = None):
        """Yield inline Markdown links to local files, excluding images and code."""
        blocked = blocked if blocked is not None else cls._code_spans(text)
        blocked_index = 0
        i = 0
        while i < len(text):
            while blocked_index < len(blocked) and i >= blocked[blocked_index][1]:
                blocked_index += 1
            if blocked_index < len(blocked) and blocked[blocked_index][0] <= i < blocked[blocked_index][1]:
                i = blocked[blocked_index][1]
                continue
            if text[i] != "[":
                i += 1
                continue
            preceding_backslashes = len(text[:i]) - len(text[:i].rstrip("\\"))
            if preceding_backslashes % 2 or (i > 0 and text[i - 1] == "!"):
                i += 1
                continue

            depth = 1
            label_end = i + 1
            escaped = False
            while label_end < len(text) and depth:
                char = text[label_end]
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                label_end += 1
            if depth or label_end >= len(text) or text[label_end] != "(":
                i += 1
                continue

            closing = cls._matching_paren(text, label_end)
            span = cls._markdown_destination(text, label_end, closing) if closing is not None else None
            if span is None:
                i += 1
                continue
            destination = text[span[0] : span[1]]
            normalized = cls._normalize_markdown_destination(destination, source_path)
            if normalized is not None:
                target, anchor = normalized
                fragment_at = destination.find("#")
                target_end = span[1] if fragment_at < 0 else span[0] + fragment_at
                yield WikilinkMatch(
                    target=target,
                    anchor=anchor,
                    start=i,
                    end=closing + 1,
                    target_start=span[0],
                    target_end=target_end,
                    syntax="markdown",
                )
            i = closing + 1

    @classmethod
    def iter_matches(cls, text: str, source_path: str = ""):
        """Yield local Wikilink and inline Markdown-link occurrences in source order."""
        blocked = cls._code_spans(text)
        matches = [*cls._iter_wikilinks(text), *cls._iter_markdown_links(text, source_path, blocked)]
        previous_end = -1
        blocked_index = 0
        for match in sorted(matches, key=lambda item: (item.start, item.end)):
            while blocked_index < len(blocked) and match.start >= blocked[blocked_index][1]:
                blocked_index += 1
            if blocked_index < len(blocked) and blocked[blocked_index][0] <= match.start < blocked[blocked_index][1]:
                continue
            if match.start < previous_end:
                continue
            previous_end = match.end
            yield match

    # -- FileLink extraction ------------------------------------------

    @classmethod
    def extract_links(cls, text: str, source_path: str) -> list[FileLink]:
        """Emit :class:`FileLink` edges for every local link in ``text``.

        Wikilink targets remain literal. Markdown destinations resolve relative
        to ``source_path`` and normalize to workspace-relative graph paths.
        Results are deduped by ``(target_path, target_anchor)`` preserving
        order. Surrounding text has no effect on the edge.
        """
        if not text:
            return []
        out: list[FileLink] = []
        seen: set[tuple] = set()
        for wm in cls.iter_matches(text, source_path):
            key = (wm.target, wm.anchor)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                FileLink(
                    source_path=source_path,
                    target_path=wm.target,
                    target_anchor=wm.anchor,
                ),
            )
        return out

    # -- Find / rewrite by literal target match ------------------------

    @classmethod
    def scan_and_rewrite(
        cls,
        text: str,
        old: str,
        new: str | None,
        source_path: str = "",
    ) -> tuple[str, int]:
        """Find and optionally rewrite local links whose graph target is ``old``.

        Returns ``(new_text, count)``. When ``new`` is ``None`` no rewrite
        happens (the original text is returned), but the count is still
        populated — used by ``find_inbound``. Wikilink matching is literal;
        Markdown destinations are compared after document-relative path
        normalization.
        """
        matches = [match for match in cls.iter_matches(text, source_path) if match.target == old]
        if new is None or not matches:
            return text, len(matches)

        rewritten = text
        for match in reversed(matches):
            replacement = new
            if match.syntax == "markdown" and source_path and not Path(source_path).is_absolute():
                replacement = posixpath.relpath(new, posixpath.dirname(source_path) or ".")
                replacement = replacement.replace(" ", "%20").replace("(", "\\(").replace(")", "\\)")
            rewritten = rewritten[: match.target_start] + replacement + rewritten[match.target_end :]
        return rewritten, len(matches)

    # -- Validation ----------------------------------------------------

    @classmethod
    def validate_src_dst(cls, src: str, dst: str) -> str | None:
        """Return an error message for bad rewrite inputs, or None when OK."""
        if not src or not dst:
            return "src and dst are required"
        if any(ch in dst for ch in cls.FORBIDDEN_IN_NEW):
            return "dst must not contain [ ] # | newline"
        if Path(src).is_absolute() or Path(dst).is_absolute():
            return "src and dst must be relative to the workspace"
        return None

    @staticmethod
    def validate_scope(scope: str) -> str | None:
        """Return an error message for a bad scope, or None when OK."""
        if scope and Path(scope).is_absolute():
            return "scope must be relative to the workspace"
        return None

    @staticmethod
    def within_scope(rel: str, scope: str) -> bool:
        """``rel`` (relative to the workspace) is inside ``scope`` (empty = anywhere)."""
        if not scope:
            return True
        prefix = scope.rstrip("/") + "/"
        return rel == scope or rel.startswith(prefix)

    # -- Async file_graph-aware operations -----------------------------

    @classmethod
    async def _inbound_sources(cls, file_store, target: str) -> list[str]:
        """Source paths the file_graph reports as referencing ``target``.

        Reverse-index lookup via ``file_graph.get_inlinks(target, scope=ALL)`` —
        ``target`` is typically virtual here (the move/delete callers query for
        references to a path that has just been removed), so ``scope=ALL`` is
        required to surface sources whose edges sit in the pending bucket.
        Each returned ``FileLink`` carries the linking node's ``source_path``;
        we dedupe to a sorted list since one source can host multiple edges
        (different anchors) to the same target. Returns ``[]`` when
        there is no file_graph attached or no source references the target.
        """
        if not file_store.file_graph:
            return []
        inlinks = await file_store.file_graph.get_inlinks(target, scope=LinkScopeEnum.ALL)
        return sorted({link.source_path for link in inlinks if link.source_path})

    @classmethod
    async def find_inbound(cls, file_store, target: str, scope: str = "") -> dict:
        """Count local links across the workspace that point at ``target``.

        The target file itself is excluded — self-references don't survive a
        delete and aren't actionable for the caller. Sources come from the
        file_graph's reverse index; per-file counts come from reading each
        candidate source because the graph dedupes repeated edges.

        Result shape::

            {
              "target": str,
              "scope":  str | None,
              "files_touched": int,    # number of OTHER files containing >=1 ref
              "links_total":   int,    # total ref count across those files
              "by_file":  [{"path": str, "count": int}, ...],
            }

        On bad inputs returns ``{"target": ..., "error": str}``.
        """
        if not target:
            return {"target": target, "error": "target is required"}
        if Path(target).is_absolute():
            return {"target": target, "error": "target must be relative to the workspace"}
        err = cls.validate_scope(scope)
        if err is not None:
            return {"target": target, "error": err}

        workspace_dir = Path(file_store.workspace_path or ".").resolve()
        by_file: list[dict] = []
        total = 0

        for rel in await cls._inbound_sources(file_store, target):
            if rel == target:
                continue  # self-references not actionable for delete cleanup
            if not cls.within_scope(rel, scope):
                continue
            try:
                text = (workspace_dir / rel).read_text(encoding="utf-8")
            except Exception:
                continue
            _, count = cls.scan_and_rewrite(text, old=target, new=None, source_path=rel)
            if count > 0:
                by_file.append({"path": rel, "count": count})
                total += count

        return {
            "target": target,
            "scope": scope or None,
            "files_touched": len(by_file),
            "links_total": total,
            "by_file": by_file,
        }

    @classmethod
    async def retarget_links(
        cls,
        file_store,
        src: str,
        dst: str,
        scope: str = "",
        dry_run: bool = False,
    ) -> dict:
        """Rewrite every wikilink pointing at ``src`` to point at ``dst``.

        Pure helper — called directly by ``file_move`` post-rename. Literal
        matching only; candidate sources come from the file_graph's reverse
        index.
        """
        err = cls.validate_src_dst(src, dst)
        if err is not None:
            return {"src": src, "dst": dst, "error": err}
        if src == dst:
            return {
                "src": src,
                "dst": dst,
                "scope": scope or None,
                "dry_run": dry_run,
                "files_touched": 0,
                "links_changed": 0,
                "by_file": [],
            }
        err = cls.validate_scope(scope)
        if err is not None:
            return {"src": src, "dst": dst, "error": err}

        workspace_dir = Path(file_store.workspace_path or ".").resolve()
        by_file: list[dict] = []
        total_changes = 0

        for rel in await cls._inbound_sources(file_store, src):
            if not cls.within_scope(rel, scope):
                continue
            abs_path = workspace_dir / rel
            try:
                text = abs_path.read_text(encoding="utf-8")
            except Exception:
                continue
            new_text, count = cls.scan_and_rewrite(text, old=src, new=dst, source_path=rel)
            if count > 0:
                by_file.append({"path": rel, "count": count})
                total_changes += count
                if not dry_run:
                    abs_path.write_text(new_text, encoding="utf-8")

        return {
            "src": src,
            "dst": dst,
            "scope": scope or None,
            "dry_run": dry_run,
            "files_touched": len(by_file),
            "links_changed": total_changes,
            "by_file": by_file,
        }
