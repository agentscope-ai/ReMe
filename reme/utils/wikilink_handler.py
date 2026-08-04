"""Wikilink handler — single source of truth for ``[[...]]`` syntax.

One class, :class:`WikilinkHandler`, owning every wikilink concern:

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
  post-rename to point inbound ``[[src]]`` at the new path). Source
  candidates come from the file_graph's reverse index — no fs scan.

Wikilink targets are taken **literally** — ``[[X]]`` →
``target="X"``, no implicit ``.md``, no short-form basename search,
no folder-note expansion. Anchor and alias survive a rewrite
verbatim. Text outside ``[[...]]`` is ignored. Recommended form: full
path relative to the workspace with extension (``[[topics/x.md]]``).
Ordinary Markdown links such as ``[label](path.md)`` are ignored.

Stale graph entries are harmless (``scan_and_rewrite`` returns
count=0 and the file is skipped), but a graph missing recent writes
will miss those sources — keep the watcher in sync.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from ..enumeration import LinkScopeEnum
from ..schema import FileLink


def _normalize_workspace_path(path: str) -> str:
    """Use POSIX separators for workspace paths on every platform."""
    return path.replace("\\", "/")


@dataclass(frozen=True)
class WikilinkMatch:
    """The graph-relevant parts and source span of one wikilink."""

    target: str
    anchor: str | None
    start: int
    end: int


class WikilinkHandler:
    """Parse, extract, rewrite, and validate wikilinks."""

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
    def iter_matches(cls, text: str):
        """Yield every non-empty ``[[...]]`` occurrence in ``text``."""
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
            )

    # -- FileLink extraction ------------------------------------------

    @classmethod
    def extract_links(cls, text: str, source_path: str) -> list[FileLink]:
        """Emit :class:`FileLink` edges for every wikilink in ``text``.

        Targets remain literal. Results are deduped by
        ``(target_path, target_anchor)`` while preserving order.
        """
        if not text:
            return []
        source_path = _normalize_workspace_path(source_path)
        out: list[FileLink] = []
        seen: set[tuple] = set()
        for wm in cls.iter_matches(text):
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
    ) -> tuple[str, int]:
        """Find and optionally rewrite wikilinks whose target is ``old``.

        Returns ``(new_text, count)``. When ``new`` is ``None`` no rewrite
        happens (the original text is returned), but the count is still
        populated — used by ``find_inbound``. Matching is literal.
        """
        old = _normalize_workspace_path(old)
        new = _normalize_workspace_path(new) if new is not None else None
        count = 0

        def sub(match: re.Match) -> str:
            nonlocal count
            if match.group("target").strip() != old:
                return match.group(0)
            count += 1
            if new is None:
                return match.group(0)
            anchor = match.group("anchor") or ""
            alias = match.group("alias") or ""
            bang = match.group("bang") or ""
            return f"{bang}[[{new}{anchor}{alias}]]"

        return cls.WIKILINK_RE.sub(sub, text), count

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
        target = _normalize_workspace_path(target)
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
        target = _normalize_workspace_path(target)
        scope = _normalize_workspace_path(scope)
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
            _, count = cls.scan_and_rewrite(text, old=target, new=None)
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
        src = _normalize_workspace_path(src)
        dst = _normalize_workspace_path(dst)
        scope = _normalize_workspace_path(scope)
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
            new_text, count = cls.scan_and_rewrite(text, old=src, new=dst)
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
