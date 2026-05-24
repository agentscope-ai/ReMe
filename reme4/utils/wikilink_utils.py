"""Wikilink graph helpers — inbound discovery + retarget rewrite.

Pure async functions usable by any step that needs to maintain
referential integrity across the vault. ``file_move`` calls
``retarget_links`` post-rename so inbound ``[[src]]`` resolves to
the new path; ``file_delete`` calls ``find_inbound`` to surface
references the caller might want to clean up first.

Matching rule. A raw wikilink target ``T`` matches ``src`` iff
``T == src`` (literal full-path match including the ``.md`` extension).
Short-link forms (``[[basename]]``) and extension-less forms
(``[[path/without-md]]``) are NOT matched — this codebase treats
wikilinks as strict full-relative-path references for simplicity and
predictability.

Rewriting. Replaces the target portion of the ``[[...]]`` with the
caller-provided ``dst`` string verbatim. ``#anchor`` and ``|alias``
suffixes pass through. Image marker (``!``) and Dataview predicate
(``pred::`` outside the brackets) are outside the ``[[...]]`` and so
are not touched. ``dst`` must be a full path relative to the vault
(matching the convention).

Source discovery uses the file_graph's reverse index — every node
whose ``links`` payload carries a ``target_path == src`` becomes a
candidate source. No filesystem walk across the vault: the graph is
the index. Stale graph entries are harmless (``_scan_text`` returns
count=0 and the file is skipped), but a graph missing recent writes
will miss those sources — keep the watcher in sync.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..enumeration import LinkScopeEnum


# Captures: optional image marker (``!``), the bare target, an optional
# ``#anchor`` slice, and an optional ``|alias`` slice. The anchor /
# alias inner classes exclude ``[`` defensively so a runaway match on
# malformed input can't swallow following links.
_WIKILINK_RE = re.compile(
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

_FORBIDDEN_IN_NEW = ("[", "]", "#", "|", "\n", "\r")


def _validate_src_dst(src: str, dst: str) -> str | None:
    """Return an error message for bad inputs, or None when OK."""
    if not src or not dst:
        return "src and dst are required"
    if any(ch in dst for ch in _FORBIDDEN_IN_NEW):
        return "dst must not contain [ ] # | newline"
    if Path(src).is_absolute() or Path(dst).is_absolute():
        return "src and dst must be relative to the vault"
    return None


def _validate_scope(scope: str) -> str | None:
    """Return an error message for a bad scope, or None when OK."""
    if scope and Path(scope).is_absolute():
        return "scope must be relative to the vault"
    return None


def _within_scope(rel: str, scope: str) -> bool:
    """``rel`` (relative to the vault) is inside ``scope`` (empty scope = anywhere)."""
    if not scope:
        return True
    prefix = scope.rstrip("/") + "/"
    return rel == scope or rel.startswith(prefix)


def _scan_text(text: str, old: str, new: str | None) -> tuple[str, int]:
    """Find (and optionally rewrite) wikilinks matching ``old`` in ``text``.

    Returns ``(new_text, count)``. When ``new`` is ``None`` no rewrite
    happens (the original text is returned), but the count is still
    populated — used by ``find_inbound``.

    Matching is literal: ``target == old``. No short-link, no implicit
    ``.md``, no folder-note expansion.
    """
    count = 0

    def sub(match: re.Match) -> str:
        nonlocal count
        target = match.group("target").strip()
        if target != old:
            return match.group(0)
        count += 1
        if new is None:
            return match.group(0)
        anchor = match.group("anchor") or ""
        alias = match.group("alias") or ""
        bang = match.group("bang") or ""
        return f"{bang}[[{new}{anchor}{alias}]]"

    new_text = _WIKILINK_RE.sub(sub, text)
    return new_text, count


async def _inbound_sources(file_store, target: str) -> list[str]:
    """Source paths the file_graph reports as referencing ``target``.

    Reverse-index lookup via ``file_graph.get_inlinks(target, scope=ALL)`` —
    ``target`` is typically virtual here (the move/delete callers query for
    references to a path that has just been removed), so ``scope=ALL`` is
    required to surface sources whose edges sit in the pending bucket.
    Each returned ``FileLink`` carries the linking node's ``source_path``;
    we dedupe to a sorted list since one source can host multiple edges
    (different anchor/predicate) to the same target. Returns ``[]`` when
    there is no file_graph attached or no source references the target.
    """
    if not file_store.file_graph:
        return []
    inlinks = await file_store.file_graph.get_inlinks(target, scope=LinkScopeEnum.ALL)
    return sorted({link.source_path for link in inlinks if link.source_path})


async def find_inbound(file_store, target: str, scope: str = "") -> dict:
    """Count wikilinks across the vault that point at ``target``.

    Literal matching: ``[[target]]`` only. The target file itself is
    excluded — self-references don't survive a delete and aren't
    actionable for the caller. Sources come from the file_graph's
    reverse index; per-file counts come from reading each candidate
    source (the graph dedupes by ``(target, predicate, anchor)`` so
    it can't count repeated bare-wikilink occurrences directly).

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
        return {"target": target, "error": "target must be relative to the vault"}
    err = _validate_scope(scope)
    if err is not None:
        return {"target": target, "error": err}

    vault_dir = Path(file_store.vault_path or ".").resolve()
    by_file: list[dict] = []
    total = 0

    for rel in await _inbound_sources(file_store, target):
        if rel == target:
            continue  # self-references not actionable for delete cleanup
        if not _within_scope(rel, scope):
            continue
        try:
            text = (vault_dir / rel).read_text(encoding="utf-8")
        except Exception:
            continue
        _, count = _scan_text(text, old=target, new=None)
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


async def retarget_links(
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
    err = _validate_src_dst(src, dst)
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
    err = _validate_scope(scope)
    if err is not None:
        return {"src": src, "dst": dst, "error": err}

    vault_dir = Path(file_store.vault_path or ".").resolve()
    by_file: list[dict] = []
    total_changes = 0

    for rel in await _inbound_sources(file_store, src):
        if not _within_scope(rel, scope):
            continue
        abs_path = vault_dir / rel
        try:
            text = abs_path.read_text(encoding="utf-8")
        except Exception:
            continue
        new_text, count = _scan_text(text, old=src, new=dst)
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
