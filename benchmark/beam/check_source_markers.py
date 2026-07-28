"""Check source markers in BEAM auto-memory notes.

Scans daily notes under one or more case workspaces and validates every
source marker against the required format::

    [[<session path>#L<start>-L<end>(,L<start>-L<end>)*]]

Reported problems:

- ``legacy_format``: old single-bracket ``[path:1-2]`` markers.
- ``bad_anchor``: ``[[...jsonl#...]]`` whose anchor is not L-prefixed
  hyphen ranges (colon separators, missing ``L``, spaces, bare line...).
- ``labelled``: marker prefixed with ``Source``/``来源`` labels.
- ``missing_file``: linked session file does not exist in the workspace.
- ``out_of_range``: cited line numbers exceed the session file length,
  or start > end.

Usage:
    python benchmark/beam/check_source_markers.py [workspace_root ...]
    (default: benchmark/memory_workspaces/beam)
"""

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent

WIKILINK_RE = re.compile(r"\[\[([^\[\]\|\n]+?)\]\]")
VALID_ANCHOR_RE = re.compile(r"^L(\d+)-L(\d+)(,L(\d+)-L(\d+))*$")
RANGE_RE = re.compile(r"L(\d+)-L(\d+)")
# Old format: [session/dialog/x.jsonl:1-2,5-6] (single brackets, colon)
LEGACY_RE = re.compile(r"(?<!\[)\[([^\[\]\n]*\.jsonl:[^\[\]\n]*)\](?!\])")
LABEL_RE = re.compile(r"(?:Source|source|来源)[:：]?\s*(\[\[[^\[\]\n]*\.jsonl[^\[\]\n]*\]\])")


def _split_frontmatter(text: str) -> str:
    """Return the note body, skipping the leading YAML frontmatter block."""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[end + 4 :]
    return text


def _session_line_count(workspace: Path, rel_path: str) -> int | None:
    f = workspace / rel_path
    if not f.is_file():
        return None
    with open(f, encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def check_note(note: Path, workspace: Path) -> tuple[list[dict], int]:
    """Return (problems, valid_marker_count) for one note file."""
    text = _split_frontmatter(note.read_text(encoding="utf-8"))
    rel_note = str(note.relative_to(workspace))
    problems: list[dict] = []
    valid = 0

    def _add(kind: str, marker: str):
        problems.append({"note": rel_note, "kind": kind, "marker": marker.strip()[:160]})

    for m in LEGACY_RE.finditer(text):
        _add("legacy_format", m.group(0))

    for m in LABEL_RE.finditer(text):
        _add("labelled", m.group(0))

    line_counts: dict[str, int | None] = {}
    for m in WIKILINK_RE.finditer(text):
        inner = m.group(1)
        if "#" not in inner:
            continue  # plain file link (e.g. day-index), not a line marker
        target, anchor = inner.split("#", 1)
        if not target.endswith(".jsonl"):
            continue
        if not VALID_ANCHOR_RE.match(anchor):
            _add("bad_anchor", m.group(0))
            continue
        if target not in line_counts:
            line_counts[target] = _session_line_count(workspace, target)
        n_lines = line_counts[target]
        if n_lines is None:
            _add("missing_file", m.group(0))
            continue
        ok = True
        for r in RANGE_RE.finditer(anchor):
            start, end = int(r.group(1)), int(r.group(2))
            if start < 1 or end > n_lines or start > end:
                _add("out_of_range", f"{m.group(0)} (file has {n_lines} lines)")
                ok = False
                break
        if ok:
            valid += 1
    return problems, valid


def main(roots: list[str]) -> int:
    """Scan every case workspace under *roots* and print a problem report."""
    total_valid = 0
    all_problems: list[dict] = []
    n_notes = 0
    for root in roots:
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = _PROJECT_ROOT / root
        for workspace in sorted(root_path.glob("*/.reme")):
            for note in sorted(workspace.rglob("daily/*/*.md")):
                n_notes += 1
                problems, valid = check_note(note, workspace)
                total_valid += valid
                all_problems.extend({**p, "workspace": workspace.parent.name} for p in problems)

    print(f"Scanned {n_notes} notes: {total_valid} valid markers, {len(all_problems)} problems")
    by_kind: dict[str, int] = {}
    for p in all_problems:
        by_kind[p["kind"]] = by_kind.get(p["kind"], 0) + 1
    for kind, count in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {kind}: {count}")
    print()
    for p in all_problems:
        print(f"[{p['kind']}] {p['workspace']}/{p['note']}\n    {p['marker']}")
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["benchmark/memory_workspaces/beam"]))
