"""CLI to inspect `LinkedFileParser` output on a real markdown file.

Run a vault file through the parser and dump its chunks + edges so you
can eyeball what the AST chunker produced (sizes, TOC skeleton wrap,
``[Part X/N]`` markers, link extraction). Not a pytest test — it's a
manual inspection script that lives in `tests/` because that's where
ad-hoc developer tools belong.

Usage::

    python tests/inspect_md_parser.py <path> [--chunk-chars N] [--no-toc]
                                              [--show-edges] [--preview N]
"""

from __future__ import annotations

import argparse
import asyncio

from reme2.component.file_parser.linked_file_parser import LinkedFileParser


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse a markdown file with LinkedFileParser and dump chunks + edges.",
    )
    ap.add_argument("path", help="Path to a markdown file.")
    ap.add_argument(
        "--chunk-chars", type=int, default=2000,
        help="Max characters per chunk content (default: 2000). "
             "Excludes TOC skeleton when embed_toc is on.",
    )
    ap.add_argument(
        "--no-toc", action="store_true",
        help="Disable the full-doc TOC skeleton wrap; chunks become plain content.",
    )
    ap.add_argument(
        "--show-edges", action="store_true",
        help="Print extracted FileEdges before chunks.",
    )
    ap.add_argument(
        "--preview", type=int, default=0,
        help="Truncate each chunk to N chars in output (0 = full text).",
    )
    args = ap.parse_args()

    parser = LinkedFileParser(
        chunk_chars=args.chunk_chars,
        embed_toc=not args.no_toc,
    )
    node, chunks = asyncio.run(parser.parse(args.path))

    print(f"file:        {node.path}")
    print(f"chunk_chars: {args.chunk_chars}")
    print(f"embed_toc:   {parser.embed_toc}")
    print(f"chunks:      {len(chunks)}")
    print(f"chars total: {sum(len(c.text) for c in chunks)}")
    if chunks:
        sizes = [len(c.text) for c in chunks]
        print(f"chars min/avg/max: {min(sizes)} / {sum(sizes)//len(sizes)} / {max(sizes)}")
    if args.show_edges:
        print(f"\nedges ({len(node.edges)}):")
        for e in node.edges:
            print(
                f"  → {e.link}"
                + (f"  predicate={e.predicate}" if e.predicate else "")
                + (f"  anchor={e.anchor}" if e.anchor else "")
            )

    for i, c in enumerate(chunks):
        print(f"\n{'=' * 72}")
        print(f"chunk {i}  lines {c.start_line}-{c.end_line}  {len(c.text)} chars")
        print("-" * 72)
        text = c.text if args.preview <= 0 else c.text[: args.preview]
        print(text)
        if args.preview > 0 and len(c.text) > args.preview:
            print(f"... ({len(c.text) - args.preview} more chars truncated)")


if __name__ == "__main__":
    main()
