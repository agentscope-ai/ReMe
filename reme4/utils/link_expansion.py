"""Expand a file's wikilink neighbors and render them as indented text.

Used by :class:`~reme.steps.index.search.SearchStep` for per-hit
context expansion. Pure helper — no step state, only ``file_store`` is
required.

Two-layer split so callers can pick what they need:

* :func:`expand_links` — data layer. Returns a structured dict keyed
  by source path, each value carrying its outlinks / inlinks with
  neighbor meta and per-edge predicate/anchor.
* :func:`render_expansion_lines` — view layer. Turns one path's
  expansion sub-dict into the same ``  → path  name=… description=…``
  block ``SearchStep`` has historically printed.
"""

import asyncio

from ..schema import FileLink, FileNode


def _group_by_neighbor(links: list[FileLink], key_attr: str) -> dict[str, list[dict]]:
    """Group edges by neighbor path (insertion-ordered), each value a list of {predicate, anchor}."""
    out: dict[str, list[dict]] = {}
    for lnk in links:
        neighbor = getattr(lnk, key_attr)
        if not neighbor:
            continue
        out.setdefault(neighbor, []).append(
            {"predicate": lnk.predicate, "anchor": lnk.target_anchor},
        )
    return out


def _node_meta(node: FileNode | None) -> dict:
    """Extract a compact meta dict (name/description) from a FileNode."""
    if node is None:
        return {}
    fm = node.front_matter
    meta: dict = {}
    if fm.name:
        meta["name"] = fm.name
    if fm.description:
        meta["description"] = fm.description
    return meta


def _format_meta_inline(meta: dict) -> str:
    """One-line render of node meta for the answer; '(no meta)' when empty."""
    parts = []
    if "name" in meta:
        parts.append(f'name="{meta["name"]}"')
    if "description" in meta:
        parts.append(f'description="{meta["description"]}"')
    return "  ".join(parts) if parts else "(no meta)"


def _format_via(edge: dict) -> str:
    """Render a single (predicate, anchor) edge as a 'via ...' descriptor."""
    bits = []
    if edge.get("predicate"):
        bits.append(f"predicate={edge['predicate']}")
    if edge.get("anchor"):
        bits.append(f"anchor=#{edge['anchor']}")
    return ", ".join(bits) if bits else "plain"


async def expand_links(
    file_store,
    paths: list[str],
    max_per_direction: int = 10,
) -> dict[str, dict]:
    """Fetch out/in links for each path and attach neighbor meta.

    Returns ``{path: {"outlinks": [...], "inlinks": [...]}, ...}`` where
    each list item is ``{"path": str, "meta": {...}, "edges": [{"predicate", "anchor"}, ...]}``.
    Empty input returns ``{}``. ``max_per_direction`` caps the neighbor
    list per direction *before* meta lookup so we don't fetch nodes
    that won't be displayed.
    """
    if not paths:
        return {}

    out_lists, in_lists = await asyncio.gather(
        asyncio.gather(*(file_store.get_outlinks(p) for p in paths)),
        asyncio.gather(*(file_store.get_inlinks(p) for p in paths)),
    )

    out_grouped = [
        dict(list(_group_by_neighbor(outs, "target_path").items())[:max_per_direction]) for outs in out_lists
    ]
    in_grouped = [dict(list(_group_by_neighbor(ins, "source_path").items())[:max_per_direction]) for ins in in_lists]

    neighbor_paths = sorted({n for g in out_grouped for n in g} | {n for g in in_grouped for n in g})
    nodes = await file_store.get_nodes(neighbor_paths) if neighbor_paths else []
    meta_by_path = {n.path: _node_meta(n) for n in nodes}

    def _attach(grouped: dict[str, list[dict]]) -> list[dict]:
        return [
            {"path": npath, "meta": meta_by_path.get(npath, {}), "edges": edges} for npath, edges in grouped.items()
        ]

    return {p: {"outlinks": _attach(og), "inlinks": _attach(ig)} for p, og, ig in zip(paths, out_grouped, in_grouped)}


def render_expansion_lines(expansion: dict, indent: str = "  ") -> list[str]:
    """Render one path's expansion sub-dict as indented lines.

    ``expansion`` is one value from :func:`expand_links` — i.e.
    ``{"outlinks": [...], "inlinks": [...]}``. Returns ``[]`` when both
    directions are empty (caller decides whether to append a blank
    line). ``indent`` controls the leading indent of the direction
    header; neighbor lines and per-edge ``via`` lines nest further.
    """
    lines: list[str] = []
    inner = indent + "  "
    edge_indent = indent + "      "
    for direction, arrow, items in (
        ("outlinks", "→", expansion.get("outlinks") or []),
        ("inlinks", "←", expansion.get("inlinks") or []),
    ):
        if not items:
            continue
        lines.append(f"{indent}{direction} ({len(items)}):")
        for item in items:
            lines.append(f"{inner}{arrow} {item['path']}  {_format_meta_inline(item['meta'])}")
            for edge in item["edges"]:
                lines.append(f"{edge_indent}via {_format_via(edge)}")
    return lines
