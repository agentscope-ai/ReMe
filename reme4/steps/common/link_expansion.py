"""First-order bidirectional neighbor expansion over a file_store.

Shared by SearchStep (batch, for rendering link blocks in answers) and ReadStep
(single-path, for injecting neighbor frontmatter into the read result).

Returns raw FileNode objects; callers project to whatever shape they need.
"""

import asyncio

from ...components.file_store import BaseFileStore
from ...schema import FileLink, FileNode


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


async def get_first_order_neighbors_batch(
    file_store: BaseFileStore,
    paths: list[str],
    *,
    max_per_direction: int | None = None,
) -> dict[str, dict]:
    """Fetch out/in first-order neighbors for each ``path`` (vault-relative).

    Returns ``{path: {"outlinks": [entry], "inlinks": [entry]}}`` where each
    entry is ``{"path": str, "node": FileNode | None, "edges": [{predicate, anchor}]}``.

    Issues per-path ``get_outlinks`` / ``get_inlinks`` concurrently, then a
    single batched ``get_nodes`` over the union of capped neighbor paths.
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
    node_by_path: dict[str, FileNode] = {n.path: n for n in nodes}

    def _attach(grouped: dict[str, list[dict]]) -> list[dict]:
        return [{"path": npath, "node": node_by_path.get(npath), "edges": edges} for npath, edges in grouped.items()]

    return {p: {"outlinks": _attach(og), "inlinks": _attach(ig)} for p, og, ig in zip(paths, out_grouped, in_grouped)}


async def get_first_order_neighbors(
    file_store: BaseFileStore,
    path: str,
    *,
    max_per_direction: int | None = None,
) -> dict:
    """Single-path convenience wrapper over ``get_first_order_neighbors_batch``."""
    batch = await get_first_order_neighbors_batch(file_store, [path], max_per_direction=max_per_direction)
    return batch.get(path, {"outlinks": [], "inlinks": []})
