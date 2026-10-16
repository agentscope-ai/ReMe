"""Retrieve candidates and verify evidence against bounded workspace files."""

import asyncio
import hashlib
from pathlib import Path

from ..file_io._file_io import get_path_lock, read_file_safe
from ..file_io._path import _check_path_permission, resolve_path
from ...schema.context_agent import ContextAgentOptions, ContextEvent, MemoryCandidate


async def read_source(workspace: Path, path: str, allowed_paths, max_bytes: int) -> tuple[str, str, float] | None:
    """Read a contained, permitted source with existing IO locks and decoding."""
    target, error = resolve_path(workspace, path)
    if error or target is None or not _check_path_permission(workspace, target, allowed_paths):
        return None
    lock = await get_path_lock(target)
    async with lock:
        checked, error = resolve_path(workspace, path)
        if error or checked != target or not _check_path_permission(workspace, target, allowed_paths):
            return None
        try:
            if not target.is_file() or target.stat().st_size > max_bytes:
                return None
            before = target.stat()
            text, _encoding = await read_file_safe(target, max_bytes=max_bytes)
            after = target.stat()
        except OSError:
            return None
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            return None
        return text, hashlib.sha256(text.encode("utf-8")).hexdigest(), after.st_mtime


async def evidence_is_current(workspace: Path, evidence: list[MemoryCandidate], allowed_paths, max_bytes: int) -> bool:
    """Recheck permissions, source versions and exact excerpts before delivery."""
    for item in evidence:
        source = await read_source(workspace, item.path, allowed_paths, max_bytes)
        if source is None or source[1] != item.content_version:
            return False
        lines = source[0].splitlines()
        if not "\n".join(lines[item.start_line - 1 : item.end_line]).startswith(item.excerpt):
            return False
    return True


async def discover_candidates(store, workspace: Path, event: ContextEvent, options: ContextAgentOptions, allowed_paths):
    """Use hybrid retrieval without mutating the main agent's search-seen state."""
    snapshot = event.task_snapshot
    queries = list(dict.fromkeys([snapshot.goal[:1000], *[q[:200] for q in snapshot.query_terms if q.strip()]]))[:3]
    if snapshot.current_tool.strip():
        queries.append(snapshot.current_tool[:500])
    rankings = []
    for query in queries:
        calls = []
        weights = []
        if options.vector_weight > 0:
            calls.append(store.vector_search(query, options.max_candidates * 3, {}))
            weights.append(options.vector_weight)
        if options.vector_weight < 1:
            calls.append(store.keyword_search(query, options.max_candidates * 3, {}))
            weights.append(1 - options.vector_weight)
        rankings.extend(zip(await asyncio.gather(*calls), weights))
    scores, chunks = {}, {}
    for ranking, weight in rankings:
        for rank, chunk in enumerate(ranking, 1):
            key = (chunk.path, chunk.start_line, chunk.end_line)
            chunks[key] = chunk
            scores[key] = scores.get(key, 0) + weight / (60 + rank)
    indexed_nodes = {node.path: node for node in await store.get_nodes(list({key[0] for key in chunks}))}
    results = []
    for key in sorted(scores, key=scores.get, reverse=True):
        chunk = chunks[key]
        if chunk.start_line < 1 or chunk.end_line < chunk.start_line:
            continue
        source = await read_source(workspace, chunk.path, allowed_paths, options.max_file_bytes)
        node = indexed_nodes.get(chunk.path)
        if source is None or node is None or source[2] != node.st_mtime:
            continue
        lines = source[0].splitlines()
        # Markdown index text may be rendered, breadcrumb-prefixed or byte-split.
        # Check index freshness via its node, then quote original source lines.
        excerpt = "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        if not excerpt.strip():
            continue
        excerpt = excerpt[: options.max_excerpt_chars]
        path = (workspace / chunk.path).resolve().relative_to(workspace.resolve()).as_posix()
        end_line = chunk.start_line + len(excerpt.splitlines()) - 1
        identity = f"{path}:{chunk.start_line}:{end_line}:{source[1]}"
        candidate_id = hashlib.sha256(identity.encode()).hexdigest()
        if candidate_id in snapshot.seen_evidence:
            continue
        results.append(
            MemoryCandidate(
                candidate_id=candidate_id,
                path=path,
                start_line=chunk.start_line,
                end_line=end_line,
                content_version=source[1],
                excerpt=excerpt,
            ),
        )
        if len(results) >= options.max_candidates:
            break
    return results
