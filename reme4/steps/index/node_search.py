"""``node_search_step`` — node-level digest search for dream Phase 2.

Specialized for dream's recall needs; **NOT** a drop-in replacement for the
general-purpose ``search`` step (which serves external RAG agents).

Five differences vs ``search``:

1. **Node-level results** — same-path chunks aggregated by max score; one
   row per digest node, not per chunk.
2. **Frontmatter included** — returns ``name + description`` inline so the
   caller can triage without a follow-up ``frontmatter_read`` per hit.
3. **Digest-only filter** — hardcoded to ``<digest_dir>/`` prefix; dream
   never wants daily / resource hits as recall candidates.
4. **No expand_links** — dream's synapse recall is looking for nodes that
   *don't* yet have wikilinks; expansion would surface already-linked
   neighbors (anti-pattern for synapse construction).
5. **No body / chunk text in response** — caller follows up with ``read``
   only on the few candidates that need deep inspection, not all.

A single hybrid (vector + BM25 RRF) recall serves **both** the dedup
judgment (`same_abstraction` label — is any candidate the same as the
new unit?) and the synapse judgment (`related` label — which candidates
should be woven as wikilinks?). They are two LLM-internal labels over
the **same candidate pool**; there is no need to split into separate
recall passes (the previous ``mode={hybrid,vector_only}`` toggle was
spurious — both judgments operate on the same recall output).

Used in dream Phase 2 (dream.yaml ``integrate_system_prompt_*``).
External agents must keep using ``search`` for chunk-level retrieval +
link expansion.
"""

import asyncio
from collections import defaultdict

from ..base_step import BaseStep
from ...components import R

_RRF_K = 60
_MAX_CANDIDATES = 200


def _rrf_merge_nodes(
    vector_paths: list[str],
    keyword_paths: list[str],
    vector_weight: float,
) -> dict[str, float]:
    """RRF fuse two ranked path lists into node-level scores."""
    scores: dict[str, float] = defaultdict(float)
    text_weight = 1.0 - vector_weight
    for rank, path in enumerate(vector_paths, start=1):
        scores[path] += vector_weight / (_RRF_K + rank)
    for rank, path in enumerate(keyword_paths, start=1):
        scores[path] += text_weight / (_RRF_K + rank)
    return scores


@R.register("node_search_step")
class NodeSearchStep(BaseStep):
    """Node-level digest-only hybrid search for dream Phase 2 recall."""

    async def execute(self):
        assert self.context is not None
        query: str = (self.context.get("query", "") or "").strip()
        limit: int = int(self.context.get("limit") or 20)
        vector_weight: float = float(self.kwargs.get("vector_weight", 0.7))
        candidate_multiplier: float = float(self.kwargs.get("candidate_multiplier", 5.0))

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Error: query cannot be empty"
            return self.context.response
        assert limit > 0, f"limit must be positive, got {limit}"

        digest_dir = getattr(self.app_context.app_config, "digest_dir", "digest") or "digest"
        digest_prefix = digest_dir.rstrip("/") + "/"

        # Over-fetch — digest filter drops a lot of raw hits.
        candidates = min(_MAX_CANDIDATES, max(1, int(limit * candidate_multiplier)))

        vector_chunks, keyword_chunks = await asyncio.gather(
            self.file_store.vector_search(query, candidates, {}),
            self.file_store.keyword_search(query, candidates, {}),
        )

        def _node_dedup(chunks: list) -> list[str]:
            """Keep first occurrence per path; respect digest prefix only.

            Self-exclusion (e.g. UPDATE target) is the LLM's job: frontmatter
            inlining lets the agent recognize self from the candidate list.
            No need to push it into a mechanical parameter.
            """
            seen: set[str] = set()
            out: list[str] = []
            for c in chunks:
                if c.path in seen:
                    continue
                if not c.path.startswith(digest_prefix):
                    continue
                seen.add(c.path)
                out.append(c.path)
            return out

        vector_paths = _node_dedup(vector_chunks)
        keyword_paths = _node_dedup(keyword_chunks)
        path_to_score = _rrf_merge_nodes(vector_paths, keyword_paths, vector_weight)

        ranked = sorted(path_to_score.items(), key=lambda kv: -kv[1])[:limit]

        # Attach frontmatter from in-memory FileNode metadata (no extra IO).
        node_paths = [p for p, _ in ranked]
        nodes = await self.file_store.get_nodes(node_paths) if node_paths else []
        path_to_fm: dict[str, dict[str, str]] = {}
        for n in nodes:
            fm = n.front_matter
            path_to_fm[n.path] = {
                "name": (getattr(fm, "name", "") or "").strip(),
                "description": (getattr(fm, "description", "") or "").strip(),
            }

        hits: list[dict] = []
        for path, score in ranked:
            fm = path_to_fm.get(path, {})
            hits.append(
                {
                    "path": path,
                    "score": round(float(score), 4),
                    "name": fm.get("name", ""),
                    "description": fm.get("description", ""),
                },
            )

        self.logger.info(
            f"[{self.name}] query={query!r} candidates={candidates} "
            f"vector_hits={len(vector_chunks)} keyword_hits={len(keyword_chunks)} "
            f"returned={len(hits)}",
        )

        lines = [
            f"=== node_search query={query!r} hits={len(hits)}/{candidates} ===",
        ]
        if not hits:
            lines.append("(no digest hits)")
        else:
            for h in hits:
                lines.append(
                    f"[{h['score']:.4f}] {h['path']}\n" f"  name: {h['name']}\n" f"  description: {h['description']}",
                )

        self.context.response.success = True
        self.context.response.answer = "\n".join(lines)
        self.context.response.metadata["hits"] = hits
        self.context.response.metadata["counts"] = {
            "vector_raw": len(vector_chunks),
            "keyword_raw": len(keyword_chunks),
            "returned": len(hits),
        }
        return self.context.response
