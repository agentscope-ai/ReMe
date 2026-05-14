"""Service-profile MCP tests.

Covers the 3-tool service surface (`reme2/config/service.yaml`):

    retrieve  — graph-aware hybrid retrieval (memory_graph_search backend)
    remember  — single write entry point (Ingestor projection):
                  mode=log     → zero-LLM event-folder upsert
                  mode=distill → LLM R-M-W (degraded path here, no LLM)
    maintain  — vault hygiene sweep (Maintainer projection)

Same shape as `test_expert.py`: each `check_*` is a one-job-or-sequence
async function returning a one-line summary. Ordering matters because
`remember.log_append` and `remember.log_refuse_distilled` rely on side
effects from `remember.log_create`.
"""

from __future__ import annotations

from pathlib import Path

from ._helpers import AppContext, decode, wait_for_index


EXPECTED_JOBS: tuple[str, ...] = ("retrieve", "remember", "maintain")


# ---------- registry ---------------------------------------------------


async def check_registry(ctx: AppContext) -> str:
    missing = [j for j in EXPECTED_JOBS if j not in ctx.jobs]
    assert not missing, f"missing jobs: {missing}"
    extras = [j for j in ctx.jobs if j not in EXPECTED_JOBS]
    assert not extras, f"curated profile leaked extra jobs: {extras} — keep the surface tight"
    return f"{len(ctx.jobs)} jobs registered"


# ---------- read: retrieve (graph-aware hybrid) -----------------------


async def check_retrieve_basic(ctx: AppContext) -> str:
    r = decode(
        await ctx.app.run_job(
            "retrieve",
            query="Alice Bob",
            max_results=5,
            min_score=0.0,
        ),
    )
    hits = r if isinstance(r, list) else (r.get("chunks") if isinstance(r, dict) else [])
    assert len(hits) > 0, r
    return f"{len(hits)} hits"


async def check_retrieve_anchored(ctx: AppContext) -> str:
    """Wikilink-anchored mode — `[[Project X]]` in the query seeds BFS."""
    r = decode(
        await ctx.app.run_job(
            "retrieve",
            query="What touches [[Project X]]?",
            max_results=5,
            min_score=0.0,
            graph_depth=1,
        ),
    )
    hits = r if isinstance(r, list) else (r.get("chunks") if isinstance(r, dict) else [])
    assert len(hits) > 0, r
    paths = {h.get("path") for h in hits if isinstance(h, dict)}
    # Project X's neighbors (Alice, Bob) should surface via 1-hop BFS
    has_neighbor = any(p and ("Alice" in p or "Bob" in p) for p in paths)
    assert has_neighbor, f"BFS didn't pull in [[Project X]] neighbors: {paths}"
    return f"{len(hits)} hits incl. project-x neighbors"


async def check_retrieve_topic_seeded(ctx: AppContext) -> str:
    """Topic-rooted mode — explicit `seeds=[...]` instead of inline wikilink."""
    project_x = ctx.abs_path("topics", "Project X", "Project X.md")
    r = decode(
        await ctx.app.run_job(
            "retrieve",
            query="collaborates",
            max_results=5,
            min_score=0.0,
            seeds=[project_x],
            graph_depth=1,
        ),
    )
    hits = r if isinstance(r, list) else (r.get("chunks") if isinstance(r, dict) else [])
    assert len(hits) > 0, r
    return f"{len(hits)} hits (seeded at Project X)"


# ---------- write: remember (mode=log: create / append / refusal) -----


async def check_remember_log_create(ctx: AppContext) -> str:
    r = decode(
        await ctx.app.run_job(
            "remember",
            mode="log",
            name="curated-event",
            description="curated-profile suite",
            content="## ops\n- ran the curated suite\n",
            topics=["[[Alice]]"],
            tags=["curated"],
            materials=[
                {"filename": "raw-prompt.md", "content": "# user prompt\n\nrun curated suite\n"},
                {"filename": "tool-output.txt", "content": "exit=0\n"},
            ],
        ),
    )
    assert isinstance(r, dict), r
    assert r.get("created") is True and r.get("action") == "created", r
    materials = r.get("materials", [])
    assert len(materials) == 2, materials
    for m in materials:
        assert Path(m).is_file(), f"material missing on disk: {m}"
    event_dir = Path(materials[0]).parent
    index_text = (event_dir / "curated-event.md").read_text(encoding="utf-8")
    assert "## Materials" in index_text, "Materials footer absent"
    assert "lifecycle: streaming" in index_text, "schema axis 'lifecycle' missing"
    assert "role: observation" in index_text, "schema axis 'role' missing"
    setattr(ctx, "_event_dir", event_dir)
    setattr(ctx, "_event_index", event_dir / "curated-event.md")
    await wait_for_index(ctx.watcher, expected_min=len(ctx.file_store))
    return f"created folder w/ {len(materials)} materials"


async def check_remember_log_append(ctx: AppContext) -> str:
    r = decode(
        await ctx.app.run_job(
            "remember",
            mode="log",
            name="curated-event",
            content="## follow-up\n- second pass\n",
            topics=["[[Bob]]"],
            tags=["follow-up"],
            materials=[
                {"filename": "tool-output.txt", "content": "second run\n"},  # collision
            ],
        ),
    )
    assert isinstance(r, dict), r
    assert r.get("created") is False and r.get("action") == "appended", r
    appended = r.get("materials", [])
    names = {Path(p).name for p in appended}
    assert "tool-output-2.txt" in names, f"collision auto-suffix failed: {names}"
    index_text = getattr(ctx, "_event_index").read_text(encoding="utf-8")
    assert "## Update —" in index_text, "Update section missing"
    assert "[[Bob]]" in index_text, "topic union failed"
    return f"appended w/ collision → {sorted(names)}"


async def check_remember_log_refuse_distilled(ctx: AppContext) -> str:
    """Curated profile lacks `memory_update_meta`, so we flip status by
    rewriting the file directly — same observable effect on remember(mode=log)."""
    index_path = Path(getattr(ctx, "_event_index"))
    text = index_path.read_text(encoding="utf-8")
    text = text.replace("status: active", "status: distilled", 1)
    index_path.write_text(text, encoding="utf-8")
    # Give the watcher a moment to re-parse before sync re-reads frontmatter.
    await wait_for_index(ctx.watcher, expected_min=len(ctx.file_store))

    r = decode(
        await ctx.app.run_job(
            "remember",
            mode="log",
            name="curated-event",
            content="should be refused",
        ),
    )
    assert isinstance(r, dict) and "error" in r, r
    assert r.get("status") == "distilled", r
    assert r.get("suggested_name"), r
    return f"refused → suggested={r['suggested_name']!r}"


# ---------- write: remember (mode=distill, degraded path) -------------


async def check_remember_distill_degraded(ctx: AppContext) -> str:
    target = ctx.abs_path("topics", "curated-ingested", "curated-ingested.md")
    r = decode(
        await ctx.app.run_job(
            "remember",
            # mode defaults to "distill"
            content="# curated-ingested\n\nproduced by the curated test suite.\n",
            target_path=target,
            metadata={
                "title": "curated-ingested",
                "lifecycle": "evolving",
                "scope": "class",
                "source": "curated",
                "role": "concept",
                "category": "concept",
            },
        ),
    )
    assert isinstance(r, dict), r
    applied = r.get("applied") or []
    assert len(applied) == 1 and applied[0].get("ok") is True, r
    assert r.get("used_llm") is False, "expected degraded path (no LLM)"
    assert Path(target).is_file(), target
    return "applied=1, used_llm=False"


# ---------- maintain (lint + decay sweep) -----------------------------


async def check_maintain_dry_run(ctx: AppContext) -> str:
    """Default ops=[lint,decay] with dry_run=true (the parameter default)
    surfaces the plan without mutating. The seeded vault has nothing
    stale and no broken wikilinks, so we just verify the envelope."""
    r = decode(await ctx.app.run_job("maintain"))
    assert isinstance(r, dict), r
    for key in ("ops_run", "scanned", "proposed", "plan", "applied", "dry_run"):
        assert key in r, f"missing key {key!r} in audit: {r}"
    assert r["dry_run"] is True, r
    assert r["ops_run"] == ["lint", "decay"], r
    assert r["applied"] == [], "dry_run should never apply"
    return f"scanned={r['scanned']}, proposed={len(r['proposed'])}"


async def check_maintain_targeted(ctx: AppContext) -> str:
    """target_prefix narrows scan; events/ subtree should yield the
    one event folder created earlier in this suite."""
    r = decode(
        await ctx.app.run_job(
            "maintain",
            target_prefix="events/",
            dry_run=True,
        ),
    )
    assert isinstance(r, dict), r
    assert r["scanned"] >= 1, r  # at least the suite's own event folder
    return f"scanned={r['scanned']} under events/"


# ---------- ordered manifest -----------------------------------------


CHECKS: list[tuple[str, callable]] = [
    ("registry", check_registry),
    ("retrieve.basic", check_retrieve_basic),
    ("retrieve.anchored", check_retrieve_anchored),
    ("retrieve.topic_seeded", check_retrieve_topic_seeded),
    ("remember.log_create", check_remember_log_create),
    ("remember.log_append", check_remember_log_append),
    ("remember.log_refuse_distilled", check_remember_log_refuse_distilled),
    ("remember.distill_degraded", check_remember_distill_degraded),
    ("maintain.dry_run", check_maintain_dry_run),
    ("maintain.targeted", check_maintain_targeted),
]
