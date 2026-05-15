"""Expert-profile MCP tests.

Covers the post-refactor 11-tool surface (5 memory + 5 file + 1 graph)
plus the auxiliary services exposed by `reme2/config/expert.yaml`:

    Hot-write   sync
    Read        memory_search, memory_graph_search, memory_get
    Memory      memory_create, memory_update_body, memory_update_meta
    File        file_download, file_upload, file_delete, file_list,
                file_move
    Graph       graph_traverse
    Lint        check_dangling, check_orphans, check_collisions,
                check_schema

Each `check_*` is an `async def` that takes a populated `AppContext`,
runs one MCP job (or a small sequence), asserts on the response, and
returns a one-line summary string. The orchestrator runs them in the
listed order and collects pass/fail.
"""

from __future__ import annotations

from pathlib import Path

from ._helpers import AppContext, decode, wait_for_index


# Manifest of every tool the expert profile must expose. Used by
# `check_registry` and as the upper bound for `wait_for_index` budgets.
EXPECTED_JOBS: tuple[str, ...] = (
    # services
    "memory_search",
    "memory_graph_search",
    # memory primitives (5 — search counted above)
    "memory_get",
    "memory_create",
    "memory_update_body",
    "memory_update_meta",
    # file primitives (5)
    "file_download",
    "file_upload",
    "file_delete",
    "file_list",
    "file_move",
    # graph (1)
    "graph_traverse",
    # event (2)
    "event_open",
    "event_complete",
    # lint (4 atomic checks)
    "check_dangling",
    "check_orphans",
    "check_collisions",
    "check_schema",
)


# ---------- registry ---------------------------------------------------


async def check_registry(ctx: AppContext) -> str:
    missing = [j for j in EXPECTED_JOBS if j not in ctx.jobs]
    assert not missing, f"missing jobs: {missing}"
    extras = [j for j in ctx.jobs if j not in EXPECTED_JOBS]
    return f"{len(ctx.jobs)} jobs registered (extras: {extras or 'none'})"


# ---------- read primitives -------------------------------------------


async def check_memory_get(ctx: AppContext) -> str:
    alice = ctx.abs_path("topics", "Alice", "Alice.md")
    r = decode(await ctx.app.run_job("memory_get", path=alice))
    assert isinstance(r, dict) and r.get("exists") is True, r
    assert "metadata" in r, "missing metadata block"
    return f"exists=True, edges={len(r.get('link', []))}"


async def check_memory_search(ctx: AppContext) -> str:
    r = decode(
        await ctx.app.run_job(
            "memory_search",
            query="collaborates",
            max_results=3,
            min_score=0.0,
        ),
    )
    hits = r if isinstance(r, list) else (r.get("chunks") if isinstance(r, dict) else [])
    assert len(hits) > 0, r
    return f"{len(hits)} hits"


async def check_memory_graph_search(ctx: AppContext) -> str:
    r = decode(
        await ctx.app.run_job(
            "memory_graph_search",
            query="Alice",
            max_results=5,
            min_score=0.0,
            graph_depth=1,
        ),
    )
    hits = r if isinstance(r, list) else (r.get("chunks") if isinstance(r, dict) else [])
    assert len(hits) > 0, r
    hops = {h.get("graph_hop") for h in hits if isinstance(h, dict)}
    return f"{len(hits)} hits, hops seen={sorted(h for h in hops if h is not None)}"


async def check_lint_dangling(ctx: AppContext) -> str:
    """Atomic lint primitive — list FileLinks pointing at non-existent nodes.
    Seeded vault is healthy, so we just verify the envelope shape."""
    r = decode(await ctx.app.run_job("check_dangling"))
    assert isinstance(r, dict) and "count" in r and "findings" in r, r
    assert isinstance(r["findings"], list), r
    return f"dangling={r['count']}"


async def check_lint_orphans(ctx: AppContext) -> str:
    """Atomic lint primitive — list nodes with no inlinks AND no outlinks."""
    r = decode(await ctx.app.run_job("check_orphans"))
    assert isinstance(r, dict) and "count" in r and "paths" in r, r
    assert isinstance(r["paths"], list), r
    return f"orphans={r['count']}"


async def check_lint_collisions(ctx: AppContext) -> str:
    """Atomic lint primitive — basenames resolving to >1 path."""
    r = decode(await ctx.app.run_job("check_collisions"))
    assert isinstance(r, dict) and "count" in r and "groups" in r, r
    assert isinstance(r["groups"], dict), r
    return f"collisions={r['count']}"


async def check_lint_schema(ctx: AppContext) -> str:
    """Atomic lint primitive — frontmatter schema violations."""
    r = decode(await ctx.app.run_job("check_schema"))
    assert isinstance(r, dict) and "count" in r and "findings" in r, r
    assert isinstance(r["findings"], list), r
    return f"schema_violations={r['count']}"


# ---------- file primitives -------------------------------------------


async def check_file_list(ctx: AppContext) -> str:
    """list_files projection: filter by tag returns indexed memories."""
    r = decode(await ctx.app.run_job("file_list", tags=["person"]))
    assert isinstance(r, dict) and r.get("count", 0) >= 2, r
    paths = {item.get("path") for item in r.get("items", [])}
    assert any("Alice.md" in p for p in paths), paths
    return f"count={r['count']}"


# ---------- graph -----------------------------------------------------


async def check_graph_traverse_out(ctx: AppContext) -> str:
    """Outgoing traversal from Alice — matches the prior `memory_links` check."""
    alice = ctx.abs_path("topics", "Alice", "Alice.md")
    r = decode(
        await ctx.app.run_job(
            "graph_traverse",
            seeds=[alice],
            max_depth=1,
            direction="out",
        ),
    )
    assert isinstance(r, list) and len(r) >= 2, r
    return f"{len(r)} outgoing edges"


async def check_graph_traverse_in(ctx: AppContext) -> str:
    """Incoming traversal to Alice — matches the prior `memory_backlinks` check."""
    alice = ctx.abs_path("topics", "Alice", "Alice.md")
    r = decode(
        await ctx.app.run_job(
            "graph_traverse",
            seeds=[alice],
            max_depth=1,
            direction="in",
        ),
    )
    assert isinstance(r, list) and len(r) >= 2, r
    return f"{len(r)} incoming edges"


# ---------- hot-write: sync (create / append / refusal) ---------------


async def check_sync_create(ctx: AppContext) -> str:
    r = decode(
        await ctx.app.run_job(
            "sync",
            name="suite-event",
            description="full-profile suite",
            content="## ops\n- ran the suite\n",
            topics=["[[Alice]]"],
            tags=["suite"],
            materials=[
                {"filename": "raw-prompt.md", "content": "# user prompt\n\nrun suite\n"},
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
    index_text = (event_dir / "suite-event.md").read_text(encoding="utf-8")
    assert "## Materials" in index_text, "Materials footer absent"
    assert "raw-prompt.md" in index_text, "Materials footer missing raw-prompt link"
    assert "lifecycle: streaming" in index_text, "schema axis 'lifecycle' missing"
    assert "role: observation" in index_text, "schema axis 'role' missing"
    setattr(ctx, "_suite_event_dir", event_dir)
    setattr(ctx, "_suite_event_index", event_dir / "suite-event.md")
    await wait_for_index(ctx.watcher, expected_min=len(ctx.file_store))
    return f"created folder w/ {len(materials)} materials"


async def check_sync_append(ctx: AppContext) -> str:
    r = decode(
        await ctx.app.run_job(
            "sync",
            name="suite-event",
            content="## follow-up\n- second pass\n",
            topics=["[[Bob]]"],
            tags=["follow-up"],
            materials=[
                {"filename": "tool-output.txt", "content": "second run\n"},
                {"filename": "summary.md", "content": "# summary\n"},
            ],
        ),
    )
    assert isinstance(r, dict), r
    assert r.get("created") is False and r.get("action") == "appended", r
    appended = r.get("materials", [])
    names = {Path(p).name for p in appended}
    assert "tool-output-2.txt" in names, f"collision auto-suffix failed: {names}"
    assert "summary.md" in names, f"new material missing: {names}"
    index_text = getattr(ctx, "_suite_event_index").read_text(encoding="utf-8")
    assert "## ops" in index_text, "original body lost"
    assert "## Update —" in index_text, "Update section missing"
    assert "[[Bob]]" in index_text, "topic union failed"
    assert "tool-output-2.txt" in index_text, "Materials footer lost the collided file"
    return f"appended w/ collision → {sorted(names)}"


async def check_sync_refuse_distilled(ctx: AppContext) -> str:
    """Flip status via memory_update_meta (patch dict), then sync should refuse."""
    index_path = str(getattr(ctx, "_suite_event_index"))
    await ctx.app.run_job(
        "memory_update_meta",
        path=index_path,
        patch={"status": "distilled"},
    )
    r = decode(
        await ctx.app.run_job(
            "sync",
            name="suite-event",
            content="should be refused",
        ),
    )
    assert isinstance(r, dict) and "error" in r, r
    assert r.get("status") == "distilled", r
    assert r.get("suggested_name"), r
    return f"refused → suggested={r['suggested_name']!r}"


# ---------- raw memory_* writes ---------------------------------------


async def check_memory_create(ctx: AppContext) -> str:
    target = ctx.abs_path("topics", "Carol", "Carol.md")
    r = decode(
        await ctx.app.run_job(
            "memory_create",
            path=target,
            metadata={
                "title": "Carol",
                "lifecycle": "evolving",
                "scope": "class",
                "source": "curated",
                "role": "profile",
                "category": "profile",
                "tags": ["person"],
            },
            content="# Carol\n\nKnows [[Alice]].\n",
        ),
    )
    assert isinstance(r, dict) and r.get("created") is True, r
    assert "error" not in r, r
    assert Path(target).is_file(), target
    await wait_for_index(ctx.watcher, expected_min=len(ctx.file_store))
    return f"created {Path(target).name}"


async def check_memory_update_body(ctx: AppContext) -> str:
    carol = ctx.abs_path("topics", "Carol", "Carol.md")
    r = decode(
        await ctx.app.run_job(
            "memory_update_body",
            path=carol,
            old_string="Knows [[Alice]].",
            new_string="Knows [[Alice]] and [[Bob]].",
        ),
    )
    assert isinstance(r, dict) and r.get("replaced", 0) >= 1, r
    assert "Bob" in Path(carol).read_text(encoding="utf-8"), "edit not on disk"
    return f"body edit applied (replaced={r['replaced']})"


async def check_memory_update_meta(ctx: AppContext) -> str:
    carol = ctx.abs_path("topics", "Carol", "Carol.md")
    r = decode(
        await ctx.app.run_job(
            "memory_update_meta",
            path=carol,
            patch={"confidence": "✅"},
        ),
    )
    assert isinstance(r, dict) and "error" not in r, r
    applied = r.get("applied") or {}
    assert applied.get("confidence", {}).get("value") == "✅", r
    assert "confidence: ✅" in Path(carol).read_text(encoding="utf-8"), "property write not on disk"
    return "set confidence=✅"


# ---------- file_move / file_delete -----------------------------------


async def check_file_move(ctx: AppContext) -> str:
    """Move (rename) Carol.md inside its folder. Default update_refs=False."""
    src = ctx.abs_path("topics", "Carol", "Carol.md")
    dst = ctx.abs_path("topics", "Carol", "Carol-renamed.md")
    r = decode(
        await ctx.app.run_job(
            "file_move",
            src=src,
            dst=dst,
        ),
    )
    assert isinstance(r, dict) and r.get("ok") is True, r
    assert Path(dst).is_file(), dst
    assert not Path(src).exists(), f"old path still on disk: {src}"
    setattr(ctx, "_carol_path", dst)
    return "Carol.md → Carol-renamed.md"


async def check_file_delete(ctx: AppContext) -> str:
    target = getattr(ctx, "_carol_path", None) or ctx.abs_path("topics", "Carol", "Carol-renamed.md")
    r = decode(await ctx.app.run_job("file_delete", vault_path=target))
    assert isinstance(r, dict) and r.get("deleted") is True, r
    assert not Path(target).exists(), target
    return "removed"


# ---------- file_download / file_upload -------------------------------


async def check_file_upload_then_download(ctx: AppContext) -> str:
    """Upload a non-md attachment under topics/Alice/, then download it back."""
    import tempfile
    payload = b"ATTACHMENT-BYTES-FROM-SUITE"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp.write(payload)
        local_src = tmp.name
    vault_path = ctx.abs_path("topics", "Alice", "alice-attachment.bin")
    up = decode(
        await ctx.app.run_job(
            "file_upload",
            local_path=local_src,
            vault_path=vault_path,
        ),
    )
    assert isinstance(up, dict) and up.get("size") == len(payload), up
    assert Path(vault_path).is_file(), vault_path

    down = decode(
        await ctx.app.run_job("file_download", vault_path=vault_path),
    )
    assert isinstance(down, dict) and down.get("local_path"), down
    assert Path(down["local_path"]).read_bytes() == payload, "round-trip bytes mismatch"
    return f"upload+download {len(payload)}B"


# ---------- schema gates (P4) -----------------------------------------


async def check_schema_path_template_refuses(ctx: AppContext) -> str:
    """memory_create rejects paths outside topics/{X}/{Y}.md,
    events/{date}/{name}/..., or Archive/..."""
    target = ctx.abs_path("notes", "freeform.md")
    r = decode(
        await ctx.app.run_job(
            "memory_create",
            path=target,
            metadata={
                "title": "freeform",
                "lifecycle": "evolving",
                "scope": "class",
                "source": "curated",
                "role": "concept",
            },
            content="should be refused",
        ),
    )
    assert isinstance(r, dict), r
    assert "error" in r and "template" in r["error"].lower(), r
    assert not Path(target).exists(), "file shouldn't have been written"
    return "refused notes/freeform.md (no template)"


async def check_schema_path_template_force_bypass(ctx: AppContext) -> str:
    """force=True bypasses the template gate."""
    target = ctx.abs_path("notes", "forced.md")
    r = decode(
        await ctx.app.run_job(
            "memory_create",
            path=target,
            metadata={
                "title": "forced",
                "lifecycle": "evolving",
                "scope": "class",
                "source": "curated",
                "role": "concept",
            },
            content="forced through",
            force=True,
        ),
    )
    assert isinstance(r, dict) and r.get("created") is True, r
    assert Path(target).is_file(), target
    return "force=True bypassed"


async def check_schema_status_skip_refused(ctx: AppContext) -> str:
    """Suite-event ended distilled (sync.refuse_distilled flipped it).
    Trying distilled → active is reverse — must refuse."""
    event_index = getattr(ctx, "_suite_event_index", None)
    assert event_index is not None, "suite-event index not staged"
    r = decode(
        await ctx.app.run_job(
            "memory_update_meta",
            path=str(event_index),
            patch={"status": "active"},
        ),
    )
    assert isinstance(r, dict), r
    applied = r.get("applied") or {}
    err = (applied.get("status") or {}).get("error", "")
    assert "transition" in err.lower(), r
    return f"refused {applied['status'].get('prior')!r} → {applied['status'].get('requested')!r}"


async def check_schema_status_invalid_value(ctx: AppContext) -> str:
    """Random string for status is refused before any state-machine check."""
    event_index = getattr(ctx, "_suite_event_index", None)
    assert event_index is not None, "suite-event index not staged"
    r = decode(
        await ctx.app.run_job(
            "memory_update_meta",
            path=str(event_index),
            patch={"status": "bogus"},
        ),
    )
    assert isinstance(r, dict), r
    applied = r.get("applied") or {}
    err = (applied.get("status") or {}).get("error", "")
    assert "invalid" in err.lower(), r
    return "refused status='bogus'"


async def check_schema_status_force_bypass(ctx: AppContext) -> str:
    """force=True bypasses the state machine."""
    event_index = getattr(ctx, "_suite_event_index", None)
    assert event_index is not None, "suite-event index not staged"
    r = decode(
        await ctx.app.run_job(
            "memory_update_meta",
            path=str(event_index),
            patch={"status": "active"},
            force=True,
        ),
    )
    assert isinstance(r, dict), r
    assert "error" not in r, r
    decode(
        await ctx.app.run_job(
            "memory_update_meta",
            path=str(event_index),
            patch={"status": "distilled"},
            force=True,
        ),
    )
    return "force=True bypassed"


# ---------- ordered manifest -----------------------------------------


# (label, async fn) — runner executes top-to-bottom; later checks may
# rely on side effects from earlier ones.
CHECKS: list[tuple[str, callable]] = [
    ("registry", check_registry),
    ("memory_get", check_memory_get),
    ("file_list", check_file_list),
    ("memory_search", check_memory_search),
    ("memory_graph_search", check_memory_graph_search),
    ("graph_traverse.out", check_graph_traverse_out),
    ("graph_traverse.in", check_graph_traverse_in),
    ("lint.dangling", check_lint_dangling),
    ("lint.orphans", check_lint_orphans),
    ("lint.collisions", check_lint_collisions),
    ("lint.schema", check_lint_schema),
    ("sync.create", check_sync_create),
    ("sync.append", check_sync_append),
    ("sync.refuse_distilled", check_sync_refuse_distilled),
    ("memory_create", check_memory_create),
    ("memory_update_body", check_memory_update_body),
    ("memory_update_meta", check_memory_update_meta),
    ("file_upload+download", check_file_upload_then_download),
    ("file_move", check_file_move),
    ("file_delete", check_file_delete),
    ("schema.path_template_refuses", check_schema_path_template_refuses),
    ("schema.path_template_force", check_schema_path_template_force_bypass),
    ("schema.status_skip_refused", check_schema_status_skip_refused),
    ("schema.status_invalid_value", check_schema_status_invalid_value),
    ("schema.status_force", check_schema_status_force_bypass),
]
