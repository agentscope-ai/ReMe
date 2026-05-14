"""Expert-profile MCP tests.

Covers all 16 jobs registered by `reme2/config/expert.yaml`:

    Hot-write   sync
    Read        memory_search, memory_graph_search,
                memory_get, memory_list, memory_links,
                memory_backlinks, memory_resolve_wikilink,
                memory_count_tokens, memory_lint
    Raw write   memory_create, memory_update, memory_property_update,
                memory_rename, memory_delete, memory_archive

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
    "sync",
    "memory_search",
    "memory_graph_search",
    "memory_get",
    "memory_list",
    "memory_backlinks",
    "memory_links",
    "memory_resolve_wikilink",
    "memory_count_tokens",
    "memory_lint",
    "memory_create",
    "memory_update",
    "memory_property_update",
    "memory_rename",
    "memory_delete",
    "memory_archive",
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


async def check_memory_list(ctx: AppContext) -> str:
    r = decode(await ctx.app.run_job("memory_list", tags=["person"]))
    assert isinstance(r, dict) and r.get("count", 0) >= 2, r
    paths = {item.get("path") for item in r.get("items", [])}
    assert any("Alice.md" in p for p in paths), paths
    return f"count={r['count']}"


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
    # At least one result should carry a graph_hop annotation
    hops = {h.get("graph_hop") for h in hits if isinstance(h, dict)}
    return f"{len(hits)} hits, hops seen={sorted(h for h in hops if h is not None)}"


async def check_memory_links(ctx: AppContext) -> str:
    alice = ctx.abs_path("topics", "Alice", "Alice.md")
    r = decode(await ctx.app.run_job("memory_links", path=alice))
    assert isinstance(r, dict) and len(r.get("links", [])) >= 2, r
    return f"{len(r['links'])} resolved outgoing links"


async def check_memory_backlinks(ctx: AppContext) -> str:
    alice = ctx.abs_path("topics", "Alice", "Alice.md")
    r = decode(await ctx.app.run_job("memory_backlinks", path=alice))
    # Bob.md and Project X.md both link to [[Alice]]
    assert isinstance(r, dict) and len(r.get("backlinks", [])) >= 2, r
    return f"{len(r['backlinks'])} incoming backlinks"


async def check_memory_resolve_wikilink(ctx: AppContext) -> str:
    r = decode(await ctx.app.run_job("memory_resolve_wikilink", wikilink="Alice"))
    assert isinstance(r, dict) and r.get("exists") is True, r
    assert "Alice.md" in (r.get("path") or ""), r
    return "stem 'Alice' → Alice.md"


async def check_memory_count_tokens(ctx: AppContext) -> str:
    r = decode(
        await ctx.app.run_job(
            "memory_count_tokens",
            text="hello world from the smoke test",
        ),
    )
    assert isinstance(r, dict) and isinstance(r.get("tokens"), int), r
    assert r["tokens"] > 0, r
    return f"text → {r['tokens']} tokens"


async def check_memory_lint(ctx: AppContext) -> str:
    """Maintainer lint pass — read-only. The seeded vault is healthy
    (no broken wikilinks / schema violations), so we just verify the
    shell wires through and returns the expected envelope."""
    r = decode(await ctx.app.run_job("memory_lint"))
    assert isinstance(r, dict), r
    assert "scanned" in r and "findings" in r, r
    assert isinstance(r["findings"], list), r
    assert r["scanned"] >= len(ctx.file_store), r
    return f"scanned={r['scanned']}, findings={len(r['findings'])}"


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
    # 4-axis schema must be on disk
    assert "lifecycle: streaming" in index_text, "schema axis 'lifecycle' missing"
    assert "role: observation" in index_text, "schema axis 'role' missing"
    # Stash for downstream checks via the context (small mutation pattern).
    ctx.abs_path("__suite_event_dir__")  # noop; readable side-effect below
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
            topics=["[[Bob]]"],  # union with [[Alice]]
            tags=["follow-up"],
            materials=[
                {"filename": "tool-output.txt", "content": "second run\n"},  # collision
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
    index_path = str(getattr(ctx, "_suite_event_index"))
    await ctx.app.run_job(
        "memory_property_update",
        path=index_path,
        key="status",
        value="distilled",
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


async def check_memory_update(ctx: AppContext) -> str:
    carol = ctx.abs_path("topics", "Carol", "Carol.md")
    r = decode(
        await ctx.app.run_job(
            "memory_update",
            path=carol,
            old_string="Knows [[Alice]].",
            new_string="Knows [[Alice]] and [[Bob]].",
        ),
    )
    assert isinstance(r, dict) and r.get("replaced", 0) >= 1, r
    assert "Bob" in Path(carol).read_text(encoding="utf-8"), "edit not on disk"
    return f"body edit applied (replaced={r['replaced']})"


async def check_memory_property_update(ctx: AppContext) -> str:
    carol = ctx.abs_path("topics", "Carol", "Carol.md")
    r = decode(
        await ctx.app.run_job(
            "memory_property_update",
            path=carol,
            key="confidence",
            value="✅",
        ),
    )
    assert isinstance(r, dict) and "error" not in r, r
    assert r.get("key") == "confidence" and r.get("value") == "✅", r
    assert "confidence: ✅" in Path(carol).read_text(encoding="utf-8"), "property write not on disk"
    return "set confidence=✅"


async def check_memory_rename(ctx: AppContext) -> str:
    src = ctx.abs_path("topics", "Carol", "Carol.md")
    dst = ctx.abs_path("topics", "Carol", "Carol-renamed.md")
    r = decode(
        await ctx.app.run_job(
            "memory_rename",
            old_path=src,
            new_path=dst,
        ),
    )
    assert isinstance(r, dict) and "error" not in r, r
    assert r.get("new_path") and Path(r["new_path"]).is_file(), r
    assert not Path(src).exists(), f"old path still on disk: {src}"
    setattr(ctx, "_carol_path", r["new_path"])
    return "Carol.md → Carol-renamed.md"


async def check_memory_archive(ctx: AppContext) -> str:
    carol = getattr(ctx, "_carol_path", ctx.abs_path("topics", "Carol", "Carol-renamed.md"))
    r = decode(await ctx.app.run_job("memory_archive", path=carol))
    assert isinstance(r, dict) and r.get("archived") is True, r
    archived_path = r.get("new_path")
    assert archived_path and "Archive" in archived_path, r
    assert Path(archived_path).is_file(), archived_path
    setattr(ctx, "_carol_archived", archived_path)
    return f"→ {Path(archived_path).resolve().relative_to(ctx.vault.resolve())}"


async def check_memory_delete(ctx: AppContext) -> str:
    target = getattr(ctx, "_carol_archived", None) or ctx.abs_path("topics", "Carol", "Carol-renamed.md")
    r = decode(await ctx.app.run_job("memory_delete", path=target))
    assert isinstance(r, dict) and r.get("deleted") is True, r
    assert not Path(target).exists(), target
    return "removed"


# ---------- schema gates (P4) -----------------------------------------


async def check_schema_path_template_refuses(ctx: AppContext) -> str:
    """memory_create rejects paths outside topics/{X}/{Y}.md,
    events/{date}/{name}/..., or Archive/..."""
    target = ctx.abs_path("notes", "freeform.md")  # outside any template
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
            "memory_property_update",
            path=str(event_index),
            key="status",
            value="active",
        ),
    )
    assert isinstance(r, dict), r
    assert "error" in r and "transition" in r["error"].lower(), r
    assert r.get("prior") == "distilled", r
    return f"refused {r.get('prior')!r} → {r.get('requested')!r}"


async def check_schema_status_invalid_value(ctx: AppContext) -> str:
    """Random string for status is refused before any state-machine check."""
    event_index = getattr(ctx, "_suite_event_index", None)
    assert event_index is not None, "suite-event index not staged"
    r = decode(
        await ctx.app.run_job(
            "memory_property_update",
            path=str(event_index),
            key="status",
            value="bogus",
        ),
    )
    assert isinstance(r, dict), r
    assert "error" in r and "invalid" in r["error"].lower(), r
    return "refused status='bogus'"


async def check_schema_status_force_bypass(ctx: AppContext) -> str:
    """force=True bypasses the state machine — useful when the agent
    intentionally needs to step outside conventions."""
    event_index = getattr(ctx, "_suite_event_index", None)
    assert event_index is not None, "suite-event index not staged"
    r = decode(
        await ctx.app.run_job(
            "memory_property_update",
            path=str(event_index),
            key="status",
            value="active",
            force=True,
        ),
    )
    assert isinstance(r, dict), r
    assert "error" not in r, r
    # restore for any downstream checks
    decode(
        await ctx.app.run_job(
            "memory_property_update",
            path=str(event_index),
            key="status",
            value="distilled",
            force=True,
        ),
    )
    return "force=True bypassed"


# ---------- ordered manifest -----------------------------------------


# (label, async fn) — runner executes top-to-bottom; later checks may
# rely on side effects from earlier ones (e.g. sync.append needs
# sync.create to have run).
CHECKS: list[tuple[str, callable]] = [
    ("registry", check_registry),
    ("memory_get", check_memory_get),
    ("memory_list", check_memory_list),
    ("memory_search", check_memory_search),
    ("memory_graph_search", check_memory_graph_search),
    ("memory_links", check_memory_links),
    ("memory_backlinks", check_memory_backlinks),
    ("memory_resolve_wikilink", check_memory_resolve_wikilink),
    ("memory_count_tokens", check_memory_count_tokens),
    ("memory_lint", check_memory_lint),
    ("sync.create", check_sync_create),
    ("sync.append", check_sync_append),
    ("sync.refuse_distilled", check_sync_refuse_distilled),
    ("memory_create", check_memory_create),
    ("memory_update", check_memory_update),
    ("memory_property_update", check_memory_property_update),
    ("memory_rename", check_memory_rename),
    ("memory_archive", check_memory_archive),
    ("memory_delete", check_memory_delete),
    ("schema.path_template_refuses", check_schema_path_template_refuses),
    ("schema.path_template_force", check_schema_path_template_force_bypass),
    ("schema.status_skip_refused", check_schema_status_skip_refused),
    ("schema.status_invalid_value", check_schema_status_invalid_value),
    ("schema.status_force", check_schema_status_force_bypass),
]
