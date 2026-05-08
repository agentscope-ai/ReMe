"""Smoke-test reme2/config/full.yaml + reme2/config/curated.yaml.

Boots Application with each profile against a temporary vault, lists
the registered jobs, then calls a representative subset to confirm
end-to-end wiring (parser → file_store → memory_io / ingest /
memory_graph_search). The Ingestor degrades gracefully when no LLM is
configured — we exercise the degraded path so the test is hermetic.

Run:
    python reme2/config/smoke_test.py
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# Eager-import side-effect modules so all @R.register() decorators run.
import reme2  # noqa: E402,F401
import reme2.mcp.steps.memory_io  # noqa: E402,F401
import reme2.mcp.steps.memory_retriever  # noqa: E402,F401
import reme2.memory.ingestor  # noqa: E402,F401
import reme2.memory.summarizer  # noqa: E402,F401
import reme2.memory.maintainer  # noqa: E402,F401
import reme2.mcp.steps  # noqa: E402,F401

from reme2.application import Application  # noqa: E402
from reme2.config import parse_args  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parent
PROFILES = {
    "full": CONFIG_DIR / "full.yaml",
    "curated": CONFIG_DIR / "curated.yaml",
}


def _seed_vault(vault: Path) -> None:
    """Drop a few markdown files so reads return something."""
    (vault / "topics" / "Alice").mkdir(parents=True, exist_ok=True)
    (vault / "topics" / "Alice" / "Alice.md").write_text(
        "---\ntitle: Alice\ncategory: profile\ntags: [person]\n---\n"
        "# Alice\n\nAlice works on [[Project X]] with [[Bob]].\n"
        "[author:: [[Alice]]]\n",
        encoding="utf-8",
    )
    (vault / "topics" / "Bob").mkdir(parents=True, exist_ok=True)
    (vault / "topics" / "Bob" / "Bob.md").write_text(
        "---\ntitle: Bob\ncategory: profile\ntags: [person]\n---\n"
        "# Bob\n\nBob collaborates with [[Alice]] on [[Project X]].\n",
        encoding="utf-8",
    )
    (vault / "topics" / "Project X").mkdir(parents=True, exist_ok=True)
    (vault / "topics" / "Project X" / "Project X.md").write_text(
        "---\ntitle: Project X\ncategory: concept\n---\n"
        "# Project X\n\nA major initiative led by [[Alice]] with [[Bob]].\n",
        encoding="utf-8",
    )


async def _wait_for_index(watcher, expected_min: int, timeout_s: float = 15.0) -> None:
    last = -1
    stable_for = 0
    for _ in range(int(timeout_s / 0.25)):
        now = len(watcher.file_store)
        if now == last and now >= expected_min:
            stable_for += 1
            if stable_for >= 4:
                return
        else:
            stable_for = 0
            last = now
        await asyncio.sleep(0.25)


def _decode(resp) -> object:
    """Job answers are JSON strings; decode for inspection. Pass through dicts/lists."""
    if isinstance(resp.answer, (dict, list)):
        return resp.answer
    if isinstance(resp.answer, str):
        try:
            return json.loads(resp.answer)
        except (json.JSONDecodeError, TypeError):
            return resp.answer
    return resp.answer


async def _run_profile(name: str, config_path: Path) -> dict:
    """Boot the profile against a fresh temp vault; exercise jobs; return summary."""
    print(f"\n========== profile: {name} ==========")

    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "vault"
        vault.mkdir()
        _seed_vault(vault)

        # Override watch_path / db_path / sidecar info; force HTTP service so
        # we never block on stdio MCP (we only call jobs directly).
        _, cfg = parse_args(
            "start",
            f"config={config_path}",
            f"components.file_watcher.default.watch_path={vault}",
            f"components.file_store.default.db_path={vault}/.reme",
        )
        cfg["service"] = {"backend": "http"}

        app = Application(**cfg)
        await app.start()
        try:
            jobs = sorted(app.context.jobs.keys())
            print(f"  registered jobs ({len(jobs)}): {jobs}")

            watcher = app.context.components["file_watcher"]["default"]
            await _wait_for_index(watcher, expected_min=3)
            print(f"  file_store nodes after sync: {len(watcher.file_store)}")
            assert len(watcher.file_store) >= 3, "watcher did not index seed files"

            results: dict = {"jobs": jobs, "checks": []}
            alice_path = str((vault / "topics" / "Alice" / "Alice.md").resolve())

            if "memory_get" in jobs:
                r = _decode(await app.run_job("memory_get", path=alice_path))
                ok = isinstance(r, dict) and r.get("exists") is True
                print(f"  memory_get(Alice.md) → exists={ok}, edges={len(r.get('link', []))}")
                results["checks"].append(("memory_get", ok))

            if "memory_list" in jobs:
                r = _decode(await app.run_job("memory_list", tags=["person"]))
                ok = isinstance(r, dict) and r.get("count", 0) >= 2
                print(f"  memory_list(tags=[person]) → count={r.get('count') if isinstance(r, dict) else '?'}")
                results["checks"].append(("memory_list", ok))

            if "memory_search" in jobs:
                r = _decode(await app.run_job("memory_search", query="collaborates", max_results=3, min_score=0.0))
                hits = r if isinstance(r, list) else (r.get("chunks") if isinstance(r, dict) else [])
                ok = len(hits) > 0
                print(f"  memory_search('collaborates') → {len(hits)} hits")
                results["checks"].append(("memory_search", ok))

            if "memory_links" in jobs:
                r = _decode(await app.run_job("memory_links", path=alice_path))
                ok = isinstance(r, dict) and len(r.get("links", [])) >= 2
                print(f"  memory_links(Alice.md) → {len(r.get('links', []))} resolved")
                results["checks"].append(("memory_links", ok))

            if "query" in jobs:
                r = _decode(await app.run_job("query", query="Alice Bob", max_results=3, min_score=0.0))
                hits = r if isinstance(r, list) else (r.get("chunks") if isinstance(r, dict) else [])
                ok = len(hits) > 0
                print(f"  query('Alice Bob') → {len(hits)} hits")
                results["checks"].append(("query", ok))

            if "ingest" in jobs:
                # Degraded path: no LLM key set in env → Ingestor falls back
                # to direct create from `target_path`. That's the hermetic path.
                target = str((vault / "topics" / "smoke" / "smoke.md").resolve())
                r = _decode(await app.run_job(
                    "ingest",
                    content="# smoke topic\n\nrecorded by config smoke test.\n",
                    target_path=target,
                    metadata={"category": "concept", "title": "smoke"},
                ))
                ok = isinstance(r, dict) and (r.get("applied") or r.get("skipped"))
                print(f"  ingest(degraded create) → applied={len(r.get('applied', [])) if isinstance(r, dict) else '?'}, "
                      f"used_llm={r.get('used_llm') if isinstance(r, dict) else '?'}")
                results["checks"].append(("ingest", bool(ok)))
                # Confirm the file landed on disk.
                assert Path(target).is_file(), "ingest did not produce the target file"

            if "sync" in jobs:
                # CREATE call.
                r = _decode(await app.run_job(
                    "sync",
                    name="smoke-event",
                    description="smoke test event",
                    content="## ops\n- ran the smoke test\n",
                    topics=["[[Alice]]"],
                    tags=["smoke"],
                    materials=[
                        {"filename": "raw-prompt.md", "content": "# raw user prompt\n\nrun the smoke test\n"},
                        {"filename": "tool-output.txt", "content": "tool ran ok\nexit=0\n"},
                    ],
                ))
                ok = (
                    isinstance(r, dict)
                    and r.get("created") is True
                    and r.get("action") == "created"
                    and len(r.get("materials", [])) == 2
                )
                materials = r.get("materials", []) if isinstance(r, dict) else []
                print(f"  sync(create smoke-event w/ 2 materials) → created={r.get('created') if isinstance(r, dict) else '?'}, "
                      f"materials={len(materials)}")
                results["checks"].append(("sync.create", bool(ok)))
                for m in materials:
                    assert Path(m).is_file(), f"event material missing: {m}"
                if materials:
                    event_dir = Path(materials[0]).parent
                    index_text = (event_dir / "smoke-event.md").read_text(encoding="utf-8")
                    assert "## Materials" in index_text, "index .md missing Materials section"
                    assert "raw-prompt.md" in index_text, "Materials section missing raw-prompt link"
                await _wait_for_index(watcher, expected_min=len(watcher.file_store) + 2)

                # APPEND call: same name, new content + new + colliding material.
                r2 = _decode(await app.run_job(
                    "sync",
                    name="smoke-event",
                    content="## follow-up\n- second pass facts\n",
                    topics=["[[Bob]]"],  # union with prior [[Alice]]
                    tags=["follow-up"],   # union with prior [smoke]
                    materials=[
                        {"filename": "tool-output.txt", "content": "second tool run\nexit=0\n"},  # collision → auto-suffix
                        {"filename": "summary.md", "content": "# summary\nsecond pass\n"},
                    ],
                ))
                ok2 = (
                    isinstance(r2, dict)
                    and r2.get("created") is False
                    and r2.get("action") == "appended"
                    and len(r2.get("materials", [])) == 2
                )
                appended_paths = r2.get("materials", []) if isinstance(r2, dict) else []
                print(f"  sync(append smoke-event w/ collision) → action={r2.get('action') if isinstance(r2, dict) else '?'}, "
                      f"new_materials={len(appended_paths)}")
                results["checks"].append(("sync.append", bool(ok2)))
                # Collision should have produced tool-output-2.txt; summary.md untouched.
                names_appended = {Path(p).name for p in appended_paths}
                assert "tool-output-2.txt" in names_appended, f"collision auto-suffix missing: {names_appended}"
                assert "summary.md" in names_appended, f"clean filename missing: {names_appended}"
                # Index should now contain BOTH the original ops section and the Update section.
                if materials:
                    index_text2 = (Path(materials[0]).parent / "smoke-event.md").read_text(encoding="utf-8")
                    assert "## ops" in index_text2, "original content lost on append"
                    assert "## Update —" in index_text2, "missing Update section after append"
                    assert "follow-up" in index_text2, "appended content not in index"
                    # Frontmatter union check
                    assert "[[Bob]]" in index_text2, "topic union failed"
                    # Materials footer should now list 4 files (2 original + summary + tool-output-2)
                    assert "tool-output-2.txt" in index_text2, "Materials footer missing collided file"
                    assert "summary.md" in index_text2, "Materials footer missing new file"

                # REFUSAL: flip status to distilled, third sync should refuse.
                index_path = str(Path(materials[0]).parent / "smoke-event.md") if materials else None
                if index_path and "memory_property_update" in jobs:
                    await app.run_job("memory_property_update", path=index_path, key="status", value="distilled")
                    r3 = _decode(await app.run_job(
                        "sync",
                        name="smoke-event",
                        content="should refuse",
                    ))
                    ok3 = isinstance(r3, dict) and "error" in r3 and r3.get("status") == "distilled"
                    print(f"  sync(refuse on distilled) → error={'error' in (r3 if isinstance(r3, dict) else {})}, "
                          f"suggested_name={r3.get('suggested_name') if isinstance(r3, dict) else '?'}")
                    results["checks"].append(("sync.refuse_distilled", bool(ok3)))

            if "topic_create" in jobs:
                r = _decode(await app.run_job(
                    "topic_create",
                    folder="Carol",
                    name="Carol",
                    category="profile",
                    description="smoke test",
                    content="# Carol\n",
                    tags=["person"],
                ))
                ok = isinstance(r, dict) and (r.get("created") is True or "path" in r)
                print(f"  topic_create(Carol) → created={r.get('created') if isinstance(r, dict) else '?'}")
                results["checks"].append(("topic_create", bool(ok)))

            return results
        finally:
            await app.close()


async def _main() -> int:
    summary: dict[str, dict] = {}
    for name, path in PROFILES.items():
        try:
            summary[name] = await _run_profile(name, path)
        except Exception as e:
            print(f"  ✗ profile '{name}' failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            summary[name] = {"error": str(e)}

    print("\n========== summary ==========")
    failed = 0
    for name, result in summary.items():
        if "error" in result:
            print(f"  {name}: ERROR — {result['error']}")
            failed += 1
            continue
        checks = result.get("checks", [])
        passed = sum(1 for _, ok in checks if ok)
        total = len(checks)
        marker = "✓" if passed == total else "✗"
        print(f"  {marker} {name}: {passed}/{total} checks passed; jobs={len(result['jobs'])}")
        for label, ok in checks:
            if not ok:
                print(f"      ✗ {label}")
                failed += 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
