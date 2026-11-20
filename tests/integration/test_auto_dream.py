"""Integration test for the three-step auto_dream job.

Runs against a real LLM. The test seeds a dream workspace, runs ``auto_dream`` for
2026-05-28, and verifies digest/catalog effects without producing proactive output.
Agent messages and generated markdown/jsonl artifacts are copied to
``tests/integration/logs/auto_dream_latest/`` for manual inspection.
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path

INTEGRATION_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = INTEGRATION_DIR / "logs" / "auto_dream_latest"
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _provenance import targets_with_provenance  # noqa: E402
from _workspace_fixture import DREAM_INPUT_PATH, workspace_env  # noqa: E402

from reme.utils.files.jsonl_zst import read_jsonl_zst  # noqa: E402

DREAM_DATE = "2026-05-28"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _print_text_file(label: str, path: Path) -> str:
    text = _read_text(path)
    print("\n" + "=" * 70)
    print(f"[{label}] {path} ({len(text)} bytes)")
    print(text)
    print("=" * 70)
    return text


def _reset_artifacts() -> None:
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _copy_artifact(src: Path, label: str) -> Path | None:
    if not src.exists():
        return None
    rel = Path(label) / src.name if src.is_file() else Path(label)
    dst = ARTIFACT_DIR / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return dst


def _copy_outputs(env, message_files: list[Path]) -> list[Path]:
    copied: list[Path] = []
    for path in message_files:
        if dst := _copy_artifact(path, "messages"):
            copied.append(dst)
    for root in ("daily", "digest", "metadata", "session", "agent_logs"):
        if dst := _copy_artifact(env.workspace_dir / root, root):
            copied.append(dst)
    return copied


def _print_message_files(paths: list[Path]) -> None:
    for idx, path in enumerate(paths, 1):
        text = _read_text(path)
        print("\n" + "=" * 70)
        print(f"[messages] {idx}: {path} ({len(text)} bytes)")
        print(text[:6000])
        if len(text) > 6000:
            print("\n[truncated]")
        print("=" * 70)


def _all_digest_text(env) -> str:
    return "\n\n".join(_read_text(p) for p in env.digest_files())


def _file_graph_links(env) -> dict[str, list[dict]]:
    """Map ``path -> links`` from every ``metadata/file_graph/*.jsonl.zst``."""
    out: dict[str, list[dict]] = {}
    graph_dir = env.workspace_dir / "metadata" / "file_graph"
    if not graph_dir.is_dir():
        return out
    for graph_path in sorted(graph_dir.glob("*.jsonl.zst")):
        for line in read_jsonl_zst(graph_path):
            if not line.strip():
                continue
            node = json.loads(line)
            out[node.get("path", "")] = node.get("links") or []
    return out


def test_auto_dream():
    """Run auto_dream end to end and verify that it does not produce proactive interests."""

    async def run():
        _reset_artifacts()
        with workspace_env() as env:
            seeded = env.seed_dream_workspace()
            app = await env.make_app()
            message_files: list[Path] = []
            try:
                print("\n" + "=" * 70)
                print("[setup] workspace_root =", env.workspace_dir)
                print("[setup] date       =", DREAM_DATE)
                print("[setup] seeded     =", json.dumps(seeded, ensure_ascii=False, indent=2))
                print("=" * 70)

                before_digest = _all_digest_text(env)
                with env.record_agents(prefix="agent_dream") as recorder:
                    response = await app.run_job(
                        "auto_dream",
                        date=DREAM_DATE,
                        hint="Integration test: preserve SOC2, JWT kid, Redis current_kid, and small-PR facts.",
                    )
                dumped = await recorder.dump()
                session_jsonl = sorted((env.workspace_dir / "mem_session" / "agentscope").glob("*.jsonl"))
                message_files = [*dumped, *session_jsonl]
                _print_message_files(message_files)

                assert response.success is True, f"auto_dream failed: {response.answer!r}\n{response.metadata!r}"
                dream = (response.metadata or {}).get("dream") or {}
                assert dream.get("date") == DREAM_DATE
                assert dream.get("files_changed", 0) >= 1, dream
                assert dream.get("units"), f"extract produced no units: {dream!r}"
                assert dream.get("integrate_results"), f"integrate produced no results: {dream!r}"
                assert dream.get("checkpoint_paths"), f"finish did not checkpoint changed paths: {dream!r}"

                day_index = env.workspace_dir / "daily" / f"{DREAM_DATE}.md"
                changed_note = env.workspace_dir / DREAM_INPUT_PATH
                interests = env.workspace_dir / "daily" / DREAM_DATE / "interests.yaml"
                catalog = env.workspace_dir / "metadata" / "file_catalog" / "dream.jsonl.zst"
                assert changed_note.is_file(), f"changed note missing: {changed_note}"
                assert not interests.exists(), f"auto_dream unexpectedly wrote interests.yaml: {interests}"
                assert catalog.is_file(), f"dream catalog missing: {catalog}"

                after_digest = _all_digest_text(env)
                new_signal = [
                    needle
                    for needle in ("SOC2", "24", "current_kid", "Redis", "300", "next steps", "kid")
                    if needle in after_digest and needle not in before_digest
                ]
                print(f"[dream] new digest signals: {new_signal}")
                assert len(new_signal) >= 2, f"digest missed expected new signal\n--- digest ---\n{after_digest}"

                # wikilink: the seeded daily note cites existing digest nodes, so it
                # should have outbound wikilink edges in the file_graph, and the dream
                # integrate agents should emit provenance and digest↔digest wikilinks
                # in the markdown they just created or updated.
                graph_links = _file_graph_links(env)
                note_links = [
                    link.get("target_path") for link in graph_links.get(DREAM_INPUT_PATH, []) if isinstance(link, dict)
                ]
                print(f"[wikilink] {DREAM_INPUT_PATH} -> {note_links}")
                assert note_links, (
                    f"seeded daily note produced no wikilink out-edges in file_graph\n"
                    f"file_graph links: {graph_links}"
                )
                assert any(
                    str(target).lstrip("!").startswith("digest/") for target in note_links
                ), f"daily note did not link out to any digest node: {note_links}"

                target_paths = [
                    str(result.get("target_path") or "")
                    for result in dream.get("integrate_results", [])
                    if result.get("target_path")
                ]
                target_texts = {
                    rel: _read_text(env.workspace_dir / rel)
                    for rel in target_paths
                    if (env.workspace_dir / rel).is_file()
                }
                digest_wikilinks = [rel for rel, text in target_texts.items() if "[[digest/" in text]
                # integrate.yaml requires every `## Sources` entry to be a complete
                # sentence containing the wikilink and explicitly forbids bare
                # `- [[...]]` lines, so match the provenance link target itself
                # (with or without an alias) rather than a specific line form.
                # The shared wikilink grammar decides what counts as a link:
                # aliases, embeds, and anchors parse to the same target, while
                # unterminated `[[...` spans do not.
                source_links = targets_with_provenance(target_texts, DREAM_INPUT_PATH)
                print(f"[wikilink] integrated targets: {target_paths}")
                print(f"[wikilink] integrated targets with [[digest/...]] links: {digest_wikilinks}")
                print(f"[wikilink] integrated targets with source links: {source_links}")
                assert target_texts, f"no integrated target files found: {target_paths}"
                assert source_links, (
                    "no source wikilink back to the changed daily note in integrated targets\n"
                    f"targets: {target_paths}"
                )
                assert digest_wikilinks, (
                    "no digest↔digest wikilink found in integrated target markdown\n" f"targets: {target_paths}"
                )

                if day_index.is_file():
                    _print_text_file("day_index.md", day_index)
                else:
                    print(f"[day_index] skipped: no direct daily notes, so {day_index} was not created")
                _print_text_file("changed input.md", changed_note)
                for path in env.digest_files():
                    _print_text_file(f"digest {path.relative_to(env.workspace_dir)}", path)
                catalog_lines = list(read_jsonl_zst(catalog))
                print("\n" + "=" * 70)
                print(f"[dream catalog.jsonl.zst] {catalog} ({len(catalog_lines)} node(s))")
                print("".join(catalog_lines))
                print("=" * 70)
            finally:
                copied = _copy_outputs(env, message_files)
                print("\n" + "=" * 70)
                print(f"[artifacts] copied {len(copied)} item(s) to {ARTIFACT_DIR}")
                for path in copied:
                    print(f"[artifacts] {path}")
                print("=" * 70)
                await env.close_all()

    asyncio.run(run())


if __name__ == "__main__":
    print("=== auto_dream integration test ===")
    test_auto_dream()
    print("\nIntegration test passed!")
