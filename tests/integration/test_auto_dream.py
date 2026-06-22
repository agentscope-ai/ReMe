"""Integration test for the 4-step auto_dream job and proactive reader.

Runs against a real LLM. The test seeds a dream workspace, runs ``auto_dream`` for
2026-05-28, verifies digest/interests/catalog effects, then runs ``proactive``.
Agent messages and generated markdown/yaml/jsonl artifacts are copied to
``tests/integration/logs/auto_dream_latest/`` for manual inspection.
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path

import yaml

INTEGRATION_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = INTEGRATION_DIR / "logs" / "auto_dream_latest"
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _workspace_fixture import DREAM_INPUT_PATH, workspace_env  # noqa: E402

from reme.utils.jsonl_zst import read_jsonl_zst  # noqa: E402

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


def test_auto_dream_and_proactive():
    """Run auto_dream end to end, save transcripts/results, then read interests via proactive."""

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
                        topic_count=3,
                        topic_diversity_days=7,
                    )
                dumped = await recorder.dump()
                session_jsonl = sorted((env.workspace_dir / "session" / "agentscope").glob("*.jsonl"))
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
                assert interests.is_file(), f"interests.yaml missing: {interests}"
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
                provenance_links = [
                    rel for rel, text in target_texts.items() if f"derived_from:: [[{DREAM_INPUT_PATH}]]" in text
                ]
                print(f"[wikilink] integrated targets: {target_paths}")
                print(f"[wikilink] integrated targets with [[digest/...]] links: {digest_wikilinks}")
                print(f"[wikilink] integrated targets with derived_from source links: {provenance_links}")
                assert target_texts, f"no integrated target files found: {target_paths}"
                assert provenance_links, (
                    "no derived_from wikilink back to the changed daily note in integrated targets\n"
                    f"targets: {target_paths}"
                )
                assert digest_wikilinks, (
                    "no digest↔digest wikilink found in integrated target markdown\n" f"targets: {target_paths}"
                )

                interests_text = _print_text_file("interests.yaml", interests)
                interests_data = yaml.safe_load(interests_text) or {}
                topics = interests_data.get("topics") or []
                assert isinstance(topics, list) and topics, f"no topics in interests.yaml\n{interests_text}"

                proactive = await app.run_job("proactive", date=DREAM_DATE, include_content=True)
                assert proactive.success is True, f"proactive failed: {proactive.answer!r}"
                assert proactive.metadata.get("path") == f"daily/{DREAM_DATE}/interests.yaml"
                assert proactive.metadata.get("topics"), f"proactive returned no topics: {proactive.metadata!r}"

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
    test_auto_dream_and_proactive()
    print("\nIntegration test passed!")
