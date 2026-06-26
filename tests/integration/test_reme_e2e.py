"""End-to-end integration test for the full ReMe memory loop.

Unlike the per-job tests in this directory (which each exercise one job in
isolation), this drives the *whole* lifecycle through the public
``ReMe.run_job`` surface, in the order a real deployment runs it:

    1. provision   — ``auto_memory``  : a conversation becomes a daily note
    2. provision   — ``auto_memory``  : a follow-up updates the same note
    3. consolidate — ``auto_dream``   : today's daily notes become digest nodes
    4. recall      — ``reindex`` + ``search`` : a distinctive fact is retrievable
    5. proactive   — ``proactive``    : surfaced interest topics are readable

The point is to prove the stages *compose*: a fact spoken in step 1 must
survive provisioning, consolidation, and indexing, and still come back out
of ``search`` in step 4. Assertions are deliberately lenient (``>= N`` hit
counts, "any stage that carried the fact") so the test asserts the loop is
intact without pinning the LLM's exact wording.

Search runs BM25-only (the default ``file_store`` leaves ``embedding_store``
empty), so this needs LLM_API_KEY but not an embedding key. Requires
LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL_NAME) in the
environment or a .env file at the repo root. Hits the real LLM API.
"""

import asyncio
import sys
from pathlib import Path

INTEGRATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _workspace_fixture import workspace_env  # noqa: E402

SESSION_ID = "project-meridian-sync"

# A distinctive, self-contained topic — invented proper nouns ("Meridian",
# "WebTransport over QUIC") so a later search hit is unambiguously traceable
# back to this conversation rather than to model priors.
_TURN_1 = [
    {
        "name": "user",
        "role": "user",
        "content": (
            "记一下 Project Meridian 的架构决定:实时协同改用 CRDT,"
            "后端选 Yjs。今天 2026-06-20 定的。"
        ),
    },
    {
        "name": "assistant",
        "role": "assistant",
        "content": "好的,已记录:Project Meridian 实时协同采用 CRDT,后端 Yjs,2026-06-20 决定。",
    },
    {
        "name": "user",
        "role": "user",
        "content": (
            "动机是旧的 OT (operational transform) 方案在离线编辑合并时冲突太多,"
            "CRDT 的最终一致性更适合多端离线场景。"
        ),
    },
]

_TURN_2 = [
    {
        "name": "user",
        "role": "user",
        "content": (
            "Meridian 传输层更新:放弃 WebSocket,改走 WebTransport over QUIC。"
            "原因是 WebSocket 的 head-of-line blocking 在弱网下让 CRDT update 批量延迟。"
        ),
    },
    {
        "name": "assistant",
        "role": "assistant",
        "content": "明白,传输层从 WebSocket 切到 WebTransport (QUIC),解决队头阻塞。",
    },
    {
        "name": "user",
        "role": "user",
        "content": "下一步:2026-06-27 前完成 WebTransport 的 fallback 到 WebSocket 的降级逻辑。",
    },
]

# Facts seeded across the two conversation turns; the loop must carry enough
# of them through to the final search.
_TOPIC_FACTS = ("Meridian", "CRDT", "Yjs", "WebTransport", "QUIC", "WebSocket")


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _all_workspace_text(env) -> str:
    """Concatenate every daily note + digest node — the consolidated memory."""
    parts = [_read(p) for p in env.daily_notes()]
    parts += [_read(p) for p in env.digest_files()]
    return "\n\n".join(parts)


async def _run_loop(env, reme) -> None:
    """The 5-stage e2e body, factored out so a local driver can reuse it
    against a non-temp workspace (see run_reme_e2e_local.py)."""
    today = env.today
    print("\n" + "=" * 70)
    print("[setup] workspace_root =", env.workspace_dir)
    print("[setup] today          =", today)
    print("=" * 70)

    # ---- 1. provision: CREATE a daily note from turn 1 ----------
    create = await reme.run_job(
        "auto_memory",
        messages=_TURN_1,
        session_id=SESSION_ID,
    )
    assert create.success is True, f"auto_memory CREATE failed: {create.answer!r}"
    meta = create.metadata or {}
    assert meta.get("created") is True, f"expected created=True, got {meta!r}"
    note_rel = f"daily/{today}/{SESSION_ID}.md"
    assert meta.get("path") == note_rel, f"unexpected note path: {meta!r}"
    note_path = env.workspace_dir / note_rel
    assert note_path.is_file(), f"daily note not written: {note_path}"
    after_create = _read(note_path)
    print(f"\n[1/5 provision-create] {note_path} ({len(after_create)} bytes)\n{after_create}")
    for fact in ("CRDT", "Yjs"):
        assert fact in after_create, f"CREATE dropped fact {fact!r}\n{after_create}"

    # ---- 2. provision: UPDATE the same note from turn 2 ---------
    update = await reme.run_job(
        "auto_memory",
        messages=_TURN_2,
        session_id=SESSION_ID,
    )
    assert update.success is True, f"auto_memory UPDATE failed: {update.answer!r}"
    umeta = update.metadata or {}
    assert umeta.get("created") is False, f"expected created=False on UPDATE, got {umeta!r}"
    assert umeta.get("path") == note_rel, f"UPDATE wrote a different note: {umeta!r}"
    after_update = _read(note_path)
    print(f"\n[2/5 provision-update] {note_path} ({len(after_update)} bytes)\n{after_update}")
    # old facts survive, new facts land
    assert "CRDT" in after_update, f"UPDATE dropped pre-existing fact\n{after_update}"
    new_hits = [f for f in ("WebTransport", "QUIC", "WebSocket") if f in after_update]
    print(f"[2/5] new transport facts landed: {new_hits}")
    assert len(new_hits) >= 2, f"UPDATE only landed {new_hits!r}\n{after_update}"

    # ---- 3. consolidate: dream today's daily notes into digest --
    dream = await reme.run_job(
        "auto_dream",
        date=today,
        hint="Integration e2e: preserve Project Meridian CRDT/Yjs/WebTransport facts.",
        topic_count=3,
    )
    assert dream.success is True, f"auto_dream failed: {dream.answer!r}\n{dream.metadata!r}"
    dmeta = (dream.metadata or {}).get("dream") or {}
    print(f"\n[3/5 consolidate] dream summary: {dmeta!r}")
    assert dmeta.get("date") == today, f"dream ran for wrong date: {dmeta!r}"
    assert dmeta.get("files_changed", 0) >= 1, f"dream changed no files: {dmeta!r}"
    digest_paths = env.digest_files()
    assert digest_paths, "consolidation produced no digest nodes"
    consolidated = _all_workspace_text(env)
    carried = [f for f in _TOPIC_FACTS if f in consolidated]
    print(f"[3/5] facts present after consolidation: {carried}")
    print(f"[3/5] digest nodes: {[str(p.relative_to(env.workspace_dir)) for p in digest_paths]}")
    assert len(carried) >= 3, f"consolidation lost the topic; only {carried!r} survived"

    # ---- 4. recall: rebuild the index, then search it -----------
    reindex = await reme.run_job("reindex")
    assert reindex.success is True, f"reindex failed: {reindex.answer!r}"

    hit_facts: list[str] = []
    searched: list[str] = []
    for query in ("Project Meridian 实时协同传输层", "CRDT Yjs WebTransport"):
        result = await reme.run_job("search", query=query, limit=5)
        assert result.success is True, f"search failed for {query!r}: {result.answer!r}"
        answer = result.answer or ""
        results_meta = (result.metadata or {}).get("results") or []
        print(
            f"\n[4/5 recall] query={query!r} -> {len(results_meta)} hit(s)\n"
            f"{answer[:1500]}",
        )
        searched.append(query)
        hit_facts += [f for f in _TOPIC_FACTS if f in answer]
    hit_facts = sorted(set(hit_facts))
    print(f"[4/5] facts recalled via search across {searched}: {hit_facts}")
    assert hit_facts, (
        "search recalled none of the seeded facts — the provision->"
        "consolidate->index->search loop is broken"
    )

    # ---- 5. proactive: read the interests surfaced by the dream --
    proactive = await reme.run_job("proactive", date=today, include_content=True)
    assert proactive.success is True, f"proactive failed: {proactive.answer!r}"
    pmeta = proactive.metadata or {}
    assert pmeta.get("path") == f"daily/{today}/interests.yaml", f"unexpected interests path: {pmeta!r}"
    topics = pmeta.get("topics") or []
    print(f"\n[5/5 proactive] topics: {topics}")
    assert topics, f"proactive surfaced no interest topics: {pmeta!r}"

    print("\n" + "=" * 70)
    print("test_reme_e2e_full_loop passed")
    print("=" * 70)


def test_reme_e2e_full_loop():
    """provision -> consolidate -> recall, end to end, via the public job API."""

    async def run():
        with workspace_env() as env:
            reme = await env.make_reme()
            try:
                await _run_loop(env, reme)
            finally:
                await env.close_all()

    asyncio.run(run())


if __name__ == "__main__":
    print("=== ReMe end-to-end integration test ===")
    test_reme_e2e_full_loop()
    print("\nIntegration test passed!")
