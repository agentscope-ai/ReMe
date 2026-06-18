"""Integration test for ``daily_topics`` final output.

This test deliberately disables the agent wrapper for the step so it
uses the deterministic fallback selector instead of a live LLM.  The
assertions check the final response metadata and the note written under
``daily/<date>/session_agent_interests.md``.
"""

import asyncio
import datetime as dt
import sys
from pathlib import Path

import frontmatter

INTEGRATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _vault_fixture import vault_env  # noqa: E402

DAY = "2026-06-17"
SESSION_ID = "interests"


def _disable_background_jobs() -> dict:
    """Turn default background jobs into inert foreground jobs for this test."""
    return {
        "index_update_loop": {"backend": "base", "steps": []},
        "resource_watch_loop": {"backend": "base", "steps": []},
        "digest_watch_loop": {"backend": "base", "steps": []},
    }


def _daily_topics_job_override() -> dict:
    """Force ``daily_topics`` through the non-LLM fallback path."""
    return {
        "daily_topics": {
            "backend": "base",
            "steps": [
                {
                    "backend": "daily_topics_step",
                    "agent_wrapper": None,
                    "topic_count": 3,
                    "diversity_days": 7,
                    "session_id": SESSION_ID,
                },
            ],
        },
    }


def _candidates() -> list[dict]:
    return [
        {
            "title": "JWT key rotation",
            "reason": "The auth notes revisit key rotation cadence.",
            "evidence": "auth notes",
            "source_path": "daily/2026-06-17/auth.md",
            "keywords": ["jwt", "rotation"],
        },
        {
            "title": "Small PR review discipline",
            "reason": "The user repeatedly prefers small PRs to reduce review risk.",
            "evidence": "refactor notes",
            "source_path": "daily/2026-06-17/refactor.md",
            "keywords": ["review", "pr-size", "soc2"],
        },
        {
            "title": "Small PR review discipline",
            "reason": "Duplicate title should be collapsed by the selector.",
            "evidence": "duplicate",
            "keywords": ["duplicate"],
        },
        {
            "title": "Redis volatile-ttl for refresh tokens",
            "reason": "Refresh token eviction policy is an operational concern.",
            "evidence": "redis notes",
            "source_path": "daily/2026-06-17/redis.md",
            "keywords": ["redis", "refresh-token", "ttl"],
        },
        {"title": "invalid candidate without a reason"},
    ]


async def _run_daily_topics_output() -> None:
    previous_day = (dt.date.fromisoformat(DAY) - dt.timedelta(days=1)).isoformat()

    with vault_env() as env:
        env.seed_daily_note(
            f"session_agent_{SESSION_ID}",
            """---
name: interests
description: previous interests
---

# Interested Topics

1. **JWT key rotation**
   - reason: already covered yesterday
   - evidence: prior auth notes
""",
            date=previous_day,
        )

        app = await env.make_app(jobs={**_disable_background_jobs(), **_daily_topics_job_override()})
        try:
            resp = await app.run_job(
                "daily_topics",
                date=DAY,
                candidates=_candidates(),
                topic_count=2,
                diversity_days=2,
                session_id=SESSION_ID,
            )
            proactive_resp = await app.run_job("proactive", date=DAY, session_id=SESSION_ID)
        finally:
            await env.close_all()

        assert resp.success is True
        assert resp.answer == f"Wrote 2 interest topic(s) to daily/{DAY}/session_agent_{SESSION_ID}.md"
        assert resp.metadata["date"] == DAY
        assert resp.metadata["path"] == f"daily/{DAY}/session_agent_{SESSION_ID}.md"
        assert resp.metadata["candidates_seen"] == 4
        assert resp.metadata["used_llm"] is False
        assert resp.metadata["skipped"] is False
        assert [topic["title"] for topic in resp.metadata["topics"]] == [
            "Small PR review discipline",
            "Redis volatile-ttl for refresh tokens",
        ]

        note_path = env.vault_dir / resp.metadata["path"]
        assert note_path.is_file()
        post = frontmatter.loads(note_path.read_text(encoding="utf-8"))
        assert post.metadata["name"] == "interests"
        assert post.metadata["date"] == DAY
        assert post.metadata["topic_count"] == 2
        assert post.metadata["diversity_days"] == 2

        body = post.content
        assert "Small PR review discipline" in body
        assert "Redis volatile-ttl for refresh tokens" in body
        assert "JWT key rotation" not in body
        assert "invalid candidate" not in body
        assert body.count("Small PR review discipline") == 1
        assert "daily/2026-06-17/refactor.md" in body
        assert "daily/2026-06-17/redis.md" in body

        day_index = env.vault_dir / "daily" / f"{DAY}.md"
        assert day_index.is_file()
        assert f"daily/{DAY}/session_agent_{SESSION_ID}.md" in day_index.read_text(encoding="utf-8")

        assert proactive_resp.success is True
        assert proactive_resp.answer == f"Read 2 proactive topic(s) from daily/{DAY}/session_agent_{SESSION_ID}.md"
        assert proactive_resp.metadata["path"] == f"daily/{DAY}/session_agent_{SESSION_ID}.md"
        assert proactive_resp.metadata["topics"] == resp.metadata["topics"]
        assert "Small PR review discipline" in proactive_resp.metadata["content"]


def test_daily_topics_writes_correct_final_output() -> None:
    """Verify daily topics are selected, written, indexed, and readable."""
    asyncio.run(_run_daily_topics_output())


if __name__ == "__main__":
    asyncio.run(_run_daily_topics_output())
