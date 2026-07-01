"""Quick test: ingest first 5 sessions of item 0, verify indexes are built."""

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from run import (
    format_messages_for_reme,
    load_eval_config,
    parse_haystack_date,
    sessions_sorted_by_time,
    should_trigger_dream,
    to_iso,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("test_5sessions")

_WORKSPACE_ROOT = _PROJECT_ROOT / "memory_workspaces"


async def main():
    eval_config = load_eval_config()
    dataset_path = _PROJECT_ROOT / eval_config["dataset"]["path"]
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    item = data[0]
    sorted_sessions = sessions_sorted_by_time(item)
    logger.info(f"Total sessions: {len(sorted_sessions)}")
    logger.info(f"First 5 sessions dates: {[s[1].isoformat() for s in sorted_sessions[:5]]}")

    # Check time format
    first_session = sorted_sessions[0]
    formatted = format_messages_for_reme(first_session[3], first_session[1])
    logger.info(f"Sample formatted message: {json.dumps(formatted[0], ensure_ascii=False, indent=2)}")

    # Prepare workspace
    item_dir = _WORKSPACE_ROOT / "item_0"
    workspace_dir = str(item_dir / ".reme")
    if item_dir.exists():
        shutil.rmtree(item_dir)
    item_dir.mkdir(parents=True, exist_ok=True)

    from reme import Application
    from reme.config import resolve_app_config
    from reme.enumeration import ComponentEnum

    reme_cfg = eval_config["reme"]
    cfg = resolve_app_config(
        config=reme_cfg["config"],
        workspace_dir=workspace_dir,
        log_to_console=True,
        log_to_file=False,
        enable_logo=False,
    )

    app = Application(**cfg)
    await app.start()

    try:
        # Ingest first 5 sessions
        for idx, (orig_idx, session_dt, session_id, messages) in enumerate(sorted_sessions[:5]):
            formatted_msgs = format_messages_for_reme(messages, session_dt)
            date_str = session_dt.strftime("%Y-%m-%d")
            logger.info(f"Ingesting session {idx+1}/5: id={session_id} date={date_str} msgs={len(formatted_msgs)}")
            logger.info(f"  created_at sample: {formatted_msgs[0]['created_at']}")

            resp = await app.run_job(
                "auto_memory",
                messages=formatted_msgs,
                session_id=session_id,
                date=date_str,
            )
            logger.info(f"  auto_memory result: success={resp.success} answer={resp.answer[:100] if resp.answer else ''}")

            # Index update after each session
            idx_resp = await app.run_job("index_update")
            logger.info(f"  index_update: success={idx_resp.success} metadata={idx_resp.metadata}")

        # Check workspace contents
        logger.info("\n" + "=" * 60)
        logger.info("WORKSPACE CONTENTS:")
        logger.info("=" * 60)
        for p in sorted(item_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(item_dir)
                size = p.stat().st_size
                logger.info(f"  {rel} ({size} bytes)")

        # Check if indexes exist
        reme_dir = item_dir / ".reme"
        metadata_dir = reme_dir / "metadata"
        logger.info(f"\nMetadata dir exists: {metadata_dir.exists()}")
        if metadata_dir.exists():
            for p in sorted(metadata_dir.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(metadata_dir)
                    size = p.stat().st_size
                    logger.info(f"  metadata/{rel} ({size} bytes)")

        # Test search
        logger.info("\n" + "=" * 60)
        logger.info("SEARCH TEST:")
        logger.info("=" * 60)
        test_query = item["question"]
        logger.info(f"Query: {test_query}")
        search_resp = await app.run_job("search", query=test_query, limit=5)
        logger.info(f"Search success: {search_resp.success}")
        logger.info(f"Search counts: {search_resp.metadata.get('counts', {})}")
        if search_resp.answer:
            logger.info(f"Search answer (first 500 chars):\n{search_resp.answer[:500]}")
        else:
            logger.info("Search returned NO answer!")

    finally:
        await app.close()


if __name__ == "__main__":
    asyncio.run(main())
