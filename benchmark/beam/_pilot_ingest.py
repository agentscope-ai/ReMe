"""Pilot: ingest a few BEAM sessions and inspect source-marker quality.

Temporary helper for prompt tuning — ingests the first N sessions of one
case into an isolated pilot workspace (never the real benchmark one).

Usage:
    python benchmark/beam/_pilot_ingest.py [case_id] [n_sessions]
"""

import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run import _PROJECT_ROOT, load_beam_chat, load_eval_config  # noqa: E402

PILOT_ROOT = _PROJECT_ROOT / "benchmark/memory_workspaces/beam_pilot"


async def main(case_id: str, n_sessions: int):
    from reme import Application
    from reme.config import resolve_app_config

    eval_config = load_eval_config()
    chat_size = eval_config["dataset"]["chat_size"]
    beam_root = _PROJECT_ROOT / eval_config["dataset"].get("beam_root", "benchmark/datasets/BEAM")
    chat_path = beam_root / "chats" / chat_size / case_id / "chat.json"

    case_dir = PILOT_ROOT / f"{chat_size}_{case_id}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    cfg = resolve_app_config(
        config=eval_config["reme"]["config"],
        workspace_dir=str(case_dir / ".reme"),
        log_to_console=False,
        log_to_file=False,
        enable_logo=False,
    )
    app = Application(**cfg)
    await app.start()
    try:
        sessions = load_beam_chat(chat_path, chat_size, case_id)[:n_sessions]
        for i, session in enumerate(sessions):
            print(f"[pilot] ingesting session {i + 1}/{len(sessions)}: "
                  f"{session['session_id']} msgs={len(session['messages'])}", flush=True)
            resp = await app.run_job(
                "auto_memory",
                messages=session["messages"],
                session_id=session["session_id"],
                date=session["date"],
            )
            print(f"[pilot] success={resp.success} chunks={resp.metadata.get('chunks')} "
                  f"answer={str(resp.answer)[:300]}", flush=True)
            await app.run_job("index_update")
    finally:
        await app.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "1",
                     int(sys.argv[2]) if len(sys.argv) > 2 else 2))
