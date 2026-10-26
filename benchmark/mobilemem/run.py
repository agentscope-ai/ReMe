"""Evaluate ReMe Image on MobileMem-Omni Visual Reasoning.

Official QA and overlap metrics: Copyright (c) 2026 ZJUNLP, MIT License.
The complete upstream notice is retained in README.md.
"""

import argparse
import asyncio
import base64
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
import io
import json
import math
import mimetypes
from pathlib import Path, PurePosixPath
import re

from PIL import Image, ImageOps, JpegImagePlugin
import yaml

from reme import Application
from reme.config import resolve_app_config

from benchmark.mobilemem._audit import (
    ImageInputError,
    USAGE_POLICY,
    error_record,
    judge_settings,
    mean_tokens,
    response_record,
)


def timestamp(value: str) -> str:
    """Preserve published start-time precision in AgentScope ISO timestamps."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()


def load_mobilemem(root: Path) -> list[dict]:
    """Keep complete raw trajectories, selecting only the native visual category."""
    root = root.expanduser().resolve()
    records = []
    for name in ("data.jsonl", "questions.jsonl"):
        rows = {}
        with (root / name).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    identity = str(row["uuid"])
                    if identity in rows:
                        raise ValueError(f"Duplicate MobileMem user: {identity}")
                    rows[identity] = row
        records.append(rows)
    histories, questions = records
    if set(histories) != set(questions):
        raise ValueError("Every MobileMem user must have a trajectory and question record")
    return [
        {
            "id": f"uuid_{identity}",
            "image_root": root / "image",
            "sessions": sorted(
                row["sessions"],
                key=lambda session: datetime.fromisoformat(timestamp(session["event_start_time"])),
            ),
            "questions": [
                question
                for question in questions[identity]["questions"]
                if question["question_type"] == "visual_reasoning"
            ],
        }
        for identity, row in histories.items()
    ]


def resolve_image(root: Path, source: str, aliases: dict) -> Path:
    """Resolve exact paths or unique published basename aliases inside image/."""
    root = root.resolve()
    relative = PurePosixPath(source)
    if relative.is_absolute() or ".." in relative.parts or "\\" in source or ":" in source:
        raise ValueError(f"Invalid MobileMem image path: {source}")
    parts = relative.parts[1:] if relative.parts[0] == "image" else relative.parts
    target = (root / Path(*parts)).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"MobileMem image escapes image/: {source}")
    if target.is_file():
        return target
    if not aliases:
        for path in root.rglob("*"):
            if path.is_file() and path.resolve().is_relative_to(root):
                if (mimetypes.guess_type(path)[0] or "").startswith("image/"):
                    aliases.setdefault(path.name, []).append(path.resolve())
    matches = aliases.get(relative.name, [])
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous MobileMem image: {source}")
    return matches[0]


def image_block(path: Path) -> dict:
    """Use original bytes, or the existing lossless MPO primary-picture conversion."""
    data = path.read_bytes()
    if len(data) > 50 * 1024 * 1024:
        raise ValueError(f"Oversized MobileMem image: {path.name}")
    opener = JpegImagePlugin.JpegImageFile if data.startswith(b"\xff\xd8\xff") else Image.open
    with opener(io.BytesIO(data)) as image:
        mime, mpo = image.get_format_mimetype(), "mp" in image.info
        if not mime or not mime.startswith("image/") or image.width * image.height > 50_000_000:
            raise ValueError(f"Invalid or oversized MobileMem image: {path.name}")
        image.verify()
    with opener(io.BytesIO(data)) as image:
        image.load()
        if mpo:
            with ImageOps.exif_transpose(image) as oriented, oriented.convert("RGB") as picture:
                buffer = io.BytesIO()
                picture.save(buffer, format="PNG")
                data, mime = buffer.getvalue(), "image/png"
    return {
        "type": "data",
        "source": {
            "type": "base64",
            "data": base64.b64encode(data).decode("ascii"),
            "media_type": mime,
        },
    }


def session_messages(session: dict, image_root: Path, aliases=None) -> list[dict]:
    """Convert original dialogue directly; annotations never enter the candidate."""
    aliases = {} if aliases is None else aliases
    messages = []
    for turn in session["dialogue"]:
        content = [{"type": "text", "text": turn.get("content", "")}]
        refs = turn.get("image_inline") or []
        refs = [refs] if isinstance(refs, str) else refs
        if turn["content_type"] == "image" and not refs:
            raise ValueError("MobileMem image turn has no image reference")
        for source in refs:
            try:
                image = image_block(resolve_image(image_root, source, aliases))
            except Exception as error:  # pylint: disable=broad-exception-caught
                raise ImageInputError(source, error) from error
            content.extend(
                [
                    {"type": "text", "text": f"[Image ID: {source}]"},
                    image,
                ],
            )
        messages.append(
            {
                "id": f"{session['session_id']}:{turn['turn']}",
                "name": turn["role"],
                "role": turn["role"],
                "content": content,
                "created_at": timestamp(session["event_start_time"]),
            },
        )
    return messages


def evidence_values(question: dict) -> list[str]:
    """Keep the official Raw2Locomo/MemBase evidence-string conversion."""
    return [
        str(item["explanation"]) if isinstance(item, dict) else item
        for item in question.get("evidence", [])
        if (isinstance(item, dict) and item.get("explanation")) or isinstance(item, str)
    ]


ANSWER_RULES = (
    "Answer Rules:\n"
    "1. If options are provided in the question, use the original option text as your answer.\n"
    "2. For multiple-choice questions (multiple-select), select all correct options.\n"
    "3. For open-ended questions, respond freely."
)


def query_prompt(question: str, context: str) -> str:
    """Use the published Text Memory QA template verbatim."""
    return (
        f"Question:\n{question}\nPlease answer the question based on the following memories:\n"
        f"{context}\n{ANSWER_RULES}"
    )


def tokenize(text: str) -> list[str]:
    """Official lowercase ASCII word and individual CJK-character tokens."""
    text = re.sub(r"[\u0000-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]", " ", str(text).lower())
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)


def score_answer(reference: str, prediction: str) -> dict[str, float]:
    """Official token-overlap F1 and BLEU-1."""
    pt, rt = tokenize(prediction), tokenize(reference)
    overlap = sum((Counter(pt) & Counter(rt)).values())
    f1 = 1.0 if not pt and not rt else 0.0
    if pt and rt and overlap:
        precision, recall = overlap / len(pt), overlap / len(rt)
        f1 = 2 * precision * recall / (precision + recall)
    bleu = overlap / len(pt) * (math.exp(1 - len(rt) / len(pt)) if len(pt) < len(rt) else 1) if pt else 0.0
    return {"f1": f1, "bleu_1": bleu}


_CURRENT = ContextVar("mobilemem_usage", default=None)
_KEYS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "noncached_input_tokens",
    "reasoning_tokens",
)


def normalized_usage(raw: dict) -> dict:
    """Keep absent cache fields unknown and avoid double-counting cache subsets."""
    incoming = raw.get("input_tokens", raw.get("prompt_tokens"))
    outgoing = raw.get("output_tokens", raw.get("completion_tokens"))
    total = raw.get("total_tokens")
    if total is None and incoming is not None and outgoing is not None:
        total = incoming + outgoing
    cached = raw.get(
        "cached_input_tokens",
        raw.get("cached_tokens", (raw.get("prompt_tokens_details") or {}).get("cached_tokens")),
    )
    return {
        "input_tokens": incoming,
        "output_tokens": outgoing,
        "total_tokens": total,
        "cached_input_tokens": cached,
        "noncached_input_tokens": incoming - cached if incoming is not None and cached is not None else None,
        "reasoning_tokens": raw.get(
            "reasoning_tokens",
            (raw.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        ),
    }


def aggregate_usage(records: list[dict]) -> dict:
    """Return totals and coverage; unavailable counters remain unknown."""
    return {
        key: (
            sum(row[key] for row in records if row.get(key) is not None)
            if any(row.get(key) is not None for row in records)
            else None
        )
        for key in _KEYS
    } | {
        "records": sum(row.get("records", 1) for row in records),
        "coverage": {
            key: sum(row.get("coverage", {}).get(key, int(row.get(key) is not None)) for row in records)
            for key in _KEYS
        },
    }


@contextmanager
def usage_scope():
    """Attribute each successful request to its invoking async task."""
    records = []
    token = _CURRENT.set(records)
    try:
        yield records
    finally:
        _CURRENT.reset(token)


def observe_usage(app) -> None:
    """Observe lifecycle-owned non-streaming clients, including the compressor."""
    for component in app.context.components.get("as_llm", {}).values():
        client = getattr(getattr(component.model, "client", None), "_client", None)
        if client is None or getattr(client, "_mobilemem_usage_observer", False):
            continue
        original = client.send

        async def send(request, *args, _send=original, **kwargs):
            response = await _send(request, *args, **kwargs)
            records = _CURRENT.get()
            if records is not None and response.status_code == 200 and response.is_stream_consumed:
                try:
                    raw = response.json()
                except (ValueError, UnicodeError):
                    raw = {}
                usage = raw.get("usage") if isinstance(raw, dict) else None
                records.append(normalized_usage(usage if isinstance(usage, dict) else {}))
            return response

        client.send = send
        client._mobilemem_usage_observer = True  # pylint: disable=protected-access


def observe_completion(app) -> None:
    """Keep native interrupted replies unscored; successful replies are unchanged."""
    for wrapper in app.context.components.get("agent_wrapper", {}).values():
        original = wrapper.reply

        async def reply(inputs, *args, _reply=original, **kwargs):
            result = await _reply(inputs, *args, **kwargs)
            last = result.get("last_message") or {}
            if str(last.get("finished_reason")) == "interrupted" or last.get("error") is not None:
                raise RuntimeError("Native Agent was interrupted; its state remains saved.")
            return result

        wrapper.reply = reply


def save(path: Path, value) -> None:
    """Write an evaluation artifact under the newly created output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def candidate_config(workspace: Path, model: str) -> dict:
    """Select one candidate model; behavioral defaults belong to the plugin preset."""
    config = resolve_app_config(
        config="mobilemem-benchmark",
        plugins=["mobilemem"],
        workspace_dir=str(workspace),
        log_config=False,
        enable_logo=False,
        log_to_console=False,
        log_to_file=True,
        components={"as_llm": {name: {"model": model} for name in ("default", "bench", "compressor")}},
    )
    config["components"].pop("as_embedding", None)
    config["components"].pop("embedding_store", None)
    return config


def judge_config(workspace: Path) -> dict:
    """Keep scoring separate from candidate memory, model and evidence tools."""
    return {
        "plugins": ["mobilemem-judge"],
        "service": {"backend": "http"},
        "workspace_dir": str(workspace),
        "enable_logo": False,
        "log_to_console": False,
        "log_to_file": True,
    }


def check_response(response):
    """Keep failed Jobs as technical errors rather than incorrect predictions."""
    if response.metadata.get("cancelled"):
        raise asyncio.CancelledError("ReMe Job cancelled")
    if not response.success:
        raise RuntimeError("ReMe Job failed; see the saved response artifact.")
    return response


async def close_application(app, directory, *, user_id=None):
    """Finish owned cleanup; keep its diagnostics separate from the primary failure."""
    if app is None:
        return
    closing = asyncio.create_task(app.close())
    try:
        await asyncio.shield(closing)
    except asyncio.CancelledError as cancellation:
        if closing.cancelled():
            save(directory / "cleanup-error.json", error_record(cancellation, stage="cleanup", user_id=user_id))
            return
        try:
            await closing
        except (Exception, asyncio.CancelledError) as error:  # pylint: disable=broad-exception-caught
            save(directory / "cleanup-error.json", error_record(error, stage="cleanup", user_id=user_id))
        raise
    except Exception as error:  # pylint: disable=broad-exception-caught
        save(directory / "cleanup-error.json", error_record(error, stage="cleanup", user_id=user_id))


async def gather_tasks(coroutines) -> None:
    """Wait for owned tasks, cancelling and joining siblings on failure or cancellation."""
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def evaluate_user(scene, directory: Path, model: str, judge, query_slots) -> None:
    """Ingest the complete chronological history into one fresh user workspace."""
    app, response, session_id = None, None, None
    stage = "startup"
    completed = 0
    memory_usage, aliases = [], {}
    try:
        app = Application(**candidate_config(directory / "workspace", model))
        await app.start()
        observe_usage(app)
        observe_completion(app)
        for ordinal, session in enumerate(scene["sessions"]):
            session_id, response = session["session_id"], None
            stage = "preprocessing"
            with usage_scope() as measured:
                try:
                    messages = await asyncio.to_thread(session_messages, session, scene["image_root"], aliases)
                    stage = "memory"
                    response = await app.run_job(
                        "auto_memory",
                        messages=messages,
                        session_id=session_id,
                        date=timestamp(session["event_start_time"])[:10],
                        include_images=True,
                        include_source_sessions=True,
                    )
                    check_response(response)
                finally:
                    save(
                        directory / "sessions" / f"{ordinal:05d}.json",
                        {
                            "session_id": session_id,
                            "response": response_record(response),
                            "usage": aggregate_usage(measured),
                        },
                    )
            memory_usage.extend(measured)
            print(f"{scene['id']}: session {ordinal + 1}/{len(scene['sessions'])}", flush=True)
        stage, session_id, response = "index", None, None
        with usage_scope() as measured:
            try:
                response = await app.run_job("index_update")
                check_response(response)
            finally:
                save(
                    directory / "index-response.json",
                    {"response": response_record(response), "usage": aggregate_usage(measured)},
                )
        memory_usage.extend(measured)
        # Compatibility marker: never write this for a partial history or failed index.
        save(directory / "memory_usage.json", aggregate_usage(memory_usage))

        async def answer_question(ordinal, question):
            nonlocal completed
            question_dir = directory / "questions" / f"{ordinal:04d}"
            answer_stage, answer_response = "answer", None
            try:
                async with query_slots:
                    with usage_scope() as measured:
                        try:
                            answer_response = await app.run_job(
                                "agentic_answer",
                                query=query_prompt(question["question"], "Use the available evidence tools."),
                                include_images=True,
                                include_source_sessions=True,
                                max_iters=30,
                            )
                            check_response(answer_response)
                        finally:
                            answer_record = {
                                "question_id": question["question_id"],
                                "response": response_record(answer_response),
                                "usage": aggregate_usage(measured),
                            }
                            save(question_dir / "answer-response.json", answer_record)
                    prediction = answer_response.answer
                    answer_stage, answer_response = "judge", None
                    try:
                        answer_response = await judge.run_job(
                            "answer_judge",
                            question=question["question"],
                            reference=question["answer"],
                            prediction=prediction,
                            evidence=evidence_values(question),
                        )
                        check_response(answer_response)
                    finally:
                        save(question_dir / "judge.json", response_record(answer_response))
                answer_stage = "metrics"
                metrics = score_answer(question["answer"], prediction)
                metrics["llm_judge"] = answer_response.metadata["llm_judge_score"]
                result = {
                    "question_id": question["question_id"],
                    "category": question["question_type"],
                    "question": question["question"],
                    "reference": question["answer"],
                    "prediction": prediction,
                    "metrics": metrics,
                    "usage": aggregate_usage(measured),
                    "judge_parse_default_zero": answer_response.metadata["parse_default_zero"],
                }
                save(question_dir / "result.json", result)
                completed += 1
                print(f"{scene['id']}: {question['question_id']} completed", flush=True)
            except (Exception, asyncio.CancelledError) as error:  # pylint: disable=broad-exception-caught
                save(
                    question_dir / "error.json",
                    error_record(
                        error,
                        stage=answer_stage,
                        user_id=scene["id"],
                        question_id=question["question_id"],
                        response=answer_response,
                    ),
                )
                if isinstance(error, asyncio.CancelledError):
                    raise
                print(f"{scene['id']}: {question['question_id']} failed ({type(error).__name__})", flush=True)

        stage, response = "answers", None
        await gather_tasks(answer_question(i, question) for i, question in enumerate(scene["questions"]))
    except (Exception, asyncio.CancelledError) as error:  # pylint: disable=broad-exception-caught
        save(
            directory / "error.json",
            error_record(error, stage=stage, user_id=scene["id"], session_id=session_id, response=response),
        )
        if isinstance(error, asyncio.CancelledError):
            raise
        print(f"{scene['id']}: {stage} failed ({type(error).__name__})", flush=True)
    finally:
        try:
            await close_application(app, directory, user_id=scene["id"])
        finally:
            save(directory / "summary.json", {"completed": completed, "expected": len(scene["questions"])})


def summarize_run(scenes, output: Path, model: str) -> dict:
    """Read each canonical receipt/result once, including a run interrupted after partial progress."""
    expected = sum(len(scene["questions"]) for scene in scenes)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in output.glob("users/*/questions/*/result.json")]
    sessions = [json.loads(path.read_text(encoding="utf-8")) for path in output.glob("users/*/sessions/*.json")]
    indexes = [json.loads(path.read_text(encoding="utf-8")) for path in output.glob("users/*/index-response.json")]
    memory = [row["usage"] for row in sessions + indexes]
    answers = [
        json.loads(path.read_text(encoding="utf-8")) for path in output.glob("users/*/questions/*/answer-response.json")
    ]
    answer_usage = [row["usage"] for row in answers]
    total = aggregate_usage(memory + answer_usage)
    summary = {
        "model": model,
        "completed": len(rows),
        "expected": expected,
        "metrics": {
            key: sum(row["metrics"][key] for row in rows) / len(rows) if rows else None
            for key in ("f1", "bleu_1", "llm_judge")
        },
        "memory_usage": aggregate_usage(memory),
        "answer_usage": aggregate_usage(answer_usage),
        "token_usage": total,
        "usage_policy": USAGE_POLICY,
        "cleanup_error_count": len(list(output.glob("users/*/cleanup-error.json")))
        + int((output / "cleanup-error.json").exists()),
        "usage_denominators": {"scheduled_users": len(scenes), "expected_questions": expected},
        "usage_counts": {
            "attempted_users": len(list(output.glob("users/*/summary.json"))),
            "attempted_sessions": len(sessions),
            "successful_sessions": sum(bool((row.get("response") or {}).get("success")) for row in sessions),
            "completed_ingestion_users": len(list(output.glob("users/*/memory_usage.json"))),
            "attempted_answers": len(answers),
            "successful_answers": sum(bool((row.get("response") or {}).get("success")) for row in answers),
            "scored_questions": len(rows),
        },
        "mean_tokens_per_user": mean_tokens(total, len(scenes)),
        "mean_tokens_per_question": mean_tokens(total, expected),
        "judge_parse_default_zero": sum(row["judge_parse_default_zero"] for row in rows),
    }
    save(output / "summary.json", summary)
    return summary


async def run(scenes, output: Path, model: str, user_concurrency: int, query_concurrency: int) -> bool:
    """Run independent users concurrently and save official metrics plus method cost."""
    judge = None
    user_slots, query_slots = asyncio.Semaphore(user_concurrency), asyncio.Semaphore(query_concurrency)
    try:
        judge = Application(**judge_config(output / "judge-workspace"))
        save(output / "judge-settings.json", judge_settings(judge))
        await judge.start()

        async def user_job(scene):
            async with user_slots:
                await evaluate_user(scene, output / "users" / scene["id"], model, judge, query_slots)

        await gather_tasks(user_job(scene) for scene in scenes)
    except (Exception, asyncio.CancelledError) as error:  # pylint: disable=broad-exception-caught
        save(output / "error.json", error_record(error, stage="run"))
        raise
    finally:
        try:
            await close_application(judge, output)
        finally:
            summary = summarize_run(scenes, output, model)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary["completed"] == summary["expected"] and summary["cleanup_error_count"] == 0


def main() -> int:
    """Create a fresh evaluation; the PR reproduces ReMe Image only."""
    defaults = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--user-concurrency", type=int, default=defaults["user_concurrency"])
    parser.add_argument("--query-concurrency", type=int, default=defaults["query_concurrency"])
    args = parser.parse_args()
    if min(args.user_concurrency, args.query_concurrency) < 1:
        parser.error("Concurrency must be positive")
    scenes = load_mobilemem(args.dataset)
    output = args.output.expanduser().resolve() / "mobilemem" / args.model
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "settings.json",
        {
            "dataset": str(args.dataset.expanduser().resolve()),
            "model": args.model,
            "candidate": "mobilemem-benchmark",
            "judge": "mobilemem-judge",
            "question_type": "visual_reasoning",
            "user_concurrency": args.user_concurrency,
            "query_concurrency": args.query_concurrency,
        },
    )
    return 0 if asyncio.run(run(scenes, output, args.model, args.user_concurrency, args.query_concurrency)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
