"""Score forced native summaries alongside the fixed seven-method task panel."""

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

from dotenv import dotenv_values
from openai import AsyncOpenAI

from benchmark.compaction.cases import write
from benchmark.compaction.run import Experiment, compose, messages
from benchmark.compaction.suite import grade
from benchmark.compaction.suite_run import ARMS, load_cases, report


def validate_base(evaluation, corpus, cases):
    """Reject incompatible base panels before launching any paid native generation."""
    protocol = json.loads((evaluation / "protocol.json").read_text())
    expected = {
        "model": "qwen3.8-max",
        "compressions": 3,
        "answer_repeats": 2,
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
    }
    rows = json.loads((evaluation / "results.json").read_text())
    planned = {(key, arm, i) for key in cases for arm in ARMS for i in range(3)}
    actual = {(r["case"], r["arm"], r["repeat"]) for r in rows}
    if len(cases) != 14 or any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Native adapter requires the matching fourteen-task Qwen 3x2 panel")
    if actual != planned or len(rows) != len(planned):
        raise ValueError("Base evaluation must contain each planned context exactly once")
    return rows


def load_native_summaries(generation, source, native_source=None):
    """Verify cached provenance and allow only an explicitly identified exam-only amendment."""
    validation_path = generation / "validation.json"
    summaries_path = generation / "summaries.json"
    if not validation_path.is_file() or not summaries_path.is_file():
        raise ValueError("Native generation set is incomplete; preserve it and use a new path")
    native_source = native_source or source
    validation = json.loads(validation_path.read_text())
    if validation.get("source_sha256") != hashlib.sha256(native_source.read_bytes()).hexdigest():
        raise ValueError("Native summary source mismatch")
    saved = json.loads(source.read_text())
    native_saved = json.loads(native_source.read_text())
    # The held-out exam is excluded from native summary generation.
    if {k: v for k, v in native_saved.items() if k != "exam"} != {k: v for k, v in saved.items() if k != "exam"}:
        raise ValueError("Native source evidence changed")
    summaries = json.loads(summaries_path.read_text())
    if len(summaries) != 3 or {r["repeat"] for r in summaries} != set(range(3)):
        raise ValueError("Native generations are incomplete")
    return summaries


async def prepare_native(args, cases, repo):
    """Prepare a new isolated native generation set or validate an existing frozen set."""
    if args.native.exists():
        for key in cases:
            load_native_summaries(
                args.native / key / "generation",
                args.evaluation / key / "source.json",
                (args.native_evaluation / key / "source.json") if args.native_evaluation else None,
            )
        return
    if not args.hermes or not args.hermes_python:
        raise ValueError("New native generations require --hermes and --hermes-python")
    native = args.native.resolve()
    if native == repo or repo in native.parents:
        raise ValueError("Native artifacts must stay outside repository")
    native.mkdir(parents=True)
    for name in ("hermes_summary.py", "README.md"):
        shutil.copy2(Path(__file__).parent / name, native / name)
    limit = asyncio.Semaphore(4)

    async def worker(key):
        async with limit:
            root = native / key
            root.mkdir()
            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
            with (root / "worker.log").open("w", encoding="utf-8") as log:
                process = await asyncio.create_subprocess_exec(
                    str(args.hermes_python),
                    "-m",
                    "benchmark.compaction.hermes_summary",
                    "--source",
                    str((args.evaluation / key / "source.json").resolve()),
                    "--output",
                    str(root / "generation"),
                    "--hermes",
                    str(args.hermes.resolve()),
                    "--env-file",
                    str(repo / ".env"),
                    "--tokenizer",
                    str(args.tokenizer.resolve()),
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=log,
                )
                try:
                    code = await asyncio.wait_for(process.wait(), 600)
                except BaseException:
                    if process.returncode is None:
                        process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), 10)
                        except asyncio.TimeoutError:
                            process.kill()
                            await process.wait()
                    raise
            if code:
                raise RuntimeError("Native worker failed: " + key)

    async with asyncio.TaskGroup() as tasks:
        for key in cases:
            tasks.create_task(worker(key))


async def run(args):
    """Keep all native generations, including non-shrinking failures, in the denominator."""
    from tokenizers import Tokenizer

    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists() or output == repo or repo in output.parents:
        raise ValueError("Use a new output directory outside repository")
    _, cases = load_cases(args.corpus.resolve())
    original = validate_base(args.evaluation, args.corpus, cases)
    await prepare_native(args, cases, repo)
    rows = deepcopy(original)
    output.mkdir(parents=True)
    for name in ("suite_native.py", "hermes_summary.py", "suite.py", "README.md"):
        shutil.copy2(Path(__file__).parent / name, output / name)
    write(
        output / "protocol.json",
        {
            "base_evaluation": str(args.evaluation.resolve()),
            "native_generations": str(args.native.resolve()),
            "methods": 8,
            "tasks": 14,
            "generations": 3,
            "answers_per_context": 2,
            "total_planned_contexts": 336,
            "total_planned_answers": 672,
            "native_strategy": (
                "Hermes small-window adapter: legacy generator, S-matched 64..256 budget, "
                "hard word limit, native body-only carrier"
            ),
            "acceptance": "native actual API input < corresponding Full and <= the predeclared case ceiling",
            "no_full_fallback": True,
            "judge": "same strict JSON field grader; no LLM judge",
        },
    )
    native_rows = []
    for key, case in cases.items():
        root = output / key
        root.mkdir()
        generated = load_native_summaries(
            args.native / key / "generation",
            args.evaluation / key / "source.json",
            (args.native_evaluation / key / "source.json") if args.native_evaluation else None,
        )
        for entry in generated:
            i = entry["repeat"]
            full = next(r for r in original if r["case"] == key and r["repeat"] == i and r["arm"] == "full")
            full_tokens = full["trials"][0]["usage"]["prompt_tokens"]
            row = {
                "case": key,
                "arm": "hermes_summary",
                "repeat": i,
                "summary": entry["summary"],
                "selected": [],
                "valid": False,
                "trials": [],
                "generation_valid": entry["valid"],
                "native_budget": entry["native_budget"],
                "summary_tokens": entry["summary_tokens"],
                "input_cap": full["input_cap"],
                "full_input_tokens": full_tokens,
            }
            write(
                root / f"c{i}.hermes_summary.context.json",
                {"messages": messages(case, compose(entry["summary"] or "", ""))},
            )
            native_rows.append(row)
    rows.extend(native_rows)
    write(output / "frozen-selections.json", rows)
    write(output / "results.json", rows)
    values = dotenv_values(repo / ".env")
    client = AsyncOpenAI(api_key=values["LLM_API_KEY"], base_url=values["LLM_BASE_URL"], timeout=90, max_retries=1)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    limit = asyncio.Semaphore(4)
    work = asyncio.Semaphore(4)

    async def answer(row):
        async with work:
            root = output / row["case"]
            label = f"c{row['repeat']}.hermes_summary"
            exp = Experiment(client, None, root, tokenizer, limit, "qwen3.8-max", row["input_cap"])
            try:
                if not row["generation_valid"]:
                    raise ValueError("Native summary generation failed its body budget")
                context = json.loads((root / f"{label}.context.json").read_text())["messages"]
                trials = []
                for j in range(2):
                    text, usage = await exp.ask(f"{label}.answer{j}", context)
                    prediction = json.loads(text)
                    trials.append(
                        {"answer": prediction, "usage": usage, "checks": grade(cases[row["case"]], prediction)},
                    )
                row["trials"] = trials
                shorter = all(t["usage"]["prompt_tokens"] < row["full_input_tokens"] for t in trials)
                fits = all(t["usage"]["prompt_tokens"] <= row["input_cap"] for t in trials)
                row.update(
                    valid=shorter and fits,
                    error=(
                        None
                        if shorter and fits
                        else "not_shorter_than_full" if not shorter else "input_ceiling_exceeded"
                    ),
                )
            except Exception as exc:
                row.update(valid=False, error=type(exc).__name__, trials=[])
            write(output / "results.json", rows)
            report(output, rows, cases, 2, include_native=True)
            print(row["case"], label, "OK" if row["valid"] else row["error"], flush=True)

    try:
        async with asyncio.TaskGroup() as tasks:
            for row in native_rows:
                tasks.create_task(answer(row))
        write(
            output / "validation.json",
            {
                "planned_contexts": len(rows),
                "valid_contexts": sum(r["valid"] for r in rows),
                "actual_answers": sum(len(r["trials"]) for r in rows),
                "native_valid": sum(r["valid"] for r in native_rows),
                "native_failed": sum(not r["valid"] for r in native_rows),
            },
        )
    finally:
        await client.close()


def main():
    """A prepared, frozen native baseline must exist before scored calls are allowed."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "evaluation", "native", "tokenizer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--hermes", type=Path)
    parser.add_argument("--hermes-python", type=Path)
    parser.add_argument("--native-evaluation", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("Use --live to authorize provider calls")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
