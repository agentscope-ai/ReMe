"""Reuse frozen Hermes summaries and Jev selections; score only H + Top 1..5."""

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from dotenv import dotenv_values
from openai import AsyncOpenAI

from benchmark.compaction.cases import render, write
from benchmark.compaction.run import Experiment, compose, messages, top_n
from benchmark.compaction.suite import grade
from benchmark.compaction.suite_native import load_native_summaries, validate_base
from benchmark.compaction.suite_run import ARMS, load_cases, report


def prepare(args, cases):
    """Verify artifacts and freeze all paired inputs before any new answer call."""
    original = validate_base(args.evaluation, args.corpus, cases)
    combined = json.loads((args.combined / "results.json").read_text())
    expected = {(k, a, i) for k in cases for a in ARMS + ["hermes_summary"] for i in range(3)}
    actual = {(r["case"], r["arm"], r["repeat"]) for r in combined}
    if actual != expected or len(combined) != len(expected):
        raise ValueError("Expected the complete frozen eight-method panel")
    if [r for r in combined if r["arm"] != "hermes_summary"] != original:
        raise ValueError("Combined panel changed the original seven-method results")
    index = {(r["case"], r["arm"], r["repeat"]): r for r in combined}
    prepared = []
    for key, case in cases.items():
        root = args.evaluation / key
        source = root / "source.json"
        saved = json.loads(source.read_text())
        if saved["messages"] != case["messages"] or saved["exam"] != case["exam"] or saved["gold"] != case["gold"]:
            raise ValueError("Corpus or evaluation questions changed")
        generation = args.native / key / "generation"
        native_source = (args.native_evaluation or args.evaluation) / key / "source.json"
        summaries = load_native_summaries(generation, source, native_source)
        for entry in summaries:
            i = entry["repeat"]
            native = index[key, "hermes_summary", i]
            if not entry["valid"] or not native["valid"] or native["summary"] != entry["summary"]:
                raise ValueError("Native summary is invalid or changed")
            scores = {}
            for path in sorted(root.glob(f"paired.c{i}.decision*.response.json")):
                scores.update(json.loads(path.read_text())["scores"])
            full_tokens = index[key, "full", i]["trials"][0]["usage"]["prompt_tokens"]
            for n in range(1, 6):
                paired = index[key, f"fusion_top{n}", i]
                chosen = top_n(saved["units"], scores, min(n, len(saved["units"])))
                if paired["selected"] != [u["id"] for u in chosen]:
                    raise ValueError("Frozen selection differs from recorded Jev ranking")
                context = messages(case, compose(entry["summary"], render(chosen)))
                prepared.append(
                    {
                        "case": key,
                        "arm": f"hermes_top{n}",
                        "repeat": i,
                        "summary": entry["summary"],
                        "selected": paired["selected"],
                        "actual_n": len(chosen),
                        "input_cap": paired["input_cap"],
                        "full_input_tokens": full_tokens,
                        "context": context,
                        "valid": False,
                        "trials": [],
                    },
                )
    return deepcopy(combined), prepared


async def run(args):
    """No summary or decision requests; every new trial only calls the answering model."""
    from tokenizers import Tokenizer

    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists() or output == repo or repo in output.parents:
        raise ValueError("Use a new output directory outside repository")
    _, cases = load_cases(args.corpus)
    rows, prepared = prepare(args, cases)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    values = dotenv_values(repo / ".env")
    if not values.get("LLM_API_KEY") or not values.get("LLM_BASE_URL"):
        raise ValueError("Missing Qwen credentials")
    output.mkdir(parents=True)
    for name in ("suite_fusion.py", "suite_run.py", "README.md"):
        shutil.copy2(Path(__file__).parent / name, output / name)
    write(
        output / "protocol.json",
        {
            "base_evaluation": str(args.evaluation.resolve()),
            "combined_evaluation": str(args.combined.resolve()),
            "native_generations": str(args.native.resolve()),
            "native_source_evaluation": str((args.native_evaluation or args.evaluation).resolve()),
            "model": "qwen3.8-max",
            "methods": 13,
            "tasks": 14,
            "generations": 3,
            "answers_per_context": 2,
            "new_contexts": 210,
            "new_answers": 420,
            "summary_calls": 0,
            "decision_calls": 0,
            "acceptance": "same existing input ceilings; mark strict reduction separately without filtering tasks",
            "ranking": "identical frozen Jev scores and selected originals for S and H",
            "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        },
    )
    for row in prepared:
        root = output / row["case"]
        root.mkdir(exist_ok=True)
        label = f"c{row['repeat']}.{row['arm']}"
        write(root / f"{label}.context.json", {"messages": row.pop("context")})
    rows.extend(prepared)
    write(output / "frozen-selections.json", rows)
    write(output / "results.json", rows)
    client = AsyncOpenAI(api_key=values["LLM_API_KEY"], base_url=values["LLM_BASE_URL"], timeout=90, max_retries=1)
    limit = asyncio.Semaphore(4)
    work = asyncio.Semaphore(4)

    async def answer(row):
        async with work:
            root = output / row["case"]
            label = f"c{row['repeat']}.{row['arm']}"
            exp = Experiment(client, None, root, tokenizer, limit, "qwen3.8-max", row["input_cap"])
            try:
                context = json.loads((root / f"{label}.context.json").read_text())["messages"]
                trials = []
                for j in range(2):
                    text, usage = await exp.ask(f"{label}.answer{j}", context)
                    prediction = json.loads(text)
                    trials.append(
                        {"answer": prediction, "usage": usage, "checks": grade(cases[row["case"]], prediction)},
                    )
                row["trials"] = trials
                row["valid"] = all(t["usage"]["prompt_tokens"] <= row["input_cap"] for t in trials)
                row["shorter_than_full"] = all(t["usage"]["prompt_tokens"] < row["full_input_tokens"] for t in trials)
                if not row["valid"]:
                    row["error"] = "input_ceiling_exceeded"
            except Exception as exc:
                row.update(valid=False, error=type(exc).__name__, trials=[])
            write(output / "results.json", rows)
            report(output, rows, cases, 2, include_native_fusion=True)
            print(row["case"], label, "OK" if row["valid"] else row["error"], flush=True)

    try:
        async with asyncio.TaskGroup() as tasks:
            for row in prepared:
                tasks.create_task(answer(row))
    finally:
        await client.close()
    load_cases(args.corpus)
    for path in output.rglob("*.json"):
        if values["LLM_API_KEY"].encode() in path.read_bytes():
            raise ValueError("Configured API key in artifacts")
    write(
        output / "validation.json",
        {
            "contexts": len(rows),
            "answers": sum(len(r["trials"]) for r in rows),
            "new_valid_contexts": sum(r["valid"] for r in prepared),
            "new_contexts_shorter_than_full": sum(r.get("shorter_than_full", False) for r in prepared),
            "no_new_summary_or_decision_calls": True,
            "no_api_keys": True,
        },
    )


def main():
    """Require explicit live opt-in; all artifacts must already exist except output."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "evaluation", "combined", "native", "tokenizer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--native-evaluation", type=Path)
    args = parser.parse_args()
    if not args.live:
        parser.error("Use --live to authorize the 420 answering calls")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
