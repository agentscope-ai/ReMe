"""Evaluate frozen-summary Top-N curves on the predeclared fixture task panel."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import statistics

from dotenv import dotenv_values
from openai import AsyncOpenAI

from benchmark.compaction.cases import event_units, render, write
from benchmark.compaction.run import Experiment, RESERVE, compose, messages, top_n
from benchmark.compaction.suite import exam, grade
from reme.components.as_decision_model import AgentScopeAsDecisionModel

ARMS = ["full", "summary"] + [f"fusion_top{n}" for n in range(1, 6)]
LABELS = ["Full context", "S"] + [f"S + Jev Top {n}" for n in range(1, 6)]


def load_cases(corpus_path):
    """Project public chat fields, verify source hashes and exclude final collection reports."""
    manifest = json.loads(corpus_path.read_text())
    root = corpus_path.parent
    result = {}
    for definition in manifest["cases"]:
        task = root / definition["id"]
        source = task / "trajectory.full.json"
        meta = json.loads((task / "collection.json").read_text())
        if not meta["completed"] or hashlib.sha256(source.read_bytes()).hexdigest() != meta["trajectory_sha256"]:
            raise ValueError("Incomplete or changed collection: " + definition["id"])
        for name, sha in definition["source_hashes"].items():
            if hashlib.sha256((task / "workspace" / name).read_bytes()).hexdigest() != sha:
                raise ValueError("Changed source fixture")
        history = json.loads(source.read_text())["messages"]
        if history[-1]["role"] != "assistant" or history[-1].get("tool_calls"):
            raise ValueError("Expected the final collection report to exclude")
        allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
        history = [{k: v for k, v in m.items() if k in allowed} for m in history[:-1]]
        for m in history:
            if m.get("tool_calls"):
                m["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {k: c["function"][k] for k in ("name", "arguments")},
                    }
                    for c in m["tool_calls"]
                ]
        if [m["role"] for m in history[:2]] != ["system", "user"]:
            raise ValueError("Missing real system/task")
        calls = [c["id"] for m in history for c in m.get("tool_calls", []) or []]
        outputs = [m["tool_call_id"] for m in history if m["role"] == "tool"]
        if len(calls) != len(set(calls)) or sorted(calls) != sorted(outputs):
            raise ValueError("Unpaired or duplicate calls")
        case = {
            **definition,
            "messages": history,
            "exam": exam(definition),
            "trajectory_sha256": meta["trajectory_sha256"],
        }
        result[definition["id"]] = case
    return manifest, result


def metrics(rows, ids, repeats, include_native=False, include_native_fusion=False):
    """Equal task weights; invalid planned contexts remain zero instead of disappearing."""
    table = []
    arms = ARMS[:2] + ["hermes_summary"] + ARMS[2:] if include_native else ARMS
    labels = LABELS[:2] + ["Hermes small-window adapter"] + LABELS[2:] if include_native else LABELS
    if include_native_fusion:
        arms = ARMS + ["hermes_summary"] + [f"hermes_top{n}" for n in range(1, 6)]
        labels = LABELS + ["H (Hermes small-window)"] + [f"H + Jev Top {n}" for n in range(1, 6)]
    for arm, label in zip(arms, labels):
        task_facts = []
        task_exact = []
        tokens = []
        valid = 0
        for task_id in ids:
            group = [r for r in rows if r["case"] == task_id and r["arm"] == arm]
            task_facts.append(
                statistics.mean(
                    (
                        sum(sum(t["checks"].values()) / len(t["checks"]) for t in r["trials"]) / repeats
                        if r["valid"]
                        else 0
                    )
                    for r in group
                ),
            )
            task_exact.append(
                statistics.mean(
                    sum(all(t["checks"].values()) for t in r["trials"]) / repeats if r["valid"] else 0 for r in group
                ),
            )
            valid += sum(r["valid"] for r in group)
            tokens.extend(t["usage"]["prompt_tokens"] for r in group for t in r["trials"])
        table.append(
            {
                "arm": arm,
                "label": label,
                "facts": statistics.mean(task_facts),
                "exact": statistics.mean(task_exact),
                "input_tokens": statistics.mean(tokens) if tokens else None,
                "valid_contexts": valid,
                "planned_contexts": sum(r["case"] in ids and r["arm"] == arm for r in rows),
            },
        )
    return table


def report(output, rows, cases, repeats, include_native=False, include_native_fusion=False):
    """Keep primary and short/query-only controls in separate tables."""
    result = {}
    lines = ["# Fourteen-task fixed-summary study", ""]
    for cohort in ("primary", "controls"):
        ids = [k for k, c in cases.items() if (c["control_reason"] is None) == (cohort == "primary")]
        result[cohort] = metrics(rows, ids, repeats, include_native, include_native_fusion)
        lines += [
            f"## {cohort.title()} ({len(ids)} tasks)",
            "",
            "| Method | Mean input tokens | Fact score | All facts correct | Valid contexts |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for r in result[cohort]:
            tokens = f"{r['input_tokens']:.0f}" if r["input_tokens"] is not None else "—"
            lines.append(
                f"| {r['label']} | {tokens} | {r['facts']:.1%} | {r['exact']:.1%} | "
                f"{r['valid_contexts']}/{r['planned_contexts']} |",
            )
        lines += [""]
    result["per_task"] = {k: metrics(rows, [k], repeats, include_native, include_native_fusion) for k in cases}
    write(output / "aggregate.json", result)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


async def run(args):
    """Freeze all inputs across tasks before any scored answers; no outcome-based tuning."""
    from tokenizers import Tokenizer

    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists() or output == repo or repo in output.parents:
        raise ValueError("Use a new directory outside repository")
    manifest, cases = load_cases(args.corpus.resolve())
    original = args.corpus.read_bytes()
    values = dotenv_values(repo / ".env")
    for key in ("LLM_API_KEY", "LLM_BASE_URL", "DECISION_MODEL_API_KEY"):
        if not values.get(key):
            raise ValueError("Missing " + key)
    output.mkdir(parents=True)
    write(output / "corpus.json", manifest)
    for name in ("suite.py", "suite_run.py", "run.py", "cases.py", "README.md"):
        shutil.copy2(Path(__file__).parent / name, output / name)
    write(
        output / "protocol.json",
        {
            "corpus_sha256": hashlib.sha256(original).hexdigest(),
            "model": args.model,
            "compressions": args.compressions,
            "answer_repeats": args.repeats,
            "top_n": [1, 2, 3, 4, 5],
            "concurrency": args.concurrency,
            "summary_allowance": "20% of rendered evidence tokens, clamped to 64..256",
            "input_ceiling": (
                "protected context + five largest whole units + summary allowance + 96 reserve; "
                "adjusted before answers if serialized S needs more space"
            ),
            "cohorts": {k: c["control_reason"] for k, c in cases.items()},
            "judge": "recursive type-strict JSON field rules",
            "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
            "evaluation": "direct Qwen; no Hermes continuation",
        },
    )
    client = AsyncOpenAI(api_key=values["LLM_API_KEY"], base_url=values["LLM_BASE_URL"], timeout=90, max_retries=1)
    limit = asyncio.Semaphore(args.concurrency)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    experiments = {}
    rows = []
    credential = {
        "api_key": values["DECISION_MODEL_API_KEY"],
        "base_url": values.get("DECISION_MODEL_BASE_URL") or None,
    }
    model = values.get("DECISION_MODEL_NAME")
    model = None if model in (None, "", "null") else model
    try:
        async with AgentScopeAsDecisionModel(credential=credential, model=model, timeout=45) as decision:
            work = asyncio.Semaphore(args.concurrency)

            async def prepare_case(task_id, case):
                async with work:
                    root = output / task_id
                    root.mkdir()
                    exp = Experiment(client, decision, root, tokenizer, limit, args.model, 100000)
                    units = event_units(case["messages"])
                    allowance = max(64, min(256, int(exp.count(render(units)) * 0.2)))
                    write(root / "source.json", {**case, "units": units})
                    base = exp.input_tokens(messages(case, compose("", "")))
                    largest = sorted(
                        units,
                        key=lambda u: exp.input_tokens(messages(case, compose("", render([u])))) - base,
                        reverse=True,
                    )[:5]
                    cap = exp.input_tokens(messages(case, compose("", render(largest)))) + allowance + RESERVE + 32
                    prepared = []
                    for i in range(args.compressions):
                        scores, _ = await exp.scores(f"paired.c{i}", case["messages"][1]["content"], units)
                        summary, _ = await exp.summarize(
                            f"paired.c{i}",
                            case["messages"][1]["content"],
                            units,
                            allowance,
                        )
                        for arm in ARMS:
                            n = int(arm[-1]) if arm.startswith("fusion") else 0
                            chosen = top_n(units, scores, min(n, len(units)))
                            s = "" if arm == "full" else summary
                            payload = None if arm == "full" else compose(s, render(chosen))
                            context = messages(case, payload)
                            label = f"c{i}.{arm}"
                            write(root / f"{label}.context.json", {"messages": context})
                            prepared.append(
                                {
                                    "case": task_id,
                                    "arm": arm,
                                    "repeat": i,
                                    "summary": s,
                                    "selected": [u["id"] for u in chosen],
                                    "actual_n": len(chosen),
                                    "estimated_input_tokens": exp.input_tokens(context),
                                    "valid": False,
                                    "trials": [],
                                },
                            )
                    cap = max(cap, max(r["estimated_input_tokens"] + RESERVE for r in prepared if r["arm"] != "full"))
                    exp.cap = cap
                    experiments[task_id] = exp
                    for r in prepared:
                        r["input_cap"] = cap
                    rows.extend(prepared)
                    write(root / "budget.json", {"allowance": allowance, "input_cap": cap, "units": len(units)})
                    print("PREPARED", task_id, len(units), "units", "summary", allowance, "cap", cap, flush=True)

            async with asyncio.TaskGroup() as tasks:
                for key, case in cases.items():
                    tasks.create_task(prepare_case(key, case))
        rows.sort(key=lambda r: (list(cases).index(r["case"]), r["repeat"], ARMS.index(r["arm"])))
        write(output / "frozen-selections.json", rows)
        write(output / "results.json", rows)
        answer_limit = asyncio.Semaphore(args.concurrency)

        async def answer(row):
            async with answer_limit:
                exp = experiments[row["case"]]
                label = f"c{row['repeat']}.{row['arm']}"
                try:
                    context = json.loads((exp.output / f"{label}.context.json").read_text())["messages"]
                    trials = []
                    for j in range(args.repeats):
                        text, usage = await exp.ask(f"{label}.answer{j}", context)
                        if row["arm"] != "full" and usage["prompt_tokens"] > exp.cap:
                            raise ValueError("Actual input ceiling exceeded")
                        prediction = json.loads(text)
                        checks = grade(cases[row["case"]], prediction)
                        trials.append({"answer": prediction, "usage": usage, "checks": checks})
                    row.update(valid=True, trials=trials)
                except Exception as exc:
                    row.update(valid=False, error=type(exc).__name__, trials=[])
                write(output / "results.json", rows)
                report(output, rows, cases, args.repeats)
                print("ANSWERED", row["case"], label, "OK" if row["valid"] else row["error"], flush=True)

        async with asyncio.TaskGroup() as tasks:
            for row in rows:
                tasks.create_task(answer(row))
        assert args.corpus.read_bytes() == original
        load_cases(args.corpus.resolve())
        for file in output.rglob("*"):
            if file.is_file():
                data = file.read_bytes()
                assert all(values[k].encode() not in data for k in ("LLM_API_KEY", "DECISION_MODEL_API_KEY"))
        write(
            output / "validation.json",
            {
                "source_unchanged": True,
                "no_api_keys": True,
                "tasks": len(cases),
                "contexts": len(rows),
                "valid": sum(r["valid"] for r in rows),
                "answers": sum(len(r["trials"]) for r in rows),
            },
        )
    finally:
        await client.close()


def main():
    """Explicit live opt-in and bounded request concurrency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.8-max")
    parser.add_argument("--compressions", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live or min(args.compressions, args.repeats, args.concurrency) < 1:
        parser.error("Use --live and positive settings")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
