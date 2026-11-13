"""Compare chunking and file-store updates against an earlier upsert implementation.

Run from the checkout: python -m scripts.benchmark_keyword_upsert_delta
Use --baseline REF --repeats 9 --sections 20 80 --output /tmp/reme-delta.json
for reproducible comparisons. Setup, checkpoint writes and HTTP are excluded;
all source files and derived state are isolated in temporary workspaces.
"""

# pylint: disable=protected-access

import argparse
import ast
import asyncio
from contextlib import chdir
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time

import numpy as np
from loguru import logger
from reme.components.file_chunker import MarkdownFileChunker
from reme.components.file_store import LocalFileStore
from reme.components.file_store import local_file_store as store_module

REPO = Path(__file__).resolve().parents[1]
SOURCE_PATH = "reme/components/file_store/local_file_store.py"
DEFAULT_BASELINE = "1e51e0dc1b1effe60f7ee425298f3b6f8ef4cbd9"


def baseline_store(commit):
    """Load only the baseline upsert, sharing unchanged helpers and dependencies."""
    baseline_source = subprocess.check_output(
        ["git", "show", f"{commit}:{SOURCE_PATH}"],
        cwd=REPO,
        text=True,
    )
    tree = ast.parse(baseline_source)
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "LocalFileStore")
    method = next(item for item in cls.body if isinstance(item, ast.AsyncFunctionDef) and item.name == "upsert")
    namespace = dict(vars(store_module))
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<baseline upsert>", "exec"), namespace)
    baseline_method = namespace["upsert"]

    class BaselineStore(LocalFileStore):
        """Real local store with its baseline upsert method restored."""

        upsert = baseline_method

    return BaselineStore


def document(sections):
    """Build synthetic bilingual Markdown below the default AST section limit."""
    phrase = "alpha beta gamma memory search deployment recovery 缓存索引更新与检索保持一致。 "
    paragraph = phrase * (1500 // len(phrase.encode("utf-8")))
    return "".join(f"## Section {i:03d}\n\n" + (paragraph + "\n\n") * 9 for i in range(sections))


def summary(samples):
    """Keep raw timings and quartiles alongside the median."""
    return {
        "median_ms": statistics.median(samples),
        "p25_ms": float(np.percentile(samples, 25)),
        "p75_ms": float(np.percentile(samples, 75)),
        "samples_ms": samples,
    }


async def trial(store_cls, base_node, base_chunks, updated):
    """Seed a fresh index, time one update and collect correctness checks."""
    with tempfile.TemporaryDirectory(prefix="reme-delta-bench-") as tmp, chdir(tmp):
        chunker = MarkdownFileChunker()
        store = store_cls(name="bench", embedding_store="", tag_index="default")
        await chunker.start()
        await store.start()
        try:
            await store.upsert([(base_node.model_copy(deep=True), [c.model_copy(deep=True) for c in base_chunks])])
            index = store.keyword_index
            slots_before = len(index._doc_ids)
            counters = {"bm25_ms": 0.0, "tokenize_count": 0, "deleted_count": 0}
            original_add, original_delete, original_tokenize = index.add_docs, index.delete_docs, index._tokenize

            def tokenize(text):
                counters["tokenize_count"] += 1
                return original_tokenize(text)

            async def add(docs):
                start = time.perf_counter()
                try:
                    return await original_add(docs)
                finally:
                    counters["bm25_ms"] += (time.perf_counter() - start) * 1000

            async def delete(ids):
                counters["deleted_count"] += len(ids)
                start = time.perf_counter()
                try:
                    return await original_delete(ids)
                finally:
                    counters["bm25_ms"] += (time.perf_counter() - start) * 1000

            index.add_docs, index.delete_docs, index._tokenize = add, delete, tokenize
            path = Path("document.md")
            path.write_text(updated, encoding="utf-8")
            start = time.perf_counter()
            new_node, new_chunks = await chunker.chunk(path)
            parsed = time.perf_counter()
            await store.upsert([(new_node, new_chunks)])
            end = time.perf_counter()
            # Query tokenization is outside the measured update and its counters.
            index._tokenize = original_tokenize
            result = {
                **counters,
                "chunk_ms": (parsed - start) * 1000,
                "upsert_ms": (end - parsed) * 1000,
                "total_ms": (end - start) * 1000,
                "slots_before": slots_before,
                "slots_after": len(index._doc_ids),
                "new_chunks": len(new_chunks),
                "retained_chunks": len({c.id for c in new_chunks} & {c.id for c in base_chunks}),
                "live_ids": set(index.document_ids),
                "total_len": index.total_len,
                "scores": {
                    q: await index.retrieve(q, limit=100000)
                    for q in ("alpha", "beta gamma", "omega", "memory deployment")
                },
            }
            return result
        finally:
            await store.close()
            await chunker.close()


def verify(baseline, optimized):
    """Compare all live IDs, corpus length and complete query score maps."""
    assert baseline["live_ids"] == optimized["live_ids"]
    assert baseline["total_len"] == optimized["total_len"]
    assert baseline["new_chunks"] == optimized["new_chunks"]
    for query, expected in baseline["scores"].items():
        actual = optimized["scores"][query]
        assert actual.keys() == expected.keys()
        assert all(abs(actual[key] - score) <= 1e-6 for key, score in expected.items())


async def main():
    """Alternate baseline and optimized runs, then print comparable measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sections", type=int, nargs="+", default=[20, 80])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(count < 1 for count in args.sections):
        parser.error("repeats and section counts must be positive")
    args.baseline = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{args.baseline}^{{commit}}"],
        cwd=REPO,
        text=True,
    ).strip()
    baseline_cls = baseline_store(args.baseline)
    logger.remove()
    report = {
        "baseline_commit": args.baseline,
        "working_tree_sha256": hashlib.sha256((REPO / SOURCE_PATH).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "repeats": args.repeats,
        "warmup_pairs": 1,
        "config": {"chunk_byte_size": 10000, "max_ast_sections": 100, "embedding": False, "tag_index": True},
        "scope": (
            "real chunker + serialized store upsert, including graph/tags/BM25; "
            "excludes source writes, setup, checkpoint, HTTP"
        ),
        "results": [],
    }
    for sections in args.sections:
        base = document(sections)
        with tempfile.TemporaryDirectory(prefix="reme-delta-input-") as tmp, chdir(tmp):
            path = Path("document.md")
            path.write_text(base, encoding="utf-8")
            chunker = MarkdownFileChunker()
            await chunker.start()
            try:
                base_node, base_chunks = await chunker.chunk(path)
            finally:
                await chunker.close()
        tail_position = base.rfind("gamma")
        scenarios = {
            "tail_edit": base[:tail_position] + "omega" + base[tail_position + len("gamma") :],
            "head_insert": "Inserted operation notice.\n\n" + base,
            "full_rewrite": base.replace("alpha", "omega"),
            "no_change": base,
        }
        for scenario, updated in scenarios.items():
            samples = {"baseline": [], "optimized": []}
            for iteration in range(args.repeats + 1):
                order = [("baseline", baseline_cls), ("optimized", LocalFileStore)]
                if iteration % 2:
                    order.reverse()
                pair = {}
                for label, store_cls in order:
                    pair[label] = await trial(store_cls, base_node, base_chunks, updated)
                verify(pair["baseline"], pair["optimized"])
                if iteration:
                    for label, values in samples.items():
                        values.append(pair[label])
            row = {
                "input_bytes": len(base.encode("utf-8")),
                "sections": sections,
                "old_chunks": len(base_chunks),
                "scenario": scenario,
            }
            for label, values in samples.items():
                row[label] = {
                    metric: summary([v[metric] for v in values])
                    for metric in ("bm25_ms", "chunk_ms", "upsert_ms", "total_ms")
                }
                for metric in (
                    "tokenize_count",
                    "deleted_count",
                    "slots_before",
                    "slots_after",
                    "new_chunks",
                    "retained_chunks",
                ):
                    counts = {v[metric] for v in values}
                    assert len(counts) == 1, (label, metric, counts)
                    row[label][metric] = counts.pop()
            row["reduction_pct"] = {
                metric: (1 - row["optimized"][metric]["median_ms"] / row["baseline"][metric]["median_ms"]) * 100
                for metric in ("bm25_ms", "upsert_ms", "total_ms")
            }
            report["results"].append(row)
            if args.output:
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            compact = {
                "KiB": round(row["input_bytes"] / 1024, 1),
                "scenario": scenario,
                "chunks": len(base_chunks),
                "tokenize": [row[k]["tokenize_count"] for k in samples],
                "bm25_ms": [round(row[k]["bm25_ms"]["median_ms"], 3) for k in samples],
                "total_ms": [round(row[k]["total_ms"]["median_ms"], 3) for k in samples],
                "reduction_pct": {k: round(v, 1) for k, v in row["reduction_pct"].items()},
            }
            print(json.dumps(compact, ensure_ascii=False), flush=True)
    if args.output:
        print(f"Results: {args.output}", flush=True)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
