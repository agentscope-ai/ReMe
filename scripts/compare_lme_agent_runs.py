#!/usr/bin/env python3
"""Compare two LongMemEval agent runs and their offloaded tool results."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


ANSWER_PATTERNS = [
    re.compile(r"45\s+minutes\s+each\s+way", re.I),
    re.compile(r"daily\s+commute[^.]{0,120}45\s+minutes", re.I | re.S),
    re.compile(r"audiobooks?[^.]{0,120}commute", re.I | re.S),
]


@dataclass
class ToolCallSummary:
    call_id: str
    name: str
    query: str
    limit: Any
    order: int
    result_file: str | None
    result_docs: int
    sessions: list[str]
    has_answer_flag: bool
    has_answer_text: bool
    answer_doc_ranks: list[int]
    evidence_snippets: list[str]


@dataclass
class RunSummary:
    session_id: str
    history_file: str
    final_answer: str
    tool_calls: list[ToolCallSummary]
    draft_texts: list[str]

    @property
    def found_answer(self) -> bool:
        return any(call.has_answer_flag or call.has_answer_text for call in self.tool_calls) or any(
            matches_answer(text) for text in self.draft_texts + [self.final_answer]
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_tool_input(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def iter_content_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        content = row.get("content")
        if isinstance(content, list):
            items.extend(item for item in content if isinstance(item, dict))
    return items


def extract_final_answer(items: list[dict[str, Any]]) -> str:
    texts = [item.get("text", "") for item in items if item.get("type") == "text" and item.get("text")]
    return texts[-1] if texts else ""


def extract_drafts(items: list[dict[str, Any]]) -> list[str]:
    drafts = []
    for item in items:
        if item.get("type") == "tool_result" and item.get("name") in {"add_draft", "read_all_draft"}:
            for output in item.get("output", []):
                if isinstance(output, dict) and output.get("text"):
                    drafts.append(output["text"])
    return drafts


def parse_concatenated_json(text: str) -> list[Any]:
    docs = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            doc, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        docs.append(doc)
        idx = end
    return docs


def read_result_docs(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text())
    outputs = data.get("output", []) if isinstance(data, dict) else []
    docs: list[dict[str, Any]] = []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        text = output.get("text", "")
        if not text:
            continue
        parsed = parse_concatenated_json(text)
        docs.extend(doc for doc in parsed if isinstance(doc, dict))
    return docs


def doc_text(doc: dict[str, Any]) -> str:
    chunks = [str(doc.get("haystack_date", "")), str(doc.get("haystack_session_id", ""))]
    for msg in doc.get("messages", []):
        if isinstance(msg, dict):
            chunks.append(str(msg.get("role", "")))
            chunks.append(str(msg.get("content", "")))
    return "\n".join(chunks)


def matches_answer(text: str) -> bool:
    return any(pattern.search(text) for pattern in ANSWER_PATTERNS)


def evidence_snippets(doc: dict[str, Any], max_snippets: int = 3) -> list[str]:
    snippets = []
    for msg in doc.get("messages", []):
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content", ""))
        if msg.get("has_answer") is True or matches_answer(content):
            snippets.append(
                f"{doc.get('haystack_date')} {doc.get('haystack_session_id')} "
                f"{msg.get('role')}: {content}"
            )
        if len(snippets) >= max_snippets:
            break
    return snippets


def summarize_call(
    call: dict[str, Any],
    order: int,
    tool_result_dir: Path,
) -> ToolCallSummary:
    call_id = str(call.get("id", ""))
    args = parse_tool_input(call.get("input"))
    result_file = tool_result_dir / f"{call_id}.json"
    docs = read_result_docs(result_file)
    sessions = []
    snippets: list[str] = []
    answer_doc_ranks: list[int] = []
    has_answer_flag = False
    has_answer_text = False
    for rank, doc in enumerate(docs, start=1):
        session = doc.get("haystack_session_id")
        if session:
            sessions.append(str(session))
        text = doc_text(doc)
        doc_has_answer_text = matches_answer(text)
        doc_has_answer_flag = False
        has_answer_text = has_answer_text or doc_has_answer_text
        for msg in doc.get("messages", []):
            if isinstance(msg, dict) and msg.get("has_answer") is True:
                doc_has_answer_flag = True
                has_answer_flag = True
        if doc_has_answer_flag or doc_has_answer_text:
            answer_doc_ranks.append(rank)
        snippets.extend(evidence_snippets(doc))

    return ToolCallSummary(
        call_id=call_id,
        name=str(call.get("name", "")),
        query=str(args.get("query", args.get("raw", ""))),
        limit=args.get("limit"),
        order=order,
        result_file=str(result_file) if result_file.exists() else None,
        result_docs=len(docs),
        sessions=list(dict.fromkeys(sessions)),
        has_answer_flag=has_answer_flag,
        has_answer_text=has_answer_text,
        answer_doc_ranks=answer_doc_ranks,
        evidence_snippets=snippets[:5],
    )


def summarize_run(history_file: Path, tool_results_root: Path) -> RunSummary:
    rows = load_jsonl(history_file)
    items = iter_content_items(rows)
    session_id = history_file.stem
    tool_result_dir = tool_results_root / session_id
    tool_calls = [
        summarize_call(item, order, tool_result_dir)
        for order, item in enumerate(items, start=1)
        if item.get("type") == "tool_call" and item.get("name") in {"vector_search", "bm25_search"}
    ]
    return RunSummary(
        session_id=session_id,
        history_file=str(history_file),
        final_answer=extract_final_answer(items),
        tool_calls=tool_calls,
        draft_texts=extract_drafts(items),
    )


def print_run(run: RunSummary) -> None:
    print(f"\n## {run.session_id}")
    print(f"final_answer: {run.final_answer!r}")
    print(f"found_answer_evidence: {run.found_answer}")
    print(f"tool_calls: {len(run.tool_calls)}")
    for call in run.tool_calls:
        marker = "ANSWER" if call.has_answer_flag or call.has_answer_text else "----"
        print(
            f"{call.order:02d} [{marker}] {call.name} "
            f"query={call.query!r} limit={call.limit!r} docs={call.result_docs} "
            f"answer_ranks={call.answer_doc_ranks} sessions={', '.join(call.sessions[:4])}"
        )
        for snippet in call.evidence_snippets:
            print(f"    evidence: {snippet}")
    if run.draft_texts:
        print("drafts:")
        for draft in run.draft_texts:
            compact = " ".join(draft.split())
            print(f"  - {compact[:300]}")


def print_comparison(a: RunSummary, b: RunSummary) -> None:
    print("\n## Comparison")
    for run in (a, b):
        answer_calls = [call for call in run.tool_calls if call.has_answer_flag or call.has_answer_text]
        first = answer_calls[0] if answer_calls else None
        if first:
            print(
                f"{run.session_id}: found evidence at call {first.order} "
                f"({first.name}, query={first.query!r}), final={run.final_answer!r}"
            )
        else:
            print(f"{run.session_id}: no answer evidence found, final={run.final_answer!r}")

    a_queries = [call.query for call in a.tool_calls]
    b_queries = [call.query for call in b.tool_calls]
    print("\nqueries only in first run:")
    for query in a_queries:
        if query and query not in b_queries:
            print(f"  - {query}")
    print("\nqueries only in second run:")
    for query in b_queries:
        if query and query not in a_queries:
            print(f"  - {query}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history_files", nargs=2, type=Path)
    parser.add_argument(
        "--tool-results-root",
        type=Path,
        default=Path("datasets/longmemeval/1/tool_results"),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    first, second = [summarize_run(path, args.tool_results_root) for path in args.history_files]
    print_run(first)
    print_run(second)
    print_comparison(first, second)

    if args.json_out:
        args.json_out.write_text(json.dumps([asdict(first), asdict(second)], indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
