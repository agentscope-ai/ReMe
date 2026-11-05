"""Stdlib-only, subprocess-isolated Python regex matching for committed text."""

from bisect import bisect_right
from itertools import accumulate
import json
import re
import sys


def search(files: dict[str, str], pattern: str, max_results: int) -> dict:
    """Merge adjacent contexts, retaining at most five lines per displayed block."""
    compiled = re.compile(pattern, re.MULTILINE)
    blocks = []
    occurrences = 0
    for path, text in files.items():
        lines = text.splitlines()
        starts = [0, *accumulate(map(len, text.splitlines(keepends=True)))]
        for match in compiled.finditer(text):
            if match.start() == match.end():
                raise ValueError("pattern must consume at least one character")
            occurrences += 1
            if occurrences > 100_000:
                raise ValueError("Too many regex matches; narrow pattern or path")
            first = bisect_right(starts, match.start())
            last = bisect_right(starts, match.end() - 1)
            start, end = max(1, first - 1), min(len(lines), last + 1)
            if (
                blocks
                and blocks[-1]["path"] == path
                and start <= blocks[-1]["end"] + 1
                and (max(end, blocks[-1]["end"]) - blocks[-1]["start"] < 5)
            ):
                block = blocks[-1]
                block["end"] = max(end, block["end"])
                block["ranges"].append((first, last))
            else:
                blocks.append({"path": path, "start": start, "end": end, "ranges": [(first, last)]})
    matches = []
    characters = 0
    for block in blocks[:max_results]:
        lines = files[block["path"]].splitlines()
        displayed_end = min(block["end"], block["start"] + 4)
        content = "\n".join(
            f"{'>' if any(first <= index <= last for first, last in block['ranges']) else ' '} "
            f"{index} | {lines[index - 1]}"
            for index in range(block["start"], displayed_end + 1)
        )
        remaining = 100_000 - characters
        if remaining <= 0:
            break
        content = content[:remaining]
        characters += len(content)
        matches.append(
            {
                "path": block["path"],
                "start_line": block["start"],
                "end_line": block["end"],
                "displayed_end_line": displayed_end,
                "occurrence_count": len(block["ranges"]),
                "content": content,
                "content_truncated": displayed_end < block["end"] or characters >= 100_000,
            },
        )
    return {
        "occurrence_count": occurrences,
        "merged_match_count": len(blocks),
        "returned_match_count": len(matches),
        "matches": matches,
        "truncated": len(blocks) > len(matches) or any(match["content_truncated"] for match in matches),
    }


def main():
    """Emit structured errors without candidate text or traceback noise."""
    arguments = json.load(sys.stdin)
    try:
        result = search(**arguments)
    except (ValueError, re.error) as exc:
        result = {"error": str(exc)}
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
