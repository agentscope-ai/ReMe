#!/usr/bin/env python3
"""Report daily new GitHub stars for a repository using the GitHub CLI."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO
from xml.sax.saxutils import escape

DEFAULT_REPOSITORY = "agentscope-ai/ReMe"
QUERY = """
query($owner: String!, $name: String!, $before: String) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    stargazerCount
    stargazers(last: 100, before: $before) {
      edges {
        starredAt
      }
      pageInfo {
        hasPreviousPage
        startCursor
      }
    }
  }
}
"""


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Print daily new-star counts for the latest N UTC calendar days.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPOSITORY,
        metavar="OWNER/REPO",
        help=f"GitHub repository (default: {DEFAULT_REPOSITORY})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="number of UTC calendar days to include (default: 365)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help="write CSV to PATH instead of standard output",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        metavar="PATH",
        help="write the SVG chart to PATH (default: CSV path with an .svg suffix)",
    )
    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be at least 1")
    if args.repo.count("/") != 1 or any(not part for part in args.repo.split("/")):
        parser.error("--repo must have the form OWNER/REPO")
    return args


def query_github(owner: str, name: str, before: str | None) -> dict[str, Any]:
    """Fetch one page of repository stargazers through ``gh api graphql``."""
    request = {
        "query": QUERY,
        "variables": {"owner": owner, "name": name, "before": before},
    }
    try:
        result = subprocess.run(
            ["gh", "api", "graphql", "--input", "-"],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("GitHub CLI 'gh' is not installed or is not on PATH") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"gh api graphql failed: {detail}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("gh returned invalid JSON") from exc

    if payload.get("errors"):
        messages = "; ".join(error.get("message", str(error)) for error in payload["errors"])
        raise RuntimeError(f"GitHub GraphQL API error: {messages}")

    repository = payload.get("data", {}).get("repository")
    if repository is None:
        raise RuntimeError(f"repository not found or inaccessible: {owner}/{name}")
    return repository


def fetch_daily_stars(repository: str, start_date: date) -> tuple[Counter[date], int]:
    """Fetch current stargazers and group their star timestamps by UTC date."""
    owner, name = repository.split("/", maxsplit=1)
    daily_stars: Counter[date] = Counter()
    before: str | None = None
    total_stars = 0

    while True:
        data = query_github(owner, name, before)
        total_stars = data["stargazerCount"]
        connection = data["stargazers"]
        edges = connection["edges"]

        reached_start = False
        for edge in edges:
            starred_at = datetime.fromisoformat(edge["starredAt"].replace("Z", "+00:00"))
            starred_date = starred_at.astimezone(UTC).date()
            if starred_date < start_date:
                reached_start = True
                continue
            daily_stars[starred_date] += 1

        page_info = connection["pageInfo"]
        if reached_start or not page_info["hasPreviousPage"]:
            break
        before = page_info["startCursor"]
        if before is None:
            raise RuntimeError("GitHub returned an invalid pagination cursor")

    return daily_stars, total_stars


def write_csv(stream: TextIO, start_date: date, days: int, counts: Counter[date]) -> None:
    """Write a row for every date in the requested period, including zero days."""
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["date", "new_stars"])
    for offset in range(days):
        current_date = start_date + timedelta(days=offset)
        writer.writerow([current_date.isoformat(), counts[current_date]])


def nice_tick_step(maximum: int, tick_count: int = 5) -> int:
    """Return a readable positive interval for numeric axis ticks."""
    rough_step = max(maximum, 1) / tick_count
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    if normalized <= 1:
        multiplier = 1
    elif normalized <= 2:
        multiplier = 2
    elif normalized <= 5:
        multiplier = 5
    else:
        multiplier = 10
    return max(1, multiplier * magnitude)


def write_svg_chart(path: Path, repository: str, dates: list[date], values: list[int]) -> None:
    """Render daily star increments as a dependency-free SVG bar chart."""
    width, height = 1280, 640
    margin_left, margin_right = 75, 30
    margin_top, margin_bottom = 70, 85
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    tick_step = nice_tick_step(max(values, default=0))
    axis_max = max(tick_step, math.ceil(max(values, default=0) / tick_step) * tick_step)
    bar_slot = plot_width / len(values)
    bar_width = max(0.8, bar_slot * 0.82)

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">'
        ),
        f"<title>{escape(repository)} daily new GitHub stars</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="sans-serif" '
            f'font-size="22" font-weight="600" fill="#24292f">{escape(repository)} daily new stars</text>'
        ),
    ]

    for tick in range(0, axis_max + 1, tick_step):
        y = margin_top + plot_height - (tick / axis_max * plot_height)
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" '
            'stroke="#d8dee4" stroke-width="1"/>',
        )
        svg.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" '
            f'font-size="12" fill="#57606a">{tick}</text>',
        )

    for index, (current_date, value) in enumerate(zip(dates, values, strict=True)):
        bar_height = value / axis_max * plot_height
        x = margin_left + index * bar_slot + (bar_slot - bar_width) / 2
        y = margin_top + plot_height - bar_height
        svg.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
            f'fill="#2f81f7"><title>{current_date.isoformat()}: {value} new stars</title></rect>',
        )

    label_count = min(12, len(dates))
    label_indexes = sorted({round(index * (len(dates) - 1) / max(label_count - 1, 1)) for index in range(label_count)})
    baseline = margin_top + plot_height
    for index in label_indexes:
        x = margin_left + (index + 0.5) * bar_slot
        svg.append(
            f'<line x1="{x:.2f}" y1="{baseline}" x2="{x:.2f}" y2="{baseline + 5}" stroke="#57606a"/>',
        )
        svg.append(
            f'<text x="{x:.2f}" y="{baseline + 20}" text-anchor="end" '
            f'transform="rotate(-35 {x:.2f} {baseline + 20})" font-family="sans-serif" '
            f'font-size="11" fill="#57606a">{dates[index].isoformat()}</text>',
        )

    svg.extend(
        [
            (
                f'<line x1="{margin_left}" y1="{baseline}" x2="{width - margin_right}" y2="{baseline}" '
                'stroke="#57606a"/>'
            ),
            (
                f'<text x="18" y="{margin_top + plot_height / 2}" text-anchor="middle" '
                f'transform="rotate(-90 18 {margin_top + plot_height / 2})" font-family="sans-serif" '
                'font-size="13" fill="#24292f">New stars</text>'
            ),
            "</svg>",
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> int:
    """Run the report and return a process exit code."""
    args = parse_args()
    end_date = datetime.now(UTC).date()
    start_date = end_date - timedelta(days=args.days - 1)

    try:
        counts, total_stars = fetch_daily_stars(args.repo, start_date)
        dates = [start_date + timedelta(days=offset) for offset in range(args.days)]
        values = [counts[current_date] for current_date in dates]
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8", newline="") as stream:
                write_csv(stream, start_date, args.days, counts)
        else:
            write_csv(sys.stdout, start_date, args.days, counts)
        chart_path = args.chart or (args.output.with_suffix(".svg") if args.output else Path("star_growth.svg"))
        write_svg_chart(chart_path, args.repo, dates, values)
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    destination = str(args.output) if args.output else "stdout"
    period_stars = sum(counts.values())
    print(
        f"Fetched {period_stars} current stargazers in {start_date}..{end_date}; "
        f"repository currently has {total_stars} stars. CSV: {destination}; chart: {chart_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
