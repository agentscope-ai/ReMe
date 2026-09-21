#!/usr/bin/env python3
"""Report daily new GitHub stars and the cumulative growth curve for a repository."""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO
from xml.sax.saxutils import escape

DEFAULT_REPOSITORY = "agentscope-ai/ReMe"
DEFAULT_PERIOD = "90d"
PERIOD_PATTERN = re.compile(r"(?P<amount>\d+)\s*(?P<unit>[dwmy])?")
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


def parse_period(period: str) -> tuple[int, str]:
    """Split a period such as ``3m`` into its amount and its unit (default ``d``)."""
    match = PERIOD_PATTERN.fullmatch(period.strip().lower())
    if match is None:
        raise ValueError(
            f"invalid period {period!r}: use a number with an optional d/w/m/y suffix, e.g. 90, 30d, 8w, 3m, 1y",
        )
    amount = int(match["amount"])
    if amount < 1:
        raise ValueError(f"invalid period {period!r}: the amount must be at least 1")
    return amount, match["unit"] or "d"


def period_start(period: str, end_date: date) -> date:
    """Return the inclusive first day of ``period``, counting back from ``end_date``.

    ``d``/``w`` are exact day counts; ``m``/``y`` are calendar months, so ``3m`` ending on
    the 21st starts on the 21st of the month three months earlier.
    """
    amount, unit = parse_period(period)
    if unit == "d":
        return end_date - timedelta(days=amount - 1)
    if unit == "w":
        return end_date - timedelta(days=amount * 7 - 1)
    months = amount if unit == "m" else amount * 12
    month_index = end_date.month - 1 - months
    year, month = end_date.year + month_index // 12, month_index % 12 + 1
    return date(year, month, min(end_date.day, calendar.monthrange(year, month)[1]))


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Print daily new-star counts and the cumulative star-growth chart for a time window.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPOSITORY,
        metavar="OWNER/REPO",
        help=f"GitHub repository (default: {DEFAULT_REPOSITORY})",
    )
    parser.add_argument(
        "--period",
        metavar="PERIOD",
        help=(
            "time window to plot, counted back from today UTC: a number with an optional "
            f"d/w/m/y suffix, e.g. 90, 30d, 8w, 3m, 1y (default: {DEFAULT_PERIOD})"
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        metavar="N",
        help="alias for --period N, i.e. the last N days including today",
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

    if args.period and args.days is not None:
        parser.error("--period and --days are mutually exclusive")
    period = args.period or (f"{args.days}d" if args.days is not None else DEFAULT_PERIOD)
    try:
        parse_period(period)
    except ValueError as exc:
        parser.error(str(exc))
    args.period = period
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


def cumulative_stars(values: list[int], total_stars: int) -> list[int]:
    """Rebuild the running star total per day from the latest total and daily increments."""
    totals = [0] * len(values)
    running = total_stars
    for index in range(len(values) - 1, -1, -1):
        totals[index] = running
        running -= values[index]
    return totals


def point_x(index: int, count: int, plot_width: float, margin_left: float) -> float:
    """Return the horizontal position of the ``index``-th point on the plot."""
    if count <= 1:
        return margin_left + plot_width / 2
    return margin_left + index * plot_width / (count - 1)


def write_svg_chart(path: Path, repository: str, dates: list[date], totals: list[int]) -> None:
    """Render the cumulative star count as a dependency-free SVG line chart.

    The vertical axis is cropped to the neighbourhood of the data instead of starting at
    zero, so a few hundred new stars stay readable next to a few thousand existing ones.
    """
    width, height = 1280, 640
    margin_left, margin_right = 75, 30
    margin_top, margin_bottom = 70, 85
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    baseline = margin_top + plot_height

    peak = max(totals, default=0)
    lowest = min(totals, default=0)
    tick_step = nice_tick_step(peak - lowest)
    axis_min = (lowest // tick_step) * tick_step
    if axis_min >= lowest:
        axis_min = max(0, axis_min - tick_step)
    axis_max = max(tick_step, math.ceil(peak / tick_step) * tick_step)
    if axis_max <= peak:
        axis_max += tick_step
    axis_max = max(axis_max, axis_min + tick_step)
    span = axis_max - axis_min

    def value_y(value: int) -> float:
        """Map a star total onto its vertical position inside the plot."""
        return baseline - (value - axis_min) / span * plot_height

    points = [
        (point_x(index, len(totals), plot_width, margin_left), value_y(total)) for index, total in enumerate(totals)
    ]
    slot = plot_width / max(len(totals) - 1, 1)

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">'
        ),
        f"<title>{escape(repository)} GitHub star growth</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="sans-serif" '
            f'font-size="22" font-weight="600" fill="#24292f">{escape(repository)} star growth</text>'
        ),
        (
            f'<text x="{width / 2}" y="54" text-anchor="middle" font-family="sans-serif" '
            f'font-size="13" fill="#57606a">{dates[0].isoformat()} to {dates[-1].isoformat()} '
            f"({len(dates)} days)</text>"
        ),
    ]

    for tick in range(axis_min, axis_max + 1, tick_step):
        y = value_y(tick)
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" '
            'stroke="#d8dee4" stroke-width="1"/>',
        )
        svg.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" '
            f'font-size="12" fill="#57606a">{tick:,}</text>',
        )

    if points:
        area = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        svg.append(
            f'<polygon points="{margin_left:.2f},{baseline:.2f} {area} '
            f'{points[-1][0]:.2f},{baseline:.2f}" fill="#2f81f7" fill-opacity="0.14"/>',
        )
        curve = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        svg.append(
            f'<polyline points="{curve}" fill="none" stroke="#2f81f7" stroke-width="2.5" '
            'stroke-linejoin="round" stroke-linecap="round"/>',
        )
        svg.append(
            f'<circle cx="{points[-1][0]:.2f}" cy="{points[-1][1]:.2f}" r="4" fill="#2f81f7"/>',
        )

    for index, (current_date, total) in enumerate(zip(dates, totals, strict=True)):
        x = points[index][0]
        svg.append(
            f'<rect x="{x - slot / 2:.2f}" y="{margin_top}" width="{max(slot, 1):.2f}" '
            f'height="{plot_height:.2f}" fill="transparent">'
            f"<title>{current_date.isoformat()}: {total:,} stars</title></rect>",
        )

    label_count = min(12, len(dates))
    label_indexes = sorted({round(index * (len(dates) - 1) / max(label_count - 1, 1)) for index in range(label_count)})
    for index in label_indexes:
        x = points[index][0]
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
                'font-size="13" fill="#24292f">Stars</text>'
            ),
        ],
    )
    if points:
        svg.append(
            f'<text x="{points[-1][0] - 8:.2f}" y="{points[-1][1] - 10:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="13" font-weight="600" fill="#24292f">'
            f"{totals[-1]:,}</text>",
        )
    svg.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> int:
    """Run the report and return a process exit code."""
    args = parse_args()
    end_date = datetime.now(UTC).date()
    start_date = period_start(args.period, end_date)
    days = (end_date - start_date).days + 1

    try:
        counts, total_stars = fetch_daily_stars(args.repo, start_date)
        dates = [start_date + timedelta(days=offset) for offset in range(days)]
        values = [counts[current_date] for current_date in dates]
        totals = cumulative_stars(values, total_stars)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8", newline="") as stream:
                write_csv(stream, start_date, days, counts)
        else:
            write_csv(sys.stdout, start_date, days, counts)
        chart_path = args.chart or (args.output.with_suffix(".svg") if args.output else Path("star_growth.svg"))
        write_svg_chart(chart_path, args.repo, dates, totals)
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    destination = str(args.output) if args.output else "stdout"
    period_stars = sum(counts.values())
    print(
        f"Fetched {period_stars} current stargazers in {start_date}..{end_date} ({days} days); "
        f"repository currently has {total_stars} stars "
        f"(grew from {totals[0] if totals else total_stars} on {start_date}). "
        f"CSV: {destination}; chart: {chart_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
