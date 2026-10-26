"""Render the fixed-set commit score/token trajectory as an editable SVG."""

from __future__ import annotations

from math import hypot
from pathlib import Path


# Source: main_result.md, "v1-v6 (BEAM 100K full fixed set)" table.
POINTS = [
    ("init", "a59889b", 55_891, 0.6460),
    ("v1", "60e6b67", 43_721, 0.6928),
    ("v2", "b45f302", 38_412, 0.7288),
    ("v3", "d245159", 31_846, 0.7214),
    ("v4", "45818d5", 35_293, 0.7256),
    ("v5", "7ab2b05", 38_670, 0.7193),
    ("v6", "e3220ef", 33_675, 0.7207),
]

WIDTH, HEIGHT = 700, 455
LEFT, RIGHT, TOP, BOTTOM = 100, 665, 40, 390
X_MIN, X_MAX = 30_000, 58_000
Y_MIN, Y_MAX = 0.640, 0.735
NODE_RADIUS = 17


def scale_x(value: int) -> float:
    return LEFT + (value - X_MIN) * (RIGHT - LEFT) / (X_MAX - X_MIN)


def scale_y(value: float) -> float:
    return BOTTOM - (value - Y_MIN) * (BOTTOM - TOP) / (Y_MAX - Y_MIN)


def unit(dx: float, dy: float) -> tuple[float, float]:
    length = hypot(dx, dy)
    return dx / length, dy / length


def arrow(start: tuple[float, float], end: tuple[float, float], control=None) -> str:
    """Draw a connector and a visible arrowhead outside the point circles."""
    sx, sy = start
    ex, ey = end
    if control is None:
        control = ((sx + ex) / 2, (sy + ey) / 2)
    cx, cy = control
    start_dx, start_dy = unit(cx - sx, cy - sy)
    end_dx, end_dy = unit(ex - cx, ey - cy)
    sx += (NODE_RADIUS + 3) * start_dx
    sy += (NODE_RADIUS + 3) * start_dy
    ex -= (NODE_RADIUS + 5) * end_dx
    ey -= (NODE_RADIUS + 5) * end_dy
    wing = 5.0
    length = 10.0
    bx, by = ex - length * end_dx, ey - length * end_dy
    nx, ny = -end_dy, end_dx
    return (
        f'<path d="M {sx:.1f},{sy:.1f} Q {cx:.1f},{cy:.1f} {ex:.1f},{ey:.1f}" '
        'fill="none" stroke="#55748a" stroke-width="2.1" stroke-linecap="round"/>'
        f'<path d="M {ex:.1f},{ey:.1f} L {bx + wing * nx:.1f},{by + wing * ny:.1f} '
        f'L {bx - wing * nx:.1f},{by - wing * ny:.1f} Z" fill="#55748a"/>'
    )


def is_pareto(tokens: int, score: float) -> bool:
    """A point is Pareto optimal if no other uses fewer tokens with at least its score."""
    return not any(
        other_tokens <= tokens
        and other_score >= score
        and (other_tokens < tokens or other_score > score)
        for _, _, other_tokens, other_score in POINTS
    )


def render() -> str:
    positions = [(scale_x(tokens), scale_y(score)) for _, _, tokens, score in POINTS]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        'aria-label="Mean query score versus tokens per query for commits init through v6, '
        'with arrows showing commit evolution">',
        '<style>'
        'text{font-family:Arial,Helvetica,sans-serif;fill:#243441}'
        '.tick{font-size:21px;fill:#52616d}'
        '.axis-title{font-size:25px;font-weight:600}'
        '.node-number{font-size:21px;font-weight:600;fill:#126d9a}'
        '.node-number.frontier{fill:#c43c39}'
        '.note{font-size:22px;fill:#52616d}'
        '</style>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
    ]

    for tick in (0.64, 0.66, 0.68, 0.70, 0.72):
        y = scale_y(tick)
        parts.extend(
            [
                f'<line x1="{LEFT}" y1="{y:.1f}" x2="{RIGHT}" y2="{y:.1f}" '
                'stroke="#e2e7eb" stroke-width="1"/>',
                f'<text x="{LEFT - 13}" y="{y + 5:.1f}" text-anchor="end" '
                f'class="tick">{tick:.2f}</text>',
            ]
        )
    for tick in (30_000, 35_000, 40_000, 45_000, 50_000, 55_000):
        x = scale_x(tick)
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{BOTTOM}" x2="{x:.1f}" y2="{BOTTOM + 7}" '
                'stroke="#52616d" stroke-width="1.5"/>',
                f'<text x="{x:.1f}" y="{BOTTOM + 27}" text-anchor="middle" '
                f'class="tick">{tick // 1000}k</text>',
            ]
        )
    parts.extend(
        [
            f'<path d="M {LEFT},{TOP} V {BOTTOM} H {RIGHT}" fill="none" '
            'stroke="#52616d" stroke-width="1.6"/>',
            f'<path d="M {LEFT},{TOP - 9} L {LEFT - 4},{TOP + 2} L {LEFT + 4},{TOP + 2} Z" '
            'fill="#52616d"/>',
            f'<path d="M {RIGHT + 9},{BOTTOM} L {RIGHT - 2},{BOTTOM - 4} '
            f'L {RIGHT - 2},{BOTTOM + 4} Z" fill="#52616d"/>',
            f'<text x="{(LEFT + RIGHT) / 2:.1f}" y="{HEIGHT - 18}" '
            'text-anchor="middle" class="axis-title">Query Token Consumption</text>',
            f'<text transform="translate(27 {(TOP + BOTTOM) / 2:.1f}) rotate(-90)" '
            'text-anchor="middle" class="axis-title">Score</text>',
            '<circle cx="455" cy="80" r="12" fill="white" stroke="#c43c39" stroke-width="2.5"/>',
            '<text x="480" y="87" class="note">Pareto Frontier</text>',
        ]
    )

    # Bend crowded transitions so they do not run through neighboring circles.
    controls = {
        2: (scale_x(35_000), scale_y(0.737)),  # v2 → v3
        3: (scale_x(33_500), scale_y(0.735)),  # v3 → v4
        5: (scale_x(36_000), scale_y(0.712)),  # v5 → v6
    }
    for i in range(len(positions) - 1):
        parts.append(arrow(positions[i], positions[i + 1], controls.get(i)))
    for index, ((_, _, tokens, score), (x, y)) in enumerate(zip(POINTS, positions)):
        frontier = is_pareto(tokens, score)
        color = '#c43c39' if frontier else '#126d9a'
        number_class = 'node-number frontier' if frontier else 'node-number'
        parts.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{NODE_RADIUS}" '
                f'fill="white" stroke="{color}" stroke-width="3"/>',
                f'<text x="{x:.1f}" y="{y + 6:.1f}" text-anchor="middle" class="{number_class}">'
                f'<tspan font-style="italic">h</tspan><tspan dy="4" font-size="13">{index}</tspan></text>',
            ]
        )

    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


if __name__ == '__main__':
    Path(__file__).with_name('commit_score_token_evolution.svg').write_text(render(), encoding='utf-8')
