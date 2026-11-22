"""The reliability diagram as a small, byte-stable SVG (no plotting dependency).

Drawn from the calibration snapshot's walk-forward rows: for each of raw,
Platt and isotonic, the ten equal-mass bins as points (mean forecast
against observed frequency), joined by a line, over the diagonal a
perfectly calibrated forecaster would sit on. Colours are the first three
slots of a colour-blind-checked categorical palette; each series also has
its own marker shape, a legend and a direct label, so identity never rests
on colour alone. The same numbers are in RESULTS.md as a table.
"""

from __future__ import annotations

from typing import Any

SERIES: tuple[tuple[str, str, str, str], ...] = (
    # (calibrator, label, colour, marker)
    ("raw", "raw", "#2a78d6", "circle"),
    ("platt", "Platt", "#eb6834", "square"),
    ("isotonic", "isotonic", "#1baf7a", "diamond"),
)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

WIDTH, HEIGHT = 560, 520
LEFT, RIGHT, TOP, BOTTOM = 64, 150, 48, 56
PLOT_W = WIDTH - LEFT - RIGHT
PLOT_H = HEIGHT - TOP - BOTTOM


def _x(p: float) -> float:
    return LEFT + p * PLOT_W


def _y(p: float) -> float:
    return TOP + (1.0 - p) * PLOT_H


def _marker(kind: str, x: float, y: float, colour: str) -> str:
    ring = f'stroke="{SURFACE}" stroke-width="2"'
    if kind == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colour}" {ring}/>'
    if kind == "square":
        return (
            f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" rx="2" '
            f'fill="{colour}" {ring}/>'
        )
    pts = f"{x:.1f},{y - 6:.1f} {x + 6:.1f},{y:.1f} {x:.1f},{y + 6:.1f} {x - 6:.1f},{y:.1f}"
    return f'<polygon points="{pts}" fill="{colour}" {ring}/>'


def reliability_svg(snap: dict[str, Any]) -> str:
    """The walk-forward reliability diagram of ``snap`` (``augur.q4.snapshot``)."""
    w = snap["walk_forward"]
    rows = {r["calibrator"]: r for r in w["rows"]}
    title = "Reliability on the walk-forward test"
    subtitle = f"{w['instances']} questions, {snap['corpus']['model'].split('/')[-1]}, no retrieval"
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="system-ui, sans-serif" font-size="12" '
        f'role="img" aria-label="{title}">',
        f"<title>{title}</title>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{SURFACE}"/>',
        f'<text x="{LEFT}" y="20" fill="{INK}" font-size="14" font-weight="600">{title}</text>',
        f'<text x="{LEFT}" y="36" fill="{MUTED}" font-size="11">{subtitle}</text>',
    ]
    for t in range(0, 11, 2):
        v = t / 10
        out.append(
            f'<line x1="{_x(0):.1f}" y1="{_y(v):.1f}" x2="{_x(1):.1f}" y2="{_y(v):.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{LEFT - 8}" y="{_y(v) + 4:.1f}" fill="{MUTED}" '
            f'text-anchor="end">{v:.1f}</text>'
        )
        out.append(
            f'<text x="{_x(v):.1f}" y="{TOP + PLOT_H + 18}" fill="{MUTED}" '
            f'text-anchor="middle">{v:.1f}</text>'
        )
    out.append(
        f'<line x1="{_x(0):.1f}" y1="{_y(0):.1f}" x2="{_x(1):.1f}" y2="{_y(1):.1f}" '
        f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 4"/>'
    )
    out.append(
        f'<text x="{_x(0.80):.1f}" y="{_y(0.83):.1f}" fill="{MUTED}" font-size="11" '
        f'transform="rotate(-42 {_x(0.80):.1f} {_y(0.83):.1f})">calibrated</text>'
    )
    out.append(
        f'<text x="{LEFT + PLOT_W / 2:.1f}" y="{HEIGHT - 16}" fill="{INK}" '
        'text-anchor="middle">mean forecast in the bin (10 equal-mass bins)</text>'
    )
    out.append(
        f'<text x="18" y="{TOP + PLOT_H / 2:.1f}" fill="{INK}" text-anchor="middle" '
        f'transform="rotate(-90 18 {TOP + PLOT_H / 2:.1f})">share that resolved Yes</text>'
    )
    for i, (name, label, colour, marker) in enumerate(SERIES):
        row = rows.get(name)
        if row is None:
            continue
        pts = [(_x(b["mean_p"]), _y(b["freq"])) for b in row["bins"]]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(
            f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="2" '
            'stroke-linejoin="round"/>'
        )
        out.extend(_marker(marker, x, y, colour) for x, y in pts)
        # Direct label at the line's right end, and a legend entry with the ECE.
        lx, ly = pts[-1]
        out.append(
            f'<text x="{lx + 9:.1f}" y="{ly + 4 + (i - 1) * 12:.1f}" fill="{INK}">{label}</text>'
        )
        gy = TOP + 20 + i * 36
        gx = LEFT + PLOT_W + 24
        out.append(_marker(marker, gx, gy - 4, colour))
        out.append(
            f'<text x="{gx + 12}" y="{gy}" fill="{INK}">{label}</text>'
            f'<text x="{gx + 12}" y="{gy + 13}" fill="{MUTED}" font-size="11">'
            f"ECE {row['ece']:.3f}</text>"
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"
