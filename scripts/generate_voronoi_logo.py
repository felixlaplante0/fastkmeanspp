"""Generate reproducible rounded Voronoi-cell paths for the project logo."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

PALETTE = [
    "#6554C0",
    "#3B82C4",
    "#E8795B",
    "#3BA99C",
    "#F0A44B",
    "#8D6BC7",
    "#D65C83",
    "#3E9E72",
    "#F1C453",
    "#D86B9B",
    "#4E87C7",
    "#C85D5D",
    "#5D9BBD",
    "#B873C2",
]


Point = tuple[float, float]
EPSILON = 1e-9
MIN_CELL_VERTICES = 3


def clip(polygon: list[Point], normal: Point, limit: float) -> list[Point]:
    """Clip a polygon to ``normal dot point <= limit``."""
    if not polygon:
        return []
    result: list[Point] = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        start_value = normal[0] * start[0] + normal[1] * start[1] - limit
        end_value = normal[0] * end[0] + normal[1] * end[1] - limit
        start_inside = start_value <= EPSILON
        end_inside = end_value <= EPSILON
        if start_inside:
            result.append(start)
        if start_inside != end_inside:
            fraction = start_value / (start_value - end_value)
            result.append(
                (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
            )
    return result


def polygon_path(polygon: list[Point]) -> str:
    """Return an SVG path for a clipped cell polygon."""
    def point(value: Point) -> str:
        return f"{value[0]:.2f} {value[1]:.2f}"

    commands = [f"M {point(polygon[0])}"]
    commands.extend(f"L {point(vertex)}" for vertex in polygon[1:])
    commands.append("Z")
    return " ".join(commands)


def rounded_square(size: float, radius: float, segments: int = 8) -> list[Point]:
    """Approximate a rounded square with a convex polygon."""
    points: list[Point] = []
    for center_x, center_y, start in (
        (size - radius, radius, -math.pi / 2),
        (size - radius, size - radius, 0),
        (radius, size - radius, math.pi / 2),
        (radius, radius, math.pi),
    ):
        for step in range(segments + 1):
            angle = start + math.pi / 2 * step / segments
            points.append(
                (
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                )
            )
    return points


def cells(seed: int, count: int, size: float, radius: float) -> list[dict[str, object]]:
    """Generate uniformly sampled, clipped Voronoi cells."""
    rng = random.Random(seed)  # noqa: S311 - reproducible visual sampling
    margin = size * 0.06
    sites = [
        (rng.uniform(margin, size - margin), rng.uniform(margin, size - margin))
        for _ in range(count)
    ]
    square = rounded_square(size, radius)
    result = []
    for i, site in enumerate(sites):
        polygon = square
        for j, other in enumerate(sites):
            if i == j:
                continue
            normal = (other[0] - site[0], other[1] - site[1])
            limit = (other[0] ** 2 + other[1] ** 2 - site[0] ** 2 - site[1] ** 2) / 2
            polygon = clip(polygon, normal, limit)
        if len(polygon) < MIN_CELL_VERTICES:
            raise RuntimeError(f"site {i} produced a degenerate cell")
        result.append(
            {
                "path": polygon_path(polygon),
                "color": PALETTE[i % len(PALETTE)],
                "site": site,
            }
        )
    return result


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cells", type=int, default=14)
    parser.add_argument("--size", type=float, default=256)
    parser.add_argument("--corner-radius", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            cells(args.seed, args.cells, args.size, args.corner_radius), indent=2
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
