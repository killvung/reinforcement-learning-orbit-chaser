"""Pure collision geometry; this module has no environment state."""

from __future__ import annotations
import math
from .rules import ARENA_CENTER, ARENA_RADIUS, BAR_WIDTH, CORE_RADIUS


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return the straight-line Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_segment_distance(point, start, end) -> float:
    """Return the shortest distance from a point to a finite line segment.

    Bars are represented by their centre-line segment. The closest point is
    clamped to the segment endpoints, so this also handles bar end-caps.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    squared_length = dx * dx + dy * dy
    if squared_length == 0:
        return distance(point, start)

    segment_offset = (point[0] - start[0], point[1] - start[1])
    projection = (segment_offset[0] * dx + segment_offset[1] * dy) / squared_length

    # Clamping keeps the nearest point on the finite bar, including its ends.
    segment_fraction = max(0.0, min(1.0, projection))
    nearest_point = (
        start[0] + segment_fraction * dx,
        start[1] + segment_fraction * dy,
    )
    return distance(point, nearest_point)


def is_blocked(point, body_radius: float, bars) -> bool:
    """Report whether a circular actor would overlap any arena obstacle.

    The actor is blocked by the outer boundary, central core, or either bar.
    `body_radius` expands each obstacle so the calculation works for both the
    player and the larger enemy without duplicating collision rules.
    """
    if (
        distance(point, ARENA_CENTER) > ARENA_RADIUS - body_radius
        or distance(point, ARENA_CENTER) < CORE_RADIUS + body_radius
    ):
        return True
    return any(
        point_segment_distance(point, start, end) < BAR_WIDTH / 2 + body_radius
        for start, end in bars
    )
