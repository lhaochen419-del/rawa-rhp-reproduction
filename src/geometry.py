from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


Point = Tuple[float, float]


@dataclass(frozen=True)
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center(self) -> Point:
        return ((self.xmin + self.xmax) * 0.5, (self.ymin + self.ymax) * 0.5)

    def contains(self, p: Point, margin: float = 0.0) -> bool:
        x, y = p
        return (
            self.xmin - margin <= x <= self.xmax + margin
            and self.ymin - margin <= y <= self.ymax + margin
        )


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_length(path: Sequence[Point]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(distance(a, b) for a, b in zip(path[:-1], path[1:])))


def point_rect_distance(point: Point, rect: Rect) -> float:
    x, y = point
    dx = max(rect.xmin - x, 0.0, x - rect.xmax)
    dy = max(rect.ymin - y, 0.0, y - rect.ymax)
    return math.hypot(dx, dy)


def point_bounds_distance(point: Point, world_size: float) -> float:
    x, y = point
    return min(x, y, world_size - x, world_size - y)


def min_clearance(point: Point, obstacles: Iterable[Rect], world_size: float) -> float:
    d = point_bounds_distance(point, world_size)
    for rect in obstacles:
        d = min(d, point_rect_distance(point, rect))
    return max(0.0, d)


def segment_midpoints(path: Sequence[Point]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(path) < 2:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    p0 = np.asarray(path[:-1], dtype=float)
    p1 = np.asarray(path[1:], dtype=float)
    delta = p1 - p0
    lengths = np.linalg.norm(delta, axis=1)
    keep = lengths > 1e-9
    if not np.any(keep):
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    dirs = delta[keep] / lengths[keep, None]
    mids = (p0[keep] + p1[keep]) * 0.5
    return mids, dirs, lengths[keep]


def turn_angle_sum(path: Sequence[Point]) -> float:
    if len(path) < 3:
        return 0.0
    _, dirs, _ = segment_midpoints(path)
    if len(dirs) < 2:
        return 0.0
    dots = np.sum(dirs[:-1] * dirs[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return float(np.sum(np.arccos(dots)))


def route_to_string(points: Sequence[Point]) -> str:
    return "|".join(f"{x:.3f}:{y:.3f}" for x, y in points)


def route_from_string(text: str) -> List[Point]:
    if not text:
        return []
    points: List[Point] = []
    for item in text.split("|"):
        x, y = item.split(":")
        points.append((float(x), float(y)))
    return points
