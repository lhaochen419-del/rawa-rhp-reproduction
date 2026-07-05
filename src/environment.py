from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

from .config import SimulationConfig, stable_seed
from .geometry import Point, Rect, distance, min_clearance
from .wind import OOD_CATEGORIES, WindField


@dataclass
class Target:
    idx: int
    point: Point
    value: float


class InspectionEnvironment:
    def __init__(
        self,
        cfg: SimulationConfig,
        seed: int,
        wind_level: str,
        obstacle_density: str,
        scenario_profile: str = "id",
    ) -> None:
        self.scenario_profile = scenario_profile
        self.ood_category = self._parse_ood_category(scenario_profile)
        self.seed = seed
        self.wind_level = wind_level
        self.obstacle_density = obstacle_density
        self.rng = np.random.default_rng(stable_seed(seed, wind_level, obstacle_density))
        self.cfg = self._profile_config(cfg)
        self.depot: Point = self._generate_depot()
        self.obstacles = self._generate_obstacles(obstacle_density)
        self.occupancy = self._rasterize_obstacles()
        self.clearance_grid = self._make_clearance_grid()
        self.wind = WindField(wind_level, stable_seed(seed, "wind", wind_level), self.cfg.world_size, profile=scenario_profile)
        self.targets = self._generate_targets()
        self.base_cost_grid = self._make_base_cost_grid()

    @property
    def cell_size(self) -> float:
        return self.cfg.cell_size

    def _profile_config(self, cfg: SimulationConfig) -> SimulationConfig:
        if self.ood_category not in {"scale_target_count", "mixed_ood"}:
            return cfg
        world_size = float(self.rng.uniform(50.0, 200.0))
        n_targets = int(self.rng.integers(16, 65))
        grid_size = int(np.clip(round(world_size / 2.0), 32, 96))
        return replace(cfg, world_size=world_size, grid_size=grid_size, n_targets=n_targets)

    def _generate_depot(self) -> Point:
        if self.ood_category not in {"random_depot", "mixed_ood"}:
            return (8.0, 8.0)
        margin = max(5.0, 0.08 * self.cfg.world_size)
        side = int(self.rng.integers(0, 4))
        if side == 0:
            return (float(self.rng.uniform(margin, self.cfg.world_size - margin)), margin)
        if side == 1:
            return (float(self.rng.uniform(margin, self.cfg.world_size - margin)), self.cfg.world_size - margin)
        if side == 2:
            return (margin, float(self.rng.uniform(margin, self.cfg.world_size - margin)))
        return (self.cfg.world_size - margin, float(self.rng.uniform(margin, self.cfg.world_size - margin)))

    def _generate_obstacles(self, density: str) -> List[Rect]:
        if density not in {"sparse", "cluttered"}:
            raise ValueError(f"Unknown density {density}")
        if self.ood_category in {"obstacle_count", "mixed_ood"}:
            count = int(self.rng.integers(5, 26))
        else:
            count = 7 if density == "sparse" else 13
        obstacles: List[Rect] = []
        attempts = 0
        while len(obstacles) < count and attempts < 2000:
            attempts += 1
            if self.ood_category in {"obstacle_count", "nonconvex_obstacles", "narrow_passage", "mixed_ood"}:
                w = float(self.rng.uniform(0.045, 0.14) * self.cfg.world_size)
                h = float(self.rng.uniform(0.045, 0.18) * self.cfg.world_size)
                margin = max(4.0, 0.04 * self.cfg.world_size)
            elif density == "sparse":
                w = float(self.rng.uniform(5.5, 11.0))
                h = float(self.rng.uniform(5.5, 14.0))
                margin = 6.0
            else:
                w = float(self.rng.uniform(6.0, 13.5))
                h = float(self.rng.uniform(6.0, 16.0))
                margin = 6.0
            x = float(self.rng.uniform(margin, self.cfg.world_size - w - margin))
            y = float(self.rng.uniform(margin, self.cfg.world_size - h - margin))
            rect = Rect(x, y, x + w, y + h)
            if rect.contains(self.depot, margin=8.0):
                continue
            if any(_rect_overlap(rect, other, margin=3.5) for other in obstacles):
                continue
            obstacles.append(rect)
            if self.ood_category in {"nonconvex_obstacles", "mixed_ood"} and self.rng.random() < 0.35 and len(obstacles) < count:
                notch_w = max(1.5, 0.35 * w)
                notch_h = max(1.5, 0.30 * h)
                notch = Rect(
                    min(self.cfg.world_size - margin, x + w * 0.65),
                    y,
                    min(self.cfg.world_size - margin, x + w * 0.65 + notch_w),
                    min(self.cfg.world_size - margin, y + notch_h),
                )
                if not notch.contains(self.depot, margin=8.0):
                    obstacles.append(notch)
        if self.ood_category in {"narrow_passage", "mixed_ood"}:
            obstacles = self._add_narrow_passage(obstacles)
        return obstacles

    def _add_narrow_passage(self, obstacles: List[Rect]) -> List[Rect]:
        if len(obstacles) >= 25:
            return obstacles
        gap = float(self.rng.uniform(2.4, 4.8))
        length = float(self.rng.uniform(0.22, 0.38) * self.cfg.world_size)
        thickness = float(self.rng.uniform(2.5, 5.0))
        cx = float(self.rng.uniform(0.30, 0.70) * self.cfg.world_size)
        cy = float(self.rng.uniform(0.30, 0.70) * self.cfg.world_size)
        left = Rect(max(1.0, cx - length / 2.0), cy - gap / 2.0 - thickness, min(self.cfg.world_size - 1.0, cx + length / 2.0), cy - gap / 2.0)
        right = Rect(max(1.0, cx - length / 2.0), cy + gap / 2.0, min(self.cfg.world_size - 1.0, cx + length / 2.0), cy + gap / 2.0 + thickness)
        for rect in (left, right):
            if not rect.contains(self.depot, margin=8.0) and len(obstacles) < 25:
                obstacles.append(rect)
        return obstacles

    def _rasterize_obstacles(self) -> np.ndarray:
        occ = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=bool)
        for gy in range(self.cfg.grid_size):
            for gx in range(self.cfg.grid_size):
                p = self.grid_to_world((gx, gy))
                occ[gy, gx] = any(rect.contains(p) for rect in self.obstacles)
        return occ

    def _make_clearance_grid(self) -> np.ndarray:
        free = ~self.occupancy
        dist_cells = ndimage.distance_transform_edt(free)
        clear = dist_cells * self.cell_size
        for gy in range(self.cfg.grid_size):
            for gx in range(self.cfg.grid_size):
                x, y = self.grid_to_world((gx, gy))
                clear[gy, gx] = min(clear[gy, gx], x, y, self.cfg.world_size - x, self.cfg.world_size - y)
                if self.occupancy[gy, gx]:
                    clear[gy, gx] = 0.0
        return clear

    def _make_base_cost_grid(self) -> np.ndarray:
        centers = np.array(
            [self.grid_to_world((gx, gy)) for gy in range(self.cfg.grid_size) for gx in range(self.cfg.grid_size)]
        )
        wind_speed = np.linalg.norm(self.wind.vector(centers), axis=1).reshape(
            self.cfg.grid_size, self.cfg.grid_size
        )
        clearance_penalty = np.exp(-np.maximum(self.clearance_grid - 1.0, 0.0) / 2.5)
        cost = 1.0 + 0.012 * wind_speed**2 + 0.25 * clearance_penalty
        cost[self.occupancy] = np.inf
        return cost

    def _generate_targets(self) -> List[Target]:
        targets: List[Target] = []
        attempts = 0
        min_target_clearance = 1.20 if self.ood_category != "id" else 1.35
        while len(targets) < self.cfg.n_targets and attempts < 10000:
            attempts += 1
            if self.obstacles and self.rng.random() < 0.78:
                rect = self.obstacles[int(self.rng.integers(0, len(self.obstacles)))]
                point = self._sample_near_rect(rect)
            else:
                point = (
                    float(self.rng.uniform(8.0, self.cfg.world_size - 6.0)),
                    float(self.rng.uniform(8.0, self.cfg.world_size - 6.0)),
                )
            if not self.in_bounds(point) or not self.is_free_world(point):
                continue
            if distance(point, self.depot) < 9.0:
                continue
            if min_clearance(point, self.obstacles, self.cfg.world_size) < min_target_clearance:
                continue
            if any(distance(point, t.point) < 2.0 for t in targets):
                continue
            value = float(self.rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], p=[0.16, 0.20, 0.28, 0.20, 0.16]))
            targets.append(Target(len(targets), point, value))
        if len(targets) < self.cfg.n_targets:
            raise RuntimeError("Could not generate enough reachable target candidates")
        return targets

    @staticmethod
    def _parse_ood_category(profile: str) -> str:
        if profile.startswith("ood:"):
            category = profile.split(":", 1)[1]
            if category not in OOD_CATEGORIES:
                raise ValueError(f"Unknown OOD category {category}")
            return category
        if profile == "ood_extreme":
            return "mixed_ood"
        return "id"

    def _sample_near_rect(self, rect: Rect) -> Point:
        side = int(self.rng.integers(0, 4))
        offset = float(self.rng.uniform(1.6, 5.6 if self.obstacle_density == "sparse" else 4.8))
        if side == 0:
            return (float(self.rng.uniform(rect.xmin, rect.xmax)), rect.ymin - offset)
        if side == 1:
            return (float(self.rng.uniform(rect.xmin, rect.xmax)), rect.ymax + offset)
        if side == 2:
            return (rect.xmin - offset, float(self.rng.uniform(rect.ymin, rect.ymax)))
        return (rect.xmax + offset, float(self.rng.uniform(rect.ymin, rect.ymax)))

    def in_bounds(self, point: Point) -> bool:
        x, y = point
        return 0.0 <= x <= self.cfg.world_size and 0.0 <= y <= self.cfg.world_size

    def is_free_world(self, point: Point) -> bool:
        if not self.in_bounds(point):
            return False
        gx, gy = self.world_to_grid(point)
        return not bool(self.occupancy[gy, gx])

    def grid_to_world(self, cell: Tuple[int, int]) -> Point:
        gx, gy = cell
        s = self.cell_size
        return ((gx + 0.5) * s, (gy + 0.5) * s)

    def world_to_grid(self, point: Point) -> Tuple[int, int]:
        x, y = point
        gx = int(np.clip(x / self.cell_size, 0, self.cfg.grid_size - 1))
        gy = int(np.clip(y / self.cell_size, 0, self.cfg.grid_size - 1))
        return gx, gy

    def clearance_at(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        scalar = False
        if pts.ndim == 1:
            pts = pts[None, :]
            scalar = True
        gx = np.clip((pts[:, 0] / self.cell_size).astype(int), 0, self.cfg.grid_size - 1)
        gy = np.clip((pts[:, 1] / self.cell_size).astype(int), 0, self.cfg.grid_size - 1)
        vals = self.clearance_grid[gy, gx]
        if scalar:
            return vals[0]
        return vals

    def nodes(self) -> List[Point]:
        return [self.depot] + [t.point for t in self.targets]

    def total_target_value(self) -> float:
        return float(sum(t.value for t in self.targets))


def _rect_overlap(a: Rect, b: Rect, margin: float = 0.0) -> bool:
    return not (
        a.xmax + margin < b.xmin
        or b.xmax + margin < a.xmin
        or a.ymax + margin < b.ymin
        or b.ymax + margin < a.ymin
    )
